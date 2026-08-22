"""Worker/model configuration and the shared BasePipeline machinery."""

from __future__ import annotations

import asyncio
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Literal, Tuple, Set

from libs.evaluator import EvaluationResult
from libs.storage import load_conversations
from libs.models import get_sampler
from libs.types import SamplerBase
from libs.browser_fetcher import close_shared_client
from libs.information_extraction import close_pdf_session

from ..logging_config import get_logger
from ..core import MonitoredQueue, Pipeline, QueueMonitor
from ..core.worker import Worker
from ..core.domain_strategy import DomainStrategy
from ..strategies import get_strategy
from ..models import (
    ConversationItem,
    ClaimItem,
    SearchTask,
    ContentItem,
    FilteredContent,
    JudgmentResult,
)
from ..workers import (
    ClaimExtractorWorker,
    WebSearcherWorker,
    WebFetcherWorker,
    SnippetExtractorWorker,
    PDFConverterWorker,
    ContentFilterWorker,
    ContentAggregatorWorker,
    JudgeWorker,
    CodingEarlyStoppingState,
    PackageVerdictCache,
)
from ..models.work_items import PDFTask, PDFResult

logger = get_logger()


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class WorkerConfig:
    """Configuration for worker counts."""
    num_extractors: int = 5
    num_searchers: int = 50
    num_fetchers: int = 20
    num_pdf_converters: int = 10
    num_filters: int = 10
    num_judges: int = 20


@dataclass 
class ModelConfig:
    """Configuration for model names."""
    extractor: str
    judge: str
    judge_fallback: str
    search: str | None = None  # Only for search-based pipelines


# =============================================================================
# Base Pipeline Class
# =============================================================================

class BasePipeline(ABC):
    """Abstract base class for evaluation pipelines.
    
    Provides common functionality for:
    - Loading and sampling conversations
    - Running the pipeline with monitoring
    - Collecting and saving results
    
    Subclasses must implement:
    - _get_model_config(): Return model configuration
    - _create_queues(): Create pipeline-specific queues
    - _create_workers(): Create pipeline-specific workers
    - _get_intermediate_queues(): Return queues to wait on before shutdown
    - _log_config(): Log pipeline-specific configuration
    """
    
    name: str = "BasePipeline"
    
    def __init__(
        self,
        input_path: Path,
        output_path: Path | None,
        worker_config: WorkerConfig,
        task_name: str,
        base_path: Path | str | None = None,
        n_conversations: int | None = None,
        max_claims_per_turn: int | None = None,
        seed: int = 42,
        monitor_interval: float = 3.0,
        claims_cache_path: Path | str | None = None,
        checkpoint_interval: int = 100,
        max_claims_per_category: int | None = None,
        judge_model: str | None = None,
        judge_fallback_model: str | None = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.worker_config = worker_config
        self.task_name = task_name
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.n_conversations = n_conversations
        self.max_claims_per_turn = max_claims_per_turn
        self.max_claims_per_category = max_claims_per_category
        self.seed = seed
        self.monitor_interval = monitor_interval
        self.checkpoint_interval = checkpoint_interval
        self.judge_model = judge_model
        self.judge_fallback_model = judge_fallback_model
        
        # Initialize strategy
        self.strategy = get_strategy(task_name, self.base_path)
        
        # Claims cache path - if None, will use default based on input path
        self._claims_cache_path = Path(claims_cache_path) if claims_cache_path else None
        
        # Will be populated during setup
        self.conversations: List[Any] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.model_config: ModelConfig | None = None
        
        # Core queues (always needed)
        self.conversation_queue: MonitoredQueue[ConversationItem] | None = None
        self.claims_queue: MonitoredQueue[ClaimItem] | None = None
        self.results_queue: MonitoredQueue[JudgmentResult] | None = None
        
        # Pipeline infrastructure
        self.pipeline: Pipeline | None = None
        self.samplers: Dict[str, SamplerBase] = {}
        
        # Cache state
        self._cached_claims: List[ClaimItem] | None = None
        
        # Result cache state (for skipping already-evaluated conversations)
        self._cached_results: List[EvaluationResult] | None = None
        self._cached_result_conv_ids: Set[int] = set()
        
        # Checkpoint state for judgments
        self._checkpoint_results: List[JudgmentResult] = []
        self._checkpoint_lock = asyncio.Lock()
        self._last_checkpoint_count: int = 0
        
        # Checkpoint state for claims extraction
        self._claims_checkpoint_lock = asyncio.Lock()
        self._last_claims_checkpoint_count: int = 0
        
        # Early stopping for coding task (saves API calls by skipping claims
        # in categories that already have a hallucination detected)
        self.early_stopping_state: CodingEarlyStoppingState | None = None
        self.package_cache: PackageVerdictCache | None = None
        if task_name == "coding":
            self.early_stopping_state = CodingEarlyStoppingState()
            self.package_cache = PackageVerdictCache()
    
    @abstractmethod
    def _get_model_config(self) -> ModelConfig:
        """Return model configuration for this pipeline type."""
        raise NotImplementedError
    
    @abstractmethod
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        """Create pipeline-specific queues. Must include 'claims' key."""
        raise NotImplementedError
    
    @abstractmethod
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        """Create pipeline-specific workers."""
        raise NotImplementedError
    
    @abstractmethod
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        """Return intermediate queues in pipeline order for join().
        
        These queues are waited on sequentially using join(), so they must
        be returned in the order items flow through the pipeline.
        When queue[i].join() completes, all its items have been processed
        and sent to queue[i+1].
        """
        raise NotImplementedError
    
    @abstractmethod
    def _log_config(self) -> None:
        """Log pipeline-specific configuration."""
        raise NotImplementedError
    
    def _load_conversations(self) -> None:
        """Load and optionally sample conversations.
        
        For coding tasks: samples N conversations per language (stratified sampling).
        For other tasks: samples first N conversations.
        """
        logger.info(f"Loading conversations from: {self.input_path}")
        self.conversations, self.metadata_list = load_conversations(self.input_path)
        logger.info(f"✓ Loaded {len(self.conversations)} conversations")

        if self.n_conversations is not None and self.n_conversations < len(self.conversations):
            if self.task_name == "coding":
                # Stratified sampling: N conversations per language
                self._sample_per_language()
            else:
                # Simple sampling: first N conversations
                self.conversations = self.conversations[:self.n_conversations]
                self.metadata_list = self.metadata_list[:self.n_conversations]
                logger.info(f"✓ Sampled first {self.n_conversations} conversations")
    
    def _sample_per_language(self) -> None:
        """Take first N conversations per language for coding tasks."""
        from collections import defaultdict
        
        # Group indices by language
        language_groups: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata_list):
            lang = meta.get("language", "unknown")
            language_groups[lang].append(i)
        
        # Take first N from each language group
        sampled_indices = []
        
        for lang, indices in sorted(language_groups.items()):
            # Take first N (indices are already in order)
            sampled = indices[:self.n_conversations]
            sampled_indices.extend(sampled)
            logger.info(f"  {lang}: {len(sampled)}/{len(indices)} conversations")
        
        # Sort to maintain original order
        sampled_indices.sort()
        
        # Apply sampling
        self.conversations = [self.conversations[i] for i in sampled_indices]
        self.metadata_list = [self.metadata_list[i] for i in sampled_indices]
        
        logger.info(f"✓ Selected {len(self.conversations)} conversations "
                   f"(first {self.n_conversations} per language, {len(language_groups)} languages)")
    
    def _create_samplers(self) -> None:
        """Create LLM samplers based on model config."""
        self.samplers["extractor"] = get_sampler(self.model_config.extractor)
        self.samplers["judge"] = get_sampler(self.model_config.judge)
        self.samplers["judge_fallback"] = get_sampler(self.model_config.judge_fallback)
        
        if self.model_config.search:
            self.samplers["search"] = get_sampler(self.model_config.search)
    
    def _get_claims_cache_path(self) -> Path:
        """Get the path for claims cache file."""
        if self._claims_cache_path:
            return self._claims_cache_path
        # Default: same directory and name as input, with _extracted_claims_cache suffix
        return self.input_path.parent / f"{self.input_path.stem}_extracted_claims_cache.jsonl"
    
    def _load_claims_cache(self) -> List[ClaimItem] | None:
        """Load claims from cache file if it exists.
        
        Returns:
            List of ClaimItem if cache exists and is valid, None otherwise.
        """
        cache_path = self._get_claims_cache_path()
        
        if not cache_path.exists():
            logger.info(f"No claims cache found at: {cache_path}")
            return None
        
        logger.info(f"Loading claims from cache: {cache_path}")
        
        try:
            claims = []
            with open(cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        claim = ClaimItem.from_cache_dict(data)
                        claims.append(claim)
                    except json.JSONDecodeError as e:
                        # Stop at corrupted line (likely partial write from interruption)
                        logger.warning(f"Corrupted cache entry at line {line_num}, "
                                      f"using {len(claims)} valid entries. Error: {e}")
                        break
            
            logger.info(f"✓ Loaded {len(claims)} claims from cache")
            return claims
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load claims cache: {e}")
            return None
    
    def _filter_cache_for_current_conversations(self, cached_claims: List[ClaimItem]) -> Tuple[List[ClaimItem], Set[int]]:
        """Filter cached claims to only include conversations in current input.
        
        Returns:
            Tuple of (filtered_claims, cached_conversation_ids)
        """
        # Get current conversation IDs from loaded data
        current_conv_ids = set()
        for i, meta in enumerate(self.metadata_list):
            conv_id = meta.get("conversation_id", i)
            current_conv_ids.add(conv_id)
        
        # Filter cached claims to only include current conversations
        filtered_claims = []
        cached_conv_ids = set()
        
        for claim in cached_claims:
            if claim.conversation_id in current_conv_ids:
                filtered_claims.append(claim)
                cached_conv_ids.add(claim.conversation_id)
        
        removed_count = len(cached_claims) - len(filtered_claims)
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} claims from removed conversations")
        
        return filtered_claims, cached_conv_ids
    
    def _filter_out_judged_claims(self, claims: List[ClaimItem]) -> List[ClaimItem]:
        """Filter out claims that have already been judged.
        
        Uses the cached results to identify which claims have been evaluated.
        
        Returns:
            List of claims that still need to be judged.
        """
        if not self._cached_results:
            return claims
        
        # Build set of claim IDs that have been judged
        judged_claim_ids = set()
        for result in self._cached_results:
            if hasattr(result, 'details') and result.details:
                claim_evals = result.details.get("claim_evaluations", [])
                for eval_item in claim_evals:
                    if isinstance(eval_item, dict):
                        claim_id = eval_item.get("claim_id")
                        if claim_id:
                            judged_claim_ids.add(claim_id)
        
        if not judged_claim_ids:
            return claims
        
        # Filter out already-judged claims
        unjudged_claims = []
        for claim in claims:
            if claim.claim_id not in judged_claim_ids:
                unjudged_claims.append(claim)
        
        removed_count = len(claims) - len(unjudged_claims)
        if removed_count > 0:
            logger.info(f"  Filtered out {removed_count} already-judged claims")
            logger.info(f"  Remaining claims to judge: {len(unjudged_claims)}")
        
        return unjudged_claims
    
    def _get_uncached_conversations(self, cached_conv_ids: Set[int]) -> List[Tuple[int, Any, Dict]]:
        """Get list of conversations that are not in cache.
        
        Returns:
            List of (index, conversation, metadata) tuples for uncached conversations
        """
        uncached = []
        for i, (conv, meta) in enumerate(zip(self.conversations, self.metadata_list)):
            conv_id = meta.get("conversation_id", i)
            if conv_id not in cached_conv_ids:
                uncached.append((i, conv, meta))
        return uncached
    
    def _save_claims_cache(self, claims: List[ClaimItem]) -> None:
        """Save claims to cache file."""
        cache_path = self._get_claims_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(claims)} claims to cache: {cache_path}")
        
        with open(cache_path, "w", encoding="utf-8") as f:
            for claim in claims:
                f.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"✓ Saved claims cache")
    
    async def _save_claims_checkpoint(self, extractor, force: bool = False) -> None:
        """Save incremental checkpoint of extracted claims.
        
        Args:
            extractor: The ClaimExtractorWorker with output tracking enabled
            force: If True, save regardless of checkpoint interval
        """
        async with self._claims_checkpoint_lock:
            # Get current extracted claims
            extracted_claims = extractor.get_all_outputs()
            current_count = len(extracted_claims)
            claims_since_last = current_count - self._last_claims_checkpoint_count
            
            # Only save if we have enough new claims or forced
            if not force and claims_since_last < self.checkpoint_interval:
                return
            
            if current_count == 0:
                return
            
            # Merge with cached claims if any
            if self._cached_claims:
                all_claims = self._cached_claims + extracted_claims
            else:
                all_claims = extracted_claims
            
            # Save to cache file
            cache_path = self._get_claims_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, "w", encoding="utf-8") as f:
                for claim in all_claims:
                    f.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
            
            self._last_claims_checkpoint_count = current_count
            logger.info(f"📁 Claims checkpoint saved: {len(all_claims)} claims ({claims_since_last} new)")
    
    async def _claims_checkpoint_collector(self, extractor) -> None:
        """Background task that periodically saves claims during extraction."""
        while True:
            try:
                # Wait for checkpoint interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Save checkpoint if enough new claims
                await self._save_claims_checkpoint(extractor)
                
            except asyncio.CancelledError:
                # Task cancelled, save final claims checkpoint
                await self._save_claims_checkpoint(extractor, force=True)
                raise
    
    def _get_output_path(self) -> Path:
        """Get the output path for results."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    def _load_results_cache(self) -> List[EvaluationResult] | None:
        """Load existing evaluation results from output file if it exists.
        
        Returns:
            List of EvaluationResult if file exists and is valid, None otherwise.
        """
        output_path = self._get_output_path()
        
        if not output_path.exists():
            logger.info(f"No existing results found at: {output_path}")
            return None
        
        logger.info(f"Loading existing results from: {output_path}")
        
        try:
            results = []
            with open(output_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Skip non-result records
                        if data.get("_type") != "evaluation_result":
                            continue
                        # Reconstruct EvaluationResult
                        result = EvaluationResult(
                            conversation_id=data["conversation_id"],
                            score=data["score"],
                            reasoning=data["reasoning"],
                            details=data.get("details"),
                            metadata=data.get("metadata"),
                        )
                        results.append(result)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Corrupted result entry at line {line_num}: {e}")
                        continue
            
            if results:
                logger.info(f"✓ Loaded {len(results)} cached evaluation results")
                return results
            return None
        
        except Exception as e:
            logger.warning(f"Failed to load results cache: {e}")
            return None
    
    def _filter_conversations_for_evaluation(self) -> None:
        """Filter out conversations that already have cached results.
        
        Updates self.conversations and self.metadata_list to only include
        conversations that need evaluation.
        
        A conversation is considered "complete" only if:
        1. It has results in the cache, AND
        2. The number of judged claims matches the number of extracted claims
           (if claims cache exists)
        """
        if not self._cached_results:
            return
        
        # Build map of conversation IDs to their judged claim counts
        cached_result_claims = {}
        for r in self._cached_results:
            cached_result_claims[r.conversation_id] = r.details.get("total_claims", 0) if hasattr(r, 'details') else 0
        
        self._cached_result_conv_ids = set(cached_result_claims.keys())
        
        # Load claims cache to get expected claim counts per conversation
        claims_cache_path = self._get_claims_cache_path()
        expected_claims_per_conv = {}
        
        if claims_cache_path.exists():
            try:
                with open(claims_cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            conv_id = data.get("conversation_id", -1)
                            expected_claims_per_conv[conv_id] = expected_claims_per_conv.get(conv_id, 0) + 1
                        except json.JSONDecodeError:
                            continue
                logger.debug(f"Loaded expected claim counts for {len(expected_claims_per_conv)} conversations from claims cache")
            except Exception as e:
                logger.warning(f"Could not load claims cache for comparison: {e}")
        
        # Filter to only include conversations that are incomplete
        filtered_convs = []
        filtered_meta = []
        complete_count = 0
        incomplete_count = 0
        
        for conv, meta in zip(self.conversations, self.metadata_list):
            conv_id = meta.get("conversation_id", 0)
            
            # Check if conversation has results
            if conv_id not in self._cached_result_conv_ids:
                # No results at all - needs evaluation
                filtered_convs.append(conv)
                filtered_meta.append(meta)
                continue
            
            # Has results - check if complete
            judged_claims = cached_result_claims.get(conv_id, 0)
            expected_claims = expected_claims_per_conv.get(conv_id, 0)
            
            if expected_claims > 0 and judged_claims < expected_claims:
                # Incomplete - some claims still need judging
                incomplete_count += 1
                filtered_convs.append(conv)
                filtered_meta.append(meta)
                logger.debug(f"  Conv {conv_id}: incomplete ({judged_claims}/{expected_claims} claims)")
            else:
                # Complete - skip this conversation
                complete_count += 1
        
        skipped_count = len(self.conversations) - len(filtered_convs)
        
        if skipped_count > 0 or incomplete_count > 0:
            logger.info(f"  Skipping {complete_count} fully completed conversations")
            if incomplete_count > 0:
                logger.info(f"  Re-processing {incomplete_count} incomplete conversations (partial results)")
            logger.info(f"  Processing {len(filtered_convs)} conversations total")
            
            # Store original lists for final result building
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
            
            # Update to filtered lists
            self.conversations = filtered_convs
            self.metadata_list = filtered_meta
        else:
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
    
    async def _load_input_queue(self) -> None:
        """Load conversations into input queue."""
        for i, (conv, meta) in enumerate(zip(self.conversations, self.metadata_list)):
            conv_id = meta.get("conversation_id", i)
            await self.conversation_queue.put(
                ConversationItem(
                    conversation_id=conv_id,
                    conversation=conv.to_message_list(),  # Convert Conversation to list[dict]
                    metadata=meta,
                    max_claims_per_turn=self.max_claims_per_turn,
                ),
                claim_id=f"conv-{conv_id}",
                conversation_id=conv_id,
            )
    
        self.conversation_queue.close()
    
    async def _collect_results(self) -> List[JudgmentResult]:
        """Collect all results from results queue."""
        all_results: List[JudgmentResult] = []
        while True:
            item = await self.results_queue.get_nowait()

            # Use this breaking instead of not self.results_queue.empty to avoid race condition
            if item is None:
                break

            all_results.append(item.data)
            self.results_queue.task_done()
    
        return all_results
    
    def _build_evaluation_results(self, judgments: List[JudgmentResult]) -> List[EvaluationResult]:
        """Build EvaluationResult objects from judgments."""
        # Group by conversation
        results_by_conv: Dict[int, List[JudgmentResult]] = {}
        for result in judgments:
            conv_id = result.conversation_id
            if conv_id not in results_by_conv:
                results_by_conv[conv_id] = []
            results_by_conv[conv_id].append(result)

        # Build EvaluationResults
        final_results: List[EvaluationResult] = []
        for meta in self.metadata_list:
            conv_id = meta.get("conversation_id", 0)
            conv_judgments = results_by_conv.get(conv_id, [])

            total_claims = len(conv_judgments)
            if total_claims == 0:
                score = 1.0
                reasoning = "No verifiable claims found"
                hallucinations = 0
                input_use_fallback_count = 0
                judge_used_websearch_fallback_count = 0
                snippets_only_count = 0
                # Coding-specific
                import_hallucinations = 0
                install_hallucinations = 0
                function_hallucinations = 0
            else:
                hallucinations = sum(1 for j in conv_judgments if j.hallucination.lower() == "yes")
                score = 1.0 - (hallucinations / total_claims)
                reasoning = f"Found {hallucinations}/{total_claims} hallucinated claims"
                input_use_fallback_count = sum(1 for j in conv_judgments if j.input_use_fallback)
                judge_used_websearch_fallback_count = sum(1 for j in conv_judgments if j.judge_used_websearch_fallback)
                snippets_only_count = sum(1 for j in conv_judgments if j.snippets_only)
                # Coding-specific hallucination counts
                import_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_import_detected)
                install_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_install_detected)
                function_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_function_usage_detected)

            # Build details dict
            details = {
                    "total_claims": total_claims,
                    "hallucinations": hallucinations,
                    "input_use_fallback_count": input_use_fallback_count,
                    "judge_used_websearch_fallback_count": judge_used_websearch_fallback_count,
                    "snippets_only_count": snippets_only_count,
                    "claim_evaluations": [j.to_dict() for j in conv_judgments],
            }
            
            # Add coding-specific details if this is a coding task
            if self.task_name == "coding":
                # Aggregate boolean flags (ANY element with hallucination = True)
                any_import_halluc = any(j.hallucinated_import_detected for j in conv_judgments)
                any_install_halluc = any(j.hallucinated_install_detected for j in conv_judgments)
                any_function_halluc = any(j.hallucinated_function_usage_detected for j in conv_judgments)
                
                # Calculate RESPONSE-LEVEL (turn-level) hallucination rates
                # Group claims by turn_number
                turns_data: Dict[int, Dict[str, bool]] = {}
                for j in conv_judgments:
                    turn = j.turn_number
                    if turn not in turns_data:
                        turns_data[turn] = {"import": False, "install": False, "function": False, "any": False}
                    if j.hallucinated_import_detected:
                        turns_data[turn]["import"] = True
                        turns_data[turn]["any"] = True
                    if j.hallucinated_install_detected:
                        turns_data[turn]["install"] = True
                        turns_data[turn]["any"] = True
                    if j.hallucinated_function_usage_detected:
                        turns_data[turn]["function"] = True
                        turns_data[turn]["any"] = True
                
                total_responses = len(turns_data)
                if total_responses > 0:
                    # Count hallucinated responses per category
                    import_halluc_responses = sum(1 for t in turns_data.values() if t["import"])
                    install_halluc_responses = sum(1 for t in turns_data.values() if t["install"])
                    function_halluc_responses = sum(1 for t in turns_data.values() if t["function"])
                    overall_halluc_responses = sum(1 for t in turns_data.values() if t["any"])
                    
                    # Calculate rates
                    import_halluc_rate = import_halluc_responses / total_responses
                    install_halluc_rate = install_halluc_responses / total_responses
                    function_halluc_rate = function_halluc_responses / total_responses
                    overall_halluc_rate = overall_halluc_responses / total_responses
                else:
                    import_halluc_responses = install_halluc_responses = function_halluc_responses = overall_halluc_responses = 0
                    import_halluc_rate = install_halluc_rate = function_halluc_rate = overall_halluc_rate = 0.0
                
                details.update({
                    "hallucinated_import_detected": any_import_halluc,
                    "hallucinated_install_detected": any_install_halluc,
                    "hallucinated_function_usage_detected": any_function_halluc,
                    # Claim-level counts (for reference)
                    "import_hallucination_count": import_hallucinations,
                    "install_hallucination_count": install_hallucinations,
                    "function_hallucination_count": function_hallucinations,
                    # Response-level (turn-level) stats
                    "total_responses": total_responses,
                    "import_hallucinated_responses": import_halluc_responses,
                    "install_hallucinated_responses": install_halluc_responses,
                    "function_hallucinated_responses": function_halluc_responses,
                    "overall_hallucinated_responses": overall_halluc_responses,
                    # Hallucination rates (hallucinated responses / total responses)
                    "import_hallucination_rate": import_halluc_rate,
                    "install_hallucination_rate": install_halluc_rate,
                    "function_hallucination_rate": function_halluc_rate,
                    "overall_hallucination_rate": overall_halluc_rate,
                })
                
                # Update reasoning for coding task with response-level rates
                if overall_halluc_responses > 0:
                    rate_parts = [f"Overall: {overall_halluc_responses}/{total_responses} ({overall_halluc_rate:.1%})"]
                    if import_halluc_responses > 0:
                        rate_parts.append(f"Import: {import_halluc_responses}/{total_responses} ({import_halluc_rate:.1%})")
                    if install_halluc_responses > 0:
                        rate_parts.append(f"Install: {install_halluc_responses}/{total_responses} ({install_halluc_rate:.1%})")
                    if function_halluc_responses > 0:
                        rate_parts.append(f"Function: {function_halluc_responses}/{total_responses} ({function_halluc_rate:.1%})")
                    reasoning = f"Response hallucination rates - {'; '.join(rate_parts)}"
                else:
                    reasoning = f"No hallucinations detected in {total_responses} responses"

            final_results.append(EvaluationResult(
                conversation_id=conv_id,
                score=score,
                reasoning=reasoning,
                details=details,
                metadata=meta,
            ))

        return final_results
    
    def _save_results(self, results: List[EvaluationResult]) -> Path:
        """Save evaluation results to file."""
        if self.output_path is None:
            output_path = self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        else:
            output_path = Path(self.output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"✓ Saved {len(results)} evaluation results")
        return output_path
    
    def _get_checkpoint_path(self) -> Path:
        """Get the path for checkpoint file (same as output file)."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    async def _save_checkpoint(self, force: bool = False) -> None:
        """Save incremental checkpoint of current results.
        
        Args:
            force: If True, save regardless of checkpoint interval
        """
        async with self._checkpoint_lock:
            current_count = len(self._checkpoint_results)
            results_since_last = current_count - self._last_checkpoint_count
            
            # Only save if we have enough new results or forced
            if not force and results_since_last < self.checkpoint_interval:
                return
            
            if current_count == 0:
                return
            
            # Build evaluation results from collected judgments
            checkpoint_eval_results = self._build_evaluation_results(self._checkpoint_results)
            
            # Merge with cached results if any
            if self._cached_results:
                all_results = self._cached_results + checkpoint_eval_results
                all_results.sort(key=lambda r: r.conversation_id)
            else:
                all_results = checkpoint_eval_results
            
            # Save to checkpoint file
            checkpoint_path = self._get_checkpoint_path()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for result in all_results:
                    record = {
                        "_type": "evaluation_result",
                        **asdict(result),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            self._last_checkpoint_count = current_count
            logger.info(f"📁 Checkpoint saved: {len(all_results)} results ({results_since_last} new)")
    
    async def _checkpoint_collector(self) -> None:
        """Background task that collects results and saves checkpoints."""
        while True:
            try:
                # Try to get a result with timeout
                try:
                    item = await asyncio.wait_for(
                        self.results_queue.get(),
                        timeout=5.0
                    )
                    # Extract the actual JudgmentResult from the QueueItem wrapper
                    self._checkpoint_results.append(item.data)
                    self.results_queue.task_done()
                    
                    # Check if we should save a checkpoint
                    await self._save_checkpoint()
                    
                except asyncio.TimeoutError:
                    # No result available, check if queue is closed and empty
                    if self.results_queue._closed and self.results_queue.empty():
                        break
                    continue
                    
            except asyncio.CancelledError:
                # Task cancelled, save final checkpoint
                await self._save_checkpoint(force=True)
                raise
    
    async def _log_summary(self, results: List[EvaluationResult], output_path: Path) -> None:
        """Log final summary."""
        total_claims = sum(r.details.get("total_claims", 0) for r in results)
        total_hallucinations = sum(r.details.get("hallucinations", 0) for r in results)
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Conversations: {len(results)}")
        logger.info(f"Total claims: {total_claims}")
        logger.info(f"Hallucinations: {total_hallucinations}")
        logger.info(f"Overall accuracy: {1 - (total_hallucinations / max(total_claims, 1)):.1%}")
        
        # Early stopping stats for coding task
        if self.early_stopping_state:
            early_stop_stats = await self.early_stopping_state.get_stats()
            skipped = early_stop_stats.get("total_claims_skipped", 0)
            fully_stopped = early_stop_stats.get("conversations_fully_stopped", 0)
            if skipped > 0:
                logger.info(f"Early stopping: {skipped} claims skipped, {fully_stopped} conversations fully stopped")
        
        # Package cache stats for coding task
        if self.package_cache:
            cache_stats = await self.package_cache.get_stats()
            whitelist_skips = cache_stats.get("whitelist_skips", 0)
            cache_hits = cache_stats.get("cache_hits", 0)
            if whitelist_skips > 0 or cache_hits > 0:
                logger.info(f"Package cache: {whitelist_skips} whitelist skips, {cache_hits} cache hits")
        
        logger.info(f"Results: {output_path}")
        logger.info("=" * 70)
    
    async def run(self) -> Path:
        """Run the full pipeline."""
        # Validate input
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Setup
        self.model_config = self._get_model_config()
        
        # Log configuration
        logger.info("=" * 70)
        logger.info(f"QUEUE-BASED EVALUATION PIPELINE: {self.name}")
        logger.info(f"Task: {self.task_name}")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_path}")
        self._log_config()
        if self.checkpoint_interval > 0:
            logger.info(f"Checkpointing: Every {self.checkpoint_interval} judgments")
        else:
            logger.info("Checkpointing: Disabled")
        logger.info("=" * 70 + "\n")
        
        # Load data
        self._load_conversations()
        logger.info("")
        
        # Try to load existing results and filter out already-evaluated conversations
        self._cached_results = self._load_results_cache()
        if self._cached_results:
            self._filter_conversations_for_evaluation()
            
            # If all conversations are cached, skip everything
            if len(self.conversations) == 0:
                logger.info(f"✓ All conversations have cached results - nothing to evaluate")
                output_path = self._get_output_path()
                await self._log_summary(self._cached_results, output_path)
                return output_path
        else:
            # Initialize for later use
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
        
        logger.info("")
        
        # Try to load claims from cache and determine what needs extraction
        raw_cached_claims = self._load_claims_cache()
        
        # Determine caching strategy
        self._cached_claims = None
        self._cached_conv_ids: Set[int] = set()
        self._uncached_conversations: List[Tuple[int, Any, Dict]] = []
        need_extraction = True
        
        if raw_cached_claims is not None:
            # Filter cache to only include current conversations
            self._cached_claims, self._cached_conv_ids = self._filter_cache_for_current_conversations(raw_cached_claims)
            
            # Filter out claims that have already been judged (for incomplete conversations)
            if self._cached_results:
                self._cached_claims = self._filter_out_judged_claims(self._cached_claims)
            
            # Find conversations that need extraction
            self._uncached_conversations = self._get_uncached_conversations(self._cached_conv_ids)
            
            if len(self._uncached_conversations) == 0:
                # All conversations are cached - no extraction needed
                need_extraction = False
                logger.info(f"✓ All {len(self.conversations)} conversations have cached claims")
            else:
                logger.info(f"  {len(self._cached_conv_ids)} conversations cached, "
                           f"{len(self._uncached_conversations)} need extraction")
        
        # Create samplers
        self._create_samplers()
        
        # Create core queues
        self.conversation_queue = MonitoredQueue("conversations")
        self.results_queue = MonitoredQueue("results")
        
        # Create pipeline-specific queues
        queues = self._create_queues()
        self.claims_queue = queues["claims"]
        
        # Create workers list - extractor only if we need extraction
        all_workers = []
        extractor = None
        
        if need_extraction:
            # Create extractor (common to all pipelines)
            # For coding tasks, limit claims per category to speed up evaluation
            max_per_cat = self.max_claims_per_category if self.task_name == "coding" else None
            extractor = ClaimExtractorWorker(
                input_queue=self.conversation_queue,
                output_queue=self.claims_queue,
                sampler=self.samplers["extractor"],
                strategy=self.strategy,
                num_workers=self.worker_config.num_extractors,
                max_claims_per_category=max_per_cat,
            )
            # Enable output tracking for caching
            extractor.enable_output_tracking()
            all_workers.append(extractor)
        
        # Create pipeline-specific workers
        workers = self._create_workers(queues)
        all_workers.extend(workers)
        
        # Create pipeline
        self.pipeline = Pipeline(name=self.name)
        
        # Register queues
        self.pipeline.add_queue(self.conversation_queue)
        for queue in queues.values():
            self.pipeline.add_queue(queue)
        self.pipeline.add_queue(self.results_queue)
        
        # Register workers
        for worker in all_workers:
            self.pipeline.add_worker(worker)
        
        self.pipeline.set_results_queue(self.results_queue)
        
        if not need_extraction:
            # All claims are cached - load directly into claims queue
            logger.info(f"Using cached claims, skipping extraction...")
            await self._load_cached_claims_to_queue()
            total_items = len(self._cached_claims)
        elif self._cached_claims:
            # Partial cache - load cached claims AND run extraction for uncached
            logger.info(f"Loading {len(self._cached_claims)} cached claims and extracting {len(self._uncached_conversations)} new conversations...")
            await self._load_cached_claims_to_queue_partial()
            await self._load_uncached_conversations_to_queue()
            total_items = len(self._cached_claims) + len(self._uncached_conversations)
        else:
            # No cache - extract all conversations
            await self._load_input_queue()
            total_items = len(self.conversations)
        logger.info(f"Starting pipeline with {total_items} {'claims' if not need_extraction else 'items'}...")
        
        # Create monitor
        monitor = QueueMonitor(
            queues=self.pipeline.queues,
            workers=self.pipeline.workers,
            total_items=total_items,
        )
        monitor.set_final_queue(self.results_queue)
        monitor.on_update(lambda s: print(str(s)))
        
        # Start everything
        await monitor.start(interval=self.monitor_interval)
        
        for worker in all_workers:
            await worker.start()
        
        # Start checkpoint collector as background task (if enabled)
        checkpoint_task = None
        claims_checkpoint_task = None
        if self.checkpoint_interval > 0:
            checkpoint_task = asyncio.create_task(self._checkpoint_collector())
        
        if need_extraction:
            # Start claims checkpoint collector during extraction (if enabled)
            if self.checkpoint_interval > 0:
                claims_checkpoint_task = asyncio.create_task(
                    self._claims_checkpoint_collector(extractor)
                )
            
            # Wait for extraction pipeline to complete using queue.join()
            await self.conversation_queue.join()
            
            # Stop claims checkpoint collector
            if claims_checkpoint_task:
                try:
                    claims_checkpoint_task.cancel()
                    await claims_checkpoint_task
                except asyncio.CancelledError:
                    pass
            
            # Save final claims cache after extraction completes
            extracted_claims = extractor.get_all_outputs()
            
            if self._cached_claims:
                all_claims = self._cached_claims + extracted_claims
                logger.info(f"Merged {len(self._cached_claims)} cached + {len(extracted_claims)} new = {len(all_claims)} total claims")
            else:
                all_claims = extracted_claims
            
            self._save_claims_cache(all_claims)
            logger.info(f"📁 Claims cache saved ({len(all_claims)} total claims)")
        
        # Wait for each intermediate queue in order
        for queue in self._get_intermediate_queues(queues):
            await queue.join()
        
        # Stop workers
        for worker in all_workers:
            await worker.stop()
        
        await monitor.stop()
        
        # Collect results - either from checkpoint collector or directly from queue
        if checkpoint_task:
            # Wait for checkpoint collector to finish and save final checkpoint
            try:
                checkpoint_task.cancel()
                await checkpoint_task
            except asyncio.CancelledError:
                pass
            
            # Final save with all results
            await self._save_checkpoint(force=True)
            
            # Build final results from checkpoint
            new_results = self._build_evaluation_results(self._checkpoint_results)
            logger.info(f"\n✓ Collected {len(self._checkpoint_results)} judgment results")
        else:
            # No checkpointing - collect results directly from queue
            all_judgments = await self._collect_results()
            logger.info(f"\n✓ Collected {len(all_judgments)} judgment results")
            new_results = self._build_evaluation_results(all_judgments)
        
        # Merge with cached results if any
        if self._cached_results:
            # Combine cached results with new results
            final_results = self._cached_results + new_results
            logger.info(f"Merged {len(self._cached_results)} cached + {len(new_results)} new = {len(final_results)} total results")
            
            # Sort by conversation_id for consistent output
            final_results.sort(key=lambda r: r.conversation_id)
        else:
            final_results = new_results
        
        output_path = self._save_results(final_results)
        await self._log_summary(final_results, output_path)
        
        # Clean up shared HTTP clients
        await close_shared_client()
        await close_pdf_session()
        
        return output_path
    
    async def _load_cached_claims_to_queue(self) -> None:
        """Load cached claims directly into the claims queue (full cache mode)."""
        for claim in self._cached_claims:
            await self.claims_queue.put(
                claim,
                claim_id=claim.claim_id,
                conversation_id=claim.conversation_id,
            )
        self.claims_queue.close()
    
    async def _load_cached_claims_to_queue_partial(self) -> None:
        """Load cached claims to queue without closing (partial cache mode).
        
        Used when we also need to extract claims for uncached conversations.
        """
        for claim in self._cached_claims:
            await self.claims_queue.put(
                claim,
                claim_id=claim.claim_id,
                conversation_id=claim.conversation_id,
            )
    
    async def _load_uncached_conversations_to_queue(self) -> None:
        """Load only uncached conversations into input queue for extraction."""
        for i, conv, meta in self._uncached_conversations:
            conv_id = meta.get("conversation_id", i)
            await self.conversation_queue.put(
                ConversationItem(
                    conversation_id=conv_id,
                    conversation=conv.to_message_list(),
                    metadata=meta,
                    max_claims_per_turn=self.max_claims_per_turn,
                ),
                claim_id=f"conv-{conv_id}",
                conversation_id=conv_id,
            )
        
        self.conversation_queue.close()
