"""Coding pipeline: judges whole turns, no claim extraction."""

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

from .base import WorkerConfig

logger = get_logger()


# =============================================================================
# Coding Direct Pipeline (OpenAI Websearch without claim extraction)
# =============================================================================

class CodingDirectPipeline:
    """Pipeline for coding hallucination detection using OpenAI websearch directly.
    
    This pipeline does NOT extract individual claims. Instead, it:
    1. Takes each assistant turn as a whole
    2. Uses OpenAI with websearch to detect all hallucinations at once
    3. Returns per-turn results with detected hallucinations
    
    This is faster and simpler but may miss some nuanced hallucinations that
    claim-by-claim evaluation would catch.
    """
    
    name = "coding_direct"
    
    def __init__(
        self,
        input_path: Path,
        output_path: Path | None = None,
        worker_config: WorkerConfig | None = None,
        task_name: str = "coding",
        n_conversations: int | None = None,
        seed: int = 42,
        monitor_interval: float = 3.0,
        base_path: Path | str | None = None,
        judge_model: str | None = None,
        checkpoint_interval: int = 100,
        # These are ignored but kept for API compatibility
        max_claims_per_turn: int | None = None,
        claims_cache_path: Path | str | None = None,
        max_claims_per_category: int | None = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.worker_config = worker_config or WorkerConfig()
        self.task_name = task_name
        self.n_conversations = n_conversations
        self.seed = seed
        self.monitor_interval = monitor_interval
        self.judge_model = judge_model
        # Persist partial results every N judged turns so the run is always resumable.
        self.checkpoint_interval = checkpoint_interval
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        
        # State
        self.conversations = []
        self.metadata_list = []
    
    def _load_conversations(self) -> None:
        """Load conversations from input file with stratified sampling for coding tasks."""
        self.conversations, self.metadata_list = load_conversations(self.input_path)
        
        if self.n_conversations and self.n_conversations < len(self.conversations):
            if self.task_name == "coding":
                # Stratified sampling: N conversations per language (same as base pipeline)
                self._sample_per_language()
            else:
                # Simple sampling: first N conversations
                random.seed(self.seed)
                indices = list(range(len(self.conversations)))
                random.shuffle(indices)
                indices = sorted(indices[:self.n_conversations])
                self.conversations = [self.conversations[i] for i in indices]
                self.metadata_list = [self.metadata_list[i] for i in indices]
                logger.info(f"✓ Sampled first {self.n_conversations} conversations")
        else:
            logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def _sample_per_language(self) -> None:
        """Take first N conversations per language for coding tasks (same as base pipeline)."""
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
    
    def _get_output_path(self) -> Path:
        """Get the output path for results."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    def _extract_assistant_turns(self) -> List[Tuple[int, int, str]]:
        """Extract all assistant turns from conversations.
        
        Returns list of (conversation_id, turn_number, content) tuples.
        """
        from ..workers import TurnItem
        
        turns = []
        for conv, meta in zip(self.conversations, self.metadata_list):
            conv_id = meta.get("conversation_id", 0)
            messages = conv.to_message_list()
            
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and len(content.strip()) > 0:
                        turns.append((conv_id, i, content))
        
        return turns
    
    def _build_evaluation_results(
        self, 
        results: List["DirectCodingResult"],
    ) -> List[EvaluationResult]:
        """Build EvaluationResult objects from direct coding results."""
        from ..workers import DirectCodingResult
        
        # Group by conversation
        results_by_conv: Dict[int, List[DirectCodingResult]] = {}
        for result in results:
            conv_id = result.conversation_id
            if conv_id not in results_by_conv:
                results_by_conv[conv_id] = []
            results_by_conv[conv_id].append(result)
        
        # Build EvaluationResults
        final_results: List[EvaluationResult] = []
        for meta in self.metadata_list:
            conv_id = meta.get("conversation_id", 0)
            conv_results = results_by_conv.get(conv_id, [])
            
            total_turns = len(conv_results)
            if total_turns == 0:
                score = 1.0
                reasoning = "No assistant turns found"
                import_hallucinations = 0
                install_hallucinations = 0
                function_hallucinations = 0
            else:
                # Separate successfully-judged turns from unjudged ones (sampler
                # error, or unparseable/empty response). Unjudged turns are NOT
                # counted as verified: they're excluded from scoring and marked
                # below so they can be re-evaluated on a second pass.
                judged_results = [r for r in conv_results if not r.error]
                num_judged = len(judged_results)

                # Count hallucinations over judged turns only
                import_hallucinations = sum(1 for r in judged_results if r.hallucinated_import_detected)
                install_hallucinations = sum(1 for r in judged_results if r.hallucinated_install_detected)
                function_hallucinations = sum(1 for r in judged_results if r.hallucinated_function_usage_detected)

                total_hallucinations = import_hallucinations + install_hallucinations + function_hallucinations
                turns_with_hallucinations = sum(1 for r in judged_results if r.has_hallucination)

                score = 1.0 - (turns_with_hallucinations / num_judged) if num_judged > 0 else 1.0
                
                # Build detailed reasoning with breakdown
                reasoning_parts = [f"Evaluated {total_turns} turn{'s' if total_turns != 1 else ''}"]
                
                if total_hallucinations == 0:
                    reasoning_parts.append(f"0 hallucinations ({import_hallucinations} import, {install_hallucinations} install, {function_hallucinations} function)")
                    reasoning_parts.append("All packages and API calls verified against official documentation")
                else:
                    reasoning_parts.append(f"{total_hallucinations} hallucination{'s' if total_hallucinations != 1 else ''} ({import_hallucinations} import, {install_hallucinations} install, {function_hallucinations} function)")
                    
                    # Collect specific issues from turn evaluations
                    issues = []
                    for r in conv_results:
                        if r.hallucinated_import_detected and r.hallucinated_imports:
                            for imp in r.hallucinated_imports[:2]:  # Limit to first 2
                                pkg = imp.get("package", "unknown")
                                issues.append(f"{pkg} import")
                        if r.hallucinated_install_detected and r.hallucinated_installs:
                            for inst in r.hallucinated_installs[:2]:  # Limit to first 2
                                pkg = inst.get("package", "unknown")
                                issues.append(f"{pkg} install")
                        if r.hallucinated_function_usage_detected and r.hallucinated_function_calls:
                            for func in r.hallucinated_function_calls[:2]:  # Limit to first 2
                                pkg = func.get("package", "unknown")
                                func_name = func.get("function", "unknown")
                                issues.append(f"{pkg}.{func_name}()")
                    
                    if issues:
                        issues_str = ", ".join(issues[:3])  # Limit to 3 issues
                        if len(issues) > 3:
                            issues_str += f" (+{len(issues) - 3} more)"
                        reasoning_parts.append(f"Issues: {issues_str}")
                
                reasoning = ". ".join(reasoning_parts) + "."
            
            # Convert turn_evaluations to claim_evaluations format for report compatibility
            claim_evaluations = []
            for turn_result in conv_results:
                turn_dict = turn_result.to_dict()
                turn_number = turn_dict.get("turn_number", 0)

                # Unjudged turn (sampler error / unparseable response): mark it so
                # a second pass can target it; do NOT emit a "verified" entry.
                if turn_result.error:
                    claim_evaluations.append({
                        "claim_id": f"turn-{conv_id}-{turn_number}-unjudged",
                        "conversation_id": conv_id,
                        "turn_idx": turn_number,
                        "turn_number": turn_number,
                        "hallucination": "Unjudged",
                        "judged": False,
                        "error": turn_result.error,
                        "claim": {"element_type": "unjudged"},
                        "reason": f"Turn not judged: {turn_result.error}",
                        "reasoning": "",
                    })
                    continue

                # Expand hallucinated items into individual claim evaluations
                # Each hallucinated item becomes a separate claim
                has_any_hallucination = turn_result.has_hallucination
                
                if has_any_hallucination:
                    # Create claim evaluation for each hallucinated import
                    for imp in turn_result.hallucinated_imports:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-import-{imp.get('package', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": True,
                            "hallucinated_install_detected": False,
                            "hallucinated_function_usage_detected": False,
                            "claim": {
                                "element_type": "import",
                                "package_name": imp.get("package", "unknown"),
                                "code_snippet": imp.get("code", ""),
                            },
                            "reason": imp.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                    
                    # Create claim evaluation for each hallucinated install
                    for inst in turn_result.hallucinated_installs:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-install-{inst.get('package', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": False,
                            "hallucinated_install_detected": True,
                            "hallucinated_function_usage_detected": False,
                            "claim": {
                                "element_type": "install",
                                "package_name": inst.get("package", "unknown"),
                                "code_snippet": inst.get("code", ""),
                            },
                            "reason": inst.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                    
                    # Create claim evaluation for each hallucinated function call
                    for func in turn_result.hallucinated_function_calls:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-function-{func.get('package', 'unknown')}-{func.get('function', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": False,
                            "hallucinated_install_detected": False,
                            "hallucinated_function_usage_detected": True,
                            "claim": {
                                "element_type": "function_call",
                                "package_name": func.get("package", "unknown"),
                                "function_name": func.get("function", "unknown"),
                                "code_snippet": func.get("code", ""),
                            },
                            "reason": func.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                else:
                    # For turns with no hallucinations, create a single "verified" claim entry
                    # This represents that the entire turn was verified
                    claim_evaluations.append({
                        "claim_id": f"turn-{conv_id}-{turn_number}-verified",
                        "conversation_id": conv_id,
                        "turn_idx": turn_number,
                        "turn_number": turn_number,
                        "hallucination": "No",
                        "hallucinated_import_detected": False,
                        "hallucinated_install_detected": False,
                        "hallucinated_function_usage_detected": False,
                        "claim": {
                            "element_type": "unknown",  # Will be inferred as non-hallucinated
                        },
                        "reason": turn_dict.get("reasoning", "All packages and API calls verified"),
                        "reasoning": turn_dict.get("reasoning", ""),
                    })
            
            # Calculate response-level stats
            total_responses = total_turns
            import_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_import_detected)
            install_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_install_detected)
            function_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_function_usage_detected)
            overall_hallucinated_responses = turns_with_hallucinations
            
            # Build details dict with both formats for compatibility
            details = {
                "total_turns": total_turns,
                # Turns that could not be judged (failed/throttled). Re-run a
                # second pass to evaluate these. Empty list == fully judged.
                "unjudged_turns": [r.turn_number for r in conv_results if r.error],
                "unjudged_turn_count": sum(1 for r in conv_results if r.error),
                "total_responses": total_responses,  # For report generator
                "total_claims": len(claim_evaluations),  # For report generator
                "import_hallucinations": import_hallucinations,
                "install_hallucinations": install_hallucinations,
                "function_hallucinations": function_hallucinations,
                "import_hallucinated_responses": import_hallucinated_responses,  # For report generator
                "install_hallucinated_responses": install_hallucinated_responses,  # For report generator
                "function_hallucinated_responses": function_hallucinated_responses,  # For report generator
                "overall_hallucinated_responses": overall_hallucinated_responses,  # For report generator
                # Conversation-level flags (for report generator)
                "hallucinated_import_detected": import_hallucinations > 0,
                "hallucinated_install_detected": install_hallucinations > 0,
                "hallucinated_function_usage_detected": function_hallucinations > 0,
                # Keep turn_evaluations for backward compatibility
                "turn_evaluations": [r.to_dict() for r in conv_results],
                # Add claim_evaluations for report generator compatibility
                "claim_evaluations": claim_evaluations,
            }
            
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
        output_path = self._get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same dir, then atomically replace, so an
        # interrupt mid-write (esp. during a checkpoint) never corrupts the
        # resumable output file.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)

        logger.info(f"✓ Saved {len(results)} evaluation results")
        return output_path

    def _load_prior_results(self, path: Path):
        """Load previously-judged turns from an existing output file (resume).

        Returns (prior_results, judged_keys) where prior_results is a list of
        DirectCodingResult for turns that were successfully judged, and
        judged_keys is the set of (conversation_id, turn_number) for those turns.
        Turns with an `error` (unjudged) are skipped so they get re-judged.
        """
        from ..workers import DirectCodingResult

        prior_results: List["DirectCodingResult"] = []
        judged_keys: set = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                details = rec.get("details", {})
                for te in details.get("turn_evaluations", []):
                    if te.get("error"):
                        continue  # unjudged -> re-judge on this pass
                    cid = te.get("conversation_id", rec.get("conversation_id", 0))
                    tn = te.get("turn_number", 0)
                    if (cid, tn) in judged_keys:
                        continue
                    prior_results.append(DirectCodingResult(
                        conversation_id=cid,
                        turn_number=tn,
                        has_hallucination=te.get("has_hallucination", False),
                        hallucinated_imports=te.get("hallucinated_imports", []),
                        hallucinated_installs=te.get("hallucinated_installs", []),
                        hallucinated_function_calls=te.get("hallucinated_function_calls", []),
                        hallucinated_import_detected=te.get("hallucinated_import_detected", False),
                        hallucinated_install_detected=te.get("hallucinated_install_detected", False),
                        hallucinated_function_usage_detected=te.get("hallucinated_function_usage_detected", False),
                        reasoning=te.get("reasoning", ""),
                        error=te.get("error"),
                        token_usage=te.get("token_usage", {}),
                    ))
                    judged_keys.add((cid, tn))
        return prior_results, judged_keys

    async def run(self) -> Path:
        """Run the direct coding pipeline."""
        from ..workers import DirectCodingJudgeWorker, TurnItem, DirectCodingResult
        
        # Validate input
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Log configuration
        logger.info("=" * 70)
        logger.info("CODING DIRECT PIPELINE (OpenAI Websearch)")
        logger.info(f"Task: {self.task_name}")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Model: {self.judge_model or 'gpt-5-mini-medium-websearch'}")
        logger.info(f"Workers: {self.worker_config.num_judges} judges")
        logger.info("=" * 70 + "\n")
        
        # Load data
        self._load_conversations()
        
        # Extract assistant turns
        turns_data = self._extract_assistant_turns()
        logger.info(f"Extracted {len(turns_data)} assistant turns to evaluate")
        
        if not turns_data:
            logger.warning("No assistant turns found to evaluate")
            return self._save_results([])

        # --- Resume: if a prior output exists, keep already-judged turns and
        # re-judge only the unjudged/missing ones, then merge. Delete the output
        # file to force a full fresh run. ---
        prior_results: List[DirectCodingResult] = []
        existing_output = self._get_output_path()
        if existing_output.exists():
            prior_results, judged_keys = self._load_prior_results(existing_output)
            before = len(turns_data)
            turns_data = [t for t in turns_data if (t[0], t[1]) not in judged_keys]
            logger.info(
                f"Resume: {len(judged_keys)} turn(s) already judged in {existing_output.name}; "
                f"re-judging {len(turns_data)}/{before} unjudged/missing turn(s)"
            )
            if not turns_data:
                logger.info("Resume: nothing left to judge — rebuilding output from existing results")
                return self._save_results(self._build_evaluation_results(prior_results))

        # Create sampler
        sampler = get_sampler(self.judge_model or "gpt-5-mini-medium-websearch")

        # Accumulate finished turns and checkpoint periodically so the run is
        # always resumable (an interrupt loses at most checkpoint_interval turns).
        self._results_accumulator: List[DirectCodingResult] = []

        async def _checkpoint():
            merged = prior_results + list(self._results_accumulator)
            self._save_results(self._build_evaluation_results(merged))
            logger.info(f"  ↳ checkpoint: persisted {len(merged)} judged turn(s)")

        # Create queues
        input_queue: MonitoredQueue[TurnItem] = MonitoredQueue("turns")
        results_queue: MonitoredQueue[DirectCodingResult] = MonitoredQueue("results")

        # Create worker
        judge = DirectCodingJudgeWorker(
            input_queue=input_queue,
            output_queue=results_queue,
            sampler=sampler,
            num_workers=self.worker_config.num_judges,
            result_sink=self._results_accumulator,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_cb=_checkpoint,
        )
        
        # Create pipeline
        # Note: Don't add results_queue to queues list - it would cause join() deadlock
        # Results are collected after pipeline completes, not consumed during run
        pipeline = Pipeline(name=self.name)
        pipeline.add_queue(input_queue)
        pipeline.add_worker(judge)
        pipeline.set_results_queue(results_queue)
        
        # Load turns into queue
        for conv_id, turn_num, content in turns_data:
            await input_queue.put(
                TurnItem(
                    conversation_id=conv_id,
                    turn_number=turn_num,
                    content=content,
                ),
                claim_id=f"turn-{conv_id}-{turn_num}",
                conversation_id=conv_id,
            )
        input_queue.close()
        
        # Run pipeline (handles starting workers, monitoring, and completion)
        logger.info(f"Starting pipeline with {len(turns_data)} turns...")
        
        try:
            await pipeline.run(
                input_queue=input_queue,
                total_items=len(turns_data),
                monitor_interval=self.monitor_interval,
            )
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user — saving turns judged so far")

        # Use the accumulator as the source of truth (every finished turn was
        # appended to it), so partial progress survives an interrupt.
        new_results = list(self._results_accumulator)
        new_count = len(new_results)
        # Merge previously-judged turns back in (resume)
        all_results = prior_results + new_results
        logger.info(f"\n✓ Collected {len(all_results)} results "
                    f"({len(prior_results)} from prior run, {new_count} newly judged)")

        # Build evaluation results
        eval_results = self._build_evaluation_results(all_results)
        
        # Save
        output_path = self._save_results(eval_results)
        
        # Log summary
        total_import_h = sum(r.details.get("import_hallucinations", 0) for r in eval_results)
        total_install_h = sum(r.details.get("install_hallucinations", 0) for r in eval_results)
        total_function_h = sum(r.details.get("function_hallucinations", 0) for r in eval_results)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Conversations: {len(eval_results)}")
        logger.info(f"Turns evaluated: {len(all_results)}")
        logger.info(f"Import hallucinations: {total_import_h}")
        logger.info(f"Install hallucinations: {total_install_h}")
        logger.info(f"Function call hallucinations: {total_function_h}")
        logger.info(f"Results: {output_path}")
        logger.info("=" * 70)
        
        return output_path


# =============================================================================
# Factory Function
# =============================================================================
