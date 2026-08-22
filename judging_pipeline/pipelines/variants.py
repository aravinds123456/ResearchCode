"""Claim-based pipeline variants: openai, serper, webscraper."""

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

from .base import BasePipeline, ModelConfig


# =============================================================================
# OpenAI Pipeline (Direct websearch via LLM)
# =============================================================================

class OpenAIPipeline(BasePipeline):
    """Pipeline using OpenAI's built-in websearch for judgment."""
    
    name = "openai"
    
    def _get_model_config(self) -> ModelConfig:
        return ModelConfig(
            extractor="gpt-5-mini-minimal",
            judge="gpt-5-mini-medium-websearch",
            judge_fallback="gpt-5-mini-medium-websearch",
            search=None,
        )
    
    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, Judge={self.worker_config.num_judges}")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # ClaimToFiltered passthrough worker
        passthrough = ClaimToFilteredWorker(
            input_queue=queues["claims"],
            output_queue=queues["filtered"],
            num_workers=self.worker_config.num_extractors,
        )
        
        # Judge with websearch
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [passthrough, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [queues["claims"], queues["filtered"]]


# =============================================================================
# Serper Pipeline (Search API with snippets)
# =============================================================================

class SerperPipeline(BasePipeline):
    """Pipeline using Serper API for web search (snippets only)."""
    
    name = "serper"
    
    def _get_model_config(self) -> ModelConfig:
        judge = self.judge_model or "gpt-5-mini-medium"
        judge_fb = self.judge_fallback_model or "gpt-5-mini-medium-websearch"
        # When a judge model is explicitly set, use it for extraction and search
        # too (avoids requiring a separate OpenAI key for those roles).
        aux = self.judge_model or "gpt-5-mini-minimal"
        return ModelConfig(
            extractor=aux,
            search=aux,
            judge=judge,
            judge_fallback=judge_fb,
        )

    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, "
                   f"Search={self.model_config.search}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, "
                   f"Search={self.worker_config.num_searchers}, "
                   f"Judge={self.worker_config.num_judges}")
        logger.info("Mode: Serper snippets only (direct to judge)")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "search": MonitoredQueue[SearchTask]("search_tasks"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # Search via Serper API
        searcher = WebSearcherWorker(
            input_queue=queues["claims"],
            output_queue=queues["search"],
            search_sampler=self.samplers["search"],
            claim_text_builder=self.strategy.build_textual_claim_for_websearch,
            strategy=self.strategy,
            num_workers=self.worker_config.num_searchers,
            rate_limit_delay=0.1,
            max_searches=1 if self.task_name == "coding" else 3,  # Coding needs fewer iterations
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        # Convert SearchTask directly to FilteredContent (skipping extraction/filtering)
        search_to_filtered = SearchToFilteredWorker(
            input_queue=queues["search"],
            output_queue=queues["filtered"],
            num_workers=self.worker_config.num_searchers,
        )
        
        # Judge
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [searcher, search_to_filtered, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [queues["claims"], queues["search"], queues["filtered"]]


# =============================================================================
# Webscraper Pipeline (Full scraping with PDF support)
# =============================================================================

class WebscraperPipeline(BasePipeline):
    """Pipeline with full web scraping and PDF extraction."""
    
    name = "webscraper"
    
    # Store aggregator separately since it's not a standard Worker
    _aggregator: ContentAggregatorWorker | None = None
    
    def _get_model_config(self) -> ModelConfig:
        judge = self.judge_model or "gpt-5-mini-medium"
        judge_fb = self.judge_fallback_model or "gpt-5-mini-medium-websearch"
        # When a judge model is explicitly set, use it for extraction and search
        # too (avoids requiring a separate OpenAI key for those roles).
        aux = self.judge_model or "gpt-5-mini-minimal"
        return ModelConfig(
            extractor=aux,
            search=aux,
            judge=judge,
            judge_fallback=judge_fb,
        )

    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, "
                   f"Search={self.model_config.search}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, "
                   f"Search={self.worker_config.num_searchers}, "
                   f"Fetch={self.worker_config.num_fetchers}, "
                   f"PDF={self.worker_config.num_pdf_converters}, "
                   f"Filter={self.worker_config.num_filters}, "
                   f"Judge={self.worker_config.num_judges}")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "search": MonitoredQueue[SearchTask]("search_tasks"),
            "content": MonitoredQueue[ContentItem]("content"),
            "pdf": MonitoredQueue[PDFTask]("pdf_tasks"),
            "pdf_result": MonitoredQueue[PDFResult]("pdf_results"),
            "aggregated": MonitoredQueue[ContentItem]("aggregated"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # Search
        searcher = WebSearcherWorker(
            input_queue=queues["claims"],
            output_queue=queues["search"],
            search_sampler=self.samplers["search"],
            claim_text_builder=self.strategy.build_textual_claim_for_websearch,
            strategy=self.strategy,
            num_workers=self.worker_config.num_searchers,
            rate_limit_delay=0.1,
            max_searches=1 if self.task_name == "coding" else 3,  # Coding needs fewer iterations
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        # Fetch with PDF queue
        fetcher = WebFetcherWorker(
            input_queue=queues["search"],
            output_queue=queues["content"],
            pdf_queue=queues["pdf"],
            num_workers=self.worker_config.num_fetchers,
            early_stopping_state=self.early_stopping_state,
        )
        
        # PDF conversion - outputs to pdf_result queue
        pdf_converter = PDFConverterWorker(
            input_queue=queues["pdf"],
            output_queue=queues["pdf_result"],
            num_workers=self.worker_config.num_pdf_converters,
        )
        
        # Content aggregator (merges HTML content with PDF results)
        self._aggregator = ContentAggregatorWorker(
            content_queue=queues["content"],
            pdf_queue=queues["pdf_result"],
            output_queue=queues["aggregated"],
        )
        
        # Filter - now reads from aggregated queue
        content_filter = ContentFilterWorker(
            input_queue=queues["aggregated"],
            output_queue=queues["filtered"],
            claim_text_builder=self.strategy.build_textual_claim_for_judging,
            num_workers=self.worker_config.num_filters,
            early_stopping_state=self.early_stopping_state,
        )
        
        # Judge
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [searcher, fetcher, pdf_converter, self._aggregator, content_filter, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [
            queues["claims"], 
            queues["search"], 
            queues["content"],
            queues["pdf"],
            queues["pdf_result"],
            queues["aggregated"],
            queues["filtered"],
        ]
    
    async def run(self) -> Path:
        """Run the full pipeline with aggregator support."""
        # Standard run but with aggregator start/stop handling
        path = await super().run()
        return path


# =============================================================================
# Helper Worker for OpenAI Pipeline
# =============================================================================

class ClaimToFilteredWorker(Worker[ClaimItem, FilteredContent]):
    """Passthrough worker that converts ClaimItem to FilteredContent."""
    
    def __init__(
        self,
        input_queue: MonitoredQueue[ClaimItem],
        output_queue: MonitoredQueue[FilteredContent],
        num_workers: int = 5,
    ):
        super().__init__(
            name="ClaimToFiltered",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
    
    async def process(self, item: ClaimItem, item_wrapper) -> FilteredContent:
        return FilteredContent(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item,
            filtered_content="",
            search_results_text="",
            use_fallback=True,  # Use websearch-enabled judge
        )


class SearchToFilteredWorker(Worker[SearchTask, FilteredContent]):
    """Worker that converts SearchTask directly to FilteredContent."""
    
    def __init__(
        self,
        input_queue: MonitoredQueue[SearchTask],
        output_queue: MonitoredQueue[FilteredContent],
        num_workers: int = 5,
    ):
        super().__init__(
            name="SearchToFiltered",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
    
    async def process(self, item: SearchTask, item_wrapper) -> FilteredContent:
        return FilteredContent(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item.claim,
            filtered_content="",
            search_results_text=item.search_results_text,
            queries=item.queries_executed,  # Pass through search queries
            use_fallback=True,  # Use snippet-based judgment
            whitelist_skip=item.whitelist_skip,  # Propagate whitelist skip
        )
