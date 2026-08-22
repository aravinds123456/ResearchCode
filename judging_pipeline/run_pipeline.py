"""Main entry point for queue-based evaluation pipeline.

Usage:
    python -m judging_pipeline.run_pipeline --input data/conversations.jsonl --type webscraper --task research_questions

Features:
    - Clear visibility into queue depths (see where bottlenecks are)
    - Easy to scale individual workers
    - Each external API has its own rate limiting
    - Real-time progress monitoring

Pipeline types:
    - openai: Claims -> Judge (with OpenAI websearch)
    - serper: Claims -> Search -> Fetch -> Filter -> Judge (Serper snippets only)
    - webscraper: Claims -> Search -> Fetch -> PDF -> Filter -> Judge (full scraping)
    - coding_direct: Turns -> Judge (OpenAI websearch, no claim extraction, coding only)

The pipeline implementations live in judging_pipeline/pipelines/; this module
only wires them to the command line.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

from .logging_config import configure_logging, get_logger
from .pipelines import (
    BasePipeline,
    CodingDirectPipeline,
    OpenAIPipeline,
    SerperPipeline,
    WebscraperPipeline,
    WorkerConfig,
)

configure_logging()
logger = get_logger()


PIPELINE_CLASSES = {
    "openai": OpenAIPipeline,
    "serper": SerperPipeline,
    "webscraper": WebscraperPipeline,
    "coding_direct": CodingDirectPipeline,
}


def create_pipeline(
    judging_type: Literal["openai", "serper", "webscraper", "coding_direct"],
    input_path: str | Path,
    task_name: str,
    output_path: str | Path | None = None,
    worker_config: WorkerConfig | None = None,
    n_conversations: int | None = None,
    max_claims_per_turn: int | None = None,
    seed: int = 42,
    monitor_interval: float = 3.0,
    base_path: str | Path | None = None,
    claims_cache_path: str | Path | None = None,
    checkpoint_interval: int = 100,
    max_claims_per_category: int | None = None,
    judge_model: str | None = None,
    judge_fallback_model: str | None = None,
) -> BasePipeline:
    """Factory function to create the appropriate pipeline."""

    base_path = Path(base_path) if base_path else Path(__file__).parent.parent

    if judging_type not in PIPELINE_CLASSES:
        raise ValueError(f"Unknown judging type: {judging_type}. Must be one of {list(PIPELINE_CLASSES.keys())}")
    
    pipeline_class = PIPELINE_CLASSES[judging_type]
    
    common = dict(
        input_path=Path(input_path),
        output_path=Path(output_path) if output_path else None,
        worker_config=worker_config or WorkerConfig(),
        task_name=task_name,
        n_conversations=n_conversations,
        max_claims_per_turn=max_claims_per_turn,
        seed=seed,
        monitor_interval=monitor_interval,
        base_path=base_path,
        claims_cache_path=Path(claims_cache_path) if claims_cache_path else None,
        checkpoint_interval=checkpoint_interval,
        max_claims_per_category=max_claims_per_category,
    )
    if judging_type == "coding_direct":
        return pipeline_class(**common, judge_model=judge_model)
    return pipeline_class(
        **common,
        judge_model=judge_model,
        judge_fallback_model=judge_fallback_model,
    )


async def run_evaluation_pipeline(
    input_path: str | Path,
    judging_type: Literal["openai", "serper", "webscraper", "coding_direct"],
    task_name: str,
    output_path: str | Path | None = None,
    num_extractors: int = 5,
    num_searchers: int = 10,
    num_fetchers: int = 20,
    num_pdf_converters: int = 3,
    num_filters: int = 10,
    num_judges: int = 20,
    n_conversations: int | None = None,
    max_claims_per_turn: int | None = None,
    seed: int = 42,
    monitor_interval: float = 3.0,
    base_path: str | Path | None = None,
    claims_cache_path: str | Path | None = None,
    checkpoint_interval: int = 100,
    max_claims_per_category: int | None = None,
    judge_model: str | None = None,
    judge_fallback_model: str | None = None,
) -> Path:
    """Convenience function to create and run a pipeline."""
    worker_config = WorkerConfig(
        num_extractors=num_extractors,
        num_searchers=num_searchers,
        num_fetchers=num_fetchers,
        num_pdf_converters=num_pdf_converters,
        num_filters=num_filters,
        num_judges=num_judges,
    )
    
    pipeline = create_pipeline(
        judging_type=judging_type,
        input_path=input_path,
        task_name=task_name,
        output_path=output_path,
        worker_config=worker_config,
        n_conversations=n_conversations,
        max_claims_per_turn=max_claims_per_turn,
        seed=seed,
        monitor_interval=monitor_interval,
        base_path=base_path,
        claims_cache_path=claims_cache_path,
        checkpoint_interval=checkpoint_interval,
        max_claims_per_category=max_claims_per_category,
        judge_model=judge_model,
        judge_fallback_model=judge_fallback_model,
    )
    
    return await pipeline.run()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run queue-based evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["openai", "serper", "webscraper", "coding_direct"],
        default="webscraper",
        help="Pipeline type. 'coding_direct' uses OpenAI websearch to judge entire turns without claim extraction.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["research_questions", "medical_guidelines", "legal_cases", "coding"],
        help="Task domain (determines prompts and logic)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--n_conversations", "-n",
        type=int,
        default=None,
        help="Number of conversations to process. For coding tasks: first N per language. For others: first N total.",
    )
    parser.add_argument(
        "--max_claims_per_turn",
        type=int,
        default=None,
        help="Maximum claims per turn to evaluate",
    )
    parser.add_argument(
        "--max-claims-per-category",
        type=int,
        default=0,
        help="For coding tasks: max claims per category (import/install/function). Default 0 (disabled). Use early stopping instead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for loading system prompts",
    )
    parser.add_argument(
        "--claims-cache",
        type=str,
        default=None,
        help="Path to claims cache file. If not provided, looks for <input>_extracted_claims_cache.jsonl",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N judgments (default: 100). Set to 0 to disable checkpointing.",
    )
    
    # Worker counts
    parser.add_argument("--extractors", type=int, default=20, help="Claim extractor workers")
    parser.add_argument("--searchers", type=int, default=100, help="Web search workers")
    parser.add_argument("--fetchers", type=int, default=50, help="Web fetch workers")
    parser.add_argument("--pdf-converters", type=int, default=10, help="PDF converter workers")
    parser.add_argument("--filters", type=int, default=100, help="Content filter workers")
    parser.add_argument("--judges", type=int, default=200, help="Judge workers")
    
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=3.0,
        help="Seconds between progress updates",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "For --type serper or webscraper: registry id for the primary claim judge "
            "(default: gpt-5-mini-medium). Ignored for openai and coding_direct."
        ),
    )
    parser.add_argument(
        "--judge-fallback-model",
        type=str,
        default=None,
        help=(
            "For --type serper or webscraper: registry id for the judge used on the web-grounding "
            "fallback path (default: gpt-5-mini-medium-websearch). Ignored for openai and coding_direct."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(level=log_level)
    
    # Fix Windows asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Handle max_claims_per_category: 0 means disabled (None)
    max_per_cat = args.max_claims_per_category if args.max_claims_per_category > 0 else None
    
    asyncio.run(
        run_evaluation_pipeline(
            input_path=args.input,
            judging_type=args.type,
            task_name=args.task,
            output_path=args.output,
            num_extractors=args.extractors,
            num_searchers=args.searchers,
            num_fetchers=args.fetchers,
            num_pdf_converters=args.pdf_converters,
            num_filters=args.filters,
            num_judges=args.judges,
            n_conversations=args.n_conversations,
            max_claims_per_turn=args.max_claims_per_turn,
            seed=args.seed,
            monitor_interval=args.monitor_interval,
            base_path=args.base_path,
            claims_cache_path=args.claims_cache,
            checkpoint_interval=args.checkpoint_interval,
            max_claims_per_category=max_per_cat,
            judge_model=args.judge_model,
            judge_fallback_model=args.judge_fallback_model,
        )
    )
