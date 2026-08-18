"""Shared paths and device settings for the forecasting pipeline."""

from __future__ import annotations

from pathlib import Path


FORECASTING_DIR = Path(__file__).resolve().parent

BATCH_RESULTS_PATH = FORECASTING_DIR / "batch_results.jsonl"
ORIGINAL_JUDGMENTS_PATH = FORECASTING_DIR / "original_judgments.jsonl"
FUTURE_TURNS_PATH = FORECASTING_DIR / "future_turns.jsonl"
CASCADE_RESULTS_PATH = FORECASTING_DIR / "factscore_cascade_results.jsonl"

MODEL_NAME = "Qwen/Qwen3.5-2B"
DEFAULT_DRAFT_MODEL = "gpt-4o-mini"


def get_device() -> str:

    import torch

    """Pick the best available device (Mac MPS, CUDA GPU, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

