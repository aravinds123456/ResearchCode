"""Paths and ID offsets for HalluHard domains."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORECASTING_DIR = Path(__file__).resolve().parent

DOMAINS = {
    "research": {
        "data_path": REPO_ROOT / "research_questions/data/research_questions_all.jsonl",
        "question_key": "research_question",
        "id_offset": 0,
        "output_path": FORECASTING_DIR / "batch_results_research.jsonl",
    },
    "legal": {
        "data_path": REPO_ROOT / "legal_cases/data/legal_cases_all.jsonl",
        "question_key": "question",
        "id_offset": 100_000,
        "output_path": FORECASTING_DIR / "batch_results_legal.jsonl",
    },
    "medical": {
        "data_path": REPO_ROOT / "medical_guidelines/data/guidelines.jsonl",
        "question_key": "question",
        "id_offset": 200_000,
        "output_path": FORECASTING_DIR / "batch_results_medical.jsonl",
    },
}

MERGED_OUTPUT = FORECASTING_DIR / "batch_results.jsonl"
