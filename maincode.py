"""Merged HallucinationResearchTest + HalluHard cascade runner.

The experiment lives in forecasting/. This file keeps `python maincode.py`
working with the HallucinationResearchTest environment variables:

    TEST_MODEL, MAX_EXAMPLES, NUM_TURNS, INPUT_PATH, OUTPUT_PATH,
    DEVICE, GEMINI_API_KEY / OPENAI_API_KEY, JUDGE_BACKEND

Prefer the explicit CLI:

    python forecasting/generate_seeds.py
    python forecasting/pipeline.py tree --max-seeds 100 --levels 3 --resume
    python forecasting/pipeline.py report --from-partial
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FORECASTING = Path(__file__).resolve().parent / "forecasting"
if str(FORECASTING) not in sys.path:
    sys.path.insert(0, str(FORECASTING))


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] not in {
        "answer", "judge", "tree", "label", "report", "seeds", "-h", "--help",
    }:
        seeds = os.environ.get("INPUT_PATH", str(FORECASTING / "batch_results.jsonl"))
        out = os.environ.get("OUTPUT_PATH", str(FORECASTING / "cascade_tree.jsonl"))
        max_seeds = os.environ.get("MAX_EXAMPLES", "100")
        levels = os.environ.get("NUM_TURNS", "3")
        extra = sys.argv[1:]
        sys.argv = [
            sys.argv[0], "tree",
            "--seeds", seeds,
            "--out", out,
            "--max-seeds", max_seeds,
            "--levels", levels,
            "--resume",
            *extra,
        ]
    from pipeline import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
