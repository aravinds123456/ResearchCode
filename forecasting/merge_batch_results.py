"""Merge scattered forecasting JSONL files into one batch_results.jsonl.

Your pipeline only reads ONE file:
  forecasting/batch_results.jsonl

Each line must look like:
  {
    "question_number": 0,
    "question": "...",
    "qwen_answer": "...",
    "gemini_judgement": "Overall label: Hallucinating"  (or Not Hallucinating)
  }

Run from repo root:
  python forecasting/merge_batch_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

FORECASTING_DIR = Path(__file__).resolve().parent
OUTPUT = FORECASTING_DIR / "batch_results.jsonl"

# Files to merge (newest / last occurrence wins for same question_number)
INPUT_FILES = [
    FORECASTING_DIR / "batch_results.jsonl",
    FORECASTING_DIR / "qwen_answers.jsonl",
    FORECASTING_DIR / "sample_results.jsonl",
    FORECASTING_DIR / "halluhard_input.jsonl",
    FORECASTING_DIR / "halluhard_factual_input.jsonl",
    FORECASTING_DIR / "halluhard_openai_input.jsonl",
    FORECASTING_DIR / "halluhard_kimi_input.jsonl",
    Path("HallucinationResearch-main/batch_results.jsonl"),
]


def normalize_record(raw: dict, fallback_id: int) -> dict | None:
    """Map different file formats into batch_results format."""
    question = (
        raw.get("question")
        or raw.get("research_question")
        or raw.get("prompt")
        or ""
    ).strip()
    answer = (
        raw.get("qwen_answer")
        or raw.get("answer")
        or raw.get("response")
        or raw.get("model_answer")
        or ""
    ).strip()
    judgment = (
        raw.get("gemini_judgement")
        or raw.get("gemini_judgment")
        or raw.get("judgement")
        or raw.get("judgment")
        or ""
    ).strip()

    if not question or not answer:
        return None

    qnum = raw.get("question_number")
    if qnum is None:
        qnum = raw.get("id", fallback_id)

    return {
        "question_number": int(qnum),
        "question": question,
        "qwen_answer": answer,
        "gemini_judgement": judgment,
    }


def main() -> None:
    merged: dict[int, dict] = {}
    fallback_id = 0

    for path in INPUT_FILES:
        if not path.exists():
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                record = normalize_record(raw, fallback_id)
                if record is None:
                    continue
                merged[record["question_number"]] = record
                count += 1
                fallback_id = max(fallback_id, record["question_number"] + 1)
        print(f"Read {count:4d} lines from {path}")

    if not merged:
        print("\nNo records found. Check that your files exist under forecasting/")
        return

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for qnum in sorted(merged):
            out.write(json.dumps(merged[qnum], ensure_ascii=False) + "\n")

    hall = sum(
        1
        for r in merged.values()
        if r["gemini_judgement"].startswith("Overall label: Hallucinating")
    )
    print(f"\nWrote {len(merged)} unique questions -> {OUTPUT}")
    print(f"Hallucinating (with Gemini label): {hall}")
    print(f"Missing gemini_judgement: {sum(1 for r in merged.values() if not r['gemini_judgement'])}")


if __name__ == "__main__":
    main()
