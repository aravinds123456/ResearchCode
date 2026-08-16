"""Step 1: Fact-check only the first (original) answers in batch_results.jsonl.

Run this before generate_future_turns.py so the pipeline knows which
questions contain unsupported claims.
"""

from __future__ import annotations

import argparse
import json

from config import BATCH_RESULTS_PATH, ORIGINAL_JUDGMENTS_PATH
from factscore_utils import has_unsupported_claim, judge_answer


def load_batch_records(limit: int | None = None) -> list[dict]:
    records: dict[int, dict] = {}
    with open(BATCH_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["question_number"]] = record

    ordered = sorted(records.values(), key=lambda row: row["question_number"])
    if limit is not None:
        return ordered[:limit]
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fact-check original answers only (pipeline step 1)."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit how many conversations to judge (default: all).",
    )
    args = parser.parse_args()

    records = load_batch_records(limit=args.max_examples)

    with open(ORIGINAL_JUDGMENTS_PATH, "w", encoding="utf-8") as output:
        for record in records:
            question_number = record["question_number"]
            original_answer = record["qwen_answer"]
            claims = judge_answer(original_answer)
            has_unsupported = has_unsupported_claim(claims)

            result = {
                "question_number": question_number,
                "question": record["question"],
                "original_answer": original_answer,
                "claims": claims,
                "has_unsupported": has_unsupported,
            }
            output.write(json.dumps(result) + "\n")
            print(
                f"question {question_number}: "
                f"{'unsupported' if has_unsupported else 'supported'}"
            )


if __name__ == "__main__":
    main()
