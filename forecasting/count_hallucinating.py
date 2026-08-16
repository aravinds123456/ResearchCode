"""Print hallucination counts and percentages from merged batch_results.jsonl."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BATCH_PATH = Path(__file__).resolve().parent / "batch_results.jsonl"


def main() -> None:
    total: Counter[str] = Counter()
    hall: Counter[str] = Counter()

    with open(BATCH_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            domain = row.get("domain", "?")
            total[domain] += 1
            total["ALL"] += 1
            if row.get("gemini_judgement", "").startswith("Overall label: Hallucinating"):
                hall[domain] += 1
                hall["ALL"] += 1

    for domain in sorted(k for k in total if k != "ALL"):
        t, h = total[domain], hall[domain]
        pct = 100 * h / t if t else 0
        print(f"{domain:10s}  {h:3d}/{t:3d} hallucinating  ({pct:.1f}%)")

    t, h = total["ALL"], hall["ALL"]
    pct = 100 * h / t if t else 0
    print(f"{'TOTAL':10s}  {h:3d}/{t:3d} hallucinating  ({pct:.1f}%)")
    print(f"Missing labels: {sum(1 for _ in open(BATCH_PATH) if not json.loads(_).get('gemini_judgement'))}")


if __name__ == "__main__":
    main()
