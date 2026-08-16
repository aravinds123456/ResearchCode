"""Merge domain batch files into one batch_results.jsonl.

Keeps hallucinating label if ANY duplicate question_number was hallucinating.
"""

from __future__ import annotations

import json
from pathlib import Path

from domain_config import DOMAINS, MERGED_OUTPUT, FORECASTING_DIR

INPUT_FILES = [
    MERGED_OUTPUT,
    *[cfg["output_path"] for cfg in DOMAINS.values()],
    FORECASTING_DIR / "qwen_answers.jsonl",
    Path("HallucinationResearch-main/batch_results.jsonl"),
]


def normalize_record(raw: dict, fallback_id: int) -> dict | None:
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
        or ""
    ).strip()
    judgment = (
        raw.get("gemini_judgement")
        or raw.get("gemini_judgment")
        or raw.get("judgement")
        or ""
    ).strip()

    if not question or not answer:
        return None

    qnum = raw.get("question_number")
    if qnum is None:
        qnum = raw.get("id", fallback_id)

    domain = raw.get("domain")
    if not domain:
        qnum = int(qnum)
        if qnum >= 200_000:
            domain = "medical"
        elif qnum >= 100_000:
            domain = "legal"
        else:
            domain = "research"

    return {
        "question_number": int(qnum),
        "domain": domain,
        "question": question,
        "qwen_answer": answer,
        "gemini_judgement": judgment,
    }


def is_hallucinating(judgment: str) -> bool:
    return judgment.strip().startswith("Overall label: Hallucinating")


def merge_record(existing: dict | None, new: dict) -> dict:
    if existing is None:
        return new

    # Keep hallucinating if ANY version was hallucinating
    if is_hallucinating(existing["gemini_judgement"]):
        new["gemini_judgement"] = existing["gemini_judgement"]
    elif not new["gemini_judgement"] and existing["gemini_judgement"]:
        new["gemini_judgement"] = existing["gemini_judgement"]

    if not new.get("domain") and existing.get("domain"):
        new["domain"] = existing["domain"]

    return new


def main() -> None:
    merged: dict[int, dict] = {}
    fallback_id = 0

    for path in INPUT_FILES:
        if not path.exists():
            continue
        count = 0
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                record = normalize_record(raw, fallback_id)
                if record is None:
                    continue
                qnum = record["question_number"]
                merged[qnum] = merge_record(merged.get(qnum), record)
                count += 1
                fallback_id = max(fallback_id, qnum + 1)
        print(f"Read {count:4d} from {path.name} (skipped {skipped} bad lines)")

    if not merged:
        print("No records found.")
        return

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as out:
        for qnum in sorted(merged):
            out.write(json.dumps(merged[qnum], ensure_ascii=False) + "\n")

    from collections import Counter

    domains = Counter(r["domain"] for r in merged.values())
    hall = sum(1 for r in merged.values() if is_hallucinating(r["gemini_judgement"]))
    missing = sum(1 for r in merged.values() if not r["gemini_judgement"])

    print(f"\nWrote {len(merged)} questions -> {MERGED_OUTPUT}")
    print("By domain:", dict(domains))
    print(f"Hallucinating: {hall}")
    print(f"Missing gemini_judgement: {missing}")


if __name__ == "__main__":
    main()
