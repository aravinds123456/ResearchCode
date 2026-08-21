"""Task 1 deliverable: generate and eyeball LLM follow-ups for real hallucinations.

Runs the claim extractor + follow-up generator on seed hallucinations from
batch_results.jsonl. No Qwen, no GPU — this is for reviewing follow-up quality
before spending compute on the full tree.

  export OPENAI_API_KEY=sk-...
  python forecasting/preview_followups.py --n 3
  python forecasting/preview_followups.py --n 5 --state new_hallucination
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from followup_defs import (
    CATEGORIES,
    PRESSURE_ROLE,
    SEED_STATE,
    TURN_STATES,
    extract_claim,
    generate_followup,
)

BATCH_RESULTS_PATH = SCRIPT_DIR / "batch_results.jsonl"
PREVIEW_PATH = SCRIPT_DIR / "followups_preview.jsonl"


def load_hallucinations(domain: str | None) -> list[dict]:
    rows = []
    with open(BATCH_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if domain and record.get("domain") != domain:
                continue
            if record.get("gemini_judgement", "").strip().startswith(
                "Overall label: Hallucinating"
            ):
                rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview LLM-generated follow-ups.")
    parser.add_argument("--n", type=int, default=3, help="Seed hallucinations to preview.")
    parser.add_argument("--domain", choices=["research", "legal", "medical"], default=None)
    parser.add_argument(
        "--state",
        choices=list(TURN_STATES),
        default=SEED_STATE,
        help="Assistant turn state to condition on (default: persisted).",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    seeds = load_hallucinations(args.domain)
    if not seeds:
        raise SystemExit("No hallucinating rows found in batch_results.jsonl")
    random.seed(42)
    random.shuffle(seeds)
    seeds = seeds[: args.n]

    print(f"Previewing {len(seeds)} hallucinations x {len(CATEGORIES)} categories")
    print(f"Conditioned on turn state: {args.state}\n")

    sources: Counter[str] = Counter()
    with open(PREVIEW_PATH, "w", encoding="utf-8") as out:
        for record in seeds:
            question = record["question"]
            answer = record["qwen_answer"]
            target = extract_claim(question, answer, model=args.model)

            print("=" * 78)
            print(f"q{record['question_number']} ({record.get('domain', '?')})")
            print(f"QUESTION: {question[:150]}")
            print(f"TARGET CLAIM: {target['claim'][:220]}")
            print(f"ENTITIES: {target['entities']}\n")

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            for category in CATEGORIES:
                result = generate_followup(
                    question,
                    target["claim"],
                    target["entities"],
                    messages,
                    category,
                    turn_state=args.state,
                    model=args.model,
                )
                sources[result["source"]] += 1
                flag = "" if result["source"].startswith("llm") else f"  [{result['validation']}]"
                print(f"  {category:<20} ({PRESSURE_ROLE[category]})")
                print(f"    {result['follow_up']}{flag}\n")

                out.write(
                    json.dumps(
                        {
                            "question_number": record["question_number"],
                            "domain": record.get("domain"),
                            "claim": target["claim"],
                            "entities": target["entities"],
                            **result,
                        }
                    )
                    + "\n"
                )

    total = sum(sources.values())
    llm = sum(count for source, count in sources.items() if source.startswith("llm"))
    print("=" * 78)
    print(f"Generated {total} follow-ups | LLM-valid: {llm} | fallback: {total - llm}")
    print(f"Saved -> {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
