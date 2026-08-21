"""Fast cascade labeling via OpenAI (no Serper). Use after generate_future_turns.py.

Same labels as factscore_serper_cascades.py but ~10x faster for quick iteration.
Document in paper as GPT-4o-mini trajectory judge (challenge-condition runs).

Example:
  python forecasting/fast_cascade_label.py
  python forecasting/fast_cascade_label.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

FUTURE_TURNS_PATH = SCRIPT_DIR / "future_turns.jsonl"
CASCADE_RESULTS_PATH = SCRIPT_DIR / "factscore_cascade_results.jsonl"

LABEL_PROMPT = """You are labeling how a hallucinating LLM conversation evolves over follow-up turns.

QUESTION:
{question}

TURN 0 (original answer — known to contain hallucinations):
{original}

FOLLOW-UP TURNS:
{turns}

Classify the trajectory:

- corrected: A later turn explicitly fixes or retracts an earlier factual error with better-supported claims.
- snowballing: A later turn adds NEW false claims that build on or depend on the original error (model doubles down or extends the mistake).
- isolated: The original error is NOT corrected, but later turns do NOT compound it with dependent new false claims.

Be strict about snowballing: only use it when later false claims logically depend on the original mistake.
Do NOT label snowballing just because the model sounds confident.

Return JSON only:
{{"final_label": "corrected|snowballing|isolated", "reason": "one sentence"}}"""


def row_key(row: dict) -> str:
    if row.get("branch_id"):
        return str(row["branch_id"])
    return f"{row['question_number']}:{row.get('follow_up_mode', '')}"


def format_turns(row: dict) -> str:
    chunks = []
    index = 1
    while f"future_turn_{index}" in row:
        fu = row.get(f"follow_up_{index}", "")
        chunks.append(
            f"FOLLOW-UP {index} / RESPONSE {index}:\n"
            f"User: {fu}\n"
            f"Assistant: {row.get(f'future_turn_{index}', '')[:2000]}"
        )
        index += 1
    return "\n\n".join(chunks)


def label_trajectory(client, row: dict, model: str) -> dict:
    prompt = LABEL_PROMPT.format(
        question=row["question"][:2000],
        original=row["original_answer"][:2500],
        turns=format_turns(row)[:8000],
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    out = json.loads(response.choices[0].message.content or "{}")
    label = out.get("final_label", "isolated").lower().strip()
    if label not in {"corrected", "snowballing", "isolated"}:
        label = "isolated"
    return {"final_label": label, "reason": out.get("reason", "")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_LABEL_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--future",
        default=None,
        help="Conversations to label (default: future_turns.jsonl; use cascade_tree.jsonl for the tree).",
    )
    parser.add_argument("--out", default=None, help="Where to write labels.")
    args = parser.parse_args()

    future_path = Path(args.future) if args.future else FUTURE_TURNS_PATH
    if not future_path.is_absolute():
        future_path = SCRIPT_DIR / future_path
    results_path = Path(args.out) if args.out else CASCADE_RESULTS_PATH
    if not results_path.is_absolute():
        results_path = SCRIPT_DIR / results_path

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    client = OpenAI(api_key=api_key)

    rows = [json.loads(line) for line in open(future_path) if line.strip()]
    done: set[str] = set()
    if args.resume and results_path.exists():
        for line in open(results_path):
            if line.strip():
                done.add(row_key(json.loads(line)))

    pending = [r for r in rows if row_key(r) not in done]
    print(f"Labeling {len(pending)}/{len(rows)} conversations (model={args.model})")

    mode = "a" if args.resume and done else "w"
    with open(results_path, mode, encoding="utf-8") as out:
        for i, row in enumerate(pending, 1):
            result = label_trajectory(client, row, args.model)
            record = {
                "question_number": row["question_number"],
                "branch_id": row.get("branch_id"),
                "dataset": row.get("dataset"),
                "domain": row.get("domain"),
                "answer_model": row.get("answer_model"),
                "follow_up_mode": row.get("follow_up_mode", "challenge"),
                "labeler": "fast_gpt",
                **result,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i}/{len(pending)}] {row_key(row)} -> {result['final_label']}")

    from collections import Counter, defaultdict

    counts = Counter()
    by_mode: dict[str, Counter] = defaultdict(Counter)
    for line in open(results_path):
        if not line.strip():
            continue
        labeled = json.loads(line)
        counts[labeled["final_label"]] += 1
        by_mode[labeled.get("follow_up_mode", "")][labeled["final_label"]] += 1
    print("Label counts:", dict(counts))
    print("By follow-up category:")
    for mode, c in sorted(by_mode.items()):
        total = sum(c.values())
        snowball = c.get("snowballing", 0)
        share = f"{100.0 * snowball / total:.0f}%" if total else "—"
        print(f"  {mode:<20} n={total:<4} snowballing={snowball} ({share})  {dict(c)}")


if __name__ == "__main__":
    main()
