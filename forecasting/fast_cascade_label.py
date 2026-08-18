"""Fast cascade labeling via OpenAI (no Serper). Use after generate_future_turns.py.

Same labels as factscore_serper_cascades.py but ~10x faster for quick iteration.
Document in paper as GPT-4o-mini trajectory judge.

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

from config import CASCADE_RESULTS_PATH, FUTURE_TURNS_PATH
from follow_up_prompts import follow_up_fields, future_turn_fields, trajectory_key

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


def format_turns(row: dict) -> str:
    fields = future_turn_fields(row)
    prompts = follow_up_fields(row)
    chunks: list[str] = []
    turn_index = 0
    for field in fields:
        if field == "original_answer":
            continue
        turn_index += 1
        user_q = prompts[turn_index - 1] if turn_index - 1 < len(prompts) else ""
        assistant = row.get(field, "")
        chunks.append(
            f"FOLLOW-UP {turn_index} / RESPONSE {turn_index}:\n"
            f"User: {user_q}\n"
            f"Assistant: {assistant[:2000]}"
        )
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


def load_done_keys() -> set[str]:
    if not CASCADE_RESULTS_PATH.exists():
        return set()
    done: set[str] = set()
    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                done.add(trajectory_key(json.loads(line)))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_LABEL_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    client = OpenAI(api_key=api_key)

    rows = [json.loads(line) for line in open(FUTURE_TURNS_PATH) if line.strip()]
    done = load_done_keys() if args.resume else set()
    pending = [r for r in rows if trajectory_key(r) not in done]
    print(f"Labeling {len(pending)}/{len(rows)} conversations (model={args.model})")

    mode = "a" if args.resume and done else "w"
    with open(CASCADE_RESULTS_PATH, mode, encoding="utf-8") as out:
        for i, row in enumerate(pending, 1):
            result = label_trajectory(client, row, args.model)
            key = trajectory_key(row)
            record = {
                "question_number": row["question_number"],
                "branch_id": row.get("branch_id", key),
                "domain": row.get("domain"),
                "follow_up_mode": row.get("follow_up_mode", "challenge"),
                "strategy": row.get("strategy") or row.get("follow_up_mode"),
                "n_turns": row.get("n_turns"),
                "labeler": "fast_gpt",
                **result,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i}/{len(pending)}] {key} -> {result['final_label']}")

    from collections import Counter

    counts = Counter()
    for line in open(CASCADE_RESULTS_PATH):
        if line.strip():
            counts[json.loads(line)["final_label"]] += 1
    print("Label counts:", dict(counts))
    print("Next: python forecasting/summarize_strategy_outcomes.py")
    print("Optional: python forecasting/predict_cascades.py")


if __name__ == "__main__":
    main()
