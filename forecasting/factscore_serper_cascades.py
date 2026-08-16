"""Step 3: Label full conversation trajectories as corrected / isolated / snowballing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

FUTURE_TURNS_PATH = SCRIPT_DIR / "future_turns.jsonl"
CASCADE_RESULTS_PATH = SCRIPT_DIR / "factscore_cascade_results.jsonl"

from factscore_utils import classify_trajectory, judge_answer

TURN_FIELDS = [
    "original_answer",
    "future_turn_1",
    "future_turn_2",
    "future_turn_3",
]


def load_done_question_numbers() -> set[int]:
    if not CASCADE_RESULTS_PATH.exists():
        return set()
    done: set[int] = set()
    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                done.add(json.loads(line)["question_number"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label conversation trajectories (pipeline step 3)."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit how many conversations to label (default: all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question_numbers already in factscore_cascade_results.jsonl.",
    )
    args = parser.parse_args()

    if not FUTURE_TURNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FUTURE_TURNS_PATH.name}. Run generate_future_turns.py first."
        )

    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        conversations = [json.loads(line) for line in file if line.strip()]

    if args.max_examples is not None:
        conversations = conversations[: args.max_examples]

    done = load_done_question_numbers() if args.resume else set()
    pending = [c for c in conversations if c["question_number"] not in done]

    print(f"Total conversations: {len(conversations)}")
    print(f"Already labeled: {len(done)}")
    print(f"Pending: {len(pending)}")

    write_mode = "a" if args.resume and done else "w"
    with open(CASCADE_RESULTS_PATH, write_mode, encoding="utf-8") as output:
        for conversation in pending:
            qnum = conversation["question_number"]
            try:
                judgments = {field: judge_answer(conversation[field]) for field in TURN_FIELDS}
                outcome = classify_trajectory(judgments)
                result = {
                    "question_number": qnum,
                    "domain": conversation.get("domain"),
                    "follow_up_mode": conversation.get("follow_up_mode", "challenge"),
                    "judgments": judgments,
                    **outcome,
                }
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(qnum, outcome["final_label"])
            except Exception as exc:
                print(f"ERROR on question {qnum}: {exc}")
                print("Saved progress so far. Re-run with --resume to continue.")
                raise


if __name__ == "__main__":
    main()
