"""Step 3: Label full conversation trajectories as corrected / isolated / snowballing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CASCADE_RESULTS_PATH, FUTURE_TURNS_PATH
from factscore_utils import classify_trajectory, judge_answer

TURN_FIELDS = [
    "original_answer",
    "future_turn_1",
    "future_turn_2",
    "future_turn_3",
]


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
    args = parser.parse_args()

    if not FUTURE_TURNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FUTURE_TURNS_PATH.name}. Run generate_future_turns.py first."
        )

    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        conversations = [json.loads(line) for line in file]

    if args.max_examples is not None:
        conversations = conversations[: args.max_examples]

    with open(CASCADE_RESULTS_PATH, "w", encoding="utf-8") as output:
        for conversation in conversations:
            judgments = {
                field: judge_answer(conversation[field])
                for field in TURN_FIELDS
            }
            outcome = classify_trajectory(judgments)

            result = {
                "question_number": conversation["question_number"],
                "follow_up_mode": conversation.get("follow_up_mode", "challenge"),
                "judgments": judgments,
                **outcome,
            }
            output.write(json.dumps(result) + "\n")
            print(conversation["question_number"], outcome["final_label"])


if __name__ == "__main__":
    main()
