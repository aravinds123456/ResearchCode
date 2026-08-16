"""Step 4: Train and evaluate the cascade forecaster on entropy/confidence features."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CASCADE_RESULTS_PATH, FUTURE_TURNS_PATH


def main() -> None:
    if not FUTURE_TURNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FUTURE_TURNS_PATH.name}. Run generate_future_turns.py first."
        )
    if not CASCADE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CASCADE_RESULTS_PATH.name}. Run factscore_serper_cascades.py first."
        )

    features: dict[int, list[float]] = {}
    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            features[row["question_number"]] = [
                row["average_confidence"],
                row["minimum_confidence"],
                row["average_entropy"],
                row["maximum_entropy"],
            ]

    labels: list[str] = []
    feature_rows: list[list[float]] = []
    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            question_number = row["question_number"]
            if question_number in features:
                feature_rows.append(features[question_number])
                labels.append(row["final_label"].lower())

    print(f"Examples: {len(labels)}")
    print("Label counts:", Counter(labels))

    if len(labels) < 2:
        print("Need at least 2 labeled examples before training.")
        raise SystemExit(1)

    if len(set(labels)) < 2:
        print("Cannot test prediction: every example has the same label.")
        raise SystemExit(1)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )

    if len(labels) <= 5:
        cv = LeaveOneOut()
        print("Using leave-one-out cross-validation (small dataset).")
    else:
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=min(5, len(labels)), shuffle=True, random_state=42)
        print(f"Using {cv.get_n_splits()} fold cross-validation.")

    predictions = cross_val_predict(model, feature_rows, labels, cv=cv)

    for actual, predicted in zip(labels, predictions):
        print(f"Actual: {actual:12} Predicted: {predicted}")

    print(classification_report(labels, predictions, zero_division=0))


if __name__ == "__main__":
    main()
