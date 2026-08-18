"""Predict trajectory outcomes from turn-1 internal signals (entropy/confidence).

Research question: At the first hallucinated turn, do internal signals forecast
whether the conversation will correct, snowball, or stay isolated?

Compares:
  - Majority baseline (always predict most common class)
  - Signal-only model (4 entropy/confidence features from turn 1)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
FUTURE_TURNS_PATH = SCRIPT_DIR / "future_turns.jsonl"
CASCADE_RESULTS_PATH = SCRIPT_DIR / "factscore_cascade_results.jsonl"

SIGNAL_FEATURES = [
    "average_confidence",
    "minimum_confidence",
    "average_entropy",
    "maximum_entropy",
]


def row_key(row: dict) -> str:
    return f"{row['question_number']}:{row.get('follow_up_mode', '')}"


def load_data() -> tuple[list[list[float]], list[str], list[int]]:
    features_by_q: dict[str, list[float]] = {}
    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            features_by_q[row_key(row)] = [
                row[name] for name in SIGNAL_FEATURES
            ]

    x_rows: list[list[float]] = []
    labels: list[str] = []
    question_numbers: list[int] = []

    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            key = row_key(row)
            if key not in features_by_q:
                continue
            x_rows.append(features_by_q[key])
            labels.append(row["final_label"].lower().strip())
            question_numbers.append(row["question_number"])

    return x_rows, labels, question_numbers


def pick_cv(labels: list[str]):
    if len(labels) <= 5:
        return LeaveOneOut(), "leave-one-out"
    n_splits = min(5, min(Counter(labels).values()))
    if n_splits < 2:
        return LeaveOneOut(), "leave-one-out"
    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        f"{n_splits}-fold stratified",
    )


def evaluate(name: str, model, x_rows, labels, cv) -> dict:
    preds = cross_val_predict(model, x_rows, labels, cv=cv)
    return {
        "name": name,
        "predictions": preds,
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "snowball_f1": f1_score(
            labels,
            preds,
            labels=sorted(set(labels)),
            average=None,
            zero_division=0,
        ),
    }


def print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    if not FUTURE_TURNS_PATH.exists():
        raise FileNotFoundError(f"Missing {FUTURE_TURNS_PATH}")
    if not CASCADE_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {CASCADE_RESULTS_PATH}")

    x_rows, labels, question_numbers = load_data()
    label_counts = Counter(labels)

    print_section("DATA")
    print(f"Examples: {len(labels)}")
    print("Label counts:", dict(label_counts))
    print("Features used:", SIGNAL_FEATURES)
    print(
        "Important: labels come from FUTURE turns; signals come from turn 1 ONLY. "
        "The model does not see future text."
    )

    if len(set(labels)) < 2:
        print("\nNeed at least 2 different labels to evaluate prediction.")
        return

    cv, cv_name = pick_cv(labels)
    print(f"\nCross-validation: {cv_name}")

    majority = DummyClassifier(strategy="most_frequent")
    signal_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )

    print_section("TASK 1: 3-CLASS (corrected / snowballing / isolated)")
    base = evaluate("Majority baseline", majority, x_rows, labels, cv)
    sig = evaluate("Signal-only (entropy + confidence)", signal_model, x_rows, labels, cv)

    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 57)
    print(f"{base['name']:<35} {base['accuracy']:>10.3f} {base['macro_f1']:>10.3f}")
    print(f"{sig['name']:<35} {sig['accuracy']:>10.3f} {sig['macro_f1']:>10.3f}")

    print("\nSignal-only per-class report:")
    print(classification_report(labels, sig["predictions"], zero_division=0))

    print_section("TASK 2: BINARY (snowballing vs everything else)")
    binary_labels = ["snowballing" if label == "snowballing" else "not_snowballing" for label in labels]
    binary_counts = Counter(binary_labels)
    print("Binary label counts:", dict(binary_counts))

    if binary_counts["snowballing"] < 2:
        print("Too few snowballing examples for stable binary forecasting.")
    else:
        cv_bin, _ = pick_cv(binary_labels)
        base_bin = evaluate(
            "Majority baseline",
            DummyClassifier(strategy="most_frequent"),
            x_rows,
            binary_labels,
            cv_bin,
        )
        sig_bin = evaluate(
            "Signal-only",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                class_weight="balanced",
                random_state=42,
            ),
            x_rows,
            binary_labels,
            cv_bin,
        )

        print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10}")
        print("-" * 57)
        print(
            f"{base_bin['name']:<35} {base_bin['accuracy']:>10.3f} {base_bin['macro_f1']:>10.3f}"
        )
        print(
            f"{sig_bin['name']:<35} {sig_bin['accuracy']:>10.3f} {sig_bin['macro_f1']:>10.3f}"
        )

        print("\nSignal-only binary report:")
        print(classification_report(binary_labels, sig_bin["predictions"], zero_division=0))

    print_section("INTERPRETATION (read this honestly)")
    print(
        "- Signals are measured at turn 1; labels describe turns 2-4. "
        "That matches the proposal IF signal model beats baseline."
    )
    print(
        "- N=30 with only 5 snowballing examples is very small. "
        "Treat results as preliminary."
    )
    print(
        "- If signal-only macro F1 is NOT higher than majority baseline, "
        "you cannot claim internal signals forecast cascades yet."
    )
    print(
        "- Accuracy alone is misleading because 70% of examples are 'corrected'."
    )


if __name__ == "__main__":
    main()
