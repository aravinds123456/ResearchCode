"""Workshop experiments: predict snowballing from turn-1 features only.

All features use ONLY the first hallucinating answer (no future-turn leakage).
Compares baselines vs signal models with proper cross-validation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
FUTURE_TURNS_PATH = SCRIPT_DIR / "future_turns.jsonl"
CASCADE_RESULTS_PATH = SCRIPT_DIR / "factscore_cascade_results.jsonl"


def load_rows() -> list[dict]:
    labels_by_q: dict[int, str] = {}
    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels_by_q[row["question_number"]] = row["final_label"].lower().strip()

    rows = []
    with open(FUTURE_TURNS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["question_number"]
            if q in labels_by_q:
                row["label"] = labels_by_q[q]
                rows.append(row)
    return rows


def feature_dict(row: dict) -> dict[str, float]:
    """Turn-1 features only."""
    avg_conf = float(row["average_confidence"])
    min_conf = float(row["minimum_confidence"])
    avg_ent = float(row["average_entropy"])
    max_ent = float(row["maximum_entropy"])
    answer = row["original_answer"]

    return {
        # Internal signals (proposal core)
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "avg_entropy": avg_ent,
        "max_entropy": max_ent,
        # Derived signals (still turn-1 only)
        "confidence_spread": avg_conf - min_conf,
        "entropy_spread": max_ent - avg_ent,
        "low_confidence_ratio": min_conf / (avg_conf + 1e-8),
        # Simple text stats (turn-1 answer only — not future turns)
        "answer_words": float(len(answer.split())),
        "answer_chars": float(len(answer)),
    }


FEATURE_SETS: dict[str, list[str]] = {
    "A_signals_4": [
        "avg_confidence",
        "min_confidence",
        "avg_entropy",
        "max_entropy",
    ],
    "B_signals_derived": [
        "avg_confidence",
        "min_confidence",
        "avg_entropy",
        "max_entropy",
        "confidence_spread",
        "entropy_spread",
        "low_confidence_ratio",
    ],
    "C_signals_plus_length": [
        "avg_confidence",
        "min_confidence",
        "avg_entropy",
        "max_entropy",
        "confidence_spread",
        "entropy_spread",
        "answer_words",
        "answer_chars",
    ],
}


def build_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    feats = [feature_dict(r) for r in rows]
    return np.array([[f[name] for name in feature_names] for f in feats], dtype=float)


def pick_cv(y: list[str], positive_label: str = "snowballing"):
    n_pos = sum(1 for label in y if label == positive_label)
    n_neg = len(y) - n_pos
    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 snowballing and 2 non-snowballing examples. "
            f"Got {n_pos} snowballing / {len(y)} total."
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def snowball_f1(y_true: list[str], y_pred: list[str]) -> float:
    return f1_score(
        y_true,
        y_pred,
        labels=["snowballing"],
        average="macro",
        zero_division=0,
    )


def evaluate(name: str, model, x: np.ndarray, y: list[str], cv) -> dict:
    preds = cross_val_predict(model, x, y, cv=cv)
    p, r, f1, _ = precision_recall_fscore_support(
        y,
        preds,
        labels=["not_snowballing", "snowballing"],
        zero_division=0,
    )
    return {
        "name": name,
        "preds": preds,
        "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
        "snowball_precision": p[1],
        "snowball_recall": r[1],
        "snowball_f1": f1[1],
    }


def main() -> None:
    rows = load_rows()
    labels3 = [r["label"] for r in rows]
    y = ["snowballing" if label == "snowballing" else "not_snowballing" for label in labels3]

    print("=" * 70)
    print("SNOWBALLING FORECASTING (turn-1 features -> snowballing label)")
    print("=" * 70)
    print(f"N = {len(rows)}")
    print("3-class counts:", Counter(labels3))
    print("Binary counts:", Counter(y))
    print()

    n_snow = Counter(y)["snowballing"]
    if n_snow < 5:
        print(
            f"WARNING: only {n_snow} snowballing examples. "
            "Re-run generate_future_turns on all ~56 hallucinating questions first."
        )

    cv = pick_cv(y)

    results = []

    # Baselines
    x_dummy = np.zeros((len(rows), 1))
    results.append(
        evaluate(
            "1_majority_baseline",
            DummyClassifier(strategy="most_frequent"),
            x_dummy,
            y,
            cv,
        )
    )
    results.append(
        evaluate(
            "2_stratified_random",
            DummyClassifier(strategy="stratified", random_state=42),
            x_dummy,
            y,
            cv,
        )
    )

    # Models x feature sets
    for feat_name, feat_cols in FEATURE_SETS.items():
        x = build_matrix(rows, feat_cols)

        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
        )

        results.append(evaluate(f"3_logistic_{feat_name}", lr, x, y, cv))
        results.append(evaluate(f"4_rf_{feat_name}", rf, x, y, cv))

    print(f"Cross-validation: {cv.get_n_splits()}-fold stratified\n")
    print(
        f"{'Model':<32} {'MacroF1':>8} {'Snow P':>8} {'Snow R':>8} {'Snow F1':>8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<32} {r['macro_f1']:>8.3f} "
            f"{r['snowball_precision']:>8.3f} {r['snowball_recall']:>8.3f} "
            f"{r['snowball_f1']:>8.3f}"
        )

    best = max(results, key=lambda r: r["snowball_f1"])
    print("\nBest snowballing F1:", best["name"], "=", round(best["snowball_f1"], 3))

    print("\nDetailed report for best model:")
    print(classification_report(y, best["preds"], zero_division=0))

    print("=" * 70)
    print("3-CLASS (reference — same data)")
    print("=" * 70)
    x = build_matrix(rows, FEATURE_SETS["B_signals_derived"])
    cv3 = StratifiedKFold(
        n_splits=min(5, min(Counter(labels3).values())),
        shuffle=True,
        random_state=42,
    )
    preds3 = cross_val_predict(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        x,
        labels3,
        cv=cv3,
    )
    print(classification_report(labels3, preds3, zero_division=0))

    print("\nHOW TO READ FOR WORKSHOP PAPER:")
    print("- Focus on 'Snow F1' column (snowballing class specifically).")
    print("- Legitimate win: best model Snow F1 > majority baseline Snow F1.")
    print("- Do NOT claim 'solved' unless Snow F1 is high AND N is large.")
    print("- If Snow F1 stays low, report as 'preliminary / challenging at N=...'")


if __name__ == "__main__":
    main()
