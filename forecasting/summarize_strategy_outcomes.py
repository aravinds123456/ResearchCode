"""Primary analysis for branched follow-up experiments: outcome × strategy.

Does not use turn-1 signals. Report every strategy arm; do not treat the pooled
snowball rate as a natural base rate.

  python forecasting/summarize_strategy_outcomes.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CASCADE_RESULTS_PATH, FUTURE_TURNS_PATH
from follow_up_prompts import EXPERIMENT_STRATEGIES, trajectory_key

OUTCOME_ORDER = ("corrected", "isolated", "snowballing")


def load_rows() -> list[dict]:
    if not CASCADE_RESULTS_PATH.exists():
        raise SystemExit(f"Missing {CASCADE_RESULTS_PATH}. Label trajectories first.")

    future_by_key: dict[str, dict] = {}
    if FUTURE_TURNS_PATH.exists():
        with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    future_by_key[trajectory_key(row)] = row

    rows: list[dict] = []
    with open(CASCADE_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            labeled = json.loads(line)
            future = future_by_key.get(trajectory_key(labeled), {})
            strategy = (
                labeled.get("strategy")
                or labeled.get("follow_up_mode")
                or future.get("strategy")
                or future.get("follow_up_mode")
                or "legacy"
            )
            rows.append(
                {
                    "key": trajectory_key(labeled),
                    "question_number": labeled.get("question_number"),
                    "domain": labeled.get("domain") or future.get("domain"),
                    "strategy": strategy,
                    "final_label": str(labeled.get("final_label", "")).lower().strip(),
                    "turn_states": future.get("turn_states", []),
                    "model_name": future.get("model_name", ""),
                }
            )
    return rows


def pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100.0 * n / total:.1f}%"


def print_table(title: str, counts: dict[str, Counter], row_order: list[str]) -> None:
    print()
    print(title)
    header = f"{'Group':<22} {'N':>5} " + " ".join(f"{lab:>14}" for lab in OUTCOME_ORDER)
    print(header)
    print("-" * len(header))
    grand: Counter[str] = Counter()
    for name in row_order:
        c = counts.get(name, Counter())
        total = sum(c.values())
        grand.update(c)
        cells = " ".join(f"{c[lab]:>5} ({pct(c[lab], total):>6})" for lab in OUTCOME_ORDER)
        print(f"{name:<22} {total:>5} {cells}")
    total = sum(grand.values())
    cells = " ".join(
        f"{grand[lab]:>5} ({pct(grand[lab], total):>6})" for lab in OUTCOME_ORDER
    )
    print(f"{'ALL (pooled)':<22} {total:>5} {cells}")
    print(
        "Pooled row averages the arms you ran; it is not an unpressured base rate. "
        "Use the neutral (and accepting) rows for that."
    )


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("No labeled trajectories.")

    by_strategy: dict[str, Counter] = defaultdict(Counter)
    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    turn_states: Counter[str] = Counter()

    for row in rows:
        label = row["final_label"] or "unknown"
        by_strategy[row["strategy"]][label] += 1
        by_domain[str(row["domain"] or "unknown")][label] += 1
        if row["model_name"]:
            by_model[row["model_name"]][label] += 1
        turn_states.update(row["turn_states"] or [])

    strategy_order = [s for s in EXPERIMENT_STRATEGIES if s in by_strategy]
    strategy_order += sorted(k for k in by_strategy if k not in strategy_order)

    print(f"Trajectories: {len(rows)}")
    print_table("OUTCOME BY FOLLOW-UP STRATEGY", by_strategy, strategy_order)
    print_table("OUTCOME BY DOMAIN", by_domain, sorted(by_domain))
    if by_model:
        print_table("OUTCOME BY MODEL", by_model, sorted(by_model))

    if turn_states:
        print()
        print("TURN-STATE COUNTS (all assistant follow-up turns)")
        for state, n in turn_states.most_common():
            print(f"  {state:<22} {n}")


if __name__ == "__main__":
    main()
