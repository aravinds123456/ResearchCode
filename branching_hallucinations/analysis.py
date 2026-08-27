"""Pure analysis over frozen artifacts. No Azure, Serper, or generation imports."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    ACTIVE_STATES,
    Action,
    ActionAudit,
    BranchNode,
    ParseStatus,
    TrajectoryJudgment,
    TrajectoryState,
    VerificationResult,
    VerificationStatus,
    VerifiedSeed,
)
from .storage import write_json


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return p, max(0.0, center - margin), min(1.0, center + margin)


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return total


def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    """Exact two-sided McNemar on discordant pairs. b = A-only, c = B-only."""
    n = b + c
    if n == 0:
        return {"b": float(b), "c": float(c), "n_discordant": 0.0, "p_value": 1.0}
    p = min(1.0, 2.0 * _binom_sf(max(b, c), n))
    return {"b": float(b), "c": float(c), "n_discordant": float(n), "p_value": p}


def seed_cluster_bootstrap(
    values_by_seed: dict[str, list[float]],
    n_boot: int = 2000,
    random_seed: int = 42,
) -> dict[str, float]:
    rng = random.Random(random_seed)
    seeds = list(values_by_seed)
    if not seeds:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_seeds": 0.0}

    def mean_of(sample: list[str]) -> float:
        vals = [item for seed in sample for item in values_by_seed[seed]]
        return sum(vals) / len(vals) if vals else 0.0

    observed = mean_of(seeds)
    draws = []
    for _ in range(n_boot):
        sample = [rng.choice(seeds) for _ in seeds]
        draws.append(mean_of(sample))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "mean": observed,
        "ci_low": lo,
        "ci_high": hi,
        "n_seeds": float(len(seeds)),
    }


def is_active(state: TrajectoryState) -> bool:
    return state in ACTIVE_STATES


def terminal_state(path_labels: list[TrajectoryState]) -> TrajectoryState | None:
    return path_labels[-1] if path_labels else None


def ever_depend(path_labels: list[TrajectoryState]) -> bool:
    return TrajectoryState.DEPEND in path_labels


def _ok_judgments(judgments: Iterable[TrajectoryJudgment]) -> list[TrajectoryJudgment]:
    return [item for item in judgments if item.parse_status != ParseStatus.FAILED]


def t1_distribution(
    nodes: list[BranchNode],
    judgments: list[TrajectoryJudgment],
) -> dict[str, Any]:
    by_node = {item.node_id: item for item in _ok_judgments(judgments)}
    counts: dict[str, Counter] = {action.value: Counter() for action in Action}
    paired: dict[str, dict[str, TrajectoryState]] = defaultdict(dict)
    for node in nodes:
        if node.depth != 1:
            continue
        judgment = by_node.get(node.node_id)
        if judgment is None:
            continue
        counts[node.action.value][judgment.label.value] += 1
        paired[node.seed_id][node.action.value] = judgment.label
    table = {}
    for action, counter in counts.items():
        n = sum(counter.values())
        table[action] = {
            state.value: {
                "count": counter[state.value],
                "wilson": wilson(counter[state.value], n),
            }
            for state in TrajectoryState
        }
        table[action]["n"] = n
        table[action]["active"] = wilson(
            counter[TrajectoryState.REPEAT.value] + counter[TrajectoryState.DEPEND.value],
            n,
        )
    paired_tests = {}
    for left, right in (("D", "N"), ("D", "V"), ("V", "N")):
        b = c = 0
        for labels in paired.values():
            if left not in labels or right not in labels:
                continue
            left_active = is_active(labels[left])
            right_active = is_active(labels[right])
            if left_active and not right_active:
                b += 1
            elif right_active and not left_active:
                c += 1
        paired_tests[f"{left}_vs_{right}_active"] = mcnemar_exact(b, c)
    return {"by_action": table, "paired_mcnemar_active": paired_tests, "n_seeds": len(paired)}


def t2_transitions(
    nodes: list[BranchNode],
    judgments: list[TrajectoryJudgment],
    random_seed: int = 42,
) -> dict[str, Any]:
    by_id = {node.node_id: node for node in nodes}
    by_node = {item.node_id: item for item in _ok_judgments(judgments)}
    cells: dict[tuple[str, str, str], int] = Counter()
    by_seed: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    ever = 0
    terminal_retract_after_depend = 0
    n_t2 = 0
    for node in nodes:
        if node.depth != 2 or not node.parent_node_id:
            continue
        child = by_node.get(node.node_id)
        parent = by_node.get(node.parent_node_id)
        if child is None or parent is None:
            continue
        key = (parent.label.value, node.action.value, child.label.value)
        cells[key] += 1
        by_seed[node.seed_id].append(key)
        n_t2 += 1
        if child.label == TrajectoryState.DEPEND or parent.label == TrajectoryState.DEPEND:
            if child.label == TrajectoryState.DEPEND:
                ever += 1
        if parent.label == TrajectoryState.DEPEND and child.label == TrajectoryState.RETRACT:
            terminal_retract_after_depend += 1
    tables: dict[str, dict[str, dict[str, int]]] = {}
    for s1 in TrajectoryState:
        tables[s1.value] = {}
        for action in Action:
            dist = {s2.value: cells[(s1.value, action.value, s2.value)] for s2 in TrajectoryState}
            tables[s1.value][action.value] = dist
    bootstrap: dict[str, Any] = {}
    for s1 in TrajectoryState:
        for action in Action:
            values = {
                seed: [1.0 if item[2] in {"REPEAT", "DEPEND"} and item[0] == s1.value and item[1] == action.value else 0.0
                       for item in rows if item[0] == s1.value and item[1] == action.value]
                for seed, rows in by_seed.items()
            }
            values = {seed: vals for seed, vals in values.items() if vals}
            bootstrap[f"P_active|{s1.value},{action.value}"] = seed_cluster_bootstrap(values, random_seed=random_seed)
    return {
        "n_t2": n_t2,
        "transitions": tables,
        "seed_cluster_bootstrap_active": bootstrap,
        "ever_depend_count": ever,
        "depend_then_retract": terminal_retract_after_depend,
        "note": "Primary endpoint is the actual terminal T2 label, not strongest-ever state.",
    }


def action_compliance(audits: list[ActionAudit]) -> dict[str, Any]:
    by_action: dict[str, Counter] = {action.value: Counter() for action in Action}
    parse_fail = 0
    for audit in audits:
        key = audit.desired_action.value
        if audit.parse_status == ParseStatus.FAILED:
            parse_fail += 1
            continue
        by_action[key]["n"] += 1
        if audit.valid:
            by_action[key]["valid"] += 1
        realized = audit.realized_action.value if audit.realized_action else "unknown"
        by_action[key][f"realized_{realized}"] += 1
    report = {}
    for action, counter in by_action.items():
        n = counter["n"]
        report[action] = {
            "n": n,
            "valid": counter["valid"],
            "compliance": (counter["valid"] / n) if n else 0.0,
            "realized": {k: v for k, v in counter.items() if k.startswith("realized_")},
        }
    report["parse_failures"] = parse_fail
    return report


def verification_attrition(results: list[VerificationResult]) -> dict[str, Any]:
    counts = Counter(item.status.value for item in results)
    parse_fail = sum(1 for item in results if item.parse_status == ParseStatus.FAILED)
    return {
        "n_candidates": len(results),
        "by_status": dict(counts),
        "parse_failures": parse_fail,
        "eligible_verified_false": counts.get(VerificationStatus.VERIFIED_FALSE.value, 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    verified_seeds: list[VerifiedSeed],
    nodes: list[BranchNode],
    judgments: list[TrajectoryJudgment],
    audits: list[ActionAudit],
    verifications: list[VerificationResult],
    out_dir: Path,
    random_seed: int = 42,
) -> dict[str, Any]:
    t1 = t1_distribution(nodes, judgments)
    t2 = t2_transitions(nodes, judgments, random_seed=random_seed)
    compliance = action_compliance(audits)
    attrition = verification_attrition(verifications)
    fallback_nodes = [
        node for node in nodes if (node.intervention_metadata or {}).get("fallback_used")
    ]
    fallback_by_action = Counter(node.action.value for node in fallback_nodes)
    fallback_n = Counter(node.action.value for node in nodes)
    fallback_rates = {
        action: (fallback_by_action[action] / fallback_n[action]) if fallback_n[action] else 0.0
        for action in fallback_n
    }
    domain_counts = Counter(seed.domain for seed in verified_seeds)
    summary = {
        "n_verified_seeds": len(verified_seeds),
        "domain_composition": dict(domain_counts),
        "n_nodes": len(nodes),
        "n_judgments_ok": len(_ok_judgments(judgments)),
        "n_judgments_failed": sum(1 for item in judgments if item.parse_status == ParseStatus.FAILED),
        "t1": t1,
        "t2": t2,
        "action_compliance": compliance,
        "fallback_rates": fallback_rates,
        "verification_attrition": attrition,
        "primary_endpoint": "actual terminal T2 label",
        "secondary_ever_depend": "ever_depend is secondary only",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    t1_rows = []
    for action, payload in t1["by_action"].items():
        for state in TrajectoryState:
            stats = payload[state.value]
            t1_rows.append(
                {
                    "action": action,
                    "state": state.value,
                    "count": stats["count"],
                    "n": payload["n"],
                    "rate": stats["wilson"][0],
                    "wilson_lo": stats["wilson"][1],
                    "wilson_hi": stats["wilson"][2],
                    "wilson_note": "descriptive only; inference is paired by seed",
                }
            )
    write_csv(out_dir / "t1_states.csv", t1_rows)
    t2_rows = []
    for s1, by_action in t2["transitions"].items():
        for action, dist in by_action.items():
            total = sum(dist.values())
            for s2, count in dist.items():
                t2_rows.append(
                    {
                        "s1": s1,
                        "a2": action,
                        "s2": s2,
                        "count": count,
                        "n": total,
                        "rate": (count / total) if total else 0.0,
                    }
                )
    write_csv(out_dir / "transitions.csv", t2_rows)
    boot_rows = [
        {"estimand": key, **value} for key, value in t2["seed_cluster_bootstrap_active"].items()
    ]
    write_csv(out_dir / "bootstrap.csv", boot_rows)
    return summary
