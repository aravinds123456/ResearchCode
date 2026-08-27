"""FactBench prompt source for the branching experiment.

Official dataset: Hugging Face `launch/FactBench` (Bayat et al., 2025).
The public CSVs are named by VERIFY tier number; the paper names them:

    tier_1.csv  Hard       532 prompts  (strong LMSYS models, e.g. GPT-4)
    tier_2.csv  Moderate   332 prompts  (accepted config alias: medium)
    tier_3.csv  Easy       136 prompts

Default subset is Moderate+Hard. Easy is stored locally but not loaded unless
asked for. FactBench scores describe the original LMSYS replies; they are not
our trajectory labels. Seed answers, grounding, the D/N/V tree, and the
judge stay on the existing branching pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


HF_DATASET = "launch/FactBench"
HF_REVISION = "67d807e76651f4649a971b3eeaeb95f72b6cba40"
DEFAULT_PATH = REPO_ROOT / "data" / "factbench" / "prompts.jsonl"

# Paper names. "medium" is accepted as Moderate.
TIER_FROM_SPLIT = {
    "tier_1": "hard",
    "tier_2": "moderate",
    "tier_3": "easy",
}
SPLIT_FROM_TIER = {name: split for split, name in TIER_FROM_SPLIT.items()}
TIER_ALIASES = {
    "hard": "hard",
    "tier_1": "hard",
    "1": "hard",
    "moderate": "moderate",
    "medium": "moderate",
    "tier_2": "moderate",
    "2": "moderate",
    "easy": "easy",
    "tier_3": "easy",
    "3": "easy",
}
DEFAULT_TIERS = ("hard", "moderate")
HALLUHARD_DOMAINS = {"research", "legal", "medical", "coding"}
EXPECTED_COUNTS = {"hard": 532, "moderate": 332, "easy": 136}


class FactBenchError(ValueError):
    """Invalid FactBench path, tier, or local file."""


def canonicalize_tier(name: str) -> str:
    key = str(name or "").strip().lower().replace(" ", "_")
    if key not in TIER_ALIASES:
        known = "hard, moderate (medium), easy"
        raise FactBenchError(f"Unknown FactBench tier {name!r}. Use {known}.")
    return TIER_ALIASES[key]


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_PATH
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _selected_tiers(domains: tuple[str, ...] | None) -> tuple[str, ...]:
    if not domains:
        return DEFAULT_TIERS
    if set(domains) <= HALLUHARD_DOMAINS:
        raise FactBenchError(
            "FactBench filters on difficulty tiers (hard, moderate, easy), "
            "not HalluHard domains. Set experiment.domains = [\"hard\", \"moderate\"]."
        )
    selected: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        tier = canonicalize_tier(domain)
        if tier not in seen:
            selected.append(tier)
            seen.add(tier)
    return tuple(selected)


def fetch_factbench(
    dest: str | Path | None = None,
    *,
    revision: str = HF_REVISION,
) -> Path:
    """Download the official CSVs and write a local JSONL with stable IDs."""
    out = _resolve_path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for split, tier in TIER_FROM_SPLIT.items():
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/{split}.csv"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        expected = EXPECTED_COUNTS[tier]
        if len(rows) != expected:
            raise FactBenchError(
                f"{split}.csv has {len(rows)} rows; paper/HF card expects {expected} {tier} prompts"
            )
        for index, row in enumerate(rows):
            question = str(row.get("user_prompts") or "").strip()
            if not question:
                raise FactBenchError(f"Empty user_prompts in {split}.csv row {index}")
            records.append(
                {
                    "id": f"fb_{tier}_{index:03d}",
                    "question": question,
                    "domain": tier,
                    "tier": tier,
                    "topic": str(row.get("topic_description") or "").strip(),
                    "prompt_score": row.get("prompt_score"),
                    "hallucination_score": row.get("hallucination_score"),
                    "source_split": split,
                    "hf_dataset": HF_DATASET,
                    "hf_revision": revision,
                }
            )
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def load_factbench(
    *,
    domains: tuple[str, ...] | None = None,
    max_questions: int | None = None,
    path: str | Path | None = None,
    source: str = "factbench",
    **_: Any,
) -> list[dict[str, Any]]:
    jsonl_path = _resolve_path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"FactBench JSONL not found: {jsonl_path}. "
            "Run `python -m branching_hallucinations.factbench` to download it."
        )
    allowed_order = _selected_tiers(domains)
    allowed = set(allowed_order)
    by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in allowed_order}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tier = canonicalize_tier(record.get("tier") or record.get("domain") or "")
            if tier not in allowed:
                continue
            question = str(record.get("question") or record.get("user_prompts") or "").strip()
            if not question:
                continue
            by_tier[tier].append(
                {
                    "question_id": str(record.get("id") or f"fb_{tier}_{len(by_tier[tier]):03d}"),
                    "domain": tier,
                    "question": question,
                    "source": source,
                    "topic": str(record.get("topic") or ""),
                    "tier": tier,
                }
            )
    questions: list[dict[str, Any]] = []
    queues = [by_tier[tier] for tier in allowed_order if by_tier[tier]]
    while queues:
        remaining: list[list[dict[str, Any]]] = []
        for queue in queues:
            questions.append(queue.pop(0))
            if max_questions is not None and len(questions) >= max_questions:
                return questions
            if queue:
                remaining.append(queue)
        queues = remaining
    if not questions:
        raise FileNotFoundError(
            f"No FactBench prompts for tiers {sorted(allowed)} in {jsonl_path}"
        )
    return questions


def main() -> int:
    path = fetch_factbench()
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            tier = json.loads(line)["tier"]
            counts[tier] = counts.get(tier, 0) + 1
    print(f"Wrote {path}")
    print("counts:", counts)
    print("default subset hard+moderate:", counts.get("hard", 0) + counts.get("moderate", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
