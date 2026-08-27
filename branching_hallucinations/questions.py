"""Question sources for seed generation.

HalluHard questions live in `halluhard/`. FactBench prompts live in
`data/factbench/`. This module is the experiment-side registry: named loaders
share the `{question_id, domain, question, source}` contract. Tree / judge /
analysis never import a dataset module.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from halluhard.questions import DOMAINS as HALLUHARD_DOMAINS
from halluhard.questions import load_halluhard

from .config import DatasetConfig, REPO_ROOT
from .factbench import DEFAULT_TIERS as FACTBENCH_DEFAULT_TIERS
from .factbench import load_factbench

QuestionLoader = Callable[..., list[dict[str, Any]]]


def _record_question(record: dict[str, Any], field_name: str) -> str:
    return str(
        record.get(field_name)
        or record.get("question")
        or record.get("research_question")
        or record.get("prompt")
        or ""
    ).strip()


def _question_row(
    *,
    question_id: int | str,
    domain: str,
    question: str,
    source: str,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "domain": domain,
        "question": question,
        "source": source,
    }


def load_jsonl(
    *,
    path: str | Path,
    question_field: str = "question",
    domain_field: str = "domain",
    id_field: str = "id",
    domains: tuple[str, ...] | None = None,
    max_questions: int | None = None,
    source: str = "jsonl",
    **_: Any,
) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.is_absolute():
        jsonl_path = REPO_ROOT / jsonl_path
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Question JSONL not found: {jsonl_path}")
    allowed = set(domains) if domains else None
    questions: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = _record_question(record, question_field)
            if not question:
                continue
            domain = str(record.get(domain_field) or "custom").strip() or "custom"
            if allowed is not None and domain not in allowed:
                continue
            raw_id = record.get(id_field)
            if raw_id is None or str(raw_id).strip() == "":
                question_id: int | str = index
            else:
                question_id = str(raw_id).replace("/", "_")
            questions.append(
                _question_row(
                    question_id=question_id,
                    domain=domain,
                    question=question,
                    source=source,
                )
            )
            if max_questions is not None and len(questions) >= max_questions:
                return questions
    if not questions:
        raise FileNotFoundError(f"No usable questions in {jsonl_path}")
    return questions


DATASET_LOADERS: dict[str, QuestionLoader] = {
    "halluhard": load_halluhard,
    "factbench": load_factbench,
    "jsonl": load_jsonl,
}


def register_source(name: str, loader: QuestionLoader) -> None:
    """Register a named dataset loader. `name` is the `[dataset].name` config value."""
    DATASET_LOADERS[name] = loader


def load_questions(
    dataset: DatasetConfig | None = None,
    domains: tuple[str, ...] | None = None,
    max_questions: int | None = None,
) -> list[dict[str, Any]]:
    spec = dataset or DatasetConfig()
    loader = DATASET_LOADERS.get(spec.name)
    if loader is None:
        known = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unknown dataset {spec.name!r}. Known: {known}")
    if domains is not None:
        selected = domains
    elif spec.name == "factbench":
        selected = FACTBENCH_DEFAULT_TIERS
    else:
        selected = ("research", "legal", "medical")
    return loader(
        domains=selected,
        max_questions=max_questions,
        path=spec.path,
        question_field=spec.question_field,
        domain_field=spec.domain_field,
        id_field=spec.id_field,
        source=spec.name,
    )
