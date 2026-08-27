"""HalluHard domain questions for seed generation.

The branching experiment only needs `{question_id, domain, question}`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import PACKAGE_DIR


@dataclass(frozen=True)
class DomainFile:
    path: Path
    question_field: str
    id_offset: int


DOMAINS: dict[str, DomainFile] = {
    "research": DomainFile(
        PACKAGE_DIR / "research_questions/data/research_questions_all.jsonl",
        "research_question",
        0,
    ),
    "legal": DomainFile(
        PACKAGE_DIR / "legal_cases/data/legal_cases_all.jsonl",
        "question",
        100_000,
    ),
    "medical": DomainFile(
        PACKAGE_DIR / "medical_guidelines/data/guidelines.jsonl",
        "question",
        200_000,
    ),
    "coding": DomainFile(
        PACKAGE_DIR / "coding/data/coding_questions.jsonl",
        "prompt",
        300_000,
    ),
}

TASK_FOR_DOMAIN = {
    "research": "research_questions",
    "legal": "legal_cases",
    "medical": "medical_guidelines",
    "coding": "coding",
}


def _question_text(record: dict[str, Any], field_name: str) -> str:
    return str(
        record.get(field_name)
        or record.get("question")
        or record.get("research_question")
        or record.get("prompt")
        or ""
    ).strip()


def load_halluhard(
    *,
    domains: tuple[str, ...] = ("research", "legal", "medical"),
    max_questions: int | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for domain in domains:
        spec = DOMAINS.get(domain)
        if spec is None:
            raise ValueError(
                f"Unknown HalluHard domain {domain!r}. Known: {', '.join(DOMAINS)}"
            )
        if not spec.path.exists():
            continue
        with spec.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                record = json.loads(line)
                question = _question_text(record, spec.question_field)
                if not question:
                    continue
                questions.append(
                    {
                        "question_id": spec.id_offset + index,
                        "domain": domain,
                        "question": question,
                        "source": "halluhard",
                    }
                )
                if max_questions is not None and len(questions) >= max_questions:
                    return questions
    if not questions:
        raise FileNotFoundError("No HalluHard domain question files found.")
    return questions
