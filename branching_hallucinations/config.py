"""Versioned experiment configuration.

Secrets stay in environment variables. Scientific choices live in TOML.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import DOMAIN_TASK, SCHEMA_VERSION

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "branching_pilot.toml"


@dataclass
class ModelRoleConfig:
    sampler: str
    deployment: str = ""
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    api: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler,
            "deployment": self.deployment,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "api": self.api,
        }


@dataclass
class DatasetConfig:
    """Where seed questions come from, and which retrieval strategy verifies them."""

    name: str = "halluhard"
    path: Path | None = None
    question_field: str = "question"
    domain_field: str = "domain"
    id_field: str = "id"
    grounding_task: str = "research_questions"
    grounding_tasks: dict[str, str] = field(default_factory=dict)

    def task_for(self, domain: str) -> str:
        if domain in self.grounding_tasks:
            return self.grounding_tasks[domain]
        if domain in DOMAIN_TASK:
            return DOMAIN_TASK[domain]
        return self.grounding_task

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path).replace("\\", "/") if self.path else "",
            "question_field": self.question_field,
            "domain_field": self.domain_field,
            "id_field": self.id_field,
            "grounding_task": self.grounding_task,
            "grounding_tasks": dict(self.grounding_tasks),
        }


@dataclass
class ExperimentConfig:
    n_seeds: int = 10
    depth: int = 2
    actions: tuple[str, ...] = ("D", "N", "V")
    random_seed: int = 42
    domains: tuple[str, ...] = ("research", "legal", "medical")
    max_questions: int | None = None
    samples_per_question: int = 1
    max_claims_per_seed: int = 8
    allow_followup_fallback: bool = False

    answer: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-oss-20b"))
    followup_writer: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-5-mini"))
    trajectory_judge: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-5-mini"))
    claim_extractor: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-5-mini"))
    grounded_verifier: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-5-mini-medium"))
    search_planner: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig("azure-gpt-5-mini"))

    grounding_method: str = "halluhard_webscraper"
    max_searches: int = 3
    trajectory_states: tuple[str, ...] = ("DROP", "RETRACT", "REPEAT", "DEPEND")

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table")
    return value


def _dataset(data: dict[str, Any]) -> DatasetConfig:
    section = data.get("dataset") or {}
    if not isinstance(section, dict):
        raise ValueError("Config section [dataset] must be a table")
    path_value = section.get("path") or None
    tasks = section.get("grounding_tasks") or {}
    if tasks and not isinstance(tasks, dict):
        raise ValueError("dataset.grounding_tasks must be a table")
    return DatasetConfig(
        name=str(section.get("name") or "halluhard"),
        path=Path(path_value) if path_value else None,
        question_field=str(section.get("question_field") or "question"),
        domain_field=str(section.get("domain_field") or "domain"),
        id_field=str(section.get("id_field") or "id"),
        grounding_task=str(section.get("grounding_task") or "research_questions"),
        grounding_tasks={str(key): str(value) for key, value in tasks.items()},
    )


def _model_role(data: dict[str, Any], name: str, default_sampler: str) -> ModelRoleConfig:
    section = data.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section [models.{name}] must be a table")
    return ModelRoleConfig(
        sampler=str(section.get("sampler") or default_sampler),
        deployment=str(section.get("deployment") or ""),
        max_tokens=section.get("max_tokens"),
        reasoning_effort=section.get("reasoning_effort"),
        temperature=section.get("temperature"),
        api=section.get("api"),
    )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = _section(data, "experiment")
    if bool(experiment.get("allow_followup_fallback", False)):
        raise ValueError(
            "allow_followup_fallback is not part of the experiment. "
            "Canned D/N/V questions are never substituted. Remove the key or set it false."
        )
    models = _section(data, "models")
    grounding = _section(data, "grounding")
    trajectory = _section(data, "trajectory")
    actions = tuple(experiment.get("actions") or ["D", "N", "V"])
    domains = tuple(experiment.get("domains") or ["research", "legal", "medical"])
    return ExperimentConfig(
        n_seeds=int(experiment.get("n_seeds") or 10),
        depth=int(experiment.get("depth") or 2),
        actions=actions,
        random_seed=int(experiment.get("random_seed") or 42),
        domains=domains,
        max_questions=experiment.get("max_questions"),
        samples_per_question=int(experiment.get("samples_per_question") or 1),
        max_claims_per_seed=int(experiment.get("max_claims_per_seed") or 8),
        allow_followup_fallback=bool(experiment.get("allow_followup_fallback", False)),
        answer=_model_role(models, "answer", "azure-gpt-oss-20b"),
        followup_writer=_model_role(models, "followup_writer", "azure-gpt-5-mini"),
        trajectory_judge=_model_role(models, "trajectory_judge", "azure-gpt-5-mini"),
        claim_extractor=_model_role(models, "claim_extractor", "azure-gpt-5-mini"),
        grounded_verifier=_model_role(models, "grounded_verifier", "azure-gpt-5-mini-medium"),
        search_planner=_model_role(models, "search_planner", "azure-gpt-5-mini"),
        grounding_method=str(grounding.get("method") or "halluhard_webscraper"),
        max_searches=int(grounding.get("max_searches") or 3),
        trajectory_states=tuple(trajectory.get("states") or ["DROP", "RETRACT", "REPEAT", "DEPEND"]),
        dataset=_dataset(data),
        source_path=config_path,
        raw=data,
    )


def prompt_path(name: str) -> Path:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path


def load_prompt(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8").strip()


def prompt_hash(name: str) -> str:
    text = load_prompt(name)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_catalog() -> dict[str, dict[str, str]]:
    catalog = {}
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = path.stem
        catalog[name] = {
            "name": name,
            "version": f"{name}.v1",
            "hash": prompt_hash(name),
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
    return catalog


def fill_prompt(name: str, **values: Any) -> str:
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))
    return text


def env_secret_present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())
