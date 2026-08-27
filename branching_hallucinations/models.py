"""Role-based SamplerBase factory for the branching experiment."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

from libs.models import MODEL_REGISTRY, get_sampler
from libs.sampler.azure_openai_sampler import AzureOpenAISampler
from libs.types import SamplerBase, SamplerResponse

from .config import ExperimentConfig, ModelRoleConfig


ROLE_DEPLOYMENT_ENV = {
    "answer": "AZURE_ANSWER_DEPLOYMENT",
    "followup_writer": "AZURE_WRITER_DEPLOYMENT",
    "trajectory_judge": "AZURE_JUDGE_DEPLOYMENT",
    "claim_extractor": "AZURE_EXTRACTOR_DEPLOYMENT",
    "grounded_verifier": "AZURE_VERIFIER_DEPLOYMENT",
    "search_planner": "AZURE_SEARCH_DEPLOYMENT",
}


def resolve_deployment(role: str, cfg: ModelRoleConfig) -> str:
    env_name = ROLE_DEPLOYMENT_ENV.get(role)
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    if cfg.deployment:
        return cfg.deployment
    if role == "answer":
        fallback = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if fallback:
            return fallback
    registry = MODEL_REGISTRY.get(cfg.sampler, {})
    return str(registry.get("deployment") or registry.get("model") or cfg.sampler)


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = set(inspect.signature(cls.__init__).parameters)
    accepted.discard("self")
    return {key: value for key, value in kwargs.items() if key in accepted}


def sampler_for_role(role: str, cfg: ModelRoleConfig) -> SamplerBase:
    """Build a SamplerBase for an experiment role.

    Azure roles never require OPENAI_API_KEY.
    """
    registry = dict(MODEL_REGISTRY.get(cfg.sampler) or {})
    backend = registry.get("backend")
    is_azure = backend == "azure" or cfg.sampler.startswith("azure-")
    if is_azure:
        kwargs = {k: v for k, v in registry.items() if k != "backend"}
        kwargs["model"] = registry.get("model") or cfg.sampler.removeprefix("azure-")
        kwargs["deployment"] = resolve_deployment(role, cfg)
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.reasoning_effort is not None:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.api:
            kwargs["api"] = cfg.api
        kwargs["websearch"] = False
        kwargs["use_reasoning_fallback"] = False
        return AzureOpenAISampler(**_filter_kwargs(AzureOpenAISampler, kwargs))
    return get_sampler(cfg.sampler)


@dataclass
class ExperimentSamplers:
    answer: SamplerBase
    followup_writer: SamplerBase
    trajectory_judge: SamplerBase
    claim_extractor: SamplerBase
    grounded_verifier: SamplerBase
    search_planner: SamplerBase

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "ExperimentSamplers":
        return cls(
            answer=sampler_for_role("answer", config.answer),
            followup_writer=sampler_for_role("followup_writer", config.followup_writer),
            trajectory_judge=sampler_for_role("trajectory_judge", config.trajectory_judge),
            claim_extractor=sampler_for_role("claim_extractor", config.claim_extractor),
            grounded_verifier=sampler_for_role("grounded_verifier", config.grounded_verifier),
            search_planner=sampler_for_role("search_planner", config.search_planner),
        )


async def complete(
    sampler: SamplerBase,
    messages: list[dict[str, str]],
    *,
    as_json: bool = False,
) -> SamplerResponse:
    response = await sampler(messages)
    if as_json and not (response.response_text or "").strip():
        raise RuntimeError("Sampler returned empty text for a JSON-required call")
    return response


def sampler_metadata(sampler: SamplerBase, response: SamplerResponse | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "sampler_class": type(sampler).__name__,
        "model": getattr(sampler, "model", None),
        "deployment": getattr(sampler, "deployment", None),
        "reasoning_effort": getattr(sampler, "reasoning_effort", None),
        "max_tokens": getattr(sampler, "max_tokens", None),
    }
    if response is not None:
        meta["response_metadata"] = {
            key: value
            for key, value in (response.response_metadata or {}).items()
            if key in {"backend", "deployment", "api", "status", "empty", "finish"}
        }
        meta["token_usage"] = response.token_usage
        meta["empty_response"] = not bool((response.response_text or "").strip())
    return meta
