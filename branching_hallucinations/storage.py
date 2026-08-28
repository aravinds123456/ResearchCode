"""Immutable stage artifacts and conversation reconstruction.

Each experiment stage writes a separate file. Trajectory judgments never
modify raw node records.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import ExperimentConfig, REPO_ROOT, prompt_catalog
from .schemas import (
    SCHEMA_VERSION,
    ActionAudit,
    BranchNode,
    CandidateClaim,
    GeneratedSeed,
    Message,
    TrajectoryJudgment,
    VerificationResult,
    VerifiedSeed,
    make_node_id,
    parse_node_id,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_typed(path: Path, factory) -> Iterator[Any]:
    for row in read_jsonl(path):
        yield factory(row)


class RunStore:
    """One run directory with stage-separated artifacts."""

    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir)
        self.seeds_dir = self.root / "seeds"
        self.tree_dir = self.root / "tree"
        self.judgments_dir = self.root / "judgments"
        self.analysis_dir = self.root / "analysis"
        self.reports_dir = self.root / "reports"
        self.logs_dir = self.root / "logs"
        self._async_lock: asyncio.Lock | None = None

    def io_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def generated_seeds_path(self) -> Path:
        return self.seeds_dir / "generated.jsonl"

    @property
    def candidate_claims_path(self) -> Path:
        return self.seeds_dir / "candidates.jsonl"

    @property
    def verification_path(self) -> Path:
        return self.seeds_dir / "verifications.jsonl"

    @property
    def verified_seeds_path(self) -> Path:
        return self.seeds_dir / "verified.jsonl"

    @property
    def nodes_path(self) -> Path:
        return self.tree_dir / "nodes.jsonl"

    def action_audit_path(self, version: str = "v1") -> Path:
        return self.judgments_dir / f"action_audit_{version}.jsonl"

    def trajectory_path(self, version: str = "v1") -> Path:
        return self.judgments_dir / f"trajectory_{version}.jsonl"

    def ensure(self) -> None:
        for path in (
            self.seeds_dir,
            self.tree_dir,
            self.judgments_dir,
            self.analysis_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ensure()
        existing = self.load_manifest()
        created_at = existing.get("created_at") or utc_now()
        manifest = {
            "experiment_id": self.root.name,
            "created_at": created_at,
            "updated_at": utc_now(),
            "git_sha": git_sha(),
            "schema_version": SCHEMA_VERSION,
            "random_seed": config.random_seed,
            "config_file": str(config.source_path) if config.source_path else "",
            "prompt_versions": prompt_catalog(),
            "answer_model": config.answer.to_dict(),
            "followup_writer_model": config.followup_writer.to_dict(),
            "trajectory_judge_model": config.trajectory_judge.to_dict(),
            "claim_extractor_model": config.claim_extractor.to_dict(),
            "grounded_verifier_model": config.grounded_verifier.to_dict(),
            "search_planner_model": config.search_planner.to_dict(),
            "dataset": config.dataset.to_dict(),
            "domains": list(config.domains),
            "grounding_method": config.grounding_method,
            "serper": {"max_searches": config.max_searches, "credential": "SERPER_API_KEY"},
            "n_seeds": config.n_seeds,
            "depth": config.depth,
            "actions": list(config.actions),
            "concurrency": config.concurrency,
            "allow_followup_fallback": config.allow_followup_fallback,
        }
        if extra:
            manifest.update(extra)
        write_json(self.manifest_path, manifest)
        if config.source_path and config.source_path.exists():
            (self.root / "config.toml").write_text(
                config.source_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return manifest

    def generated_seeds(self) -> list[GeneratedSeed]:
        return [GeneratedSeed.from_dict(row) for row in read_jsonl(self.generated_seeds_path)]

    def candidate_claims(self) -> list[CandidateClaim]:
        return [CandidateClaim.from_dict(row) for row in read_jsonl(self.candidate_claims_path)]

    def verifications(self) -> list[VerificationResult]:
        return [VerificationResult.from_dict(row) for row in read_jsonl(self.verification_path)]

    def verified_seeds(self) -> list[VerifiedSeed]:
        return [VerifiedSeed.from_dict(row) for row in read_jsonl(self.verified_seeds_path)]

    def nodes(self) -> list[BranchNode]:
        return [BranchNode.from_dict(row) for row in read_jsonl(self.nodes_path)]

    def nodes_by_id(self) -> dict[str, BranchNode]:
        return {node.node_id: node for node in self.nodes()}

    def seeds_by_id(self) -> dict[str, VerifiedSeed]:
        return {seed.seed_id: seed for seed in self.verified_seeds()}

    def generated_by_id(self) -> dict[str, GeneratedSeed]:
        return {seed.seed_id: seed for seed in self.generated_seeds()}

    def completed_ids(self, path: Path, key: str) -> set[str]:
        return {str(row.get(key)) for row in read_jsonl(path) if row.get(key)}

    def append_generated(self, seed: GeneratedSeed) -> None:
        append_jsonl(self.generated_seeds_path, seed.to_dict())

    def append_claim(self, claim: CandidateClaim) -> None:
        append_jsonl(self.candidate_claims_path, claim.to_dict())

    def append_verification(self, result: VerificationResult) -> None:
        append_jsonl(self.verification_path, result.to_dict())

    def append_verified(self, seed: VerifiedSeed) -> None:
        append_jsonl(self.verified_seeds_path, seed.to_dict())

    def append_node(self, node: BranchNode) -> None:
        existing = self.completed_ids(self.nodes_path, "node_id")
        if node.node_id in existing:
            raise ValueError(f"Duplicate node_id {node.node_id}")
        append_jsonl(self.nodes_path, node.to_dict())

    def append_action_audit(self, audit: ActionAudit, version: str = "v1") -> None:
        append_jsonl(self.action_audit_path(version), audit.to_dict())

    def append_judgment(self, judgment: TrajectoryJudgment, version: str = "v1") -> None:
        append_jsonl(self.trajectory_path(version), judgment.to_dict())

    def action_audits(self, version: str = "v1") -> list[ActionAudit]:
        return [ActionAudit.from_dict(row) for row in read_jsonl(self.action_audit_path(version))]

    def judgments(self, version: str = "v1") -> list[TrajectoryJudgment]:
        return [TrajectoryJudgment.from_dict(row) for row in read_jsonl(self.trajectory_path(version))]


def conversation_for_seed(seed: VerifiedSeed | GeneratedSeed) -> list[Message]:
    return [
        Message(role="user", content=seed.question),
        Message(role="assistant", content=seed.seed_answer),
    ]


def conversation_for(
    node_id: str,
    seed: VerifiedSeed | GeneratedSeed,
    nodes_by_id: dict[str, BranchNode],
) -> list[Message]:
    """Canonical transcript for a node: seed Q/A plus the path to this node.

    Used by both intervention generation and trajectory judgment.
    """
    messages = conversation_for_seed(seed)
    parsed_seed, path = parse_node_id(node_id)
    if parsed_seed != seed.seed_id:
        raise ValueError(f"node {node_id} does not belong to seed {seed.seed_id}")
    for depth in range(1, len(path) + 1):
        current_id = make_node_id(seed.seed_id, path[:depth])
        node = nodes_by_id.get(current_id)
        if node is None:
            raise KeyError(f"Missing node {current_id} while reconstructing {node_id}")
        messages.append(Message(role="user", content=node.user_message))
        messages.append(Message(role="assistant", content=node.assistant_response))
    return messages


def conversation_before_user_turn(
    node: BranchNode,
    seed: VerifiedSeed,
    nodes_by_id: dict[str, BranchNode],
) -> list[Message]:
    """Transcript before the user message that created this node."""
    if node.parent_node_id:
        return conversation_for(node.parent_node_id, seed, nodes_by_id)
    return conversation_for_seed(seed)


def messages_as_dicts(messages: list[Message]) -> list[dict[str, str]]:
    return [message.to_dict() for message in messages]


def render_conversation(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {message.content}")
    return "\n\n".join(lines)
