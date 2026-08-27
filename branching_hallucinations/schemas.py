"""Scientific objects for the Branching Hallucinations experiment.

No provider-specific types live here. Invalid actions and trajectory labels
are rejected at construction time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


SCHEMA_VERSION = "branching_hallucinations.v1"


class Action(str, Enum):
    D = "D"
    N = "N"
    V = "V"


class TrajectoryState(str, Enum):
    DROP = "DROP"
    RETRACT = "RETRACT"
    REPEAT = "REPEAT"
    DEPEND = "DEPEND"


class VerificationStatus(str, Enum):
    VERIFIED_FALSE = "VERIFIED_FALSE"
    SUPPORTED = "SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIABLE = "UNVERIFIABLE"


class ParseStatus(str, Enum):
    OK = "ok"
    RETRIED = "retried"
    FAILED = "failed"


class AtomicityStatus(str, Enum):
    ATOMIC = "atomic"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


ACTIVE_STATES = (TrajectoryState.REPEAT, TrajectoryState.DEPEND)
INACTIVE_STATES = (TrajectoryState.DROP, TrajectoryState.RETRACT)

# Within-response resolution only. Never used to overwrite terminal path state.
WITHIN_RESPONSE_PRECEDENCE = (
    TrajectoryState.DEPEND,
    TrajectoryState.REPEAT,
    TrajectoryState.RETRACT,
    TrajectoryState.DROP,
)

DOMAIN_TASK = {
    "research": "research_questions",
    "legal": "legal_cases",
    "medical": "medical_guidelines",
    "coding": "coding",
}


def _require_enum(enum_cls: type[Enum], value: Any) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ValueError(f"Invalid {enum_cls.__name__} {value!r}; expected one of: {allowed}") from error


def make_seed_id(question_id: int | str, sample_index: int = 0) -> str:
    if sample_index:
        return f"seed{question_id}:{sample_index}"
    return f"seed{question_id}"


def make_claim_id(seed_id: str, index: int) -> str:
    return f"{seed_id}/c{index}"


def make_node_id(seed_id: str, path: list[str] | tuple[str, ...]) -> str:
    if not path:
        raise ValueError("node path must contain at least one action")
    actions = [Action(item).value for item in path]
    return f"{seed_id}/{'/'.join(actions)}"


def parent_node_id(seed_id: str, path: list[str] | tuple[str, ...]) -> str | None:
    if len(path) <= 1:
        return None
    return make_node_id(seed_id, path[:-1])


def parse_node_id(node_id: str) -> tuple[str, list[str]]:
    if "/" not in node_id:
        raise ValueError(f"Invalid node_id {node_id!r}; expected seed_id/path")
    seed_id, remainder = node_id.split("/", 1)
    path = remainder.split("/")
    if not seed_id or not path or any(part not in Action._value2member_map_ for part in path):
        raise ValueError(f"Invalid node_id {node_id!r}")
    return seed_id, path


def expected_label(
    *,
    uses_claim_as_premise: bool,
    reaffirms_claim: bool,
    explicit_retraction: bool,
) -> TrajectoryState:
    if uses_claim_as_premise:
        return TrajectoryState.DEPEND
    if reaffirms_claim:
        return TrajectoryState.REPEAT
    if explicit_retraction:
        return TrajectoryState.RETRACT
    return TrajectoryState.DROP


def _as_dict(record: Any) -> dict[str, Any]:
    payload = asdict(record)
    for key, value in list(payload.items()):
        if isinstance(value, Enum):
            payload[key] = value.value
    return payload


def _filter_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in known}


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(role=str(data["role"]), content=str(data.get("content") or ""))


@dataclass
class EvidenceSpan:
    text: str
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceSpan":
        return cls(
            text=str(data.get("text") or ""),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass
class GeneratedSeed:
    seed_id: str
    question_id: int | str
    domain: str
    question: str
    seed_answer: str
    answer_model: str
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedSeed":
        return cls(**_filter_fields(cls, data))


@dataclass
class CandidateClaim:
    claim_id: str
    seed_id: str
    text: str
    source_span: str = ""
    entities: list[str] = field(default_factory=list)
    atomicity_status: AtomicityStatus = AtomicityStatus.UNKNOWN
    extractor_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atomicity_status = _require_enum(AtomicityStatus, self.atomicity_status)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["atomicity_status"] = self.atomicity_status.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateClaim":
        payload = _filter_fields(cls, data)
        if "atomicity_status" in payload:
            payload["atomicity_status"] = AtomicityStatus(payload["atomicity_status"])
        return cls(**payload)


@dataclass
class VerificationResult:
    claim_id: str
    status: VerificationStatus
    reason: str = ""
    queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    evidence_passages: list[str] = field(default_factory=list)
    verification_method: str = "halluhard_webscraper"
    verifier_model: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    verifier_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _require_enum(VerificationStatus, self.status)  # type: ignore[assignment]
        self.parse_status = _require_enum(ParseStatus, self.parse_status)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["status"] = self.status.value
        payload["parse_status"] = self.parse_status.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        payload = _filter_fields(cls, data)
        if "status" in payload:
            payload["status"] = VerificationStatus(payload["status"])
        if "parse_status" in payload:
            payload["parse_status"] = ParseStatus(payload["parse_status"])
        return cls(**payload)


@dataclass
class VerifiedSeed:
    """Frozen experimental starting point. Only VERIFIED_FALSE seeds enter the tree."""

    seed_id: str
    question_id: int | str
    domain: str
    question: str
    seed_answer: str
    tracked_claim: str
    tracked_claim_id: str
    verification_status: VerificationStatus = VerificationStatus.VERIFIED_FALSE
    verification_reason: str = ""
    queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    evidence_passages: list[str] = field(default_factory=list)
    answer_model_metadata: dict[str, Any] = field(default_factory=dict)
    verification_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.verification_status = _require_enum(VerificationStatus, self.verification_status)  # type: ignore[assignment]
        if self.verification_status != VerificationStatus.VERIFIED_FALSE:
            raise ValueError(
                "VerifiedSeed requires verification_status=VERIFIED_FALSE; "
                f"got {self.verification_status.value}"
            )
        if not (self.tracked_claim or "").strip():
            raise ValueError("VerifiedSeed requires a non-empty tracked_claim")

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["verification_status"] = self.verification_status.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedSeed":
        payload = _filter_fields(cls, data)
        if "verification_status" in payload:
            payload["verification_status"] = VerificationStatus(payload["verification_status"])
        return cls(**payload)


@dataclass
class BranchNode:
    """One assistant response in the tree. Contains no trajectory label."""

    node_id: str
    seed_id: str
    parent_node_id: str | None
    depth: int
    path: list[str]
    action: Action
    user_message: str
    assistant_response: str
    intervention_metadata: dict[str, Any] = field(default_factory=dict)
    answer_model_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = _require_enum(Action, self.action)  # type: ignore[assignment]
        self.path = [Action(item).value for item in self.path]
        expected_id = make_node_id(self.seed_id, self.path)
        if self.node_id != expected_id:
            raise ValueError(f"node_id {self.node_id!r} does not match path {expected_id!r}")
        if self.path[-1] != self.action.value:
            raise ValueError("path terminal action must match action")
        if self.depth != len(self.path):
            raise ValueError("depth must equal path length")
        expected_parent = parent_node_id(self.seed_id, self.path)
        if self.parent_node_id != expected_parent:
            raise ValueError(
                f"parent_node_id {self.parent_node_id!r} does not match {expected_parent!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["action"] = self.action.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchNode":
        payload = _filter_fields(cls, data)
        if "action" in payload:
            payload["action"] = Action(payload["action"])
        return cls(**payload)


@dataclass
class TrajectoryJudgment:
    node_id: str
    seed_id: str
    explicit_retraction: bool
    reaffirms_claim: bool
    uses_claim_as_premise: bool
    label: TrajectoryState
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    reason: str = ""
    judge_model: str = ""
    judge_prompt_version: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    judge_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _require_enum(TrajectoryState, self.label)  # type: ignore[assignment]
        self.parse_status = _require_enum(ParseStatus, self.parse_status)  # type: ignore[assignment]
        if self.parse_status != ParseStatus.FAILED:
            derived = expected_label(
                uses_claim_as_premise=self.uses_claim_as_premise,
                reaffirms_claim=self.reaffirms_claim,
                explicit_retraction=self.explicit_retraction,
            )
            if self.label != derived:
                raise ValueError(
                    f"label {self.label.value} is inconsistent with "
                    f"uses_claim_as_premise={self.uses_claim_as_premise}, "
                    f"reaffirms_claim={self.reaffirms_claim}, "
                    f"explicit_retraction={self.explicit_retraction}; expected {derived.value}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["label"] = self.label.value
        payload["parse_status"] = self.parse_status.value
        payload["evidence_spans"] = [span.to_dict() if isinstance(span, EvidenceSpan) else span for span in self.evidence_spans]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryJudgment":
        payload = _filter_fields(cls, data)
        if "label" in payload:
            payload["label"] = TrajectoryState(payload["label"])
        if "parse_status" in payload:
            payload["parse_status"] = ParseStatus(payload["parse_status"])
        spans = []
        for span in payload.get("evidence_spans") or []:
            spans.append(span if isinstance(span, EvidenceSpan) else EvidenceSpan.from_dict(span))
        payload["evidence_spans"] = spans
        return cls(**payload)


@dataclass
class ActionAudit:
    node_id: str
    desired_action: Action
    realized_action: Action | None
    valid: bool
    reason: str = ""
    evidence: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    auditor_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.desired_action = _require_enum(Action, self.desired_action)  # type: ignore[assignment]
        if self.realized_action is not None:
            self.realized_action = _require_enum(Action, self.realized_action)  # type: ignore[assignment]
        self.parse_status = _require_enum(ParseStatus, self.parse_status)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["desired_action"] = self.desired_action.value
        payload["realized_action"] = self.realized_action.value if self.realized_action else None
        payload["parse_status"] = self.parse_status.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionAudit":
        payload = _filter_fields(cls, data)
        payload["desired_action"] = Action(payload["desired_action"])
        if payload.get("realized_action"):
            payload["realized_action"] = Action(payload["realized_action"])
        if "parse_status" in payload:
            payload["parse_status"] = ParseStatus(payload["parse_status"])
        return cls(**payload)


@dataclass
class InterventionResult:
    text: str
    desired_action: Action
    draft_attempts: list[str] = field(default_factory=list)
    fallback_used: bool = False
    validation_failures: list[str] = field(default_factory=list)
    writer_model_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.desired_action = _require_enum(Action, self.desired_action)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload = _as_dict(self)
        payload["desired_action"] = self.desired_action.value
        return payload
