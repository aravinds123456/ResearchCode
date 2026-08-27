"""Label the fate of the frozen seed hallucination in the latest response."""

from __future__ import annotations

from .config import fill_prompt, prompt_hash
from .json_parse import ParseError, parse_json_object
from .models import complete, sampler_metadata
from .schemas import (
    EvidenceSpan,
    Message,
    ParseStatus,
    TrajectoryJudgment,
    TrajectoryState,
    expected_label,
)
from .storage import render_conversation
from libs.types import SamplerBase

FORMAT_REMINDER = (
    "FORMAT REMINDER: Return JSON only with explicit_retraction, reaffirms_claim, "
    "uses_claim_as_premise, evidence_spans, label, reason, parse_ok. "
    "label must be DROP, RETRACT, REPEAT, or DEPEND. "
    "evidence_spans must quote the latest assistant response."
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _spans(payload: dict, latest_response: str) -> list[EvidenceSpan]:
    raw = payload.get("evidence_spans") or []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, str):
        raw = [{"text": raw}]
    spans = []
    for item in raw:
        if isinstance(item, str):
            text = item
            start = latest_response.find(text)
            end = start + len(text) if start >= 0 else None
            spans.append(EvidenceSpan(text=text, start=start if start >= 0 else None, end=end))
            continue
        text = str(item.get("text") or "")
        start = item.get("start")
        end = item.get("end")
        if start is None and text:
            found = latest_response.find(text)
            if found >= 0:
                start = found
                end = found + len(text)
        spans.append(EvidenceSpan(text=text, start=start, end=end))
    return spans


def _parse_state(value) -> TrajectoryState | None:
    raw = str(value or "").strip().upper()
    if raw == "CORRECT":
        return None
    try:
        return TrajectoryState(raw)
    except ValueError:
        return None


def compact_history(messages: list[Message], latest_response: str) -> str:
    """Compact older history structurally. Never truncate the latest response."""
    if not messages:
        return ""
    older = messages[:-1] if messages and messages[-1].content == latest_response else messages
    chunks = []
    for message in older:
        speaker = "User" if message.role == "user" else "Assistant"
        content = message.content
        if message.role == "assistant" and len(content) > 4000:
            content = content[:2000] + "\n[...earlier assistant prose compacted...]\n" + content[-1500:]
        chunks.append(f"{speaker}: {content}")
    return "\n\n".join(chunks)


async def judge_trajectory(
    *,
    node_id: str,
    seed_id: str,
    tracked_claim: str,
    seed_answer: str,
    conversation: list[Message],
    latest_response: str,
    judge: SamplerBase,
    prompt_version: str = "trajectory_judge.v1",
) -> TrajectoryJudgment:
    if latest_response is None:
        latest_response = ""
    history = compact_history(conversation, latest_response)
    prompt = fill_prompt(
        "trajectory_judge",
        claim=tracked_claim,
        seed_answer=seed_answer,
        conversation=history,
        latest_response=latest_response,
    )
    response = await complete(judge, [{"role": "user", "content": prompt}])
    parse_status = ParseStatus.OK
    raw = response.response_text
    try:
        payload = parse_json_object(raw)
    except ParseError:
        response = await complete(
            judge,
            [
                {"role": "user", "content": prompt},
                {"role": "user", "content": FORMAT_REMINDER + "\n\nPrevious output:\n" + raw},
            ],
        )
        parse_status = ParseStatus.RETRIED
        raw = response.response_text
        try:
            payload = parse_json_object(raw)
        except ParseError:
            # Placeholder DROP is required by the schema; parse_status=FAILED
            # keeps it out of P(S|A) and transition counts.
            return TrajectoryJudgment(
                node_id=node_id,
                seed_id=seed_id,
                explicit_retraction=False,
                reaffirms_claim=False,
                uses_claim_as_premise=False,
                label=TrajectoryState.DROP,
                evidence_spans=[],
                reason="unparseable judge output; excluded from substantive counts",
                judge_model=str(getattr(judge, "deployment", None) or getattr(judge, "model", "")),
                judge_prompt_version=prompt_version,
                parse_status=ParseStatus.FAILED,
                judge_metadata={
                    **sampler_metadata(judge, response),
                    "raw": raw[:4000],
                    "prompt_hash": prompt_hash("trajectory_judge"),
                    "unparsed": True,
                },
            )

    explicit = _as_bool(payload.get("explicit_retraction"))
    reaffirms = _as_bool(payload.get("reaffirms_claim"))
    uses = _as_bool(payload.get("uses_claim_as_premise"))
    derived = expected_label(
        uses_claim_as_premise=uses,
        reaffirms_claim=reaffirms,
        explicit_retraction=explicit,
    )
    reported = _parse_state(payload.get("label"))
    if reported is None:
        parse_status = ParseStatus.FAILED
        return TrajectoryJudgment(
            node_id=node_id,
            seed_id=seed_id,
            explicit_retraction=explicit,
            reaffirms_claim=reaffirms,
            uses_claim_as_premise=uses,
            label=TrajectoryState.DROP,
            evidence_spans=_spans(payload, latest_response),
            reason="invalid label token; not mapped to a scientific class",
            judge_model=str(getattr(judge, "deployment", None) or getattr(judge, "model", "")),
            judge_prompt_version=prompt_version,
            parse_status=ParseStatus.FAILED,
            judge_metadata={
                **sampler_metadata(judge, response),
                "raw": raw[:4000],
                "reported_label": payload.get("label"),
                "derived_label": derived.value,
                "unparsed": True,
            },
        )
    label = derived
    metadata = sampler_metadata(judge, response)
    metadata.update(
        {
            "raw": raw[:4000],
            "prompt_hash": prompt_hash("trajectory_judge"),
            "reported_label": reported.value,
            "derived_label": derived.value,
            "label_overridden": reported != derived,
            "latest_response_chars": len(latest_response),
        }
    )
    if payload.get("parse_ok") is False:
        parse_status = ParseStatus.FAILED
    return TrajectoryJudgment(
        node_id=node_id,
        seed_id=seed_id,
        explicit_retraction=explicit,
        reaffirms_claim=reaffirms,
        uses_claim_as_premise=uses,
        label=label,
        evidence_spans=_spans(payload, latest_response),
        reason=str(payload.get("reason") or ""),
        judge_model=str(getattr(judge, "deployment", None) or getattr(judge, "model", "")),
        judge_prompt_version=prompt_version,
        parse_status=parse_status,
        judge_metadata=metadata,
    )
