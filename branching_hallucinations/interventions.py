"""D/N/V follow-up writing and post-hoc action audit.

The writer gets the tracked claim, the transcript, and a move-specific prompt.
It never receives a trajectory label. There is no regex filter on the question.
There is no canned D/N/V question. Unparseable writer output is retried once
for format, then the run stops.
"""

from __future__ import annotations

from libs.types import SamplerBase

from .config import fill_prompt, prompt_hash
from .json_parse import ParseError, parse_json_object
from .models import complete, sampler_metadata
from .schemas import Action, ActionAudit, InterventionResult, Message, ParseStatus
from .storage import render_conversation


FOLLOWUP_PROMPT = {
    Action.D: "followup_d",
    Action.N: "followup_n",
    Action.V: "followup_v",
}

FORMAT_REMINDER = (
    'FORMAT REMINDER: Return JSON only: {"follow_up": "<one natural question>"}. '
    "Do not return prose outside the JSON object."
)


class FollowupGenerationError(RuntimeError):
    """Writer returned nothing usable after a format retry."""


def _follow_up_from_writer(raw: str) -> str:
    try:
        payload = parse_json_object(raw)
    except ParseError:
        return ""
    return str(payload.get("follow_up") or "").strip()


async def generate_intervention(
    action: Action,
    tracked_claim: str,
    conversation: list[Message],
    writer: SamplerBase,
) -> InterventionResult:
    prompt_name = FOLLOWUP_PROMPT[action]
    prompt = fill_prompt(
        prompt_name,
        claim=tracked_claim,
        conversation=render_conversation(conversation),
    )
    messages = [{"role": "user", "content": prompt}]
    response = await complete(writer, messages)
    metadata = sampler_metadata(writer, response)
    metadata["prompt_hash"] = prompt_hash(prompt_name)
    metadata["parse_status"] = ParseStatus.OK.value
    raw = response.response_text
    text = _follow_up_from_writer(raw)
    drafts = [text or raw[:500]]
    if not text:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": FORMAT_REMINDER + "\n\nPrevious output:\n" + raw},
        ]
        response = await complete(writer, messages)
        metadata = sampler_metadata(writer, response)
        metadata["prompt_hash"] = prompt_hash(prompt_name)
        metadata["parse_status"] = ParseStatus.RETRIED.value
        raw = response.response_text
        text = _follow_up_from_writer(raw)
        drafts.append(text or raw[:500])
    if text:
        return InterventionResult(
            text=text,
            desired_action=action,
            draft_attempts=drafts,
            fallback_used=False,
            validation_failures=[],
            writer_model_metadata=metadata,
        )
    raise FollowupGenerationError(
        f"Follow-up writer returned no JSON follow_up for action {action.value} "
        "after a format retry. No canned question is substituted."
    )


async def audit_action(
    tracked_claim: str,
    conversation_before_user_message: list[Message],
    user_message: str,
    desired_action: Action,
    auditor: SamplerBase,
) -> ActionAudit:
    if auditor is None:
        raise ValueError("Action audit requires an auditor sampler; skipped audits are not stored as labels.")
    prompt = fill_prompt(
        "action_audit",
        claim=tracked_claim,
        conversation=render_conversation(conversation_before_user_message),
        desired_action=desired_action.value,
        user_message=user_message,
    )
    response = await complete(auditor, [{"role": "user", "content": prompt}])
    parse_status = ParseStatus.OK
    try:
        payload = parse_json_object(response.response_text)
    except ParseError:
        reminder = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": "FORMAT REMINDER: Return JSON only with realized_action, valid, reason, evidence."},
        ]
        response = await complete(auditor, reminder)
        parse_status = ParseStatus.RETRIED
        try:
            payload = parse_json_object(response.response_text)
        except ParseError:
            return ActionAudit(
                node_id="",
                desired_action=desired_action,
                realized_action=None,
                valid=False,
                reason="unparseable action audit; not mapped to a scientific label",
                evidence=(user_message or "")[:240],
                parse_status=ParseStatus.FAILED,
                auditor_metadata=sampler_metadata(auditor, response),
            )
    try:
        realized = Action(str(payload.get("realized_action") or "").strip().upper())
    except ValueError:
        realized = None
        parse_status = ParseStatus.FAILED
    valid = bool(payload.get("valid")) and realized == desired_action
    return ActionAudit(
        node_id="",
        desired_action=desired_action,
        realized_action=realized,
        valid=valid,
        reason=str(payload.get("reason") or ""),
        evidence=str(payload.get("evidence") or "")[:500],
        parse_status=parse_status,
        auditor_metadata=sampler_metadata(auditor, response),
    )
