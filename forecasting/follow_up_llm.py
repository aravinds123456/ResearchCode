"""OpenAI helpers: turn-state classifier and follow-up question drafter."""

from __future__ import annotations

from follow_up_prompts import (
    build_classifier_prompt,
    build_draft_prompt,
    fallback_followup,
    normalize_turn_state,
)


def classify_turn(
    question: str,
    original: str,
    messages: list[dict[str, str]],
    latest: str,
) -> tuple[str, str]:
    from factscore_utils import ask

    result = ask(build_classifier_prompt(question, original, messages, latest))
    state = normalize_turn_state(str(result.get("turn_state", "")))
    reason = str(result.get("reason", "")).strip()
    return state, reason


def draft_follow_up(
    strategy: str,
    turn_state: str,
    messages: list[dict[str, str]],
    question: str,
    original: str,
    turn_index: int,
) -> str:
    from factscore_utils import ask

    try:
        result = ask(
            build_draft_prompt(question, original, messages, strategy, turn_state)
        )
        question_text = str(result.get("question", "")).strip()
        if question_text:
            return question_text
    except Exception as exc:
        print(f"  draft failed ({strategy} t{turn_index}): {exc}; using template")
    return fallback_followup(strategy, turn_state, turn_index)
