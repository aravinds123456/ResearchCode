"""Pure branching loop: strategy held constant, turn-state adaptive drafts.

No torch/openai imports so this can be unit-tested without GPU or API keys.
"""

from __future__ import annotations

from collections.abc import Callable

from follow_up_prompts import DEFAULT_SEED_STATE, fallback_followup, normalize_turn_state

ClassifyFn = Callable[[str, str, list[dict[str, str]], str], tuple[str, str]]
DraftFn = Callable[[str, str, list[dict[str, str]], str, str, int], str]
GenerateFn = Callable[[list[dict[str, str]]], str]


def run_adaptive_branch(
    *,
    question: str,
    original_answer: str,
    strategy: str,
    n_turns: int,
    generate_fn: GenerateFn,
    draft_fn: DraftFn | None = None,
    classify_fn: ClassifyFn | None = None,
    seed_state: str = DEFAULT_SEED_STATE,
) -> dict:
    """Run one strategy branch for n_turns.

    draft_fn(strategy, turn_state, messages, question, original, turn_index) -> follow-up
    classify_fn(question, original, messages, latest_assistant) -> (state, reason)
    generate_fn(messages) -> assistant text
    """
    if n_turns < 1:
        raise ValueError("n_turns must be >= 1")

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": original_answer},
    ]
    state = normalize_turn_state(seed_state)
    record: dict = {
        "strategy": strategy,
        "n_turns": n_turns,
        "seed_turn_state": state,
        "follow_ups": [],
        "turn_states": [],
        "turn_state_reasons": [],
    }

    for turn_index in range(1, n_turns + 1):
        if draft_fn is None:
            follow_up = fallback_followup(strategy, state, turn_index)
        else:
            follow_up = draft_fn(
                strategy, state, messages, question, original_answer, turn_index
            )
            follow_up = (follow_up or "").strip() or fallback_followup(
                strategy, state, turn_index
            )

        messages.append({"role": "user", "content": follow_up})
        response = generate_fn(messages)
        messages.append({"role": "assistant", "content": response})

        if classify_fn is None:
            next_state, reason = "not_applicable", "template-only: no classifier"
        else:
            next_state, reason = classify_fn(
                question, original_answer, messages, response
            )
            next_state = normalize_turn_state(next_state)

        record[f"follow_up_{turn_index}"] = follow_up
        record[f"future_turn_{turn_index}"] = response
        record[f"turn_state_{turn_index}"] = next_state
        record[f"turn_state_reason_{turn_index}"] = reason
        record["follow_ups"].append(follow_up)
        record["turn_states"].append(next_state)
        record["turn_state_reasons"].append(reason)
        state = next_state

    return record
