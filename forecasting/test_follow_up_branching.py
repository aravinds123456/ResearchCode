"""Unit tests for branched follow-up generation (no GPU / API required)."""

from __future__ import annotations

import unittest

from follow_up_branch import run_adaptive_branch
from follow_up_prompts import (
    DRAFT_FOLLOWUP_PROMPT,
    EXPERIMENT_STRATEGIES,
    REACTIVE_FOLLOWUPS,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_TEMPLATES,
    branch_id,
    build_draft_prompt,
    fallback_followup,
    future_turn_fields,
    normalize_turn_state,
    parse_strategies,
    trajectory_key,
)


class ParseStrategiesTests(unittest.TestCase):
    def test_all(self):
        self.assertEqual(parse_strategies("all"), list(EXPERIMENT_STRATEGIES))

    def test_subset_and_alias(self):
        self.assertEqual(
            parse_strategies("neutral, challenge"),
            ["neutral", "skeptical"],
        )

    def test_unknown(self):
        with self.assertRaises(ValueError):
            parse_strategies("aggressive")


class KeyTests(unittest.TestCase):
    def test_branch_id(self):
        self.assertEqual(branch_id(100010, "skeptical"), "100010:skeptical")

    def test_trajectory_key_branched(self):
        row = {"question_number": 8, "branch_id": "8:neutral"}
        self.assertEqual(trajectory_key(row), "8:neutral")

    def test_trajectory_key_legacy(self):
        self.assertEqual(trajectory_key({"question_number": 8}), "8")


class PromptTests(unittest.TestCase):
    def test_five_strategies_and_four_states(self):
        self.assertEqual(len(EXPERIMENT_STRATEGIES), 5)
        self.assertEqual(len(REACTIVE_FOLLOWUPS), 4)
        for name in EXPERIMENT_STRATEGIES:
            self.assertIn(name, STRATEGY_DESCRIPTIONS)
            self.assertEqual(len(STRATEGY_TEMPLATES[name]), 5)

    def test_drafter_must_not_induce_snowball(self):
        self.assertIn("Does NOT try to make the assistant compound an error", DRAFT_FOLLOWUP_PROMPT)
        self.assertIn("Does NOT ask the assistant to invent extra facts", DRAFT_FOLLOWUP_PROMPT)

    def test_normalize_turn_state(self):
        self.assertEqual(normalize_turn_state("new-hallucination"), "new_hallucination")
        self.assertEqual(normalize_turn_state("nope"), "not_applicable")

    def test_fallback_uses_strategy_script(self):
        text = fallback_followup("dependency-seeking", "persisted", 1)
        self.assertIn("depend", text.lower())

    def test_draft_prompt_includes_strategy_and_state(self):
        prompt = build_draft_prompt(
            "What is X?",
            "X is Y.",
            [
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is Y."},
            ],
            "skeptical",
            "persisted",
        )
        self.assertIn("skeptical", prompt)
        self.assertIn("persisted", prompt)


class BranchLoopTests(unittest.TestCase):
    def test_template_only_five_turns(self):
        calls = {"n": 0}

        def generate_fn(messages):
            calls["n"] += 1
            return f"reply-{calls['n']}"

        record = run_adaptive_branch(
            question="Q?",
            original_answer="Wrong answer.",
            strategy="neutral",
            n_turns=5,
            generate_fn=generate_fn,
        )
        self.assertEqual(calls["n"], 5)
        self.assertEqual(record["n_turns"], 5)
        self.assertEqual(len(record["follow_ups"]), 5)
        self.assertEqual(record["future_turn_5"], "reply-5")
        self.assertEqual(record["strategy"], "neutral")
        self.assertEqual(record["turn_states"], ["not_applicable"] * 5)

    def test_adaptive_draft_and_classify(self):
        states = ["persisted", "new_hallucination", "corrected"]

        def classify_fn(question, original, messages, latest):
            idx = len([m for m in messages if m["role"] == "assistant"]) - 2
            return states[idx], f"reason-{idx}"

        def draft_fn(strategy, turn_state, messages, question, original, turn_index):
            return f"{strategy}:{turn_state}:{turn_index}"

        def generate_fn(messages):
            return messages[-1]["content"].upper()

        record = run_adaptive_branch(
            question="Q?",
            original_answer="A0",
            strategy="accepting",
            n_turns=3,
            generate_fn=generate_fn,
            draft_fn=draft_fn,
            classify_fn=classify_fn,
            seed_state="persisted",
        )
        self.assertEqual(record["follow_up_1"], "accepting:persisted:1")
        self.assertEqual(record["turn_state_1"], "persisted")
        self.assertEqual(record["follow_up_2"], "accepting:persisted:2")
        self.assertEqual(record["turn_state_2"], "new_hallucination")
        self.assertEqual(record["follow_up_3"], "accepting:new_hallucination:3")
        self.assertEqual(record["turn_state_3"], "corrected")

    def test_every_hallucination_gets_five_branch_ids(self):
        q = 100010
        ids = [branch_id(q, s) for s in parse_strategies("all")]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)


class FieldHelperTests(unittest.TestCase):
    def test_future_turn_fields_scales(self):
        row = {"original_answer": "a", "future_turn_1": "b", "future_turn_2": "c"}
        self.assertEqual(
            future_turn_fields(row),
            ["original_answer", "future_turn_1", "future_turn_2"],
        )


if __name__ == "__main__":
    unittest.main()
