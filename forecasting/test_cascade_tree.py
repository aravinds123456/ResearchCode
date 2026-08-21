"""Tests for the cascade tree runner (no API key, GPU, or model download)."""

from __future__ import annotations

import unittest

import build_cascade_tree as tree
from followup_defs import CATEGORIES


class SeedLoadingTests(unittest.TestCase):
    def test_halluhard_schema(self):
        seed = tree.normalize_seed(
            {
                "question_number": 7,
                "question": "Who discovered X?",
                "qwen_answer": "Scientist B did.",
                "domain": "research",
            },
            "halluhard",
            0,
        )
        self.assertEqual(seed["question_number"], 7)
        self.assertEqual(seed["original_answer"], "Scientist B did.")
        self.assertEqual(seed["domain"], "research")

    def test_generic_schema(self):
        seed = tree.normalize_seed(
            {"id": "x1", "prompt": "Who discovered X?", "response": "Scientist B did."},
            "other_dataset",
            0,
        )
        self.assertEqual(seed["question_number"], "x1")
        self.assertEqual(seed["dataset"], "other_dataset")
        self.assertEqual(seed["domain"], "other_dataset")

    def test_row_without_answer_is_skipped(self):
        self.assertIsNone(tree.normalize_seed({"question": "Q?"}, "d", 0))

    def test_hallucination_filter(self):
        self.assertTrue(
            tree.is_hallucinating({"gemini_judgement": "Overall label: Hallucinating"})
        )
        self.assertFalse(
            tree.is_hallucinating({"gemini_judgement": "Overall label: Not hallucinating"})
        )
        self.assertTrue(tree.is_hallucinating({"label": "yes"}))
        self.assertFalse(tree.is_hallucinating({"label": "no"}))
        # Pre-filtered datasets carry no verdict field.
        self.assertTrue(tree.is_hallucinating({"question": "Q?"}))


class BranchIdTests(unittest.TestCase):
    def test_id_separates_model_dataset_and_category(self):
        seed = {"dataset": "halluhard", "question_number": 12}
        first = tree.branch_id(seed, "Qwen3.5-2B", "neutral")
        self.assertEqual(first, "Qwen3.5-2B:halluhard:12:neutral")
        self.assertNotEqual(first, tree.branch_id(seed, "Qwen3.5-2B", "skeptical"))
        self.assertNotEqual(first, tree.branch_id(seed, "other-model", "neutral"))

    def test_every_category_is_a_distinct_branch(self):
        seed = {"dataset": "d", "question_number": 1}
        ids = {tree.branch_id(seed, "m", category) for category in CATEGORIES}
        self.assertEqual(len(ids), 5)


class BranchLoopTests(unittest.TestCase):
    def setUp(self):
        self.seed = {
            "dataset": "d",
            "question_number": 1,
            "domain": "research",
            "question": "Who discovered X?",
            "original_answer": "Scientist B discovered X in 1953.",
        }
        self.claim = {"claim": "Scientist B discovered X.", "entities": ["Scientist B"]}
        self._generate = tree.generate_followup
        self._classify = tree.classify_turn_state

    def tearDown(self):
        tree.generate_followup = self._generate
        tree.classify_turn_state = self._classify

    def test_dry_run_produces_one_row_per_level(self):
        row = tree.run_branch(
            self.seed, self.claim, "neutral", 4, tree.StubAnswerer(), "gpt-4o-mini", True
        )
        self.assertEqual(len(row["follow_ups"]), 4)
        self.assertEqual(len(row["turn_states"]), 4)
        self.assertIn("future_turn_4", row)
        self.assertNotIn("future_turn_5", row)
        self.assertEqual(row["follow_up_source_1"], "dry_run")

    def test_state_from_each_turn_drives_the_next_followup(self):
        states = ["new_hallucination", "corrected", "not_applicable"]
        seen_states = []

        def fake_generate(question, claim, entities, messages, category, turn_state, model):
            seen_states.append(turn_state)
            return {"follow_up": f"q-{turn_state}", "source": "llm", "validation": "ok"}

        def fake_classify(question, claim, messages, latest, model):
            return {"turn_state": states[len(seen_states) - 1], "reason": "test"}

        tree.generate_followup = fake_generate
        tree.classify_turn_state = fake_classify

        row = tree.run_branch(
            self.seed, self.claim, "skeptical", 3, tree.StubAnswerer(), "gpt-4o-mini", False
        )
        # Level 1 uses the seed state; later levels use the previous classification.
        self.assertEqual(seen_states, ["persisted", "new_hallucination", "corrected"])
        self.assertEqual(row["turn_states"], states)
        self.assertEqual(row["follow_up_2"], "q-new_hallucination")

    def test_category_is_fixed_within_a_branch(self):
        used = []

        def fake_generate(question, claim, entities, messages, category, turn_state, model):
            used.append(category)
            return {"follow_up": "q", "source": "llm", "validation": "ok"}

        tree.generate_followup = fake_generate
        tree.classify_turn_state = lambda *a, **k: {"turn_state": "persisted", "reason": ""}
        tree.run_branch(
            self.seed, self.claim, "dependency-seeking", 5, tree.StubAnswerer(), "m", False
        )
        self.assertEqual(used, ["dependency-seeking"] * 5)

    def test_conversation_grows_with_each_turn(self):
        lengths = []

        def fake_generate(question, claim, entities, messages, category, turn_state, model):
            lengths.append(len(messages))
            return {"follow_up": "q", "source": "llm", "validation": "ok"}

        tree.generate_followup = fake_generate
        tree.classify_turn_state = lambda *a, **k: {"turn_state": "persisted", "reason": ""}
        tree.run_branch(self.seed, self.claim, "neutral", 3, tree.StubAnswerer(), "m", False)
        # Starts at question + seed answer, then grows by a user/assistant pair per level.
        self.assertEqual(lengths, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
