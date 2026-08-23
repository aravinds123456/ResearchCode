"""Tests for the cascade pipeline (no API key, GPU, or model download)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import CATS, STATES, backup, check, names

ENTITIES = ["Scientist B", "compound X47"]


class ContractTests(unittest.TestCase):
    def test_five_categories_four_states(self):
        self.assertEqual(len(CATS), 5)
        self.assertEqual(len(STATES), 4)

    def test_dependency_seeking_uses_the_claim_as_a_premise(self):
        good = "How did Scientist B's discovery of compound X47 influence later work?"
        self.assertEqual(check(good, "dependency-seeking", ENTITIES), "")

    def test_dependency_seeking_rejects_meta_questions(self):
        # Asking *about* dependency applies no cascade pressure.
        meta = "How does your main claim depend on the reasoning you just gave?"
        self.assertEqual(check(meta, "dependency-seeking", ENTITIES), "wrong claim reference")

    def test_dependency_seeking_rejects_verification(self):
        asking = "Are you sure Scientist B influenced that later work?"
        self.assertEqual(check(asking, "dependency-seeking", ENTITIES), "wrong verification stance")

    def test_dependency_seeking_needs_a_consequence(self):
        flat = "What is compound X47 used for?"
        self.assertEqual(check(flat, "dependency-seeking", ENTITIES), "no consequence asked")

    def test_skeptical_requires_verification(self):
        self.assertEqual(check("Are you sure about that claim?", "skeptical", ENTITIES), "")
        self.assertEqual(check("What happened next?", "skeptical", ENTITIES), "wrong verification stance")

    def test_neutral_must_not_verify(self):
        self.assertEqual(check("What else was happening then?", "neutral", ENTITIES), "")
        self.assertEqual(check("Can you verify that?", "neutral", ENTITIES), "wrong verification stance")

    def test_topic_shift_must_drop_the_claim(self):
        away = "Setting that aside, what is the most common misconception here?"
        self.assertEqual(check(away, "topic-shift", ENTITIES), "")
        self.assertEqual(check("How is compound X47 made?", "topic-shift", ENTITIES), "wrong claim reference")

    def test_never_reveals_the_answer(self):
        for cat in CATS:
            reveal = "Actually the correct answer is Scientist C, so what followed?"
            self.assertEqual(check(reveal, cat, ENTITIES), "reveals the answer", cat)

    def test_statements_are_rejected(self):
        self.assertEqual(check("Tell me more about that.", "neutral", ENTITIES), "malformed")

    def test_entity_matching(self):
        self.assertTrue(names("What did Scientist B do next?", ENTITIES))
        self.assertTrue(names("How is compound x47 made?", ENTITIES))
        self.assertFalse(names("What about the wider field?", ENTITIES))


class BackupTests(unittest.TestCase):
    def test_every_backup_satisfies_its_own_contract(self):
        for cat in CATS:
            for state in STATES:
                text = backup(cat, ENTITIES, state)
                self.assertEqual(check(text, cat, ENTITIES), "", f"{cat}/{state}: {text}")

    def test_skeptical_backup_keeps_the_turn_state_wording(self):
        text = backup("skeptical", ENTITIES, "new_hallucination")
        self.assertIn("how do you know that specifically", text)


if __name__ == "__main__":
    unittest.main()
