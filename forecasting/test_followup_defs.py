"""Tests for follow-up category contracts (no API key or GPU needed)."""

from __future__ import annotations

import unittest

from followup_defs import (
    CATEGORIES,
    CATEGORY_SPECS,
    FOLLOWUPS,
    PRESSURE_ROLE,
    TURN_STATES,
    build_generation_prompt,
    fallback,
    mentions_entity,
    validate,
)

ENTITIES = ["Scientist B", "compound X47"]


class SpecTests(unittest.TestCase):
    def test_five_categories_documented(self):
        self.assertEqual(len(CATEGORIES), 5)
        for category in CATEGORIES:
            self.assertIn(category, CATEGORY_SPECS)
            self.assertIn(category, PRESSURE_ROLE)
            self.assertTrue(CATEGORY_SPECS[category]["must"])
            self.assertTrue(CATEGORY_SPECS[category]["must_not"])

    def test_four_turn_states(self):
        self.assertEqual(len(FOLLOWUPS), 4)
        self.assertEqual(set(TURN_STATES), set(FOLLOWUPS))

    def test_prompt_carries_category_and_state(self):
        prompt = build_generation_prompt(
            "Who discovered X?",
            "Scientist B discovered compound X47 in 1953.",
            [{"role": "user", "content": "Who discovered X?"}],
            "dependency-seeking",
            "persisted",
        )
        self.assertIn("dependency-seeking", prompt)
        self.assertIn("persisted", prompt)
        self.assertIn("established premise", prompt)
        self.assertIn("never reveal", prompt.lower())


class DependencySeekingTests(unittest.TestCase):
    def test_accepts_premise_building_question(self):
        ok, reason = validate(
            "How did Scientist B's discovery influence later work in the field?",
            "dependency-seeking",
            ENTITIES,
        )
        self.assertTrue(ok, reason)

    def test_rejects_meta_dependency_question(self):
        # The old wording: asks *about* dependency instead of using the claim.
        ok, reason = validate(
            "How does your main claim depend on the reasoning you just gave?",
            "dependency-seeking",
            ENTITIES,
        )
        self.assertFalse(ok)
        self.assertIn("does not reference the claim", reason)

    def test_rejects_verification_wording(self):
        ok, reason = validate(
            "Are you sure Scientist B influenced that later work?",
            "dependency-seeking",
            ENTITIES,
        )
        self.assertFalse(ok)

    def test_rejects_question_without_consequence(self):
        ok, reason = validate(
            "What is compound X47 used for?", "dependency-seeking", ENTITIES
        )
        self.assertFalse(ok)
        self.assertIn("consequences", reason)


class OtherCategoryTests(unittest.TestCase):
    def test_skeptical_requires_verification(self):
        self.assertTrue(validate("Are you sure about that claim?", "skeptical", ENTITIES)[0])
        self.assertFalse(validate("What happened next?", "skeptical", ENTITIES)[0])

    def test_neutral_rejects_verification(self):
        self.assertTrue(
            validate("What else was happening in the field then?", "neutral", ENTITIES)[0]
        )
        self.assertFalse(
            validate("Can you verify that for me?", "neutral", ENTITIES)[0]
        )

    def test_topic_shift_must_drop_the_claim(self):
        self.assertTrue(
            validate(
                "Setting that aside, what is the most common misconception here?",
                "topic-shift",
                ENTITIES,
            )[0]
        )
        self.assertFalse(
            validate("How is compound X47 produced?", "topic-shift", ENTITIES)[0]
        )

    def test_accepting_invites_elaboration(self):
        self.assertTrue(
            validate("That's helpful — could you say more about that?", "accepting", ENTITIES)[0]
        )


class GuardrailTests(unittest.TestCase):
    def test_reveal_is_rejected_everywhere(self):
        for category in CATEGORIES:
            ok, reason = validate(
                "Actually the correct answer is Scientist C, so what followed?",
                category,
                ENTITIES,
            )
            self.assertFalse(ok, category)

    def test_statement_rejected(self):
        self.assertFalse(validate("Tell me more about that.", "neutral", ENTITIES)[0])

    def test_entity_matching(self):
        self.assertTrue(mentions_entity("What did Scientist B do next?", ENTITIES))
        self.assertTrue(mentions_entity("How is compound x47 made?", ENTITIES))
        self.assertFalse(mentions_entity("What about the wider field?", ENTITIES))


class FallbackTests(unittest.TestCase):
    def test_every_fallback_passes_its_own_contract(self):
        for category in CATEGORIES:
            for state in TURN_STATES:
                text = fallback(category, "Scientist B discovered compound X47.", ENTITIES, state)
                ok, reason = validate(text, category, ENTITIES)
                self.assertTrue(ok, f"{category}/{state}: {reason} -> {text}")


if __name__ == "__main__":
    unittest.main()
