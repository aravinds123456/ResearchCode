from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import DatasetConfig, load_config
from branching_hallucinations.factbench import (
    DEFAULT_PATH,
    EXPECTED_COUNTS,
    FactBenchError,
    canonicalize_tier,
    load_factbench,
)
from branching_hallucinations.questions import load_questions


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "id": "fb_hard_000",
            "question": "List all people that got awarded more than one Nobel Prize.",
            "domain": "hard",
            "tier": "hard",
            "topic": "Awards",
        },
        {
            "id": "fb_moderate_000",
            "question": "Make a comparison table of Galaxy S21 and iPhone 11",
            "domain": "moderate",
            "tier": "moderate",
            "topic": "Phones",
        },
        {
            "id": "fb_easy_000",
            "question": 'Explain the difference between a "state" and a "country".',
            "domain": "easy",
            "tier": "easy",
            "topic": "Travel",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FactBenchLoaderTests(unittest.TestCase):
    def test_medium_is_moderate(self):
        self.assertEqual(canonicalize_tier("medium"), "moderate")
        self.assertEqual(canonicalize_tier("Moderate"), "moderate")
        self.assertEqual(canonicalize_tier("tier_1"), "hard")

    def test_default_subset_is_hard_and_moderate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_fixture(path)
            questions = load_factbench(path=path)
        self.assertEqual([q["domain"] for q in questions], ["hard", "moderate"])
        self.assertEqual(questions[0]["source"], "factbench")
        self.assertNotIn("easy", {q["domain"] for q in questions})

    def test_medium_alias_selects_moderate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_fixture(path)
            questions = load_factbench(path=path, domains=("medium",))
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["domain"], "moderate")

    def test_halluhard_domains_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_fixture(path)
            with self.assertRaises(FactBenchError):
                load_factbench(path=path, domains=("research", "legal", "medical"))

    def test_registry_default_is_factbench_medium_hard(self):
        questions = load_questions(DatasetConfig(name="factbench", path=DEFAULT_PATH))
        self.assertEqual(len(questions), EXPECTED_COUNTS["hard"] + EXPECTED_COUNTS["moderate"])
        self.assertEqual(questions[0]["source"], "factbench")
        self.assertEqual(questions[0]["domain"], "hard")
        self.assertEqual(questions[1]["domain"], "moderate")
        self.assertTrue(questions[0]["question"])
        self.assertNotIn("easy", {q["domain"] for q in questions})

    def test_vendored_counts_match_official_tiers(self):
        all_tiers = load_factbench(path=DEFAULT_PATH, domains=("hard", "moderate", "easy"))
        counts = {}
        for row in all_tiers:
            counts[row["domain"]] = counts.get(row["domain"], 0) + 1
        self.assertEqual(counts, EXPECTED_COUNTS)
        self.assertEqual(all_tiers[0]["question_id"], "fb_hard_000")
        self.assertEqual(all_tiers[0]["domain"], "hard")
        self.assertEqual(all_tiers[1]["domain"], "moderate")
        self.assertEqual(
            all_tiers[1]["question"],
            "Give me a vegetarian lunch idea that contains ~400 calories and at least 30 grams of protein per serving.",
        )

    def test_pilot_factbench_config(self):
        cfg = load_config(ROOT / "configs" / "branching_pilot_factbench.toml")
        self.assertEqual(cfg.dataset.name, "factbench")
        self.assertEqual(cfg.domains, ("hard", "moderate"))
        self.assertEqual(cfg.dataset.task_for("hard"), "research_questions")
        self.assertEqual(cfg.dataset.task_for("moderate"), "research_questions")
        self.assertEqual(cfg.n_seeds, 10)
        self.assertFalse(cfg.allow_followup_fallback)
        self.assertEqual(cfg.trajectory_states, ("DROP", "RETRACT", "REPEAT", "DEPEND"))


if __name__ == "__main__":
    unittest.main()
