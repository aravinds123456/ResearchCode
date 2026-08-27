from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import DatasetConfig
from branching_hallucinations.questions import (
    HALLUHARD_DOMAINS,
    load_jsonl,
    load_questions,
    register_source,
)


class QuestionSourceTests(unittest.TestCase):
    def test_halluhard_research_questions_exist(self):
        questions = load_questions(
            DatasetConfig(name="halluhard"),
            domains=("research",),
            max_questions=3,
        )
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0]["source"], "halluhard")
        self.assertEqual(questions[0]["domain"], "research")
        self.assertTrue(questions[0]["question"])

    def test_unknown_halluhard_domain(self):
        with self.assertRaises(ValueError):
            load_questions(DatasetConfig(name="halluhard"), domains=("not-a-domain",))

    def test_jsonl_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qs.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "q/1",
                        "domain": "scienceqa",
                        "prompt": "What is the boiling point of water at 1 atm?",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            questions = load_jsonl(
                path=path,
                question_field="prompt",
                domains=("scienceqa",),
                source="scienceqa",
            )
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["question_id"], "q_1")
        self.assertEqual(questions[0]["domain"], "scienceqa")
        self.assertEqual(questions[0]["source"], "scienceqa")

    def test_register_source(self):
        def fake_loader(**_: object):
            return [
                {
                    "question_id": "x",
                    "domain": "custom",
                    "question": "Name a city.",
                    "source": "fake",
                }
            ]

        register_source("fake", fake_loader)
        try:
            questions = load_questions(DatasetConfig(name="fake"))
            self.assertEqual(questions[0]["question"], "Name a city.")
        finally:
            from branching_hallucinations import questions as qmod

            qmod.DATASET_LOADERS.pop("fake", None)

    def test_halluhard_domain_files_are_present(self):
        missing = [
            name
            for name, spec in HALLUHARD_DOMAINS.items()
            if name != "coding" and not spec.path.exists()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
