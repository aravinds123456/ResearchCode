from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import load_config


class ConfigTests(unittest.TestCase):
    def test_pilot_config_loads(self):
        cfg = load_config(ROOT / "configs" / "branching_pilot.toml")
        self.assertEqual(cfg.n_seeds, 10)
        self.assertEqual(cfg.depth, 2)
        self.assertEqual(cfg.actions, ("D", "N", "V"))
        self.assertEqual(cfg.grounding_method, "halluhard_webscraper")
        self.assertEqual(cfg.answer.sampler, "azure-gpt-oss-20b")
        self.assertEqual(cfg.trajectory_judge.sampler, "azure-gpt-5-mini")
        self.assertNotIn("CORRECT", cfg.trajectory_states)
        self.assertEqual(cfg.dataset.name, "halluhard")
        self.assertEqual(cfg.domains, ("research", "legal", "medical"))
        self.assertEqual(cfg.dataset.task_for("legal"), "legal_cases")
        self.assertEqual(cfg.dataset.task_for("custom"), "research_questions")
        self.assertFalse(cfg.allow_followup_fallback)

    def test_followup_fallback_flag_is_rejected(self):
        import tempfile

        src = (ROOT / "configs" / "branching_pilot.toml").read_text(encoding="utf-8")
        src = src.replace(
            "max_claims_per_seed = 8",
            "max_claims_per_seed = 8\nallow_followup_fallback = true",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text(src, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
