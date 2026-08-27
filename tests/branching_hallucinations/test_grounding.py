from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.grounding import _status_from_payload
from branching_hallucinations.schemas import VerificationStatus


class GroundingStatusTests(unittest.TestCase):
    def test_only_canonical_statuses_are_accepted(self):
        for status in VerificationStatus:
            self.assertEqual(
                _status_from_payload({"status": status.value}),
                status,
            )

    def test_true_false_aliases_are_not_mapped(self):
        self.assertIsNone(_status_from_payload({"status": "TRUE"}))
        self.assertIsNone(_status_from_payload({"status": "FALSE"}))
        self.assertIsNone(_status_from_payload({"status": "CONTRADICTED"}))
        self.assertIsNone(_status_from_payload({"status": "UNSUPPORTED"}))


if __name__ == "__main__":
    unittest.main()
