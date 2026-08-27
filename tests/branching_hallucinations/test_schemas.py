from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.schemas import (
    Action,
    BranchNode,
    ParseStatus,
    TrajectoryJudgment,
    TrajectoryState,
    VerificationStatus,
    VerifiedSeed,
    expected_label,
    make_node_id,
    parent_node_id,
    parse_node_id,
)


class SchemaTests(unittest.TestCase):
    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            Action("X")

    def test_invalid_trajectory_label_rejected(self):
        with self.assertRaises(ValueError):
            TrajectoryState("CORRECT")

    def test_stable_node_ids(self):
        self.assertEqual(make_node_id("seed149", ["D"]), "seed149/D")
        self.assertEqual(make_node_id("seed149", ["D", "V"]), "seed149/D/V")
        self.assertEqual(parent_node_id("seed149", ["D", "V"]), "seed149/D")
        self.assertIsNone(parent_node_id("seed149", ["D"]))
        seed, path = parse_node_id("seed149/D/V")
        self.assertEqual(seed, "seed149")
        self.assertEqual(path, ["D", "V"])

    def test_parent_path_consistency_enforced(self):
        with self.assertRaises(ValueError):
            BranchNode(
                node_id="seed1/D/V",
                seed_id="seed1",
                parent_node_id="seed1/N",
                depth=2,
                path=["D", "V"],
                action=Action.V,
                user_message="q",
                assistant_response="a",
            )

    def test_verified_seed_rejects_supported(self):
        with self.assertRaises(ValueError):
            VerifiedSeed(
                seed_id="seed1",
                question_id=1,
                domain="research",
                question="q",
                seed_answer="a",
                tracked_claim="c",
                tracked_claim_id="seed1/c0",
                verification_status=VerificationStatus.SUPPORTED,
            )

    def test_expected_label_precedence(self):
        self.assertEqual(
            expected_label(uses_claim_as_premise=True, reaffirms_claim=True, explicit_retraction=True),
            TrajectoryState.DEPEND,
        )
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=True, explicit_retraction=True),
            TrajectoryState.REPEAT,
        )
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=False, explicit_retraction=True),
            TrajectoryState.RETRACT,
        )
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=False, explicit_retraction=False),
            TrajectoryState.DROP,
        )

    def test_inconsistent_judgment_rejected_unless_unparsed(self):
        with self.assertRaises(ValueError):
            TrajectoryJudgment(
                node_id="seed1/D",
                seed_id="seed1",
                explicit_retraction=False,
                reaffirms_claim=True,
                uses_claim_as_premise=True,
                label=TrajectoryState.REPEAT,
                parse_status=ParseStatus.OK,
            )
        failed = TrajectoryJudgment(
            node_id="seed1/D",
            seed_id="seed1",
            explicit_retraction=False,
            reaffirms_claim=False,
            uses_claim_as_premise=False,
            label=TrajectoryState.DROP,
            parse_status=ParseStatus.FAILED,
        )
        self.assertEqual(failed.parse_status, ParseStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
