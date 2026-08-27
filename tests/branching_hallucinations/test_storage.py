from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.schemas import (
    Action,
    BranchNode,
    TrajectoryJudgment,
    TrajectoryState,
    VerificationStatus,
    VerifiedSeed,
)
from branching_hallucinations.storage import RunStore, conversation_for


def _seed() -> VerifiedSeed:
    return VerifiedSeed(
        seed_id="seed149",
        question_id=149,
        domain="research",
        question="q",
        seed_answer="seed answer",
        tracked_claim="tracked claim C",
        tracked_claim_id="seed149/c0",
        verification_status=VerificationStatus.VERIFIED_FALSE,
    )


class StorageTests(unittest.TestCase):
    def test_no_duplicate_node_ids(self):
        seed = _seed()
        node = BranchNode(
            node_id="seed149/D",
            seed_id="seed149",
            parent_node_id=None,
            depth=1,
            path=["D"],
            action=Action.D,
            user_message="u",
            assistant_response="a",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            store.ensure()
            store.append_verified(seed)
            store.append_node(node)
            with self.assertRaises(ValueError):
                store.append_node(node)

    def test_judgments_do_not_alter_raw_nodes(self):
        node = BranchNode(
            node_id="seed149/D",
            seed_id="seed149",
            parent_node_id=None,
            depth=1,
            path=["D"],
            action=Action.D,
            user_message="u",
            assistant_response="a",
        )
        judgment = TrajectoryJudgment(
            node_id="seed149/D",
            seed_id="seed149",
            explicit_retraction=False,
            reaffirms_claim=True,
            uses_claim_as_premise=False,
            label=TrajectoryState.REPEAT,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            store.ensure()
            store.append_node(node)
            raw = store.nodes_path.read_text(encoding="utf-8")
            store.append_judgment(judgment, version="v1")
            store.append_judgment(judgment, version="v2")
            self.assertEqual(store.nodes_path.read_text(encoding="utf-8"), raw)
            self.assertTrue(store.trajectory_path("v1").exists())
            self.assertTrue(store.trajectory_path("v2").exists())
            self.assertNotIn("REPEAT", store.nodes()[0].to_dict())

    def test_conversation_reconstruction(self):
        seed = _seed()
        t1 = BranchNode(
            node_id="seed149/D",
            seed_id="seed149",
            parent_node_id=None,
            depth=1,
            path=["D"],
            action=Action.D,
            user_message="t1 user",
            assistant_response="t1 assistant",
        )
        t2 = BranchNode(
            node_id="seed149/D/V",
            seed_id="seed149",
            parent_node_id="seed149/D",
            depth=2,
            path=["D", "V"],
            action=Action.V,
            user_message="t2 user",
            assistant_response="t2 assistant",
        )
        convo = conversation_for("seed149/D/V", seed, {t1.node_id: t1, t2.node_id: t2})
        self.assertEqual([m.role for m in convo], ["user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertEqual(convo[0].content, "q")
        self.assertEqual(convo[-1].content, "t2 assistant")
        self.assertEqual(convo[2].content, "t1 user")


if __name__ == "__main__":
    unittest.main()
