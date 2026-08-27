from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.analysis import analyze, mcnemar_exact, t1_distribution, t2_transitions
from branching_hallucinations.schemas import (
    Action,
    ActionAudit,
    BranchNode,
    ParseStatus,
    TrajectoryJudgment,
    TrajectoryState,
)


def _node(path: str, response="a") -> BranchNode:
    parts = path.split("/")[1:]
    seed = path.split("/")[0]
    parent = None if len(parts) == 1 else seed + "/" + "/".join(parts[:-1])
    return BranchNode(
        node_id=path,
        seed_id=seed,
        parent_node_id=parent,
        depth=len(parts),
        path=parts,
        action=Action(parts[-1]),
        user_message="u",
        assistant_response=response,
    )


def _judgment(node_id: str, label: TrajectoryState) -> TrajectoryJudgment:
    uses = label is TrajectoryState.DEPEND
    reaffirms = label is TrajectoryState.REPEAT or uses
    retract = label is TrajectoryState.RETRACT
    return TrajectoryJudgment(
        node_id=node_id,
        seed_id=node_id.split("/")[0],
        explicit_retraction=retract,
        reaffirms_claim=reaffirms and not retract,
        uses_claim_as_premise=uses,
        label=label,
        parse_status=ParseStatus.OK,
    )


class AnalysisTests(unittest.TestCase):
    def test_terminal_state_not_strongest_ever(self):
        nodes = [_node("seed1/D"), _node("seed1/D/V")]
        judgments = [
            _judgment("seed1/D", TrajectoryState.DEPEND),
            _judgment("seed1/D/V", TrajectoryState.RETRACT),
        ]
        t2 = t2_transitions(nodes, judgments)
        self.assertEqual(t2["transitions"]["DEPEND"]["V"]["RETRACT"], 1)
        self.assertEqual(t2["depend_then_retract"], 1)
        self.assertIn("actual terminal", t2["note"])

    def test_parse_failures_excluded(self):
        nodes = [_node("seed1/D"), _node("seed1/N")]
        judgments = [
            _judgment("seed1/D", TrajectoryState.REPEAT),
            TrajectoryJudgment(
                node_id="seed1/N",
                seed_id="seed1",
                explicit_retraction=False,
                reaffirms_claim=False,
                uses_claim_as_premise=False,
                label=TrajectoryState.DROP,
                parse_status=ParseStatus.FAILED,
            ),
        ]
        t1 = t1_distribution(nodes, judgments)
        self.assertEqual(t1["by_action"]["D"]["n"], 1)
        self.assertEqual(t1["by_action"]["N"]["n"], 0)

    def test_mcnemar_is_paired(self):
        result = mcnemar_exact(5, 1)
        self.assertLess(result["p_value"], 0.25)
        self.assertEqual(result["n_discordant"], 6.0)

    def test_analyze_writes_tables(self):
        nodes = [_node("seed1/D"), _node("seed1/N"), _node("seed1/V"), _node("seed1/D/D")]
        judgments = [
            _judgment("seed1/D", TrajectoryState.REPEAT),
            _judgment("seed1/N", TrajectoryState.DROP),
            _judgment("seed1/V", TrajectoryState.RETRACT),
            _judgment("seed1/D/D", TrajectoryState.DEPEND),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = analyze(
                verified_seeds=[],
                nodes=nodes,
                judgments=judgments,
                audits=[],
                verifications=[],
                out_dir=out,
            )
            self.assertTrue((out / "t1_states.csv").exists())
            self.assertTrue((out / "transitions.csv").exists())
            self.assertEqual(summary["primary_endpoint"], "actual terminal T2 label")


if __name__ == "__main__":
    unittest.main()
