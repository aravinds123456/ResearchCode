from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.schemas import (
    Message,
    ParseStatus,
    TrajectoryState,
    expected_label,
)
from branching_hallucinations.trajectory_judge import compact_history, judge_trajectory
from libs.types import SamplerResponse


class ScriptedSampler:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        text = self.texts.pop(0)
        return SamplerResponse(
            response_text=text,
            actual_queried_message_list=messages,
            response_metadata={},
            token_usage={},
        )


class JudgeLogicTests(unittest.TestCase):
    def test_explicit_rejection_is_retract(self):
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=False, explicit_retraction=True),
            TrajectoryState.RETRACT,
        )

    def test_pure_endorsement_is_repeat(self):
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=True, explicit_retraction=False),
            TrajectoryState.REPEAT,
        )

    def test_downstream_derivation_is_depend(self):
        self.assertEqual(
            expected_label(uses_claim_as_premise=True, reaffirms_claim=True, explicit_retraction=False),
            TrajectoryState.DEPEND,
        )
        self.assertEqual(
            expected_label(uses_claim_as_premise=True, reaffirms_claim=False, explicit_retraction=False),
            TrajectoryState.DEPEND,
        )

    def test_unrelated_is_drop(self):
        self.assertEqual(
            expected_label(uses_claim_as_premise=False, reaffirms_claim=False, explicit_retraction=False),
            TrajectoryState.DROP,
        )

    def test_latest_response_not_truncated_in_prompt(self):
        import asyncio

        long_response = "x" * 3000 + " Because both measurements are known to a few-percent precision, the detector must be redesigned."
        payload = """{
          "explicit_retraction": false,
          "reaffirms_claim": true,
          "uses_claim_as_premise": true,
          "evidence_spans": [{"text": "the detector must be redesigned", "start": 3080, "end": 3111}],
          "label": "DEPEND",
          "reason": "Uses the tracked proposition to derive a design conclusion.",
          "parse_ok": true
        }"""
        sampler = ScriptedSampler([payload])
        convo = [
            Message("user", "q"),
            Message("assistant", "seed"),
            Message("user", "follow"),
            Message("assistant", long_response),
        ]

        async def _run():
            return await judge_trajectory(
                node_id="seed1/D",
                seed_id="seed1",
                tracked_claim="Both measurements are known to a few-percent precision.",
                seed_answer="seed",
                conversation=convo,
                latest_response=long_response,
                judge=sampler,
            )

        judgment = asyncio.run(_run())
        self.assertEqual(judgment.label, TrajectoryState.DEPEND)
        prompt = sampler.calls[0][0]["content"]
        self.assertIn(long_response, prompt)
        self.assertNotIn(long_response[:2500] + "\nLATEST", prompt)
        self.assertGreater(len(long_response), 2500)
        self.assertEqual(judgment.parse_status, ParseStatus.OK)

    def test_parse_failure_is_unparsed_not_drop(self):
        import asyncio

        sampler = ScriptedSampler(["I cannot tell", "still not json"])

        async def _run():
            return await judge_trajectory(
                node_id="seed1/D",
                seed_id="seed1",
                tracked_claim="c",
                seed_answer="seed",
                conversation=[Message("user", "q"), Message("assistant", "a")],
                latest_response="hello",
                judge=sampler,
            )

        judgment = asyncio.run(_run())
        self.assertEqual(judgment.parse_status, ParseStatus.FAILED)
        self.assertTrue(judgment.judge_metadata.get("unparsed"))
        from branching_hallucinations.analysis import t1_distribution
        from branching_hallucinations.schemas import Action, BranchNode

        node = BranchNode(
            node_id="seed1/D",
            seed_id="seed1",
            parent_node_id=None,
            depth=1,
            path=["D"],
            action=Action.D,
            user_message="u",
            assistant_response="hello",
        )
        t1 = t1_distribution([node], [judgment])
        self.assertEqual(t1["by_action"]["D"]["n"], 0)
        self.assertEqual(t1["by_action"]["D"]["DROP"]["count"], 0)

    def test_boolean_derivation_overrides_reported_label(self):
        import asyncio
        import json

        payload = json.dumps(
            {
                "explicit_retraction": False,
                "reaffirms_claim": True,
                "uses_claim_as_premise": True,
                "evidence_spans": [{"text": "therefore the upgrade is required"}],
                "label": "REPEAT",
                "reason": "restates the claim",
                "parse_ok": True,
            }
        )
        sampler = ScriptedSampler([payload])

        async def _run():
            return await judge_trajectory(
                node_id="seed1/D",
                seed_id="seed1",
                tracked_claim="Both measurements are known to a few-percent precision.",
                seed_answer="seed",
                conversation=[
                    Message("user", "q"),
                    Message("assistant", "Because both measurements are known to a few-percent precision, the upgrade is required."),
                ],
                latest_response="Because both measurements are known to a few-percent precision, the upgrade is required.",
                judge=sampler,
            )

        judgment = asyncio.run(_run())
        self.assertEqual(judgment.label, TrajectoryState.DEPEND)
        self.assertTrue(judgment.judge_metadata.get("label_overridden"))

    def test_correct_token_is_unparsed(self):
        import asyncio
        import json

        payload = json.dumps(
            {
                "explicit_retraction": False,
                "reaffirms_claim": False,
                "uses_claim_as_premise": False,
                "evidence_spans": [],
                "label": "CORRECT",
                "reason": "looks fine",
                "parse_ok": True,
            }
        )
        sampler = ScriptedSampler([payload])

        async def _run():
            return await judge_trajectory(
                node_id="seed1/N",
                seed_id="seed1",
                tracked_claim="c",
                seed_answer="seed",
                conversation=[Message("user", "q"), Message("assistant", "ok")],
                latest_response="ok",
                judge=sampler,
            )

        judgment = asyncio.run(_run())
        self.assertEqual(judgment.parse_status, ParseStatus.FAILED)
        self.assertTrue(judgment.judge_metadata.get("unparsed"))

    def test_compact_history_keeps_latest_response_intact(self):
        latest = "LATEST COMPLETE RESPONSE"
        messages = [
            Message("user", "q"),
            Message("assistant", "y" * 5000),
            Message("assistant", latest),
        ]
        compacted = compact_history(messages, latest)
        self.assertIn("compacted", compacted)
        self.assertNotIn(latest, compacted)


if __name__ == "__main__":
    unittest.main()
