from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.schemas import (
    VerificationStatus,
    VerifiedSeed,
)
from branching_hallucinations.storage import RunStore, conversation_for, conversation_for_seed
from branching_hallucinations.tree import expected_node_count, generate_tree
from libs.types import SamplerResponse


class ScriptedSampler:
    def __init__(self, texts=None, echo_prefix="answer"):
        self.texts = list(texts or [])
        self.calls = []
        self.echo_prefix = echo_prefix

    async def __call__(self, messages):
        self.calls.append(messages)
        if self.texts:
            text = self.texts.pop(0)
        else:
            last = messages[-1]["content"] if messages else ""
            if last.startswith("Write the next user question") or "FORMAT REMINDER" in last:
                text = json.dumps({"follow_up": f"What follows from that ({self.echo_prefix})?"})
            else:
                text = f"{self.echo_prefix}: {last[:80]}"
        return SamplerResponse(
            response_text=text,
            actual_queried_message_list=messages,
            response_metadata={"backend": "mock"},
            token_usage={},
        )


def _seed() -> VerifiedSeed:
    return VerifiedSeed(
        seed_id="seed149",
        question_id=149,
        domain="research",
        question="What is the precision of the two B_s measurements?",
        seed_answer="Both measurements are known to a few-percent precision.",
        tracked_claim="Both B_s to mu mu measurements are known to a few-percent precision.",
        tracked_claim_id="seed149/c0",
        verification_status=VerificationStatus.VERIFIED_FALSE,
        verification_reason="retrieved PDG value disagrees",
    )


class TreeTests(unittest.TestCase):
    def test_expected_counts(self):
        self.assertEqual(expected_node_count(3, 2), 12)

    def test_mock_tree_has_3_t1_and_9_t2_and_no_judge(self):
        seed = _seed()
        writer = ScriptedSampler()
        answer = ScriptedSampler(echo_prefix="gpt-oss")
        judge = ScriptedSampler()

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                store = RunStore(tmp)
                store.ensure()
                created = await generate_tree(
                    store,
                    [seed],
                    answer_model=answer,
                    writer=writer,
                    actions=("D", "N", "V"),
                    depth=2,
                )
                nodes = store.nodes()
                t1 = [n for n in nodes if n.depth == 1]
                t2 = [n for n in nodes if n.depth == 2]
                return created, nodes, t1, t2, store

        import asyncio
        created, nodes, t1, t2, store = asyncio.run(_run())
        self.assertEqual(len(t1), 3)
        self.assertEqual(len(t2), 9)
        self.assertEqual(len(nodes), 12)
        self.assertEqual(len(created), 12)
        self.assertEqual(judge.calls, [])
        for node in nodes:
            self.assertNotIn("label", node.to_dict())

        self.assertGreaterEqual(len(writer.calls), 12)
        t1_answer = next(n.assistant_response for n in t1 if n.action.value == "D")
        writer_prompts = [call[0]["content"] for call in writer.calls]
        self.assertTrue(any(t1_answer in prompt for prompt in writer_prompts))
        self.assertFalse(any("REPEAT" in prompt and "CURRENT TRAJECTORY" in prompt for prompt in writer_prompts))
        self.assertFalse(any("previous_trajectory_state" in prompt for prompt in writer_prompts))

        by_id = {node.node_id: node for node in nodes}
        t2_node = next(n for n in t2 if n.path == ["D", "V"])
        convo = conversation_for(t2_node.node_id, seed, by_id)
        self.assertEqual(convo[0].content, seed.question)
        self.assertEqual(convo[1].content, seed.seed_answer)
        self.assertEqual(convo[-1].content, t2_node.assistant_response)

    def test_resume_skips_completed_nodes(self):
        seed = _seed()
        import asyncio
        import tempfile

        async def _run():
            with tempfile.TemporaryDirectory() as tmp:
                store = RunStore(tmp)
                store.ensure()
                writer = ScriptedSampler()
                answer = ScriptedSampler(echo_prefix="gpt-oss")
                await generate_tree(store, [seed], answer_model=answer, writer=writer, depth=2)
                first = len(answer.calls)
                await generate_tree(store, [seed], answer_model=answer, writer=writer, depth=2)
                return first, len(answer.calls), len(store.nodes())

        first, second, n_nodes = asyncio.run(_run())
        self.assertEqual(n_nodes, 12)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
