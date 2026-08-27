from __future__ import annotations

import asyncio
import inspect
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branching_hallucinations.config import load_prompt
from branching_hallucinations.interventions import (
    FOLLOWUP_PROMPT,
    FollowupGenerationError,
    generate_intervention,
)
from branching_hallucinations.schemas import Action, Message
from libs.types import SamplerResponse


CLAIM = (
    "The two independent measurements of the B_s to mu mu branching fraction "
    "are both known to a few-percent precision."
)


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


class InterventionContractTests(unittest.TestCase):
    def test_writer_does_not_accept_previous_trajectory_state(self):
        params = inspect.signature(generate_intervention).parameters
        self.assertNotIn("previous_trajectory_state", params)
        self.assertNotIn("state", params)
        self.assertNotIn("allow_fallback", params)
        self.assertNotIn("allow_followup_fallback", params)

    def test_empty_writer_retries_then_errors(self):
        writer = ScriptedSampler(['{"follow_up": ""}', "not json"])
        convo = [Message("user", "q"), Message("assistant", "a")]

        async def _run():
            return await generate_intervention(Action.V, CLAIM, convo, writer=writer)

        with self.assertRaises(FollowupGenerationError):
            asyncio.run(_run())
        self.assertEqual(len(writer.calls), 2)
        self.assertIn("FORMAT REMINDER", writer.calls[1][-1]["content"])

    def test_format_retry_recovers_usable_follow_up(self):
        writer = ScriptedSampler(
            [
                "Can you check whether that branching fraction claim is accurate?",
                '{"follow_up": "Can you verify whether those two measurements are known to a few-percent precision?"}',
            ]
        )
        convo = [Message("user", "q"), Message("assistant", "a")]

        async def _run():
            return await generate_intervention(Action.V, CLAIM, convo, writer=writer)

        result = asyncio.run(_run())
        self.assertFalse(result.fallback_used)
        self.assertIn("verify whether those two measurements", result.text)
        self.assertEqual(result.writer_model_metadata["parse_status"], "retried")
        self.assertEqual(len(writer.calls), 2)

    def test_no_canned_followup_helper(self):
        import branching_hallucinations.interventions as mod

        self.assertFalse(hasattr(mod, "fallback_intervention"))

    def test_writer_text_is_used_without_regex_filter(self):
        writer = ScriptedSampler(
            ['{"follow_up": "Are you sure that those two measurements are known to a few-percent precision?"}']
        )
        convo = [Message("user", "q"), Message("assistant", "a")]

        async def _run():
            return await generate_intervention(Action.D, CLAIM, convo, writer=writer)

        result = asyncio.run(_run())
        self.assertFalse(result.fallback_used)
        self.assertIn("Are you sure", result.text)

    def test_followup_prompts_are_action_specific(self):
        d = load_prompt("followup_d")
        n = load_prompt("followup_n")
        v = load_prompt("followup_v")
        self.assertEqual(FOLLOWUP_PROMPT[Action.D], "followup_d")
        self.assertIn("Treat this proposition as true", d)
        self.assertNotIn("DESIRED USER MOVE", d)
        self.assertNotIn("- N:", d)
        self.assertIn("neither challenges nor builds", n)
        self.assertNotIn("Treat this proposition as true", n)
        self.assertIn("verify", v.lower())
        self.assertNotIn("Treat this proposition as true", v)


if __name__ == "__main__":
    unittest.main()
