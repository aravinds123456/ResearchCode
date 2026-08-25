"""Azure GPT-OSS empty-content handling (no network, no Azure key)."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import (
    azure_answer_text,
    azure_chat_create,
    azure_create_kwargs,
    content_to_text,
    describe_completion,
    empty_retry_tokens,
    extract_reasoning_text,
    extract_visible_text,
    followup_max_new_tokens,
    is_unsupported_parameter,
)


def make_response(
    content="",
    finish_reason="length",
    reasoning_tokens=300,
    reasoning_content="",
    completion_tokens=None,
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(
        completion_tokens=completion_tokens if completion_tokens is not None else reasoning_tokens,
        completion_tokens_details=details,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(f"unexpected extra Azure call: {kwargs}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ExtractTests(unittest.TestCase):
    def test_visible_text_ignores_hidden_reasoning(self):
        message = SimpleNamespace(
            content="",
            reasoning_content="The capital is Paris because...",
        )
        self.assertEqual(extract_visible_text(message), "")
        self.assertIn("Paris", extract_reasoning_text(message))

    def test_list_content_parts(self):
        self.assertEqual(
            content_to_text([{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]),
            "Hello\nworld",
        )

    def test_describe_length_with_reasoning_tokens(self):
        response = make_response(content="", finish_reason="length", reasoning_tokens=300)
        text = describe_completion(response.choices[0], response.usage)
        self.assertIn("finish_reason=length", text)
        self.assertIn("reasoning_tokens=300", text)


class PayloadTests(unittest.TestCase):
    def test_prefers_max_tokens_and_low_effort(self):
        kwargs = azure_create_kwargs(
            "gpt-oss-20b",
            [{"role": "user", "content": "hi"}],
            32768,
            temperature=0.7,
            send_temperature=False,
            reasoning_effort="low",
        )
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_empty_retry_is_at_least_4096_and_capped(self):
        self.assertEqual(empty_retry_tokens(300), 4096)
        self.assertEqual(empty_retry_tokens(3000), 6000)
        self.assertEqual(empty_retry_tokens(32768), 32768)

    def test_followup_budget_uses_max_tokens_default(self):
        self.assertGreaterEqual(followup_max_new_tokens(), 32768)

    def test_temperature_rejection_is_detected(self):
        error = ValueError("Unsupported value: 'temperature' does not support 0.7")
        self.assertTrue(is_unsupported_parameter(error, "temperature"))
        self.assertFalse(is_unsupported_parameter(error, "reasoning_effort"))


class AzureAnswerTests(unittest.TestCase):
    def test_retries_empty_content_then_returns_visible_text(self):
        client = ScriptedClient(
            [
                make_response(content="", finish_reason="length", reasoning_tokens=300),
                make_response(content="The treaty was signed in 1815.", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "When was it signed?"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "The treaty was signed in 1815.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 300)
        self.assertEqual(client.calls[1]["max_tokens"], 4096)
        self.assertEqual(client.calls[0]["reasoning_effort"], "low")

    def test_does_not_use_reasoning_as_the_public_answer_when_fallback_off(self):
        client = ScriptedClient(
            [
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_tokens=300,
                    reasoning_content="hidden chain of thought",
                ),
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_tokens=4096,
                    reasoning_content="hidden chain of thought",
                ),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "")

    def test_last_resort_uses_reasoning_content_after_retry(self):
        client = ScriptedClient(
            [
                make_response(content="", finish_reason="length", reasoning_content="first"),
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_content="The compound is X47.",
                ),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=True,
        )
        self.assertEqual(text, "The compound is X47.")

    def test_drops_unsupported_temperature_and_retries(self):
        client = ScriptedClient(
            [
                ValueError("Unsupported parameter: 'temperature' is not supported for this model"),
                make_response(content="Visible answer.", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            2048,
            temperature=0.7,
            model_name="gpt-oss-20b",
            send_temperature=True,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "Visible answer.")
        self.assertIn("temperature", client.calls[0])
        self.assertNotIn("temperature", client.calls[1])

    def test_moves_reasoning_effort_to_extra_body_on_typeerror(self):
        client = ScriptedClient(
            [
                TypeError("create() got an unexpected keyword argument 'reasoning_effort'"),
                make_response(content="ok", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            2048,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "ok")
        self.assertNotIn("reasoning_effort", client.calls[1])
        self.assertEqual(client.calls[1].get("extra_body", {}).get("reasoning_effort"), "low")


class AzureCreateLoopTests(unittest.TestCase):
    def test_max_tokens_falls_back_to_max_completion_tokens(self):
        client = ScriptedClient(
            [
                ValueError("Unsupported parameter: max_tokens"),
                make_response(content="ok", finish_reason="stop"),
            ]
        )
        response = azure_chat_create(
            client,
            {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "q"}],
                "max_tokens": 32768,
            },
        )
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertNotIn("max_tokens", client.calls[1])
        self.assertEqual(client.calls[1]["max_completion_tokens"], 32768)


if __name__ == "__main__":
    unittest.main()
