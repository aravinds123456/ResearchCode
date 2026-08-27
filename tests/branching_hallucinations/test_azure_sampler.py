from __future__ import annotations

import unittest
from types import SimpleNamespace

from libs.sampler.azure_openai_sampler import (
    content_to_text,
    extract_reasoning_text,
    extract_visible_text,
    is_unsupported_parameter,
    uses_responses_api,
)


class AzureSamplerHelperTests(unittest.TestCase):
    def test_visible_text_ignores_hidden_reasoning(self):
        message = SimpleNamespace(content="", reasoning_content="The capital is Paris because...")
        self.assertEqual(extract_visible_text(message), "")
        self.assertIn("Paris", extract_reasoning_text(message))

    def test_list_content_parts(self):
        self.assertEqual(
            content_to_text([{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]),
            "Hello\nworld",
        )

    def test_gpt5_uses_responses_gptoss_uses_chat(self):
        self.assertTrue(uses_responses_api("gpt-5-mini"))
        self.assertFalse(uses_responses_api("gpt-oss-20b"))

    def test_websearch_rejected(self):
        from libs.sampler.azure_openai_sampler import AzureOpenAISampler

        with self.assertRaises((ValueError, RuntimeError)):
            AzureOpenAISampler(model="gpt-5-mini", websearch=True)

    def test_unsupported_parameter_detection(self):
        error = Exception("Unsupported parameter: 'temperature'")
        self.assertTrue(is_unsupported_parameter(error, "temperature"))
        self.assertFalse(is_unsupported_parameter(error, "foo"))


if __name__ == "__main__":
    unittest.main()
