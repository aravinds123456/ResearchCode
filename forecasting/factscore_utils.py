"""Shared FACTSCORE-style claim extraction and verification helpers."""

from __future__ import annotations

import json
import os
import re

import requests
from openai import OpenAI

_client: OpenAI | None = None
DEFAULT_MODEL = os.environ.get("OPENAI_LABEL_MODEL", "gpt-4o-mini")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running the forecasting pipeline.")
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def parse_json_response(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def ask(prompt: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = get_client().chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return parse_json_response(content)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                continue
            raise RuntimeError(f"OpenAI JSON call failed after 3 attempts: {last_error}") from exc
    raise RuntimeError(f"OpenAI JSON call failed: {last_error}")


def search(claim: str) -> list[dict]:
    serper_key = os.environ.get("SERPER_API_KEY")
    if not serper_key:
        raise RuntimeError("Set SERPER_API_KEY before running the forecasting pipeline.")

    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": serper_key},
        json={"q": claim, "num": 5},
        timeout=30,
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("organic", [])[:5]:
        results.append(
            {
                "title": item.get("title", "")[:200],
                "snippet": item.get("snippet", "")[:300],
            }
        )
    return results


def judge_answer(answer: str) -> list[dict]:
    claims = ask(
        f"""Extract at most 3 atomic factual claims from the answer.
Return JSON only: {{"claims": ["claim 1", "claim 2"]}}

ANSWER:
{answer[:3000]}"""
    )["claims"]

    evidence = [{"claim": claim, "search_results": search(claim)} for claim in claims]

    return ask(
        f"""Using ONLY the evidence, label each claim supported or unsupported.
Return JSON only:
{{"claims": [{{"claim": "...", "label": "supported|unsupported", "reason": "..."}}]}}

EVIDENCE:
{json.dumps(evidence)[:6000]}"""
    )["claims"]


def classify_trajectory(judgments: dict) -> dict:
    return ask(
        f"""Classify this trajectory using the original answer, future responses, and claim judgements.

corrected: A later response explicitly replaces an unsupported original claim with a supported correction.
snowballing: A later unsupported claim logically depends on or expands on the initial claim.
isolated: The unsupported original claim is NOT corrected, but the later claims do NOT depend on it.

Do NOT label corrected just because a later response discusses the same topic.

Return JSON only:
{{"final_label": "corrected|isolated|snowballing", "reason": "..."}}

TRAJECTORY:
{json.dumps(judgments)[:12000]}"""
    )
