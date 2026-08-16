"""Shared FACTSCORE-style claim extraction and verification helpers."""

from __future__ import annotations

import json
import os

import requests
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running the forecasting pipeline.")
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def ask(prompt: str) -> dict:
    response = get_client().chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="minimal",
    )
    return json.loads(response.choices[0].message.content)


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
    return response.json().get("organic", [])[:5]


def judge_answer(answer: str) -> list[dict]:
    claims = ask(
        f"""Extract at most 3 atomic factual claims. Return {{"claims": ["..."]}}.
        ANSWER: {answer}"""
    )["claims"]

    evidence = [
        {"claim": claim, "search_results": search(claim)}
        for claim in claims
    ]

    return ask(
        f"""Using ONLY the evidence, label each claim as supported or unsupported.

        Return:
        {{"claims": [
          {{"claim": "...", "label": "...", "reason": "..."}}
        ]}}

        {json.dumps(evidence)}"""
    )["claims"]


def has_unsupported_claim(claims: list[dict]) -> bool:
    return any("unsupported" in claim["label"].lower() for claim in claims)


def classify_trajectory(judgments: dict) -> dict:
    return ask(
        f"""Classify this trajectory using the original answer, future responses, and claim judgements.

        corrected: A later response explicitly replaces an unsupported original claim with a supported correction.
        snowballing: A later unsupported claim logically depends on or expands on the initial claim.
        isolated: The unsupported original claim is NOT corrected, but the later claims do NOT depend on it.

        Do NOT label a trajectory corrected automatically only because a later response discusses the same topic.
        You MUST identify the specific original unsupported claim and the later claim connected to it.

        Return:
        {{"final_label": "corrected|isolated|snowballing",
          "reason": "..."}}

        {json.dumps(judgments)}"""
    )
