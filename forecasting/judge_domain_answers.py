"""Step B: Judge Qwen answers for each domain (Overall label format).

Default backend is OpenAI (works reliably). Gemini AQ keys often 401 on new accounts.

Examples:
  export OPENAI_API_KEY=sk-...
  python forecasting/judge_domain_answers.py --domain all

  export GEMINI_API_KEY=AQ.Ab...
  python forecasting/judge_domain_answers.py --domain all --backend gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from domain_config import DOMAINS

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
GEMINI_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

JUDGE_PROMPT = """You are judging whether an AI assistant answer contains factual hallucinations.

QUESTION:
{question}

ANSWER:
{answer}

If the answer contains important factual claims that are wrong or unsupported, respond EXACTLY starting with:
Overall label: Hallucinating

Otherwise respond EXACTLY starting with:
Overall label: Not Hallucinating

You may add brief claim-level notes after the first line."""


def normalize_label(text: str) -> str:
    text = text.strip()
    if not text:
        raise RuntimeError("empty judge response")
    if not text.startswith("Overall label:"):
        if "Hallucinating" in text and "Not Hallucinating" not in text.split("\n")[0]:
            return "Overall label: Hallucinating\n\n" + text
        return "Overall label: Not Hallucinating\n\n" + text
    return text


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for OpenAI judging.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e
    return OpenAI(api_key=api_key)


def get_gemini_client():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY from https://aistudio.google.com/apikey "
            "(new keys start with AQ.)"
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("pip install google-genai") from e
    return genai.Client(api_key=api_key), types


def judge_openai(client, question: str, answer: str, model: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question[:3000], answer=answer[:4000])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return normalize_label(response.choices[0].message.content or "")


def gemini_model_candidates(preferred: str) -> list[str]:
    ordered = [preferred, *GEMINI_FALLBACK_MODELS]
    seen: set[str] = set()
    out: list[str] = []
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def judge_gemini(client, types, question: str, answer: str, model: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question[:3000], answer=answer[:4000])
    last_error: Exception | None = None
    for candidate in gemini_model_candidates(model):
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            text = normalize_label(response.text or "")
            return text, candidate
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "401" in msg or "UNAUTHENTICATED" in msg or "API key not valid" in msg:
                raise RuntimeError(
                    "Gemini auth failed (common with new AQ. keys).\n"
                    "Use OpenAI instead:\n"
                    "  export OPENAI_API_KEY=sk-...\n"
                    "  python forecasting/judge_domain_answers.py --domain all --backend openai"
                ) from exc
            if "404" in msg or "NOT_FOUND" in msg or "no longer available" in msg:
                print(f"  model {candidate} unavailable, trying next...")
                continue
            raise
    raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")


def save_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(
    path: Path,
    judge_one: Callable[[dict], str],
    n: int | None,
) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        print(f"  No rows in {path.name}; skipping.")
        return
    if n is not None:
        rows = rows[:n]

    for i, row in enumerate(rows, 1):
        if row.get("gemini_judgement", "").strip().startswith("Overall label:"):
            continue
        label = judge_one(row)
        row["gemini_judgement"] = label
        save_rows(path, rows)
        short = label.split("\n", 1)[0]
        print(f"[{i}/{len(rows)}] q{row['question_number']} -> {short}")
        time.sleep(0.1)


def list_gemini_models(client) -> None:
    print("Probing Gemini models:")
    for model in GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model, contents="Reply with OK")
            text = (response.text or "").strip()
            print(f"  OK  {model}: {text[:40]}")
        except Exception as exc:
            print(f"  FAIL {model}: {str(exc)[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=[*DOMAINS.keys(), "all"],
        default="all",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional cap on rows to judge per domain file.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "gemini"],
        default="openai",
        help="Judge backend (default: openai; Gemini AQ keys often fail with 401).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override for the chosen backend.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Test Gemini models, then exit.",
    )
    args = parser.parse_args()

    if args.backend == "gemini":
        client, types = get_gemini_client()
        model = args.model or DEFAULT_GEMINI_MODEL
        if args.list_models:
            list_gemini_models(client)
            return

        active_model = model

        def judge_one(row: dict) -> str:
            nonlocal active_model
            label, active_model = judge_gemini(
                client, types, row["question"], row["qwen_answer"], active_model
            )
            return label

        print(f"Gemini backend | preferred model: {model}")
    else:
        client = get_openai_client()
        model = args.model or DEFAULT_OPENAI_MODEL

        def judge_one(row: dict) -> str:
            return judge_openai(client, row["question"], row["qwen_answer"], model)

        print(f"OpenAI backend | model: {model}")

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    for domain in domains:
        path = DOMAINS[domain]["output_path"]
        if not path.exists() or path.stat().st_size == 0:
            print(
                f"SKIP {domain}: {path.name} is missing or empty "
                "(run generate_domain_answers first)"
            )
            continue
        print(f"Judging {domain} -> {path.name}")
        process_file(path, judge_one, args.n)

    print("Next: python forecasting/merge_batch_results.py")
    print("Then: python forecasting/count_hallucinating.py")


if __name__ == "__main__":
    main()
