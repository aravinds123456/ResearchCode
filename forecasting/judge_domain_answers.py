"""Step B: Gemini-judge Qwen answers for each domain (Overall label format).

Requires GOOGLE_API_KEY or GEMINI_API_KEY from https://aistudio.google.com/apikey

Example:
  python forecasting/judge_domain_answers.py --domain all
  python forecasting/judge_domain_answers.py --domain all --model gemini-3.1-flash-lite
  python forecasting/judge_domain_answers.py --list-models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from domain_config import DOMAINS

# New accounts get AQ.* auth keys and newer model IDs (not gemini-2.5-*).
DEFAULT_MODEL = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = [
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


def get_gemini_client():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key in {"your_key", "your_key_here"}:
        raise RuntimeError(
            "Set a valid GEMINI_API_KEY from https://aistudio.google.com/apikey\n"
            "New keys start with AQ. (auth keys). Example:\n"
            "  export GEMINI_API_KEY='AQ.Ab...'"
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("pip install google-genai") from e

    return genai.Client(api_key=api_key), types


def model_candidates(preferred: str) -> list[str]:
    ordered = [preferred, *FALLBACK_MODELS]
    seen: set[str] = set()
    out: list[str] = []
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def generate_with_fallback(client, types, prompt: str, preferred_model: str) -> tuple[str, str]:
    last_error: Exception | None = None
    for model in model_candidates(preferred_model):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("empty response text")
            return text, model
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                raise RuntimeError(
                    "Gemini rejected your API key.\n"
                    "- Create/copy a fresh key at https://aistudio.google.com/apikey\n"
                    "- New keys start with AQ. (that is normal)\n"
                    "- Run: export GEMINI_API_KEY='AQ.Ab...'\n"
                    "- Upgrade SDK: pip install -U google-genai"
                ) from exc
            if "404" in msg or "NOT_FOUND" in msg or "no longer available" in msg:
                print(f"  model {model} unavailable, trying next...")
                continue
            raise
    raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")


def judge(client, types, question: str, answer: str, model: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question[:3000], answer=answer[:4000])
    text, used_model = generate_with_fallback(client, types, prompt, model)
    if not text.startswith("Overall label:"):
        if "Hallucinating" in text and "Not Hallucinating" not in text.split("\n")[0]:
            return "Overall label: Hallucinating\n\n" + text, used_model
        return "Overall label: Not Hallucinating\n\n" + text, used_model
    return text, used_model


def save_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(client, types, path: Path, model: str, n: int | None) -> str:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        print(f"  No rows in {path.name}; skipping.")
        return model
    if n is not None:
        rows = rows[:n]

    used_model = model
    for i, row in enumerate(rows, 1):
        if row.get("gemini_judgement", "").strip().startswith("Overall label:"):
            continue
        label, used_model = judge(client, types, row["question"], row["qwen_answer"], used_model)
        row["gemini_judgement"] = label
        save_rows(path, rows)
        short = label.split("\n", 1)[0]
        print(f"[{i}/{len(rows)}] q{row['question_number']} -> {short}")
        time.sleep(0.2)
    return used_model


def list_models(client) -> None:
    print("Probing models (first working model wins):")
    for model in FALLBACK_MODELS:
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
        "--model",
        default=DEFAULT_MODEL,
        help=f"Preferred Gemini model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Test which Gemini models work with your API key, then exit.",
    )
    args = parser.parse_args()

    client, types = get_gemini_client()

    if args.list_models:
        list_models(client)
        return

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    print(f"Preferred Gemini model: {args.model}")

    active_model = args.model
    for domain in domains:
        path = DOMAINS[domain]["output_path"]
        if not path.exists() or path.stat().st_size == 0:
            print(
                f"SKIP {domain}: {path.name} is missing or empty "
                "(run generate_domain_answers first)"
            )
            continue
        print(f"Judging {domain} -> {path.name}")
        active_model = process_file(client, types, path, active_model, args.n)

    print(f"Finished with model: {active_model}")
    print("Next: python forecasting/merge_batch_results.py")


if __name__ == "__main__":
    main()
