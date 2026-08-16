"""Step B: Gemini-judge Qwen answers for each domain (Overall label format).

Requires GOOGLE_API_KEY or GEMINI_API_KEY.

Example:
  python forecasting/judge_domain_answers.py --domain research
  python forecasting/judge_domain_answers.py --domain all
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
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("pip install google-genai") from e

    return genai.Client(api_key=api_key)


def judge(client, question: str, answer: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question[:3000], answer=answer[:4000])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = response.text.strip()
    if not text.startswith("Overall label:"):
        if "Hallucinating" in text and "Not Hallucinating" not in text.split("\n")[0]:
            return "Overall label: Hallucinating\n\n" + text
        return "Overall label: Not Hallucinating\n\n" + text
    return text


def process_file(client, path: Path) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    with open(path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            if row.get("gemini_judgement", "").strip().startswith("Overall label:"):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            label = judge(client, row["question"], row["qwen_answer"])
            row["gemini_judgement"] = label
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            short = label.split("\n", 1)[0]
            print(f"[{i}/{len(rows)}] q{row['question_number']} -> {short}")
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=[*DOMAINS.keys(), "all"],
        default="all",
    )
    args = parser.parse_args()

    client = get_gemini_client()
    domains = list(DOMAINS) if args.domain == "all" else [args.domain]

    for domain in domains:
        path = DOMAINS[domain]["output_path"]
        if not path.exists():
            print(f"SKIP {domain}: missing {path.name} (run generate_domain_answers first)")
            continue
        print(f"Judging {domain} -> {path.name}")
        process_file(client, path)

    print("Next: python forecasting/merge_batch_results.py")


if __name__ == "__main__":
    main()
