"""Step A: Run Qwen on research / legal / medical questions (one domain at a time).

Example:
  python forecasting/generate_domain_answers.py --domain research
  python forecasting/generate_domain_answers.py --domain legal
  python forecasting/generate_domain_answers.py --domain medical
  python forecasting/generate_domain_answers.py --domain research --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from domain_config import DOMAINS

MODEL_NAME = "Qwen/Qwen3.5-2B"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_questions(domain: str, n: int | None) -> list[dict]:
    cfg = DOMAINS[domain]
    rows = []
    with open(cfg["data_path"], encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            raw = json.loads(line)
            question = raw[cfg["question_key"]].strip()
            rows.append(
                {
                    "question_number": cfg["id_offset"] + i,
                    "domain": domain,
                    "question": question,
                }
            )
    return rows


def generate_answer(tokenizer, model, device: str, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_len = model_inputs["input_ids"].shape[1]

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        return_dict_in_generate=True,
    )
    tokens = outputs.sequences[0, input_len:]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen answers for one HalluHard domain.")
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS),
        required=True,
        help="research, legal, or medical",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional cap on questions (default: all questions in the domain file).",
    )
    args = parser.parse_args()

    questions = load_questions(args.domain, args.n)
    if not questions:
        raise SystemExit(f"No questions loaded for domain={args.domain}")

    device = get_device()
    print(f"Domain: {args.domain} | Questions: {len(questions)} | Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    out_path = DOMAINS[args.domain]["output_path"]
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(questions, 1):
            answer = generate_answer(tokenizer, model, device, row["question"])
            record = {
                "question_number": row["question_number"],
                "domain": row["domain"],
                "question": row["question"],
                "qwen_answer": answer,
                "gemini_judgement": "",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(questions)}] wrote question {record['question_number']}")

    print(f"Saved -> {out_path}")
    print("Next: python forecasting/judge_domain_answers.py --domain", args.domain)


if __name__ == "__main__":
    main()
