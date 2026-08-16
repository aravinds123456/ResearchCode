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


def load_existing_question_numbers(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["question_number"])
    return done


def load_questions(domain: str, n: int | None) -> list[dict]:
    cfg = DOMAINS[domain]
    data_path = cfg["data_path"]
    if not data_path.exists():
        raise SystemExit(
            f"Domain data file not found for {domain}:\n  {data_path}\n"
            "Run from the halluhard repo root and make sure you cloned the full repo."
        )

    rows = []
    with open(data_path, encoding="utf-8") as f:
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the domain output file and skip question_numbers already written.",
    )
    args = parser.parse_args()

    cfg = DOMAINS[args.domain]
    questions = load_questions(args.domain, args.n)
    if not questions:
        raise SystemExit(f"No questions loaded for domain={args.domain}")

    out_path = cfg["output_path"]
    done = load_existing_question_numbers(out_path) if args.resume else set()
    pending = [q for q in questions if q["question_number"] not in done]
    if not pending:
        print(f"Nothing to do: {out_path.name} already has all {len(questions)} questions.")
        print("Next: python forecasting/judge_domain_answers.py --domain", args.domain)
        return

    device = get_device()
    print(f"Domain: {args.domain}")
    print(f"Data: {cfg['data_path']}")
    print(f"Output: {out_path}")
    print(f"Pending: {len(pending)}/{len(questions)} | Device: {device}")
    if out_path.exists() and out_path.stat().st_size == 0 and not args.resume:
        print(f"Removing empty placeholder file: {out_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    write_mode = "a" if args.resume and done else "w"
    with open(out_path, write_mode, encoding="utf-8") as out:
        for i, row in enumerate(pending, 1):
            answer = generate_answer(tokenizer, model, device, row["question"])
            record = {
                "question_number": row["question_number"],
                "domain": row["domain"],
                "question": row["question"],
                "qwen_answer": answer,
                "gemini_judgement": "",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(pending)}] wrote question {record['question_number']}")

    print(f"Saved -> {out_path} ({out_path.stat().st_size} bytes)")
    print("Next: python forecasting/judge_domain_answers.py --domain", args.domain)


if __name__ == "__main__":
    main()
