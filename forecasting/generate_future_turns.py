"""Step 2: Generate future conversation turns.

Default (mentor design): for each hallucinating turn-0 answer, grow one branch
per follow-up strategy (accepting, dependency-seeking, neutral, skeptical,
topic-shift). Strategy is held constant. After each assistant turn, classify
state and have an LLM draft the next user question in that strategy's voice.

Unbiased by construction: every hallucination gets the same five strategies;
the drafter is told not to induce compounding errors; no branches are pruned.

Examples:
  export OPENAI_API_KEY=sk-...
  python forecasting/generate_future_turns.py
  python forecasting/generate_future_turns.py --turns 5 --max-examples 2
  python forecasting/generate_future_turns.py --strategies skeptical,neutral --template-only
  python forecasting/generate_future_turns.py --legacy --follow-up-mode neutral --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BATCH_RESULTS_PATH,
    DEFAULT_TURNS,
    FUTURE_TURNS_PATH,
    MODEL_NAME,
    get_device,
)
from follow_up_branch import run_adaptive_branch
from follow_up_prompts import (
    DEFAULT_SEED_STATE,
    STRATEGY_TEMPLATES,
    branch_id,
    parse_strategies,
    trajectory_key,
)


def load_hallucinating_questions(domain: str | None = None) -> set[int]:
    hallucinating_questions: set[int] = set()
    with open(BATCH_RESULTS_PATH, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if domain and record.get("domain") != domain:
                continue
            judgment = record.get("gemini_judgement", "").strip()
            if judgment.startswith("Overall label: Hallucinating"):
                hallucinating_questions.add(record["question_number"])
    return hallucinating_questions


def load_existing_keys() -> set[str]:
    if not FUTURE_TURNS_PATH.exists():
        return set()
    existing: set[str] = set()
    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                existing.add(trajectory_key(json.loads(line)))
    return existing


def load_latest_records(domain: str | None = None) -> dict[int, dict]:
    latest_records: dict[int, dict] = {}
    with open(BATCH_RESULTS_PATH, encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            if domain and record.get("domain") != domain:
                continue
            latest_records[record["question_number"]] = record
    return latest_records


def load_qwen(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model


def generate_response(tokenizer, model, device: str, messages: list[dict[str, str]]) -> str:
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    input_length = model_inputs["input_ids"].shape[1]
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False,
        return_dict_in_generate=True,
    )

    generated_tokens = outputs.sequences[0, input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def calculate_features(tokenizer, model, device: str, question: str, answer: str) -> dict:
    import torch

    question_tokens = tokenizer(question + "\n", return_tensors="pt")["input_ids"].to(device)
    all_tokens = tokenizer(question + "\n" + answer, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids=all_tokens).logits

    answer_start = question_tokens.shape[1]
    answer_logits = logits[0, answer_start - 1 : -1]
    answer_tokens = all_tokens[0, answer_start:]

    probabilities = torch.softmax(answer_logits, dim=-1)
    token_confidences = probabilities.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
    token_entropies = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1)

    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
    }


def write_jsonl_row(path: Path, row: dict, *, overwrite_first: bool) -> None:
    mode = "w" if overwrite_first else "a"
    with open(path, mode, encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_legacy(args, tokenizer, model, device: str, candidates: list[dict]) -> int:
    follow_ups = STRATEGY_TEMPLATES[args.follow_up_mode]
    if args.turns:
        follow_ups = follow_ups[: args.turns]
    processed = 0
    first_write = args.overwrite or not FUTURE_TURNS_PATH.exists()
    existing = set() if args.overwrite else load_existing_keys()

    for record in candidates:
        if args.max_examples is not None and processed >= args.max_examples:
            break
        qnum = record["question_number"]
        if str(qnum) in existing:
            continue

        question = record["question"]
        original_answer = record["qwen_answer"]
        features = calculate_features(tokenizer, model, device, question, original_answer)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": original_answer},
        ]
        future_turns: dict[str, str] = {}
        for index, follow_up in enumerate(follow_ups, start=1):
            messages.append({"role": "user", "content": follow_up})
            response = generate_response(tokenizer, model, device, messages)
            messages.append({"role": "assistant", "content": response})
            future_turns[f"follow_up_{index}"] = follow_up
            future_turns[f"future_turn_{index}"] = response

        result = {
            "question_number": qnum,
            "branch_id": str(qnum),
            "domain": record.get("domain", "research"),
            "question": question,
            "original_answer": original_answer,
            "follow_up_mode": args.follow_up_mode,
            "strategy": args.follow_up_mode,
            "n_turns": len(follow_ups),
            "model_name": args.model,
            **future_turns,
            **features,
        }
        write_jsonl_row(FUTURE_TURNS_PATH, result, overwrite_first=first_write)
        first_write = False
        processed += 1
        print(f"Finished legacy example {processed} (question {qnum})")
    return processed


def run_branched(args, tokenizer, model, device: str, candidates: list[dict]) -> int:
    strategies = parse_strategies(args.strategies)
    existing = set() if args.overwrite else load_existing_keys()
    first_write = args.overwrite or not FUTURE_TURNS_PATH.exists()
    processed_hall = 0
    processed_branches = 0

    classify_fn = None
    draft_fn = None
    if not args.template_only:
        from follow_up_llm import classify_turn, draft_follow_up

        classify_fn = classify_turn
        draft_fn = draft_follow_up

    def generate_fn(messages: list[dict[str, str]]) -> str:
        return generate_response(tokenizer, model, device, messages)

    for record in candidates:
        if args.max_examples is not None and processed_hall >= args.max_examples:
            break

        qnum = record["question_number"]
        question = record["question"]
        original_answer = record["qwen_answer"]
        pending_strategies = [
            strategy
            for strategy in strategies
            if branch_id(qnum, strategy) not in existing
        ]
        if not pending_strategies:
            continue

        features = calculate_features(tokenizer, model, device, question, original_answer)
        wrote_any = False
        for strategy in pending_strategies:
            print(f"  q{qnum} / {strategy}")
            branch = run_adaptive_branch(
                question=question,
                original_answer=original_answer,
                strategy=strategy,
                n_turns=args.turns,
                generate_fn=generate_fn,
                draft_fn=draft_fn,
                classify_fn=classify_fn,
                seed_state=args.seed_state,
            )
            result = {
                "question_number": qnum,
                "branch_id": branch_id(qnum, strategy),
                "domain": record.get("domain", "research"),
                "question": question,
                "original_answer": original_answer,
                "follow_up_mode": strategy,
                "model_name": args.model,
                "draft_mode": "template" if args.template_only else "llm",
                **features,
                **branch,
            }
            write_jsonl_row(FUTURE_TURNS_PATH, result, overwrite_first=first_write)
            first_write = False
            existing.add(result["branch_id"])
            processed_branches += 1
            wrote_any = True
            print(
                f"  finished {result['branch_id']} states={branch['turn_states']}"
            )

        if wrote_any:
            processed_hall += 1
            print(f"Finished hallucination {processed_hall} (question {qnum})")

    print(f"Wrote {processed_branches} branches from {processed_hall} hallucinations")
    return processed_branches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate future turns (branched strategies by default)."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max hallucinations to process (each still gets every selected strategy).",
    )
    parser.add_argument(
        "--follow-up-mode",
        choices=sorted(STRATEGY_TEMPLATES),
        default="challenge",
        help="Legacy linear mode only: challenge or neutral (or any named strategy).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Old linear 3-turn script (no branching, no LLM drafter).",
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help="Comma-separated strategies or 'all' (default: all five).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        help="Follow-up turns per branch (default: 5).",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Use fixed templates instead of LLM draft/classify (no OpenAI key).",
    )
    parser.add_argument(
        "--seed-state",
        choices=["corrected", "persisted", "new_hallucination", "not_applicable"],
        default=DEFAULT_SEED_STATE,
        help="Turn-0 state used to draft follow-up 1 (default: persisted).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace future_turns.jsonl instead of appending.",
    )
    parser.add_argument(
        "--domain",
        choices=["research", "legal", "medical"],
        default=None,
        help="Optional: only generate for one domain.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", MODEL_NAME),
        help="Hugging Face model id for assistant replies (default: Qwen/Qwen3.5-2B).",
    )
    args = parser.parse_args()

    if args.turns < 1:
        raise SystemExit("--turns must be >= 1")

    random.seed(42)
    device = get_device()
    hallucinating_questions = load_hallucinating_questions(domain=args.domain)
    latest_records = load_latest_records(domain=args.domain)

    print(f"Using device: {device}")
    print(f"Assistant model: {args.model}")
    print(f"Hallucinating questions found: {len(hallucinating_questions)}")
    if not args.legacy:
        print(f"Strategies: {parse_strategies(args.strategies)}")
        print(f"Turns: {args.turns} | draft: {'template' if args.template_only else 'llm'}")

    tokenizer, model = load_qwen(args.model, device)

    candidates = [
        latest_records[q]
        for q in hallucinating_questions
        if q in latest_records
    ]
    random.shuffle(candidates)

    if args.overwrite and FUTURE_TURNS_PATH.exists():
        FUTURE_TURNS_PATH.unlink()

    if args.legacy:
        run_legacy(args, tokenizer, model, device, candidates)
    else:
        if not args.template_only and not os.environ.get("OPENAI_API_KEY", "").strip():
            raise SystemExit(
                "Set OPENAI_API_KEY for LLM-drafted follow-ups, "
                "or pass --template-only to use fixed scripts."
            )
        run_branched(args, tokenizer, model, device, candidates)

    print(f"Saved -> {FUTURE_TURNS_PATH}")
    print("Next: python forecasting/fast_cascade_label.py   # or factscore_serper_cascades.py")
    print("Then: python forecasting/summarize_strategy_outcomes.py")


if __name__ == "__main__":
    main()
