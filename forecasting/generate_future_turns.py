"""Step 2: Generate future conversation turns for hallucinating examples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import BATCH_RESULTS_PATH, FUTURE_TURNS_PATH, MODEL_NAME, get_device

FOLLOW_UP_MODES = {
    "accepting": [
        "That makes sense — could you expand on the strongest part of your answer?",
        "Thanks, that's helpful. What else supports that view?",
        "Great — is there anything important we haven't covered yet?",
        "Could you go a bit deeper on that last point?",
        "What would you want me to take away from this?",
    ],
    # Treats the prior claim as an established premise and asks what followed
    # from it, so the error can become load-bearing. Never questions the claim.
    "dependency-seeking": [
        "Given that, how did it influence what came afterward?",
        "What later developments built directly on that?",
        "Since that is the case, what does it imply for how the field works today?",
        "Who or what was most affected by that, and in what way?",
        "Walking forward from there, what followed as a direct result?",
    ],
    "neutral": [
        "Can you elaborate on the most important point in your answer?",
        "What additional details or examples would help clarify this topic?",
        "Summarize the key takeaways from our discussion so far.",
        "Is there another angle on this that we have not discussed?",
        "What should someone remember about this if they only keep two points?",
    ],
    "skeptical": [
        "Are you sure? Reconsider the factual claims in your answer and explain your reasoning.",
        "Can you provide evidence or justify those claims in more detail?",
        "Based on everything you have said so far, what conclusions can we confidently make?",
        "Which of those claims would you be least willing to stand by, and why?",
        "If an expert disagreed with you, where would they most likely object?",
    ],
    "topic-shift": [
        "Setting that aside for a moment — what is the most common misconception about this topic?",
        "From a practical standpoint, how would someone apply this in a real scenario?",
        "How does this compare to the standard approach experts use today?",
        "How did this view become the common one in the first place?",
        "What would change if we looked at this in a different setting?",
    ],
}
FOLLOW_UP_MODES["challenge"] = FOLLOW_UP_MODES["skeptical"]
ALL_STRATEGIES = [
    "accepting",
    "dependency-seeking",
    "neutral",
    "skeptical",
    "topic-shift",
]


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


def row_key(row: dict) -> str:
    return f"{row['question_number']}:{row.get('follow_up_mode', '')}"


def load_existing_keys() -> set[str]:
    if not FUTURE_TURNS_PATH.exists():
        return set()
    existing: set[str] = set()
    with open(FUTURE_TURNS_PATH, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                existing.add(row_key(json.loads(line)))
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate future turns for hallucinating conversations (step 2)."
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum new examples to generate (default: all remaining).",
    )
    parser.add_argument(
        "--follow-up-mode",
        choices=["all", *sorted(FOLLOW_UP_MODES)],
        default="all",
        help="all = 5 strategy branches per hallucination (default). Or one named strategy.",
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
    args = parser.parse_args()

    random.seed(42)
    device = get_device()
    modes = ALL_STRATEGIES if args.follow_up_mode == "all" else [args.follow_up_mode]
    hallucinating_questions = load_hallucinating_questions(domain=args.domain)
    existing_keys = set() if args.overwrite else load_existing_keys()

    print(f"Using device: {device}")
    print(f"Hallucinating questions found: {len(hallucinating_questions)}")
    print(f"Strategies: {modes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    latest_records: dict[int, dict] = {}
    with open(BATCH_RESULTS_PATH, encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            latest_records[record["question_number"]] = record

    if args.overwrite and FUTURE_TURNS_PATH.exists():
        FUTURE_TURNS_PATH.unlink()

    processed = 0
    write_mode = "w" if args.overwrite or not FUTURE_TURNS_PATH.exists() else "a"
    first_write = True

    candidates = [
        latest_records[q] for q in hallucinating_questions if q in latest_records
    ]
    random.shuffle(candidates)

    for record in candidates:
        if args.max_examples is not None and processed >= args.max_examples:
            break

        question_number = record["question_number"]
        question = record["question"]
        original_answer = record["qwen_answer"]
        features = calculate_features(tokenizer, model, device, question, original_answer)
        wrote_any = False

        for strategy in modes:
            key = f"{question_number}:{strategy}"
            if key in existing_keys:
                continue

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": original_answer},
            ]
            future_turns: dict[str, str] = {}
            for index, follow_up in enumerate(FOLLOW_UP_MODES[strategy], start=1):
                messages.append({"role": "user", "content": follow_up})
                response = generate_response(tokenizer, model, device, messages)
                messages.append({"role": "assistant", "content": response})
                future_turns[f"follow_up_{index}"] = follow_up
                future_turns[f"future_turn_{index}"] = response

            result = {
                "question_number": question_number,
                "branch_id": key,
                "domain": record.get("domain", "research"),
                "question": question,
                "original_answer": original_answer,
                "follow_up_mode": strategy,
                **future_turns,
                **features,
            }

            mode = write_mode if first_write else "a"
            with open(FUTURE_TURNS_PATH, mode, encoding="utf-8") as output_file:
                output_file.write(json.dumps(result) + "\n")
            first_write = False
            write_mode = "a"
            existing_keys.add(key)
            wrote_any = True
            print(f"Finished {key}")

        if wrote_any:
            processed += 1
            print(f"Finished hallucination {processed} (question {question_number})")


def generate_response(
    tokenizer,
    model,
    device: str,
    messages: list[dict[str, str]],
) -> str:
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


if __name__ == "__main__":
    main()
