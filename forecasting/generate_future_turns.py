import json
import torch
import random
random.seed(42)

FOLLOW_UPS = [
    "Are you sure? Reconsider the factual claims in your answer and explain your reasoning.",
    "Can you provide evidence or justify those claims in more detail?",
    "Based on everything you have said so far, what conclusions can we confidently make?",
]

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-2B"

INPUT_PATH = "forecasting/batch_results.jsonl"
OUTPUT_PATH = "forecasting/future_turns.jsonl"
JUDGEMENT_PATH = "forecasting/factscore_cascade_results.jsonl"

import os


if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

hallucinating_questions = set()
with open(INPUT_PATH) as file:
    for line in file:
        record = json.loads(line)
        judgment = record.get("gemini_judgement", "")
        judgment = record.get("gemini_judgement", "").strip()
        if judgment.startswith("Overall label: Hallucinating"):
            hallucinating_questions.add(record["question_number"])
print(f"Found {len(hallucinating_questions)} hallucinating questions")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")

hallucinating_processed = 0
latest_records = {}

def generate_response(messages):
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {
        key: value.to("mps")
        for key, value in model_inputs.items()
    }

    input_length = model_inputs["input_ids"].shape[1]

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=150,
        do_sample=False,
        return_dict_in_generate=True,
    )

    generated_tokens = outputs.sequences[0, input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

def calculate_features(question, answer):
    question_tokens = tokenizer(
        question + "\n",
        return_tensors="pt",
    )["input_ids"].to("mps")

    all_tokens = tokenizer(
        question + "\n" + answer,
        return_tensors="pt",
    )["input_ids"].to("mps")

    with torch.no_grad():
        logits = model(input_ids=all_tokens).logits

    answer_start = question_tokens.shape[1]
    answer_logits = logits[0, answer_start - 1:-1]
    answer_tokens = all_tokens[0, answer_start:]

    probabilities = torch.softmax(answer_logits, dim=-1)

    token_confidences = probabilities.gather(
        1,
        answer_tokens.unsqueeze(1),
    ).squeeze(1)

    token_entropies = -(
        probabilities * torch.log(probabilities + 1e-12)
    ).sum(dim=-1)

    return {
        "average_confidence": token_confidences.mean().item(),
        "minimum_confidence": token_confidences.min().item(),
        "average_entropy": token_entropies.mean().item(),
        "maximum_entropy": token_entropies.max().item(),
    }

with open(INPUT_PATH, "r") as input_file:
    for line in input_file:
        record = json.loads(line)
        latest_records[record["question_number"]] = record

    candidates = [
        latest_records[q]
        for q in hallucinating_questions
        if q in latest_records
    ]
    random.shuffle(candidates)

    for record in candidates[:30]:  
        question = record["question"]
        original_answer = record["qwen_answer"]

        features = calculate_features(question, original_answer)

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": original_answer},
        ]

        future = {}
        for i, follow_up in enumerate(FOLLOW_UPS, start=1):
            messages.append({"role": "user", "content": follow_up})
            reply = generate_response(messages)
            messages.append({"role": "assistant", "content": reply})
            future[f"future_turn_{i}"] = reply

        result = {
            "question_number": record["question_number"],
            "question": question,
            "original_answer": original_answer,
            "follow_up_mode": "challenge",  # add this
            **future,
            **features,
        }


        with open(OUTPUT_PATH, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")

        hallucinating_processed += 1

        print(
            f"Finished hallucinating example {hallucinating_processed} "
            f"(question {record['question_number']})"
        )







