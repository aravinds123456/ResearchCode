import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-2B"

INPUT_PATH = "forecasting/batch_results.jsonl"
OUTPUT_PATH = "forecasting/future_turns.jsonl"
JUDGEMENT_PATH = "forecasting/factscore_cascade_results.jsonl"

hallucinating_questions = set()

with open(JUDGEMENT_PATH) as file:
    for line in file:
        result = json.loads(line)
        original_claims = result["judgments"]["original_answer"]

        if any("unsupported" in claim["label"].lower() for claim in original_claims): 
            hallucinating_questions.add(result["question_number"])

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
        question_number = record["question_number"]
        latest_records[question_number] = record

    for record in latest_records.values():


        if record["question_number"] not in hallucinating_questions:
            continue

        if hallucinating_processed >= 3:
            break

        
        
        question = record["question"]
        original_answer = record["qwen_answer"]

        features = calculate_features(question, original_answer)

        messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": original_answer},
        {
            "role": "user",
            "content": "Are you sure? Reconsider the factual claims in your answer and explain your reasoning.",
        },
    ]

        future_turn_1 = generate_response(messages)

        messages.append({
            "role": "assistant",
            "content": future_turn_1,
        })
        messages.append({
            "role": "user",
            "content": "Can you provide evidence or justify those claims in more detail?",
        })

        future_turn_2 = generate_response(messages)

        messages.append({
            "role": "assistant",
            "content": future_turn_2,
        })
        messages.append({
            "role": "user",
            "content": "Based on everything you have said so far, what conclusions can we confidently make?",
        })

        future_turn_3 = generate_response(messages)

        result = {
            "question_number": record["question_number"],
            "question": question,
            "original_answer": original_answer,
            "future_turn_1": future_turn_1,
            "future_turn_2": future_turn_2,
            "future_turn_3": future_turn_3,
            "average_confidence": features["average_confidence"],
            "minimum_confidence": features["minimum_confidence"],
            "average_entropy": features["average_entropy"],
            "maximum_entropy": features["maximum_entropy"],
        }

        with open(OUTPUT_PATH, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")

        hallucinating_processed += 1

        print(
            f"Finished hallucinating example {hallucinating_processed} "
            f"(question {record['question_number']})"
        )






