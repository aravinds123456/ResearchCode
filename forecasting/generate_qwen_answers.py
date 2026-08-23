import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-2B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")


INPUT = "research_questions/data/research_questions_all.jsonl"
OUTPUT = "forecasting/qwen_answers.jsonl"



with open(INPUT, "r") as input_file:
    for i, line in enumerate(input_file):
        if i >= 2:
            break

        data = json.loads(line)
        prompt = data["research_question"]
        source = data["abstract"]

        messages = [
            {"role": "system", "content": "Answer the specific question directly in 150-200 words. Add an inline citation after EVERY factual claim."},
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to("mps")
            for key, value in inputs.items()
        }

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.05,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )

        input_length = inputs["input_ids"].shape[1]
        answer_tokens = outputs.sequences[0, input_length:]

        token_confidences = []
        token_entropies = []

        for token_id, scores in zip(answer_tokens, outputs.scores):
            score_probabilities = torch.softmax(scores[0], dim=-1)

            token_confidence = score_probabilities[token_id].item()
            token_confidences.append(token_confidence)

            token_entropy = -(
                score_probabilities
                * torch.log(score_probabilities + 1e-12)
            ).sum().item()

            token_entropies.append(token_entropy)

        average_confidence = sum(token_confidences) / len(token_confidences)
        minimum_confidence = min(token_confidences)
        average_entropy = sum(token_entropies) / len(token_entropies)
        maximum_entropy = max(token_entropies)

        

        answer = tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True,
        )

        result = {
            "question_number": i,
            "question": prompt,
            "source_abstract": source,
            "qwen_answer": answer,
            "average_confidence": average_confidence,
            "minimum_confidence": minimum_confidence,
            "average_entropy": average_entropy,
            "maximum_entropy": maximum_entropy,
            "token_ids": answer_tokens.tolist(),
            "token_confidences": token_confidences,
            "token_entropies": token_entropies
        }

        with open(OUTPUT, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")

        print(f"Finished {i + 1}")















