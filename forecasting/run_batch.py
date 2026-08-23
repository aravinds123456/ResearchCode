import json
import torch

from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")
client = genai.Client()

INPUT = "research_questions/data/research_questions_all.jsonl"
OUTPUT = "forecasting/batch_results.jsonl"

hallucinating_count = 0
processed_count = 0

with open(INPUT, "r") as input_file:
    for i, line in enumerate(input_file):
        if i >= 2:
            break

        data = json.loads(line)
        prompt = data["research_question"]
        source = data["abstract"]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to("mps")
            for key, value in inputs.items()
        }

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            return_dict_in_generate=True,
            output_scores=True,
        )

        token_confidences = []
        token_entropies = []

        for scores in outputs.scores:
            score_probabilities = torch.softmax(scores[0], dim=-1)

            token_confidence = torch.max(score_probabilities).item()
            token_confidences.append(token_confidence)

            token_entropy = -(
                score_probabilities
                * torch.log(score_probabilities + 1e-12)
            ).sum().item()

            token_entropies.append(token_entropy)

        average_confidence = (
            sum(token_confidences) / len(token_confidences)
        )
        minimum_confidence = min(token_confidences)
        average_entropy = (
            sum(token_entropies) / len(token_entropies)
        )
        maximum_entropy = max(token_entropies)

        answer = tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"""
Model's answer:
{answer}

Source Abstract:
{source}

Step 1: Extract every factual claim from the model answer.

Step 2: For each claim, determine whether it is
Hallucinating or Not Hallucinating.

Rules:
- Do NOT label a claim Hallucinating only because it is absent from the abstract.
- If the abstract does not resolve the claim, use Google Search.
- Judge only from the abstract or retrieved web evidence.
- Do NOT rely on unsupported internal memory.
- Do NOT treat repetition or wording differences as hallucinations.
- Do NOT rewrite or correct the claims.

Begin with exactly one of:

Overall label: Hallucinating
Overall label: Not Hallucinating

Only display claims labeled Hallucinating.

If no claims are hallucinating, return only:
Overall label: Not Hallucinating
""",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ]
            ),
        )

        response_text = response.text or ""
        processed_count += 1

        first_line = (
            response_text.strip().splitlines()[0]
            if response_text.strip()
            else ""
        )

        if first_line == "Overall label: Hallucinating":
            hallucinating_count += 1
            print(response_text)

        result = {
            "question_number": i,
            "question": prompt,
            "qwen_answer": answer,
            "gemini_judgement": response_text,
            "average_confidence": average_confidence,
            "minimum_confidence": minimum_confidence,
            "average_entropy": average_entropy,
            "maximum_entropy": maximum_entropy,
        }

        with open(OUTPUT, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")

        print(f"Finished {i + 1}")

print(f"Hallucinating: {hallucinating_count}/{processed_count}")

if processed_count > 0:
    print(
        f"Hallucination rate: "
        f"{hallucinating_count / processed_count:.1%}"
    )

