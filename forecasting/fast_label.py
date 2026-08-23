import json
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def parse_json(text):
    text = text.strip()
    if not text:
        raise ValueError("Empty response from API")

    # Remove ```json ... ``` blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    return json.loads(text)

with open("forecasting/future_turns.jsonl") as f:
    rows = [json.loads(line) for line in f]

with open("forecasting/factscore_cascade_results.jsonl", "w") as out:
    for row in rows:
        prompt = f"""Classify this multi-turn conversation.

original: {row['original_answer'][:1500]}
turn1: {row['future_turn_1'][:1000]}
turn2: {row['future_turn_2'][:1000]}
turn3: {row['future_turn_3'][:1000]}

Labels:
- corrected: later turn fixes or retracts an earlier wrong claim
- snowballing: later turn builds on or repeats an earlier wrong claim
- isolated: wrong claim stays but later turns don't depend on it

Return ONLY valid JSON:
{{"final_label": "corrected|snowballing|isolated", "reason": "one sentence"}}"""

        try:
            r = client.responses.create(
                model=os.environ.get("OPENAI_LABEL_MODEL", "gpt-5-mini"),
                input=prompt,
                reasoning={"effort": "minimal"},
                text={"format": {"type": "json_object"}},
            )
            result = parse_json(r.output_text)
            result["question_number"] = row["question_number"]
            out.write(json.dumps(result) + "\n")
            print(row["question_number"], result["final_label"])

        except Exception as e:
            print(f"FAILED question {row['question_number']}: {e}")
            if "r" in locals():
                print("Raw response:", repr(r.choices[0].message.content))




                