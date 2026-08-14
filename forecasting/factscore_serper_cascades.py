import json
import os
import requests
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
serper_key = os.environ["SERPER_API_KEY"]

def ask(prompt):
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="minimal",
    )
    return json.loads(response.choices[0].message.content)


def search(claim):
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": serper_key},
        json={"q": claim, "num": 5},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("organic", [])[:5]

def judge_answer(answer):
    claims = ask(
        f"""Extract at most 3 atomic factual claims. Return {{"claims": ["..."]}}.
        ANSWER: {answer}"""
    )["claims"]

    evidence = [
        {"claim": claim, "search_results": search(claim)}
        for claim in claims
    ]

    return ask(
        f"""Using ONLY the evidence, label each claim that is supported or unsupported.  
        
        Return:
        {{"claims": [
          {{"claim": "...", "label": "...", "reason": "..."}}
        ]}}

        {json.dumps(evidence)}"""

    )["claims"]

with open("forecasting/future_turns.jsonl") as file:
    conversations = [json.loads(line) for line in file][:10]

with open("forecasting/factscore_cascade_results.jsonl", "w") as output:
    for conversation in conversations:
        judgments = {
            name: judge_answer(conversation[name])
            for name in [
                "original_answer",
                "future_turn_1",
                "future_turn_2",
                "future_turn_3",
            ]
        }

        outcome = ask(
            f"""Classify this trajectory using the original answer, future responses, and claim judgements.

            corrected: A later response explicitly replaces an unsupported original claim with a supported correction.
            snowballing: A later unsupported claim logically depends on or expans on the initial claimt.
            isolated: The unsupported original claim is NOT corrected, but the later claims do NOT depend on it.

            Do NOT label a trajectory corrected automatically only because a later response discusses the same topic. 
            You MUST identify the specific original unsupported claim and the later claim conencted to it.

            Return:
            {{"final_label": "corrected|isolated|snowballing",
              "reason": "..."}}

            {json.dumps(judgments)}"""
        )

        result = {
            "question_number": conversation["question_number"],
            "judgments": judgments,
            **outcome,
        }

        output.write(json.dumps(result) + "\n")
        print(conversation["question_number"], outcome["final_label"])



