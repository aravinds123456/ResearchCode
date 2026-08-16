##Note: CHECK PROMPT IN MEETING

import json

from google import genai
client = genai.Client()

with open("forecasting/qwen_answer.txt", "r") as file:
    answer = file.read()

with open("research_questions/data/research_questions_all.jsonl", "r") as file:
    data_first_line = file.readline()


data_final = json.loads(data_first_line)    
source = data_final["abstract"]

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",

    contents=f"""
    Model's answer: {answer}
    
    Source Abstract: {source}
    
    Step 1: Your first task is to extract every factual claim from the model's answer.

    Step 2: For each claim, determine whether it is Hallucinating or Not Hallucinating.

    Here are your clear rules to follow:
    -Compare the claim against the reference abstract.
    -If the abstract does NOT mention the claim, you may use only well-established, verified sites.
    -Do NOT invent your own evidence or assumptions.
    -Do NOT treat repetition, writing, or wording differences as hallucinations.
    -Do NOT rewrite or correct the claims.

    For each claim, output:

    Claim: 
    Label: Hallucinating/Not Hallucinating
    Reason: 

    """
)


print(response.text)




