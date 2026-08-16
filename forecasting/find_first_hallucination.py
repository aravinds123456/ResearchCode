import json

with open("forecasting/sample_results.jsonl", "r") as file:
    data = json.load(file)
    print(data["claim_evaluations"])
    claims = data["claim_evaluations"]

    for claim in claims:
        if claim["hallucination"] == "Yes":
            print(claims)
            break

    

    




