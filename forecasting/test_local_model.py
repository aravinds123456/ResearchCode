import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-2B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("mps")


with open("research_questions/data/research_questions_all.jsonl", "r") as file:
    data_first_line = file.readline()
    data_final = json.loads(data_first_line)        
    prompt = data_final["research_question"]
    print(prompt)



inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to("mps") for key, value in inputs.items()}
outputs = model.generate(
    **inputs,
    max_new_tokens=352,
    return_dict_in_generate=True,
    output_scores=True,
)

with torch.no_grad():
    model_outputs = model(**inputs, output_hidden_states=True)

token_logits = model_outputs.logits[0, -1]
probabilities = torch.softmax(token_logits, dim=-1) #dim just. applies the conversiont o all vocab tokens thats all
confidence_score = torch.max(probabilities)
print(confidence_score.item())
print(len(outputs.scores))

response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
with open("forecasting/qwen_answer.txt", "w") as file:
    file.write(response)
print(model_outputs.hidden_states[-1].shape)
print(response)

#print(type(response))
#print(repr(response))










