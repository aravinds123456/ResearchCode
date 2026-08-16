import json

with open("forecasting/batch_results.jsonl") as source:
    with open("forecasting/halluhard_input.jsonl", "w") as output:
        for i, line in enumerate(source):
            if i >= 10:
                break
            data = json.loads(line)
            record = {
                "conversation_id": i,
                "turns":[
                    {"role": "user", "content": data["question"], "turn_index": 0},
                    {"role": "assistant", "content": data["qwen_answer"], "turn_index": 1}
                ],
                "task_name": "research_questions",
                "model_name": "Qwen3.5-2B",
                "metadata": data,
            }
            output.write(json.dumps(record) + "\n")


            