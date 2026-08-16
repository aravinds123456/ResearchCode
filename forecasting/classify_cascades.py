import json
from collections import Counter

INPUT_FILE = "forecasting/cascade_evidence.jsonl"
OUTPUT_FILE = "forecasting/cascade_results.jsonl"

def determine_outcome(was_corrected, created_dependent_errors):
    if was_corrected:
        return "corrected"

    if created_dependent_errors:
        return "snowballing"

    return "isolated"

def main():
    label_totals = Counter()

    with open(INPUT_FILE, "r") as input_file, open(OUTPUT_FILE, "w") as output_file:
        for line in input_file:
            conversation = json.loads(line)

            final_label = determine_outcome(conversation["was_corrected"], conversation["created_dependent_errors"],)

            conversation["final_label"] = final_label
            label_totals[final_label] += 1
            output_file.write(json.dumps(conversation) + "\n")

        total_conversations = sum(label_totals.values())

        print("Total Conversations: {total_conversations}")

        for label in ("corrected", "isolated", "snowballing"):
            conversation_count = label_totals[label]
            percentage = (100 * conversation_count / total_conversations if total_conversations else 0)

        print(f"{label}: {conversation_count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()


