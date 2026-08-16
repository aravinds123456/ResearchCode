import json
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report

features = {}

with open("forecasting/future_turns.jsonl") as file:
    for line in file:
        row = json.loads(line)
        features[row["question_number"]] = [
            row["average_confidence"],
            row["minimum_confidence"],
            row["average_entropy"],
            row["maximum_entropy"],
        ]

X, y = [], []

with open("forecasting/factscore_cascade_results.jsonl") as file:
    for line in file:
        row = json.loads(line)
        question = row["question_number"]

        if question in features:
            X.append(features[question])
            y.append(row["final_label"].lower())

print("Label counts:", Counter(y))

if len(set(y)) < 2:
    print("Cannot test prediction: every example has the same label.")
    raise SystemExit

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    class_weight="balanced",
    random_state=42,
)

predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())

for actual, predicted in zip(y, predictions):
    print(f"Actual: {actual:12} Predicted: {predicted}")

print(classification_report(y, predictions, zero_division=0))

