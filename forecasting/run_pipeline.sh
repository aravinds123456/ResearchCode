#!/usr/bin/env bash
# Run the full forecasting pipeline from the forecasting/ directory.
set -euo pipefail

cd "$(dirname "$0")"

echo "Step 1/4: Fact-check original answers..."
python factscore_original_only.py "$@"

echo "Step 2/4: Generate future turns..."
python generate_future_turns.py "$@"

echo "Step 3/4: Label trajectories..."
python factscore_serper_cascades.py "$@"

echo "Step 4/4: Train forecaster..."
python predict_cascades.py

echo "Done."
