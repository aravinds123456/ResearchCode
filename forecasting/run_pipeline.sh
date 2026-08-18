#!/usr/bin/env bash
# Run the full forecasting pipeline from the forecasting/ directory.
set -euo pipefail

cd "$(dirname "$0")"

echo "Step 1/5: Fact-check original answers..."
python factscore_original_only.py "$@"

echo "Step 2/5: Generate branched future turns..."
python generate_future_turns.py "$@"

echo "Step 3/5: Label trajectories..."
python factscore_serper_cascades.py --resume "$@"

echo "Step 4/5: Summarize outcomes by strategy..."
python summarize_strategy_outcomes.py

echo "Step 5/5: Optional signal forecaster..."
python predict_cascades.py

echo "Done."
