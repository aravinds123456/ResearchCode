#!/usr/bin/env bash
# Export a standalone 50-seed / 3-turn cascade repo you can publish with:
#   gh repo create hallucination-cascade-50x3 --public --source="$DEST" --push
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/../hallucination-cascade-50x3}"
mkdir -p "$DEST/forecasting/results" \
  "$DEST/research_questions/data" \
  "$DEST/legal_cases/data" \
  "$DEST/medical_guidelines/data"
cp "$ROOT"/forecasting/{cascade.py,pipeline.py,report.py,runtime.py,features.py,generate_seeds.py,test_cascade.py,test_pipeline.py,__init__.py} "$DEST/forecasting/"
cp "$ROOT"/forecasting/batch_results.jsonl "$DEST/forecasting/"
cp "$ROOT"/forecasting/batch_results_*.jsonl "$DEST/forecasting/"
cp "$ROOT"/forecasting/results/cascade_partial_run.json "$DEST/forecasting/results/" 2>/dev/null || true
cp "$ROOT"/maincode.py "$DEST/"
cp "$ROOT"/research_questions/data/research_questions_all.jsonl "$DEST/research_questions/data/"
cp "$ROOT"/legal_cases/data/legal_cases_all.jsonl "$DEST/legal_cases/data/"
cp "$ROOT"/medical_guidelines/data/guidelines.jsonl "$DEST/medical_guidelines/data/"
cp "$ROOT"/scripts/cascade_repo_readme.md "$DEST/README.md"
cp "$ROOT"/scripts/cascade_repo_requirements.txt "$DEST/requirements.txt"
cat > "$DEST/.gitignore" <<'EOF'
.env
.venv/
venv/
__pycache__/
*.pyc
.DS_Store
future_turns*.jsonl
seeds_*.jsonl
cascade_tree.jsonl
cascade_labels.jsonl
EOF
echo "Standalone repo written to $DEST"
echo "Publish: cd $DEST && git init -b main && git add . && git commit -m '50 seeds x 3 turns' && gh repo create hallucination-cascade-50x3 --public --source=. --push"
