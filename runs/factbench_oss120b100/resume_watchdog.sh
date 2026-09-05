#!/usr/bin/env bash
# Resume FactBench gpt-oss-120b from saved artifacts. Restart the chain on crash.
set -u
cd /Users/sussybaka/ResearchCode-run

: "${AZURE_OPENAI_API_KEY:?missing AZURE_OPENAI_API_KEY}"
: "${AZURE_OPENAI_ENDPOINT:?missing AZURE_OPENAI_ENDPOINT}"
: "${SERPER_API_KEY:?missing SERPER_API_KEY}"

export AZURE_ANSWER_DEPLOYMENT='openai--gpt-oss-120b'
export AZURE_WRITER_DEPLOYMENT='gpt-5-mini'
export AZURE_JUDGE_DEPLOYMENT='gpt-5-mini'
export AZURE_EXTRACTOR_DEPLOYMENT='gpt-5-mini'
export AZURE_VERIFIER_DEPLOYMENT='gpt-5-mini'
export AZURE_SEARCH_DEPLOYMENT='gpt-5-mini'
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

PYBIN=/Users/sussybaka/halluhard-run/.venv/bin/python
CFG=runs/factbench_oss120b100/config.toml
RUN=runs/factbench_oss120b100
DONE="$RUN/reports/summary.json"

run_stage() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  "$PYBIN" -m branching_hallucinations "$@" --config "$CFG" --run "$RUN" --concurrency 8
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== watchdog attempt ${attempt} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  if [[ -f "$DONE" ]]; then
    echo "=== already complete: $DONE ==="
    exit 0
  fi
  if run_stage init-run init-run \
    && run_stage generate-seeds generate-seeds \
    && run_stage extract-claims extract-claims \
    && run_stage verify-seeds verify-seeds \
    && run_stage generate-tree generate-tree \
    && run_stage audit-actions audit-actions \
    && run_stage judge-trajectories judge-trajectories \
    && echo "=== analyze ===" \
    && "$PYBIN" -m branching_hallucinations analyze --config "$CFG" --run "$RUN"; then
    echo "=== done ==="
    exit 0
  fi
  echo "=== crashed; restarting in 20s ==="
  sleep 20
done
