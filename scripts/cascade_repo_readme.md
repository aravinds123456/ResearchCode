# Hallucination Cascade Forecasting (100 seeds x 3 turns)

Standalone experiment: HalluHard 5-strategy tree + HallucinationResearchTest DROP/CORRECT/REPEAT/DEPEND labels.

**Default design: 100 hallucinating seeds x 5 follow-up strategies x 3 turns = 500 branches / 1500 answers.**

Strategies: dependency-seeking, neutral, skeptical, accepting, topic-shift.

**Default models:** student `Qwen/Qwen3.5-2B` (`TEST_MODEL`) with thinking/reasoning **off**, judge `gpt-5-mini` (`OPENAI_LABEL_MODEL`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

## Run

```bash
python forecasting/pipeline.py answer --domain all --resume
python forecasting/pipeline.py judge  --domain all --resume
python forecasting/pipeline.py tree   --max-seeds 100 --levels 3 --resume
python forecasting/pipeline.py report
```

`python maincode.py` defaults to 100 seeds and 3 turns (`MAX_EXAMPLES`, `NUM_TURNS`).

```bash
TEST_MODEL=Qwen/Qwen3.5-2B python forecasting/generate_seeds.py
python forecasting/pipeline.py tree --seeds forecasting/seeds_qwen-qwen3.5-2b.jsonl --max-seeds 100 --levels 3 --resume
```

```bash
python -m unittest forecasting.test_pipeline forecasting.test_cascade
```
