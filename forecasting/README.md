# Cascade experiment

Open this file on GitHub at [forecasting/README.md](https://github.com/aravinds123456/halluhard/blob/main/forecasting/README.md). This folder is the **user-move cascade** study in [aravinds123456/halluhard](https://github.com/aravinds123456/halluhard). It sits on top of HalluHard questions (research, legal, medical). It is not the HalluHard paper’s generic follow-up benchmark.

If you just cloned the repo and want to know what *this* experiment is: Qwen has already said something false. We freeze that lie and start **five separate chats** from the same point. After three turns we label what happened to that lie.

The original HalluHard generate → judge → HTML report path is in the [root README](../README.md).

## What we are asking

The new part is the **fork**: same model lie, five user styles, compare endings. Cascades themselves are not a new phenomenon.

```
seed lie (turn 0, frozen)
 ├── dependency-seeking → turn 1 → turn 2 → turn 3
 ├── neutral            → turn 1 → turn 2 → turn 3
 ├── accepting          → turn 1 → turn 2 → turn 3
 ├── skeptical          → turn 1 → turn 2 → turn 3
 └── topic-shift        → turn 1 → turn 2 → turn 3
```

## User styles

| Style | What the user does |
|---|---|
| `dependency-seeking` | Treats the lie as true and asks what followed from it |
| `neutral` | Asks something nearby; does not build on or challenge the lie |
| `accepting` | Goes along with it |
| `skeptical` | Asks the model to verify or rethink it |
| `topic-shift` | Drops that claim and asks about something else |

Do not mix styles inside one branch if you want a per-style table.

## Turn labels

Each follow-up answer is judged **only against the seed false claim**:

| Label | Meaning |
|---|---|
| `DROP` | The lie is no longer used or fixed (it faded) |
| `CORRECT` | The model retracts or replaces the lie |
| `REPEAT` | The model says the same false thing again |
| `DEPEND` | The model uses the lie as a premise for new content (cascade) |

The branch outcome is derived from the three turn labels (`DEPEND` beats `REPEAT` beats `CORRECT` beats `DROP`).

## Default run

- **100** hallucinating seeds × **5** styles × **3** turns = **500** branches / **1500** Qwen answers
- Answering model: `Qwen/Qwen3.5-2B` (`TEST_MODEL`)
- Qwen thinking / reasoning: **off** (`enable_thinking=False`)
- Judge and follow-up writer: `gpt-5-mini` (`OPENAI_LABEL_MODEL`) via the OpenAI Responses API
- Seeds are sampled round-robin across research / legal / medical

A mid-sentence Qwen answer is a **token-cap cutoff** (tree follow-ups default to 256 new tokens). The judge scores the text that is there, not a guessed ending. Cutoff alone is not a hallucination.

## How to run

Needs a GPU or Apple MPS for Qwen, and `OPENAI_API_KEY` for the judge. This repo’s supported install is [pixi](https://pixi.sh) (see the [root README](../README.md)). A slim venv also works if you only want this folder:

```bash
# Option A — full repo (pixi)
curl -fsSL https://pixi.sh/install.sh | sh
export OPENAI_API_KEY=...
# export HF_TOKEN=...   # only if the answering-model repo is gated

# Option B — cascade-only venv
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/cascade_repo_requirements.txt
export OPENAI_API_KEY=...
```

Then:

```bash
# 1) New answers + seed labels (cap how many questions you score)
MAX_QUESTIONS=400 python forecasting/generate_seeds.py

# 2) Five-style tree (resume-safe; safe to restart with --resume)
python forecasting/pipeline.py tree \
  --seeds forecasting/seeds_qwen-qwen3.5-2b.jsonl \
  --out forecasting/cascade_tree_100x3.jsonl \
  --max-seeds 100 \
  --levels 3 \
  --resume

# 3) Tables, Wilson CIs, HTML/PDF
python forecasting/pipeline.py report --tree forecasting/cascade_tree_100x3.jsonl
```

`generate_seeds.py` writes `forecasting/seeds_<model-slug>.jsonl`. Only rows labeled Hallucinating are used as tree seeds.

```bash
python maincode.py
# maps TEST_MODEL, MAX_EXAMPLES, NUM_TURNS, INPUT_PATH, OUTPUT_PATH
```

```bash
pixi run forecast-test
# or: python -m unittest forecasting.test_pipeline forecasting.test_cascade
python forecasting/pipeline.py tree --dry-run --max-seeds 2
```

The old 61-seed PDF snapshot (no GPU):

```bash
pixi run forecast-report
# or: python forecasting/pipeline.py report --from-partial
```

## Pipeline commands

| Command | Role |
|---|---|
| `generate_seeds.py` | Per-model answers + Hallucinating / Not Hallucinating |
| `pipeline.py answer` | Turn-0 answers from HalluHard splits (older path) |
| `pipeline.py judge` | Label those answers with `P_JUDGE` |
| `pipeline.py tree` | Five branches × N turns |
| `pipeline.py label` | Optional extra branch labels |
| `pipeline.py report` | Outcome tables |

## Files

| Path | What it is |
|---|---|
| `cascade.py` | Strategies, contracts, labels, sampling |
| `runtime.py` | Load Qwen, thinking off, OpenAI / Gemini judges |
| `generate_seeds.py` | Seed generation |
| `pipeline.py` | CLI |
| `features.py` | Teacher-forced P(emitted token), not max-softmax |
| `report.py` | CIs, McNemar, HTML/PDF |
| `seeds_qwen-qwen3.5-2b.jsonl` | Current 2B seed pool |
| `cascade_tree_100x3.jsonl` | Current tree (one JSON object per branch) |
| `results/cascade_partial_run.json` | Captured 61-seed PDF run (Qwen3.5-2B, older setup) |

Do not mix a 4B seed file into a 2B tree. Do not point `--seeds` at Desktop backups from another model or judge.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `TEST_MODEL` / `QWEN_MODEL` | `Qwen/Qwen3.5-2B` | Model that answers |
| `OPENAI_LABEL_MODEL` | `gpt-5-mini` | Judge + follow-up drafts |
| `ENABLE_THINKING` | off | Set `1` to allow Qwen thinking |
| `MAX_QUESTIONS` | all HalluHard items | Cap seed generation |
| `MAX_EXAMPLES` | `100` | Seeds in the tree |
| `NUM_TURNS` | `3` | Follow-ups per branch |
| `MAX_NEW_TOKENS` | `400` | Tree uses `max(256, this/2)` per follow-up |
| `SEED_MAX_NEW_TOKENS` | `300` | Seed answer length |

## HalluHard vs this folder

The rest of the repo is the **HalluHard benchmark**: generate cited multi-turn chats, web-ground claims, HTML reports.

This folder **reuses HalluHard questions** but runs a different protocol (fixed lie, five user styles, DROP/CORRECT/REPEAT/DEPEND). Labels here are about the **seed claim**, not a full-essay grade of the follow-up.
