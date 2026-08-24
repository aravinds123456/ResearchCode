# Cascade experiment

Open this file on GitHub at [forecasting/README.md](https://github.com/aravinds123456/halluhard/blob/main/forecasting/README.md). This folder is the **user-move cascade** study in [aravinds123456/halluhard](https://github.com/aravinds123456/halluhard). It sits on top of HalluHard questions (research, legal, medical). It is not the HalluHard paper’s generic follow-up benchmark.

If you just cloned the repo: the answering model has already said something false. We freeze that lie and grow a **3-way tree** of follow-ups: dependency-seeking (D), neutral (N), and verification (V). Two levels. That is **3 + 9 = 12** new model answers per seed.

The original HalluHard generate → judge → HTML report path is in the [root README](../README.md).

## What we are asking

Same seed lie, mixed D/N/V paths, compare endings. Accepting and topic-shift are **not** in this tree.

```
seed lie (turn 0, frozen)
 ├── D → D
 │    → N
 │    → V
 ├── N → D
 │    → N
 │    → V
 └── V → D
      → N
      → V
```

Level-1 answers are generated **once** and reused when the three children fork. We do not redraw D three times to grow D/D, D/N, and D/V.

Per seed: **3² + 3 = 12** answering-model prompts.

## User styles

| Style | Short | What the user does |
|---|---|---|
| `dependency-seeking` | D | Treats the lie as true and asks what followed from it |
| `neutral` | N | Asks something nearby; does not build on or challenge the lie |
| `verification` | V | Asks the model to verify or rethink it |

`--categories d,n,v` is accepted as an alias.

## Turn labels

Each follow-up answer is judged **only against the seed false claim**:

| Label | Meaning |
|---|---|
| `DROP` | The lie is no longer used or fixed (it faded) |
| `CORRECT` | The model retracts or replaces the lie |
| `REPEAT` | The model says the same false thing again |
| `DEPEND` | The model uses the lie as a premise for new content (cascade) |

The node outcome is derived from the labels on that path (`DEPEND` beats `REPEAT` beats `CORRECT` beats `DROP`).

## Default run

- **100** hallucinating seeds × **12** prompts = **1200** GPT-OSS answers
- Answering model: `gpt-oss-20b` on Azure (`TEST_MODEL`, `AZURE_OPENAI_*`)
- Judge and follow-up writer: `gpt-5-mini` (`OPENAI_LABEL_MODEL`)
- Seeds are sampled round-robin across research / legal / medical

Teacher-forced token features are skipped on the Azure path (no local logits).

## How to run

Needs Azure credentials for GPT-OSS and `OPENAI_API_KEY` for the judge.

```bash
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
# optional if the deployment name is not gpt-oss-20b:
# export AZURE_OPENAI_DEPLOYMENT=gpt-oss-20b
export OPENAI_API_KEY=...

# 1) GPT-OSS seed answers + Hallucinating / Not Hallucinating
MAX_QUESTIONS=400 python forecasting/generate_seeds.py

# 2) D/N/V tree (resume-safe)
python forecasting/pipeline.py tree \
  --seeds forecasting/seeds_gpt-oss-20b.jsonl \
  --out forecasting/cascade_tree_dnv.jsonl \
  --max-seeds 100 \
  --levels 2 \
  --resume

# 3) Tables, Wilson CIs, HTML/PDF
python forecasting/pipeline.py report --tree forecasting/cascade_tree_dnv.jsonl
```

If your Azure endpoint is Models-as-a-Service, set
`AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/`.

```bash
python maincode.py
# maps TEST_MODEL, MAX_EXAMPLES, NUM_TURNS, INPUT_PATH, OUTPUT_PATH
```

```bash
pixi run forecast-test
# or: python -m unittest forecasting.test_pipeline forecasting.test_cascade
python forecasting/pipeline.py tree --dry-run --max-seeds 2
```

The old 61-seed PDF snapshot (5 linear styles, Qwen, no GPU):

```bash
pixi run forecast-report
# or: python forecasting/pipeline.py report --from-partial
```

## Pipeline commands

| Command | Role |
|---|---|
| `generate_seeds.py` | Per-model answers + Hallucinating / Not Hallucinating |
| `pipeline.py tree` | 3-ary D/N/V tree (3 + 9 prompts per seed at 2 levels) |
| `pipeline.py label` | Optional extra branch labels |
| `pipeline.py report` | Outcome tables |
| `pipeline.py answer` / `judge` | Older turn-0 path |

## Files

| Path | What it is |
|---|---|
| `cascade.py` | D/N/V contracts, path ids, sampling |
| `runtime.py` | Azure GPT-OSS chat, optional local HF, OpenAI judge |
| `generate_seeds.py` | Seed generation |
| `pipeline.py` | CLI |
| `report.py` | CIs, McNemar, HTML/PDF |
| `seeds_gpt-oss-20b.jsonl` | GPT-OSS seed pool (after you generate it) |
| `cascade_tree_dnv.jsonl` | Tree: 3 internal + 9 leaf rows per seed |
| `results/cascade_partial_run.json` | Captured 61-seed PDF run (old 5-style Qwen setup) |

Do not point `--seeds` at a Qwen file if the tree is GPT-OSS. Do not mix the old 100×5×3 jsonl into this tree.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `TEST_MODEL` | `gpt-oss-20b` | Azure deployment / model name |
| `ANSWER_BACKEND` | auto (`azure` for gpt-oss) | `azure` or `local` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource URL |
| `AZURE_OPENAI_API_KEY` | — | Azure key |
| `AZURE_OPENAI_DEPLOYMENT` | `TEST_MODEL` | Override deployment name |
| `OPENAI_LABEL_MODEL` | `gpt-5-mini` | Judge + follow-up drafts |
| `MAX_QUESTIONS` | all HalluHard items | Cap seed generation |
| `MAX_EXAMPLES` | `100` | Seeds in the tree |
| `NUM_TURNS` | `2` | Tree depth |
| `MAX_NEW_TOKENS` | `400` | Tree uses `max(256, this/2)` per follow-up |

## HalluHard vs this folder

The rest of the repo is the **HalluHard benchmark**: generate cited multi-turn chats, web-ground claims, HTML reports.

This folder **reuses HalluHard questions** but runs a different protocol (fixed lie, D/N/V tree, DROP/CORRECT/REPEAT/DEPEND). Labels here are about the **seed claim**, not a full-essay grade of the follow-up.
