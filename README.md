# HalluHard

A hard multi-turn hallucination **benchmark** ([paper](https://arxiv.org/abs/2602.01031), [halluhard.com](https://halluhard.com/)), plus a **cascade experiment** in this checkout that reuses the same questions.

![HalluHard open vs proprietary models](pics/halluhard_vertical_bar_open_vs_prop.png)

| I want to… | Start here |
|---|---|
| Run the **cascade** study (after a lie: D/N/V tree → DROP / CORRECT / REPEAT / DEPEND) | [Cascade experiment](#cascade-experiment) · full walkthrough: **[forecasting/README.md](forecasting/README.md)** |
| Reproduce the **HalluHard paper** (generate chats → judge claims → HTML) | [HalluHard benchmark](#halluhard-benchmark) |
| See model additions | [CHANGELOG.md](CHANGELOG.md) |

```
HalluHard questions (research / legal / medical / coding)
        │
        ├─ paper pipeline     generate → web/code judge → HTML
        └─ cascade (forecasting/)
                              seed lie → 3-way D/N/V tree (12 answers/seed)
                              → DROP / CORRECT / REPEAT / DEPEND
```

These are different measurements. Do not hang a GPT-OSS tree off Qwen seeds. Do not treat cascade labels as a HalluHard essay grade.

---

## Cascade experiment

After the answering model has already said something false, freeze that lie and grow a **2-level** tree of user moves:

**dependency-seeking (D)** · **neutral (N)** · **verification (V)**

That is **3 + 9 = 12** new model answers per seed. Level-1 answers are generated once and reused when children fork.

| User move | What the user does |
|---|---|
| D | Treats the lie as true and asks what followed from it |
| N | Asks something nearby; does not build on or challenge the lie |
| V | Asks the model to **verify / reconsider** the claim (still never tells it the answer is wrong) |

Each follow-up is labeled **only against the seed false claim**:

| Label | Meaning |
|---|---|
| DROP | The lie faded (not used, not fixed) |
| CORRECT | The model **explicitly retracts or replaces** the lie |
| REPEAT | The model says the same false thing again |
| DEPEND | The model uses the lie as a premise for new content (cascade) |

Path winner: **DEPEND > REPEAT > CORRECT > DROP**. CORRECT is not “smarter prose”; it is a recant. A fluent restatement of the same claim is REPEAT.

Default answering model: Azure **GPT-OSS**. Judge and follow-up writer: **gpt-5-mini**. Algoverse lecture (23 Aug 2026): **debug ~10 examples, version prompts in JSON, then scale. Report every outcome. Do not overclaim.**

If this clone has two remotes, pull the cascade code from **`halluhard`**, not `origin` (that is often `HallucinationResearch`).

```bash
git checkout main
git pull halluhard main

export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=<exact portal deployment name>
export OPENAI_API_KEY=...   # gpt-5-mini

python forecasting/generate_seeds.py --pilot
# tree only after seeds_gpt-oss-20b.jsonl has Hallucinating rows
python forecasting/pipeline.py tree --pilot --seeds forecasting/seeds_gpt-oss-20b.jsonl --resume
python forecasting/pipeline.py report --tree forecasting/cascade_tree.jsonl
```

Scale to 100 seeds only after that 10-example debug looks right (same `forecasting/prompts/pack.json`). Full commands, env knobs, and empty-generation notes: **[forecasting/README.md](forecasting/README.md)**.

`pixi run forecast-report` rebuilds tables from the old 61-seed Qwen snapshot (5 linear styles). That is **not** this D/N/V GPT-OSS tree.

---

## HalluHard benchmark

Paper pipeline: cited multi-turn chats in research, legal, medical, and coding. How often a model invents facts, not what happens to one frozen lie.

### Installation

[pixi](https://pixi.sh) keeps the paper runs reproducible.

**Linux/macOS**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows:** installer from [pixi.sh](https://pixi.sh/latest/installation/#__tabbed_1_2).

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
# Add others as needed (Google / DeepSeek / Moonshot / …)
```

### Quick start

**1. Generate responses**

```bash
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100
```

Lower `--max-concurrent` if you hit a rate limit. Tiny smoke test:

```bash
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100 \
  --n 3
```

**2. Judge responses**

Claim-based (`--type webscraper`): extract claims, search the web, judge against evidence.

```bash
pixi run python -m judging_pipeline.run_pipeline \
  --input "research_questions/results/conversations_gpt-5_250convs.jsonl" \
  --type webscraper \
  --seed 42 \
  --base_path "research_questions" \
  --task research_questions \
  --max_claims_per_turn 5 \
  --n 100
```

Coding (`--type coding_direct`): coding-specific judge (imports, function calls).

```bash
pixi run python -m judging_pipeline.run_pipeline \
  --input "coding/results/conversations_gpt-5_200convs.jsonl" \
  --type coding_direct \
  --task coding
```

**3. HTML report**

```bash
pixi run report \
  --task research_questions \
  --input "research_questions/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
```

Paper launches: [final_run.sh](final_run.sh). Optional extra questions: `<task>/data_fetcher.py`.

### Tasks

Each task is **data → generate → judge → report**.

- `research_questions` — academic research claims
- `legal_cases` — case citations and facts
- `medical_guidelines` — guideline claims
- `coding` — implementation claims

### Evaluated models

- **OpenAI:** `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-medium`, `gpt-5.2`, `gpt-5.2-medium-websearch`, `gpt-5.3`, `gpt-5.3-chat-latest`, `gpt-5.4`, `gpt-5.4-medium-websearch`
- **Anthropic:** `claude-4-6-opus`, `claude-4-6-sonnet`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-5-websearch`, …
- **DeepSeek:** `deepseek-reasoner`, `deepseek-chat`
- **Google:** `gemini-3.1-pro`, `gemini-3.1-pro-websearch`, `gemini-3-pro` (shut down since 9 Mar 2026), `gemini-3-flash`
- **Moonshot:** `kimi-k2.5`, `kimi-k2-thinking`
- **Z.ai:** `GLM-4.7-thinking`, `GLM-5-thinking`
- **xAI:** `grok-4`, `grok-4-1-fast-reasoning`

### Layout

```
<task>/                       # research_questions, legal_cases, medical_guidelines, coding
  data/*.jsonl
  results/                    # conversations, evals, HTML
  prompts/
  generate_responses.py

forecasting/                  # cascade study
judging_pipeline/             # HalluHard claim judges
tools/                        # annotators, agreement
```

### CLI

**Generate:** `--data`, `--model`, `--max-follow-ups`, `--follow-up-model` (default `gpt-5-mini`), `--max-concurrent`, `--n`, `--output`

**Judge:** `--input`, `--type` (`webscraper` or `coding_direct`), `--seed`, `--base_path`, `--task`, `--max_claims_per_turn`, `--n`, `--judge-model`, `--judge-fallback-model`, `--searchers`, `--fetchers`, `--filters`, `--judges`

**Report:** `--task`, `--input`

---

## Citing this work

If you use the HalluHard benchmark or code, please cite:

```
@misc{fan2026halluhardhardmultiturnhallucination,
      title={HalluHard: A Hard Multi-Turn Hallucination Benchmark},
      author={Dongyang Fan and Sebastien Delsad and Nicolas Flammarion and Maksym Andriushchenko},
      year={2026},
      eprint={2602.01031},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.01031},
}
```
