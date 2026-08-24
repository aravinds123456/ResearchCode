# HalluHard: A Hard Multi-Turn Hallucination Benchmark

**Cascade experiment docs (this checkout):** [forecasting/README.md](forecasting/README.md)

![HalluHard open vs proprietary models](pics/halluhard_vertical_bar_open_vs_prop.png)

A framework for evaluating hallucinations in multi-turn conversations across challenging domains.

Check out our website for the updates on newly released models: https://halluhard.com/

See [CHANGELOG.md](CHANGELOG.md) for the history of model additions and benchmark updates.

## What's in this repository

This checkout has two layers that share the same question files:

| Layer | What it measures | Start here |
|---|---|---|
| **HalluHard benchmark** | How often a model invents facts in cited multi-turn chats (research, legal, medical, coding) | The rest of this README |
| **Cascade experiment** | After the model has already made a false claim, what five different user styles do to that same lie over three turns | **[forecasting/README.md](forecasting/README.md)** |

The published HalluHard paper is the first layer. The cascade code reuses those questions but runs a different protocol: freeze one seed lie, fork five pure user styles (dependency-seeking, neutral, skeptical, accepting, topic-shift), and label each turn `DROP` / `CORRECT` / `REPEAT` / `DEPEND`.

```
research_questions/   legal_cases/   medical_guidelines/   coding/
    HalluHard tasks: generate conversations → judge claims → HTML report

forecasting/
    Cascade study: generate seeds → 5-style tree → outcome tables

judging_pipeline/     tools/     report.py
    Shared claim judges, annotators, and the HalluHard HTML reporter
```

If you cloned this repo to run or read the cascade study, skip to [forecasting/README.md](forecasting/README.md). The original one-shot / multi-turn HalluHard pipeline is documented below.

## Preparation

We use pixi to make sure our experiments are reproducible across all enviroments.

### Installation

Install [pixi](https://pixi.sh) package manager:

**Linux/macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows:**
Download the installer from [pixi.sh](https://pixi.sh/latest/installation/#__tabbed_1_2) and run it.

### Configuration

Running models from hosted providers typically requires API credentials. Export the relevant environment variables for the providers/models you plan to use.

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
# Add others as needed (e.g., Google/DeepSeek/Moonshot), depending on what you run.
```


## Quick Start

### 1. Generate Responses

Generate multi-turn conversations using different model:

```bash
# Example: Research Questions task
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100 \
```

Tips: 
- You may need to change `--max-concurrent` to a smaller value if a rate limit is reached. 
- Start with a tiny run first (the example below generates 3 conversations, each with 2 follow-up questions):

```bash
# Example: Research Questions task
pixi run python -m research_questions.generate_responses \
  --data research_questions/data/research_questions_all.jsonl \
  --model gpt-5 \
  --max-follow-ups 2 \
  --max-concurrent 100 \
  --n 3
```

### 2. Judge Responses

HalluHard supports two judging modes:

A) Claim-based verification (`--type webscraper`)

Extracts atomic claims (citation + supported content) per turn, searches the web, and judges claims against retrieved evidence. This is intended for tasks that require citation grounding.

```bash
# Evaluate using web scraper method
pixi run python -m judging_pipeline.run_pipeline \
  --input "research_questions/results/conversations_gpt-5_250convs.jsonl" \
  --type webscraper \
  --seed 42 \
  --base_path "research_questions" \
  --task research_questions \
  --max_claims_per_turn 5 \
  --n 100
```

B) Response-based verification (`--type coding_direct`)

Directly evaluates coding-task responses using a coding-specific judge (e.g., checking package installation/importing and function calling behaviors). This mode is intended for the coding task.

```bash
# Evaluate coding task using direct coding judge
pixi run python -m judging_pipeline.run_pipeline \
  --input "coding/results/conversations_gpt-5_200convs.jsonl" \
  --type coding_direct \
  --task coding \
```

### 3. Generate Reports

Generate an HTML report from an evaluation output file:

```bash
pixi run report \
  --task research_questions \
  --input "research_questions/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
```

## Cascade forecasting

The cascade study lives under [`forecasting/`](forecasting/). Full walkthrough, labels, and file map: **[forecasting/README.md](forecasting/README.md)**.

Default design: **100** domain-balanced hallucinating seeds × **5** pure user styles × **3** turns. Answering model `Qwen/Qwen3.5-2B` (thinking off). Judge and follow-up writer `gpt-5-mini`.

```bash
# No GPU: rebuild tables from the captured 61-seed snapshot
pixi run forecast-report
# or: python forecasting/pipeline.py report --from-partial

# Full experiment (needs a local GPU or Apple MPS + OPENAI_API_KEY)
MAX_QUESTIONS=400 python forecasting/generate_seeds.py
python forecasting/pipeline.py tree \
  --seeds forecasting/seeds_qwen-qwen3.5-2b.jsonl \
  --out forecasting/cascade_tree_100x3.jsonl \
  --max-seeds 100 --levels 3 --resume
python forecasting/pipeline.py report --tree forecasting/cascade_tree_100x3.jsonl
```

`python maincode.py` maps HallucinationResearchTest env vars (`TEST_MODEL`, `MAX_EXAMPLES`, `INPUT_PATH`, `OUTPUT_PATH`, `NUM_TURNS`) onto `pipeline.py tree`. Do not mix a 4B seed file into a 2B tree. Do not mix follow-up styles inside one branch if you want a per-style table.

## Available Tasks

- `research_questions` - Academic research question claims
- `legal_cases` - Legal case citations and facts
- `medical_guidelines` - Medical guideline claims
- `coding` - Code implementation claims

Each task follows the same workflow: 

**data preparation → response generation → judging → reporting**

## Evaluated Models

The framework supports multiple LLM providers and models:

- **OpenAI**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-medium`, `gpt-5.2`, `gpt-5.2-medium-websearch`, `gpt-5.3`, `gpt-5.3-chat-latest`,`gpt-5.4`, `gpt-5.4-medium-websearch`
- **Anthropic**: `claude-4-6-opus`, `claude-4-6-sonnet`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-5-websearch`, …
- **DeepSeek**: `deepseek-reasoner`, `deepseek-chat`
- **Google**: `gemini-3.1-pro`, `gemini-3.1-pro-websearch`, `gemini-3-pro` (shut down since March 9, 2026), `gemini-3-flash`
- **Moonshot**: `kimi-k2.5`, `kimi-k2-thinking`
- **Z.ai**: `GLM-4.7-thinking`, `GLM-5-thinking`
- **xAI**: `grok-4`, `grok-4-1-fast-reasoning` 

## Project Structure

```
<task>/                       # research_questions, legal_cases, medical_guidelines, coding
  ├── data/                    # Input data
  │   └── *.jsonl             # Task-specific question datasets
  ├── results/                 # Generated conversations and evaluations
  │   ├── conversations_<model>_<n>convs.jsonl
  │   ├── conversations_<model>_<n>convs_eval_<type>.jsonl
  │   └── reports/             # HTML reports
  ├── prompts/                 # Task-specific prompts
  └── generate_responses.py    # Response generation script

forecasting/                   # Cascade study (see forecasting/README.md)
judging_pipeline/              # Claim judges used by the HalluHard path
tools/                         # Annotators and judge-agreement scripts
```

## CLI Reference

### Response generation parameters
- `--data`: Path to input data file
- `--model`: Model name to use
- `--max-follow-ups`: Number of follow-up questions per conversation (typically 2)
- `--follow-up-model`: Model that simulates the user for follow-up questions (optional; default is `gpt-5-mini` when omitted)
- `--max-concurrent`: Number of concurrent API requests (varies by model rate limits)
- `--n`: Number of conversations to generate (optional, defaults to all)
- `--output`: Custom output path (optional)

### Judging pipeline parameters
- `--input`: Path to conversations file to evaluate
- `--type`: Evaluation method (`webscraper` or `coding_direct`)
- `--seed`: Random seed for reproducibility
- `--base_path`: Base directory for task
- `--task`: Task name
- `--max_claims_per_turn`: Maximum claims per turn (typically 5)
- `--n`: Number of conversations to evaluate (optional)
- `--judge-model`: When `--type` is `serper` or `webscraper` — registry id for the primary claim judge (default: `gpt-5-mini-medium`). Ignored for `openai` and `coding_direct`.
- `--judge-fallback-model`: When `--type` is `serper` or `webscraper` — registry id for the judge on the web-grounding fallback path (default: `gpt-5-mini-medium-websearch`). Ignored for `openai` and `coding_direct`.
- Worker parameters: `--searchers`, `--fetchers`, `--filters`, `--judges`

### Report generation parameters
- `--task`: Task name
- `--input`: Path to evaluation results file

## Launch Experiments

See [final_run.sh](final_run.sh) for the final launches used in the paper.

## (Optional) Data Generation

For full transparency, we provide our data generation pipelines under `<task_name>/data_fetcher.py`. Readers are welcome to re-use this script to generate more questions for their own use. 

## Citing this work
If you find our code useful, please cite our work
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

