# Add an evaluation dataset

Tree, judge, and analysis never import HalluHard or FactBench modules. They only see `{question_id, domain, question, source}`.

## Fastest path: a JSONL

1. Write `data/myset/prompts.jsonl` with one object per line. Required: a question field. Useful: `id`, `domain`.
2. Copy a config and point `[dataset]` at it:

```toml
[experiment]
domains = ["custom"]   # must match the domain field in the JSONL (or you get zero questions)

[dataset]
name = "jsonl"
path = "data/myset/prompts.jsonl"
question_field = "question"
domain_field = "domain"
id_field = "id"
grounding_task = "research_questions"
```

3. Use a new `--run` directory. Do not append a second dataset into an existing HalluHard run.

`grounding_task` selects the HalluHard **search style** (`research_questions`, `legal_cases`, `medical_guidelines`, `coding`). For general web prompts, keep `research_questions`.

## Named loader (FactBench-style)

Use this when you need filtering, IDs, or a fetch step.

1. Add `branching_hallucinations/<name>.py` with `load_<name>(*, domains, max_questions, path, source, **_)` returning the same row dicts as HalluHard.
2. Register it in `branching_hallucinations/questions.py`:

```python
DATASET_LOADERS = {
    "halluhard": load_halluhard,
    "factbench": load_factbench,
    "jsonl": load_jsonl,
    "myset": load_myset,
}
```

3. Vendor the prompts under `data/<name>/` (or document a fetch command). Pin a revision if you download.
4. Add `configs/branching_pilot_<name>.toml` and `configs/branching_final_<name>.toml`.
5. Set `experiment.domains` to **this loader’s** filter keys. HalluHard uses `research` / `legal` / `medical`. FactBench uses `hard` / `moderate` / `easy`. Mixing those lists is a config bug; FactBench raises if you pass HalluHard domains.
6. Tests: count, first prompt text, and that the default subset is what you claim (FactBench default is Hard+Moderate, Easy stored but not loaded).

Do **not** import the new dataset from `tree.py`, `trajectory_judge.py`, or `analysis.py`.

## Switching datasets mid-project

Keep HalluHard and FactBench as **separate runs**. Seed ids, claim ids, and node ids are not namespaced by dataset name beyond what you put in `question_id`.
