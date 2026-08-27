# Add or change a model role

Experiment code must not construct `AsyncAzureOpenAI` itself. Roles go through `branching_hallucinations.models.sampler_for_role`.

## Change a role in a run

Edit the matching `[models.<role>]` table:

```toml
[models.answer]
sampler = "azure-gpt-oss-20b"
max_tokens = 32768
reasoning_effort = "low"
```

Roles: `answer`, `followup_writer`, `trajectory_judge`, `claim_extractor`, `grounded_verifier`, `search_planner`.

The **deployment name** is taken from, in order:

1. `AZURE_<ROLE>_DEPLOYMENT` (see `ROLE_DEPLOYMENT_ENV` in `models.py`)
2. `models.<role>.deployment` in TOML
3. For `answer` only: `AZURE_OPENAI_DEPLOYMENT`
4. The sampler registry entry

Keep `grounded_verifier` on a stronger reasoning effort than the other minis if they share a deployment. That call is the scientific gate for `C`.

## Register a new Azure sampler id

1. Add an entry to `MODEL_REGISTRY` in `libs/models.py`:

```python
"azure-gpt-oss-20b": {
    "backend": "azure",
    "model": "gpt-oss-20b",
    "reasoning_effort": "low",
    "max_tokens": 32768,
    "websearch": False,
},
```

2. Point TOML `sampler = "azure-..."`.
3. Experiment construction **forces** `websearch=False` and `use_reasoning_fallback=False`. Do not rely on a registry default to turn those on.

Non-Azure backends in `get_sampler` are HalluHard leftovers. Do not use them for this experiment: they may require `OPENAI_API_KEY` or websearch.

## Do not

- Use OpenAI/Azure web_search as grounding
- Feed hidden GPT-OSS reasoning as the visible answer
- Put `OPENAI_API_KEY` on the Azure path
