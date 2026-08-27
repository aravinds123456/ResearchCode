# Branching Hallucinations

Stage-separated study of what happens to a **verified-false** seed claim `C` when the user continues with D / N / V moves.

HalluHard and FactBench are **prompt sources**, not the measurement. Trajectory labels (DROP / RETRACT / REPEAT / DEPEND) are about that one frozen lie.

```
questions ──► GPT-OSS answer ──► atomic claims ──► grounded verifier
                                              │
                                    only VERIFIED_FALSE C
                                              ▼
                                 2-level D/N/V tree (no judge)
                                              ▼
                      post-hoc DROP / RETRACT / REPEAT / DEPEND
```

**Docs**

| Doc | When to read it |
|---|---|
| [docs/experiment.md](docs/experiment.md) | What is measured, fail-closed rules, what is *not* a label |
| [docs/add-dataset.md](docs/add-dataset.md) | Add or switch an evaluation prompt set |
| [docs/add-model.md](docs/add-model.md) | Add or change an Azure role / sampler |
| [docs/add-prompt.md](docs/add-prompt.md) | Add or edit a prompt file |
| [docs/change-or-remove.md](docs/change-or-remove.md) | Retire a dataset, role, prompt, or config flag |
| [docs/configs.md](docs/configs.md) | TOML keys and run layout |

## Layout

```
branching_hallucinations/   experiment (seeds, tree, judge, analysis, CLI)
configs/                    HalluHard and FactBench TOML
data/factbench/             FactBench prompts (Hard / Moderate / Easy)
docs/                       how to add / remove / change pieces
runs/                       gitignored artifacts
tests/branching_hallucinations/

halluhard/                  questions + Serper/fetch/PDF retrieval
libs/                       Azure samplers, Serper, HTML/PDF helpers
```

## Credentials

Azure for every LLM role. Serper for search. `OPENAI_API_KEY` is not used on this path.

```bash
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=<gpt-oss deployment>
export AZURE_WRITER_DEPLOYMENT=<gpt-5-mini deployment>
export AZURE_JUDGE_DEPLOYMENT=<gpt-5-mini deployment>
export AZURE_EXTRACTOR_DEPLOYMENT=<gpt-5-mini deployment>
export AZURE_VERIFIER_DEPLOYMENT=<gpt-5-mini deployment>
export AZURE_SEARCH_DEPLOYMENT=<gpt-5-mini deployment>
export SERPER_API_KEY=...
```

`AZURE_ANSWER_DEPLOYMENT` overrides the gpt-oss deployment if set. Role-specific `AZURE_*_DEPLOYMENT` vars override TOML.

## Models

| Role | Pilot sampler | Why |
|---|---|---|
| `answer` | `azure-gpt-oss-20b` | Seed answers and tree replies (the subject model) |
| `followup_writer`, `claim_extractor`, `search_planner`, `trajectory_judge` | `azure-gpt-5-mini` (`reasoning_effort=minimal`) | Structured JSON |
| `grounded_verifier` | `azure-gpt-5-mini-medium` | Same gpt-5-mini **deployment**, more reasoning. This call decides whether `C` is `VERIFIED_FALSE`. |

The grounded verifier does **not** search. Search/fetch already happened. Unsupported without contradiction is `UNVERIFIABLE`, not false.

## Run a HalluHard pilot

```bash
python -m unittest discover -s tests/branching_hallucinations -p "test_*.py"

python -m branching_hallucinations init-run \
  --config configs/branching_pilot.toml --run runs/pilot

python -m branching_hallucinations generate-seeds \
  --config configs/branching_pilot.toml --run runs/pilot

python -m branching_hallucinations extract-claims \
  --config configs/branching_pilot.toml --run runs/pilot

python -m branching_hallucinations verify-seeds \
  --config configs/branching_pilot.toml --run runs/pilot

python -m branching_hallucinations generate-tree \
  --config configs/branching_pilot.toml --run runs/pilot

python -m branching_hallucinations audit-actions \
  --config configs/branching_pilot.toml --run runs/pilot --version v1

python -m branching_hallucinations judge-trajectories \
  --config configs/branching_pilot.toml --run runs/pilot --version v1

python -m branching_hallucinations analyze \
  --config configs/branching_pilot.toml --run runs/pilot --trajectory-version v1

python -m branching_hallucinations export-audit \
  --config configs/branching_pilot.toml --run runs/pilot --trajectory-version v1
```

Scale with `configs/branching_final.toml` only after a ~10-seed manual audit of frozen `C` and a few D/N/V turns.

Resume is by artifact id. Re-running a stage skips completed seeds/claims/nodes.

## Run FactBench Medium/Hard

Same stages. Different config and run directory.

```bash
python -m branching_hallucinations init-run \
  --config configs/branching_pilot_factbench.toml --run runs/pilot-factbench
```

Then repeat `generate-seeds` … `analyze` with that config and `--run`. Final size: `configs/branching_final_factbench.toml`.

FactBench is used as **prompts only**. VERIFY, LMSYS replies, `prompt_score`, and `hallucination_score` are not labels in this experiment. Refresh the local copy with `python -m branching_hallucinations.factbench`.

## Fail-closed (no result-changing fallbacks)

| Failure | What happens |
|---|---|
| Follow-up JSON missing after one format retry | Run stops. No canned D/N/V question. |
| Verifier status not exactly `VERIFIED_FALSE` / `SUPPORTED` / `AMBIGUOUS` / `UNVERIFIABLE` | Treated as unparseable. Claim does not enter the tree. |
| No retrieved evidence, or search crash | `UNVERIFIABLE`. Not false. |
| Trajectory JSON missing after one format retry | Stored as `parse_status=failed`. **Excluded** from `P(S\|A)` and transitions. |
| `allow_followup_fallback = true` in TOML | Config load errors. |
| Azure `websearch` / hidden-reasoning-as-answer | Forced off for experiment roles. |

Format retries (JSON envelope only) and Azure rate-limit retries do **not** invent scientific labels.

## HalluHard paper pipeline

Optional, separate from this experiment: [`halluhard/README.md`](halluhard/README.md). Paper: [arXiv:2602.01031](https://arxiv.org/abs/2602.01031).

FactBench prompts: [arXiv:2410.22257](https://arxiv.org/abs/2410.22257).
