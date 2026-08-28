# Configs and run layout

## Files

| Config | Dataset | `n_seeds` |
|---|---|---|
| `configs/branching_pilot.toml` | HalluHard research/legal/medical | 10 |
| `configs/branching_final.toml` | HalluHard | 100 |
| `configs/branching_pilot_factbench.toml` | FactBench Hard+Moderate | 10 |
| `configs/branching_final_factbench.toml` | FactBench Hard+Moderate | 100 |

Copy one of these rather than editing keys ad hoc in an existing run.

## `[experiment]`

| Key | Meaning |
|---|---|
| `n_seeds` | Stop after this many **verified-false** seeds (verify) / generated seeds (generate) |
| `depth` | Tree depth. Pilot/final use 2 |
| `actions` | User moves, default `["D","N","V"]` |
| `random_seed` | Analysis bootstrap only, not Azure sampling |
| `domains` | Loader filter. HalluHard: `research`,`legal`,`medical`. FactBench: `hard`,`moderate` |
| `max_claims_per_seed` | Cap on extracted candidates |
| `concurrency` | Max in-flight I/O units per stage (Azure/Serper/fetch). Default `8`. Override with `--concurrency`. Does not change freeze order or tree parent/child waits |
| `allow_followup_fallback` | **Must be absent or false.** `true` raises |

## `[dataset]`

| Key | Meaning |
|---|---|
| `name` | `halluhard` / `factbench` / `jsonl` / a registered loader |
| `path` | JSONL path for `jsonl` or FactBench |
| `question_field` / `domain_field` / `id_field` | JSONL columns |
| `grounding_task` | HalluHard search strategy name |
| `grounding_tasks` | Optional per-domain override table |

`dataset.task_for(domain)` uses `grounding_tasks`, then HalluHard `DOMAIN_TASK`, then `grounding_task`.

## `[models.*]`, `[grounding]`, `[trajectory]`

See [add-model.md](add-model.md). Grounding method is `halluhard_webscraper` (Serper + fetch/PDF). Trajectory `states` must stay DROP/RETRACT/REPEAT/DEPEND.

## Run directory

Created by `init-run --run runs/<name>`:

```
runs/<name>/
  manifest.json
  generated_seeds.jsonl
  candidate_claims.jsonl
  verifications.jsonl
  verified_seeds.jsonl
  nodes.jsonl
  action_audits/<version>.jsonl
  trajectories/<version>.jsonl
  analysis/
  reports/
```

`--version v1` on audit/judge writes a new JSONL so a prompt change does not mix with old labels. `analyze --trajectory-version v1 --audit-version v1` must match.
