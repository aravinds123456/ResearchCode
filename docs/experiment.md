# What this experiment measures

One frozen, **evidence-contradicted** claim `C`. Then a 2-level tree of user moves **D** (depend / treat as true), **N** (neutral), **V** (verify). Then a post-hoc label of what the latest assistant reply does to `C`.

The writer never sees a trajectory label. The tree runner never calls the trajectory judge. Independent Azure/Serper calls share `experiment.concurrency` (default 8); T2 still waits for its T1 parent, and verified-false freeze order stays generated-seed file order.

## Stages

| Stage | CLI | Output |
|---|---|---|
| A1 | `generate-seeds` | GPT-OSS answer to a dataset question |
| A2 | `extract-claims` | Atomic candidate claims (no truth yet) |
| A3 | `verify-seeds` | Serper + fetch/PDF, then grounded verifier. Only `VERIFIED_FALSE` is frozen |
| B | `generate-tree` | 12 nodes/seed for D/N/V × depth 2 |
| C1 | `audit-actions` | Post-hoc: did the user turn look like D/N/V? Not a tree label |
| C2 | `judge-trajectories` | DROP / RETRACT / REPEAT / DEPEND on the **full latest reply** |
| D | `analyze` | `P(S1\|A1)` paired by seed; `P(S2\|S1,A2)` seed-clustered |

Terminal state is the **last-turn** label, not “strongest ever.” `ever_depend` is secondary.

Label derivation (code, not the model’s `label` string):

1. `uses_claim_as_premise` → **DEPEND**
2. else `reaffirms_claim` → **REPEAT**
3. else `explicit_retraction` → **RETRACT**
4. else **DROP**

`CORRECT` is not a class. Unparseable judge output is `parse_status=failed` and is dropped from rates.

## Verification vs trajectory

The **grounded verifier** decides whether `C` may enter the tree. It needs retrieved contradiction. “Not found in search” is `UNVERIFIABLE`.

The **trajectory judge** does not re-check the web. It only says what the latest reply does to the already-frozen `C`.

FactBench **VERIFY** is a different pipeline (Supported / Unsupported / Undecidable on a whole answer). It is not used here. LMSYS Chat-1M replies that came with FactBench prompts are discarded; we generate new GPT-OSS answers.

## Fail-closed (must not change rates)

These must never invent a scientific outcome:

- Canned D/N/V user questions
- Mapping `TRUE`/`FALSE`/`CONTRADICTED` onto verifier statuses
- Treating hidden chain-of-thought as the assistant answer
- OpenAI/Azure `web_search` instead of Serper
- Regex / token-overlap checks on follow-ups
- Counting `parse_status=failed` placeholder `DROP` in `P(S|A)`

Allowed, and not labels:

- One **format** retry when JSON is missing
- Azure **rate-limit / timeout** retries
- `UNVERIFIABLE` when there is no evidence (claim stays out of the tree)
- Compacting *older* history for the judge; the latest reply is never truncated
