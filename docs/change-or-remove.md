# Change or remove pieces

Silent leftovers are worse than a missing feature. If you delete a path, make the old config **error**.

## Remove a dataset

1. Unregister it from `DATASET_LOADERS` in `questions.py`.
2. Delete or archive `data/<name>/`.
3. Delete `configs/branching_*_<name>.toml`.
4. Leave HalluHard configs pointing at `name = "halluhard"`.
5. Do not keep a loader that returns empty lists for unknown domains; raise.

Unknown `[dataset].name` already raises in `load_questions`.

## Remove a model role

Do not leave `sampler_for_role` falling through to `get_sampler` for a half-deleted Azure id. Delete the registry key and the TOML table together.

## Remove a scientific fallback

If you find code that can invent a user turn, a `VERIFIED_FALSE`, or a DROP/RETRACT/REPEAT/DEPEND, delete it and add a test that the helper is gone (see `test_no_canned_followup_helper`). Config flags that re-enable it should raise at `load_config`, not log-and-continue.

## Change trajectory classes

`schemas.TrajectoryState` and `expected_label` are the source of truth. Update `trajectory_judge.txt`, analysis tables, and tests together. Do not add `CORRECT`. Do not teach analysis to treat `parse_status=failed` as DROP.

## Wipe a run

`runs/` is gitignored. Delete the run directory. Stages resume by id; leftover JSONL from a different config will silently skip work. Prefer a new `--run` name after a config change.
