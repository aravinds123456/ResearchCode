# Add or edit a prompt

Prompts live in `branching_hallucinations/prompts/<name>.txt`. `fill_prompt(name, **values)` replaces `{placeholders}` only. No extra templating.

| File | Used by | Placeholders |
|---|---|---|
| `claim_extraction.txt` | `seeds.extract_claims_for_seed` | (question/answer appended in code) |
| `grounded_verifier.txt` | `grounding.verify_claim` | `claim`, `context`, `queries`, `search_results`, `filtered_content` |
| `followup_d.txt` / `followup_n.txt` / `followup_v.txt` | `interventions.generate_intervention` | `claim`, `conversation` |
| `action_audit.txt` | `interventions.audit_action` | `claim`, `conversation`, `desired_action`, `user_message` |
| `trajectory_judge.txt` | `trajectory_judge.judge_trajectory` | `claim`, `seed_answer`, `conversation`, `latest_response` |

Hashes of these files are stored on artifacts (`prompt_hash`). Changing a prompt without a new `--version` on judge/audit will mix prompt versions in one JSONL if you resume. For a new judge prompt, use `--version v2`.

## Add a new named prompt

1. Create `prompts/foo.txt`.
2. Call `fill_prompt("foo", ...)` from the stage that needs it.
3. Do not put trajectory labels in follow-up prompts.
4. Do not add a regex that accepts/rejects the model’s question. If JSON is missing, retry format once, then stop.

## Remove a prompt

Delete the file and every `fill_prompt("...")` / `FOLLOWUP_PROMPT` mapping. Tests in `tests/branching_hallucinations/test_interventions.py` check that D/N/V prompts stay action-specific.
