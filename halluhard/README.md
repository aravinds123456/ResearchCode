# HalluHard (upstream)

This directory is the HalluHard benchmark, reused here for:

1. **Seed questions** — `research_questions/data`, `legal_cases/data`, `medical_guidelines/data`, `coding/data`. Loaded by `halluhard.questions`.
2. **Retrieval** — `judging_pipeline` (Serper search, page/PDF fetch, filtering). The branching experiment’s grounded verifier then judges the atomic claim against that evidence.

The D/N/V experiment itself is `branching_hallucinations/` at the repo root.

## Paper reproduction (optional)

Generate chats, run the HalluHard claim judge, write HTML:

```bash
pixi run python -m halluhard.research_questions.generate_responses \
  --data halluhard/research_questions/data/research_questions_all.jsonl \
  --model gpt-5 --max-follow-ups 2 --max-concurrent 100

pixi run python -m halluhard.judging_pipeline.run_pipeline \
  --input "halluhard/research_questions/results/conversations_gpt-5_250convs.jsonl" \
  --type webscraper --task research_questions \
  --base_path halluhard/research_questions --n 100

pixi run python halluhard/report.py \
  --task research_questions \
  --input "halluhard/research_questions/results/conversations_gpt-5_250convs_eval_webscraper.jsonl"
```

Leaderboard notes: [CHANGELOG.md](CHANGELOG.md). Bulk launches: [final_run.sh](final_run.sh).
