"""Generate and judge seed hallucinations with the model under test.

HallucinationResearchTest's seed sampler, adapted to HalluHard's three
domains. Scoring another model's confidence on Qwen's text is a different
quantity from that model's own generation confidence, so each model needs
to produce its own seeds before the branching stage runs.

    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" python forecasting/generate_seeds.py
    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" \
        python forecasting/pipeline.py tree --seeds forecasting/seeds_meta-llama-llama-3.1-8b-instruct.jsonl --resume
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from cascade import (
    DEFAULT_TEST_MODEL,
    DIR,
    DOMAINS,
    ENABLE_THINKING,
    env_float,
    env_int,
    env_str,
    model_slug,
    seed_identifier,
    strip_question_prefix,
    strip_thinking,
    write,
)
from prompts_pack import (
    DEFAULT_PILOT_QUESTIONS,
    fill_prompt,
    log_experiment,
    prompt_ids,
    prompt_pack_version,
    prompt_text,
    require_pilot,
    write_pilot_stage,
)

MODEL_NAME = env_str("TEST_MODEL", env_str("QWEN_MODEL", DEFAULT_TEST_MODEL))
SEED_SCHEMA_VERSION = 3
# GPT-OSS hidden reasoning counts against this. 300 often returns empty content.
SEED_MAX_NEW_TOKENS = env_int("SEED_MAX_NEW_TOKENS", 2048)
MAX_QUESTIONS = env_int("MAX_QUESTIONS", 0)
SAMPLES_PER_QUESTION = env_int("SAMPLES_PER_QUESTION", 1)
TEMPERATURE = env_float("TEMPERATURE", 0.7)
TOP_P = env_float("TOP_P", 0.95)
TOP_K = env_int("TOP_K", 0)
BASE_SEED = env_int("BASE_SEED", 1234)
DOMAIN_FILTER = env_str("SEED_DOMAIN", "all")

SEEDS_PATH = Path(env_str("SEEDS_PATH", str(DIR / f"seeds_{model_slug(MODEL_NAME)}.jsonl")))

SEED_JUDGE_TEMPLATE = prompt_text("seed_judge")

SEED_LABEL_PATTERN = re.compile(
    r"Overall label:\s*(Not\s+Hallucinating|Hallucinating)",
    re.IGNORECASE,
)
SEED_REASON_PATTERN = re.compile(r"Reason:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_seed_judgement(text: str) -> tuple[str, str]:
    match = SEED_LABEL_PATTERN.search(text)
    if match:
        normalized = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        label = "Not Hallucinating" if normalized.startswith("not") else "Hallucinating"
    elif re.search(r"\bnot\s+hallucinat", text, re.I):
        label = "Not Hallucinating"
    elif re.search(r"\bhallucinat", text, re.I):
        label = "Hallucinating"
    else:
        print(f"Warning: unparseable seed judgement, treating as clean. Raw: {text[:160]}")
        label = "Not Hallucinating"
    reason_match = SEED_REASON_PATTERN.search(text)
    reason = reason_match.group(1).strip().split("\n")[0] if reason_match else ""
    return label, reason


def load_questions() -> dict[int, tuple[str, str]]:
    """question_number -> (question, domain), using HalluHard offsets."""
    questions = {}
    wanted = DOMAINS if DOMAIN_FILTER == "all" else {DOMAIN_FILTER: DOMAINS[DOMAIN_FILTER]}
    for domain, (path, key, offset) in wanted.items():
        if not path.exists():
            continue
        for index, line in enumerate(path.open(encoding="utf-8")):
            record = json.loads(line)
            question = (record.get(key) or record.get("question") or "").strip()
            if question:
                questions[offset + index] = (question, domain)
    if not questions:
        raise FileNotFoundError("No HalluHard domain question files found.")
    return dict(sorted(questions.items()))


def load_existing_seeds():
    if not SEEDS_PATH.exists():
        return set(), {}
    processed = set()
    answers_by_question = {}
    for record in (json.loads(line) for line in SEEDS_PATH.open() if line.strip()):
        if record.get("seed_schema_version", 0) != SEED_SCHEMA_VERSION:
            continue
        if record.get("model_name") != MODEL_NAME:
            continue
        processed.add((record["question_number"], record.get("sample_index", 0)))
        answers_by_question.setdefault(record["question_number"], set()).add(
            record.get("model_answer") or record.get("qwen_answer", "")
        )
    return processed, answers_by_question


def sample_seed_value(question_number: int, sample_index: int) -> int:
    return (BASE_SEED * 1_000_003 + int(question_number) * 1_009 + sample_index) % (2**31 - 1)


def raw_step_logits(outputs):
    logits = getattr(outputs, "logits", None)
    if logits:
        return logits
    print("Warning: transformers did not return raw logits; falling back to processed scores.")
    return outputs.scores


def generate_seed_answer(question: str, question_number: int, sample_index: int):
    import runtime

    rng_seed = sample_seed_value(question_number, sample_index)
    if runtime.uses_azure_answer(MODEL_NAME):
        answer = runtime.answer_chat(
            [{"role": "user", "content": question}],
            max_new_tokens=SEED_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
        )
        return strip_question_prefix(question, answer), {}, rng_seed

    import torch
    from features import generation_features
    from runtime import init_model

    init_model(MODEL_NAME)
    tokenizer = runtime.tokenizer
    messages = [{"role": "user", "content": question}]
    model_inputs = runtime.build_model_inputs(messages)
    model_inputs = {key: value.to(runtime.device) for key, value in model_inputs.items() if hasattr(value, "to")}
    input_length = model_inputs["input_ids"].shape[1]
    torch.manual_seed(rng_seed)
    sampling_kwargs = {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P}
    if TOP_K:
        sampling_kwargs["top_k"] = TOP_K
    with torch.no_grad():
        outputs = runtime.model.generate(
            **model_inputs,
            max_new_tokens=SEED_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            **sampling_kwargs,
        )
    generated_token_ids = outputs.sequences[0, input_length:]
    answer = strip_thinking(tokenizer.decode(generated_token_ids, skip_special_tokens=True)).strip()
    special_ids = set(tokenizer.all_special_ids or [])
    features = generation_features(raw_step_logits(outputs), generated_token_ids, special_ids)
    return strip_question_prefix(question, answer), features, rng_seed


def judge_seed(question: str, answer: str):
    from runtime import call_gemini, gpt, judge_backend

    prompt = fill_prompt("seed_judge", question=question, answer=answer)
    if judge_backend() == "gemini":
        raw_text = call_gemini(prompt)
    else:
        raw_text = str(gpt(prompt, as_json=False)).strip()
    label, reason = parse_seed_judgement(raw_text)
    return label, reason, raw_text


def main():
    questions = load_questions()
    processed_samples, answers_by_question = load_existing_seeds()
    question_items = list(questions.items())
    pilot = "--pilot" in sys.argv
    skip_pilot = "--skip-pilot" in sys.argv
    dry_run = env_str("DRY_RUN", "") == "1"
    limit = DEFAULT_PILOT_QUESTIONS if pilot else MAX_QUESTIONS
    require_pilot(stage="seeds", n=limit or 10**9, dry_run=dry_run, skip_pilot=skip_pilot)
    if limit:
        question_items = question_items[:limit]
    pending = [
        (number, question, domain, sample_index)
        for number, (question, domain) in question_items
        for sample_index in range(SAMPLES_PER_QUESTION)
        if (number, sample_index) not in processed_samples
    ]
    from runtime import (
        active_judge_model,
        azure_deployment,
        azure_reasoning_effort,
        azure_send_temperature,
        judge_backend,
        setup_gemini,
        uses_azure_answer,
    )

    print(f"Test model: {MODEL_NAME}")
    if uses_azure_answer(MODEL_NAME):
        print(f"Azure deployment: {azure_deployment(MODEL_NAME)} (override with AZURE_OPENAI_DEPLOYMENT)")
        print(f"Azure reasoning_effort: {azure_reasoning_effort() or 'off'}")
        if azure_send_temperature():
            print(f"Azure temperature: {TEMPERATURE}")
        else:
            print("Azure temperature: omitted (GPT-OSS often rejects it; set AZURE_SEND_TEMPERATURE=1 to sample)")
    print(f"Judge model: {active_judge_model()}")
    print(f"Thinking: {'on' if ENABLE_THINKING else 'off'}")
    print(f"Prompt pack: v{prompt_pack_version()} ids={prompt_ids()}")
    print("Workflow: debug prompts on ~10 examples (--pilot), then scale.")
    print(f"Questions: {len(question_items)} HalluHard items, {len(pending)} pending generations")
    print(
        f"Seed max_completion_tokens: {SEED_MAX_NEW_TOKENS} "
        "(hidden GPT-OSS reasoning counts against this; empty content usually means the cap was too low)"
    )
    print(f"Output: {SEEDS_PATH.name}")
    log_experiment(
        "seed_pilot" if pilot else "seed_full",
        n=len(question_items),
        pending=len(pending),
        dry_run=dry_run,
        model=MODEL_NAME,
    )
    if dry_run:
        print("DRY_RUN=1 set; validation passed, exiting before model/API calls.")
        return
    if not pending:
        print("Nothing to do.")
        return

    if judge_backend() == "gemini":
        setup_gemini()

    label_counts = Counter()
    duplicate_count = 0
    for index, (question_number, question, domain, sample_index) in enumerate(pending, start=1):
        answer, features, rng_seed = generate_seed_answer(question, question_number, sample_index)
        progress = f"[{index}/{len(pending)}] q{question_number}#{sample_index}"
        if not answer:
            print(
                f"{progress}: empty generation, skipping "
                f"(no visible assistant text after retry; GPT-OSS spent the token budget on "
                f"hidden reasoning or returned an empty message. "
                f"SEED_MAX_NEW_TOKENS={SEED_MAX_NEW_TOKENS})"
            )
            continue
        seen_answers = answers_by_question.setdefault(question_number, set())
        is_duplicate = answer in seen_answers
        seen_answers.add(answer)
        if is_duplicate:
            duplicate_count += 1
            record = {
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "question_number": question_number,
                "sample_index": sample_index,
                "domain": domain,
                "question": question,
                "model_answer": answer,
                "qwen_answer": answer,
                "model_name": MODEL_NAME,
                "duplicate_answer": True,
                "gemini_judgement": "Overall label: Not Hallucinating",
                "prompt_pack_version": prompt_pack_version(),
                "prompt_ids": prompt_ids(),
            }
            write(SEEDS_PATH, record, SEEDS_PATH.exists())
            print(f"{progress}: duplicate of an earlier sample, not judged")
            continue
        label, reason, judge_raw = judge_seed(question, answer)
        label_counts[label] += 1
        record = {
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "question_number": question_number,
            "sample_index": sample_index,
            "domain": domain,
            "question": question,
            "model_answer": answer,
            "qwen_answer": answer,
            "model_name": MODEL_NAME,
            "judge_model_name": active_judge_model(),
            "enable_thinking": ENABLE_THINKING,
            "max_new_tokens": SEED_MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "rng_seed": rng_seed,
            "duplicate_answer": False,
            "gemini_judgement": f"Overall label: {label}",
            "judge_reason": reason,
            "judge_raw": judge_raw,
            "prompt_pack_version": prompt_pack_version(),
            "prompt_ids": prompt_ids(),
        }
        record["seed_id"] = seed_identifier(record)
        if features:
            record.update({f"gen_{name}": value for name, value in features.items()})
        write(SEEDS_PATH, record, True)
        print(f"{progress}: {label}" + (f" - {reason[:80]}" if reason else ""))

    total = sum(label_counts.values())
    hallucinating = label_counts["Hallucinating"]
    print(f"\nJudged {total} answers: {hallucinating} hallucinating, {total - hallucinating} clean")
    if total:
        print(f"Hallucination rate: {hallucinating / total:.1%}")
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate answer(s).")
    print(f"Wrote {SEEDS_PATH}")
    if pilot:
        write_pilot_stage("seeds", n=len(question_items), judged=total, model=MODEL_NAME)
        print("Recorded 10-example seed-prompt debug in forecasting/results/pilot.json")
        print(f"\nNext: python forecasting/pipeline.py tree --pilot --seeds {SEEDS_PATH} --resume")
    else:
        print(
            f"\nNext: python forecasting/pipeline.py tree --pilot --seeds {SEEDS_PATH} --resume"
            "\nThen scale: python forecasting/pipeline.py tree --seeds "
            f"{SEEDS_PATH} --max-seeds 100 --levels 2 --resume"
        )


if __name__ == "__main__":
    main()
