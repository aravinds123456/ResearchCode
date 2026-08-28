"""Stage A: generate seed answers and extract atomic candidate claims."""

from __future__ import annotations

from .concurrency import bounded_map, clamp_concurrency
from .config import DatasetConfig, fill_prompt, prompt_hash
from .json_parse import ParseError, parse_json_list
from .models import complete, sampler_metadata
from .questions import load_questions
from .schemas import (
    AtomicityStatus,
    CandidateClaim,
    GeneratedSeed,
    make_claim_id,
    make_seed_id,
)
from .storage import RunStore
from libs.types import SamplerBase


async def generate_seeds(
    store: RunStore,
    *,
    answer_model: SamplerBase,
    domains: tuple[str, ...],
    n_seeds: int,
    max_questions: int | None = None,
    samples_per_question: int = 1,
    answer_name: str = "",
    dataset: DatasetConfig | None = None,
    concurrency: int = 8,
) -> list[GeneratedSeed]:
    store.ensure()
    done = store.completed_ids(store.generated_seeds_path, "seed_id")
    questions = load_questions(dataset, domains=domains, max_questions=max_questions)
    pending: list[tuple[dict, int, str]] = []
    for question in questions:
        for sample_index in range(samples_per_question):
            seed_id = make_seed_id(question["question_id"], sample_index)
            if seed_id in done:
                continue
            pending.append((question, sample_index, seed_id))

    limit = clamp_concurrency(concurrency)
    created: list[GeneratedSeed] = []
    cursor = 0

    async def _one(item: tuple[dict, int, str]) -> GeneratedSeed | None:
        question, sample_index, seed_id = item
        response = await complete(
            answer_model,
            [{"role": "user", "content": question["question"]}],
        )
        text = (response.response_text or "").strip()
        if not text:
            return None
        return GeneratedSeed(
            seed_id=seed_id,
            question_id=question["question_id"],
            domain=question["domain"],
            question=question["question"],
            seed_answer=text,
            answer_model=answer_name or str(getattr(answer_model, "deployment", None) or getattr(answer_model, "model", "")),
            generation_metadata={
                **sampler_metadata(answer_model, response),
                "dataset": question.get("source") or (dataset.name if dataset else "halluhard"),
                **{
                    key: question[key]
                    for key in ("topic", "tier")
                    if question.get(key)
                },
            },
        )

    while len(done) + len(created) < n_seeds and cursor < len(pending):
        need = n_seeds - len(done) - len(created)
        take = min(limit, need, len(pending) - cursor)
        batch = pending[cursor : cursor + take]
        cursor += take
        results = await bounded_map(batch, _one, concurrency=take)
        for seed in results:
            if seed is None:
                continue
            if len(done) + len(created) >= n_seeds:
                break
            store.append_generated(seed)
            created.append(seed)
    return created


async def extract_claims_for_seed(
    seed: GeneratedSeed,
    extractor: SamplerBase,
    max_claims: int = 8,
) -> list[CandidateClaim]:
    prompt = fill_prompt(
        "claim_extraction",
    ) + (
        f"\n\nUSER QUESTION:\n{seed.question}\n\nASSISTANT ANSWER:\n{seed.seed_answer}\n"
    )
    response = await complete(extractor, [{"role": "user", "content": prompt}])
    try:
        items = parse_json_list(response.response_text)
    except ParseError:
        reminder = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "FORMAT REMINDER: Return JSON {\"claims\": [...]} only. "
                "Do not decide truth. If none, return {\"claims\": []}.",
            },
        ]
        response = await complete(extractor, reminder)
        try:
            items = parse_json_list(response.response_text)
        except ParseError:
            return []
    claims: list[CandidateClaim] = []
    meta = sampler_metadata(extractor, response)
    meta["prompt_hash"] = prompt_hash("claim_extraction")
    for index, item in enumerate(items[:max_claims]):
        if isinstance(item, str):
            text = item.strip()
            source_span = ""
            entities: list[str] = []
            atomicity = AtomicityStatus.UNKNOWN
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("claim") or "").strip()
            source_span = str(item.get("source_span") or "")
            entities = [str(e) for e in (item.get("entities") or [])]
            try:
                atomicity = AtomicityStatus(str(item.get("atomicity_status") or "unknown"))
            except ValueError:
                atomicity = AtomicityStatus.UNKNOWN
        else:
            continue
        if not text:
            continue
        claims.append(
            CandidateClaim(
                claim_id=make_claim_id(seed.seed_id, index),
                seed_id=seed.seed_id,
                text=text,
                source_span=source_span,
                entities=entities,
                atomicity_status=atomicity,
                extractor_metadata=meta,
            )
        )
    return claims


async def extract_claims_for_seeds(
    seeds: list[GeneratedSeed],
    extractor: SamplerBase,
    *,
    max_claims: int = 8,
    concurrency: int = 8,
) -> list[tuple[GeneratedSeed, list[CandidateClaim]]]:
    async def _one(seed: GeneratedSeed) -> tuple[GeneratedSeed, list[CandidateClaim]]:
        claims = await extract_claims_for_seed(seed, extractor, max_claims=max_claims)
        return seed, claims

    return await bounded_map(seeds, _one, concurrency=clamp_concurrency(concurrency))
