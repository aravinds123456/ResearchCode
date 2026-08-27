"""HalluHard grounding adapter.

Claim extraction is a separate stage. This module verifies exactly the
atomic proposition it is given. If retrieval yields no usable evidence,
the result is UNVERIFIABLE. There is no OpenAI websearch fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from halluhard.judging_pipeline.core.queue import MonitoredQueue, QueueItem
from halluhard.judging_pipeline.models.work_items import (
    ClaimItem,
    ContentItem,
    FilteredContent,
    PDFResult,
    PDFTask,
    SearchTask,
)
from halluhard.judging_pipeline.strategies import get_strategy
from libs.types import SamplerBase

from .config import REPO_ROOT, fill_prompt, prompt_hash
from .json_parse import ParseError, parse_json_object
from .models import complete, sampler_metadata
from .schemas import (
    DOMAIN_TASK,
    ParseStatus,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

FORMAT_REMINDER = (
    "FORMAT REMINDER: Return JSON only with keys status, reason, "
    "evidence_passages, sources. status must be one of "
    "VERIFIED_FALSE, SUPPORTED, AMBIGUOUS, UNVERIFIABLE. "
    "Do not mark VERIFIED_FALSE merely because a specific claim is unsupported."
)


def _dummy_queue(name: str) -> MonitoredQueue:
    return MonitoredQueue(name)


def _queue_item(data, claim_id: str = "", conversation_id: int = 0) -> QueueItem:
    return QueueItem(data=data, claim_id=claim_id, conversation_id=conversation_id)


def _claim_item(claim: str, context: str, domain: str) -> ClaimItem:
    return ClaimItem(
        claim_id="verify",
        conversation_id=0,
        turn_number=0,
        data={
            "claimed_content": claim,
            "full_citation": "",
            "claimed_title": "",
            "claimed_authors": "",
            "claimed_year": "",
            "claimed_url": "",
            "context": context,
        },
        metadata={"domain": domain},
    )


def _has_usable_evidence(search: SearchTask | None, filtered: FilteredContent | None) -> bool:
    snippets = ""
    if search is not None:
        snippets = (search.search_results_text or "").strip()
    if filtered is not None:
        snippets = snippets or (filtered.search_results_text or "").strip()
        if (filtered.filtered_content or "").strip():
            return True
    if not snippets or snippets == "No search results found.":
        return False
    return True


def _sources_from_search(search: SearchTask | None) -> list[str]:
    if search is None:
        return []
    urls = list(search.urls_to_fetch or []) + list(search.pdf_urls or [])
    for result_set in search.search_results_raw or []:
        for organic in result_set.get("organic") or []:
            link = organic.get("link")
            if link and link not in urls:
                urls.append(link)
    return urls[:12]


async def _search(
    claim_item: ClaimItem,
    search_sampler: SamplerBase,
    domain: str,
    max_searches: int,
    grounding_task: str | None = None,
) -> SearchTask | None:
    from halluhard.judging_pipeline.workers.web_searcher import WebSearcherWorker

    task_name = grounding_task or DOMAIN_TASK.get(domain, "research_questions")
    strategy = get_strategy(task_name, REPO_ROOT)
    searcher = WebSearcherWorker(
        input_queue=_dummy_queue("claims"),
        output_queue=_dummy_queue("search"),
        search_sampler=search_sampler,
        claim_text_builder=strategy.build_textual_claim_for_websearch,
        strategy=strategy,
        max_searches=max_searches,
        num_workers=1,
    )
    await searcher.setup()
    try:
        return await searcher.process(claim_item, _queue_item(claim_item, claim_item.claim_id))
    finally:
        await searcher.teardown()


async def _fetch_and_filter(
    search: SearchTask,
    domain: str,
    grounding_task: str | None = None,
) -> FilteredContent:
    from halluhard.judging_pipeline.workers.content_filter import ContentFilterWorker
    from halluhard.judging_pipeline.workers.pdf_processor import PDFConverterWorker
    from halluhard.judging_pipeline.workers.web_fetcher import WebFetcherWorker

    task_name = grounding_task or DOMAIN_TASK.get(domain, "research_questions")
    strategy = get_strategy(task_name, REPO_ROOT)
    pdf_queue: MonitoredQueue[PDFTask] = _dummy_queue("pdf")
    fetcher = WebFetcherWorker(
        input_queue=_dummy_queue("search"),
        output_queue=_dummy_queue("content"),
        pdf_queue=pdf_queue,
        num_workers=1,
    )
    await fetcher.setup()
    try:
        content = await fetcher.process(search, _queue_item(search, search.claim_id))
    finally:
        # BrowserFetcher has no required teardown beyond process completion.
        pass
    if content is None:
        content = ContentItem(
            claim_id=search.claim_id,
            conversation_id=search.conversation_id,
            claim=search.claim,
            search_results_text=search.search_results_text,
            queries=search.queries_executed,
        )
    pdf_contents = []
    converter = PDFConverterWorker(
        input_queue=_dummy_queue("pdf_in"),
        output_queue=_dummy_queue("pdf_out"),
        num_workers=1,
    )
    while not pdf_queue.empty:
        wrapper = await pdf_queue.get()
        pdf_task: PDFTask = wrapper.data
        result: PDFResult = await converter.process(pdf_task, _queue_item(pdf_task, pdf_task.claim_id))
        if result.success and result.content:
            pdf_contents.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": "",
                    "content": result.content,
                }
            )
    content.pdf_contents = pdf_contents
    filt = ContentFilterWorker(
        input_queue=_dummy_queue("agg"),
        output_queue=_dummy_queue("filtered"),
        claim_text_builder=strategy.build_textual_claim_for_judging,
        num_workers=1,
    )
    return await filt.process(content, _queue_item(content, content.claim_id))


def _status_from_payload(payload: dict[str, Any]) -> VerificationStatus | None:
    raw = str(payload.get("status") or "").strip().upper()
    try:
        return VerificationStatus(raw)
    except ValueError:
        return None


async def _judge_with_evidence(
    claim: str,
    context: str,
    search: SearchTask | None,
    filtered: FilteredContent | None,
    verifier: SamplerBase,
) -> tuple[dict[str, Any] | None, str, ParseStatus, dict[str, Any]]:
    queries = []
    search_results = ""
    filtered_content = ""
    if search is not None:
        queries = list(search.queries_executed or [])
        search_results = search.search_results_text or ""
    if filtered is not None:
        queries = list(filtered.queries or queries)
        search_results = filtered.search_results_text or search_results
        filtered_content = filtered.filtered_content or ""
    prompt = fill_prompt(
        "grounded_verifier",
        claim=claim,
        context=context,
        queries="\n".join(f"- {q}" for q in queries) or "(none)",
        search_results=search_results or "(none)",
        filtered_content=filtered_content or "(none)",
    )
    messages = [{"role": "user", "content": prompt}]
    response = await complete(verifier, messages)
    status = ParseStatus.OK
    raw = response.response_text
    try:
        payload = parse_json_object(raw)
        if _status_from_payload(payload) is None:
            raise ParseError("missing/invalid status")
    except ParseError:
        reminder = messages + [{"role": "user", "content": FORMAT_REMINDER + "\n\nPrevious output:\n" + raw}]
        response = await complete(verifier, reminder)
        raw = response.response_text
        status = ParseStatus.RETRIED
        try:
            payload = parse_json_object(raw)
            if _status_from_payload(payload) is None:
                raise ParseError("missing/invalid status")
        except ParseError:
            return None, raw, ParseStatus.FAILED, sampler_metadata(verifier, response)
    return payload, raw, status, sampler_metadata(verifier, response)


async def verify_claim(
    claim: str,
    context: str = "",
    *,
    domain: str = "research",
    search_sampler: SamplerBase,
    verifier: SamplerBase,
    method: str = "halluhard_webscraper",
    max_searches: int = 3,
    claim_id: str = "",
    grounding_task: str | None = None,
) -> VerificationResult:
    """Verify exactly this atomic proposition with retrieved evidence."""
    claim_item = _claim_item(claim, context, domain)
    task_name = grounding_task or DOMAIN_TASK.get(domain, "research_questions")
    search: SearchTask | None = None
    filtered: FilteredContent | None = None
    try:
        search = await _search(
            claim_item, search_sampler, domain, max_searches, grounding_task=task_name
        )
    except Exception as error:
        logger.warning("HalluHard search failed: %s", error)
        return VerificationResult(
            claim_id=claim_id,
            status=VerificationStatus.UNVERIFIABLE,
            reason=f"search failure: {error}",
            verification_method=method,
            parse_status=ParseStatus.FAILED,
            verifier_metadata={"error": str(error)},
        )

    if method == "halluhard_webscraper" and search is not None:
        try:
            filtered = await _fetch_and_filter(search, domain, grounding_task=task_name)
        except Exception as error:
            logger.warning("HalluHard fetch/filter failed, using snippets only: %s", error)
            filtered = FilteredContent(
                claim_id=claim_item.claim_id,
                conversation_id=0,
                claim=claim_item,
                filtered_content="",
                search_results_text=search.search_results_text if search else "",
                queries=search.queries_executed if search else [],
                use_fallback=False,
            )

    if not _has_usable_evidence(search, filtered):
        return VerificationResult(
            claim_id=claim_id,
            status=VerificationStatus.UNVERIFIABLE,
            reason="No usable retrieved evidence. Not falling back to an ungrounded or websearch judge.",
            queries=list(search.queries_executed) if search else [],
            sources=_sources_from_search(search),
            verification_method=method,
            parse_status=ParseStatus.OK,
            verifier_metadata={"no_evidence": True},
        )

    payload, raw, parse_status, meta = await _judge_with_evidence(
        claim, context, search, filtered, verifier
    )
    meta["prompt_hash"] = prompt_hash("grounded_verifier")
    meta["raw"] = raw[:4000]
    if parse_status == ParseStatus.FAILED or payload is None:
        return VerificationResult(
            claim_id=claim_id,
            status=VerificationStatus.UNVERIFIABLE,
            reason="Verifier output was unparseable and was not mapped to a scientific label.",
            queries=list(search.queries_executed) if search else [],
            sources=_sources_from_search(search),
            verification_method=method,
            parse_status=ParseStatus.FAILED,
            verifier_metadata=meta,
        )
    status = _status_from_payload(payload)
    assert status is not None
    passages = payload.get("evidence_passages") or []
    if isinstance(passages, str):
        passages = [passages]
    sources = payload.get("sources") or _sources_from_search(search)
    if isinstance(sources, str):
        sources = [sources]
    return VerificationResult(
        claim_id=claim_id,
        status=status,
        reason=str(payload.get("reason") or ""),
        queries=list(search.queries_executed) if search else [],
        sources=[str(item) for item in sources],
        evidence_passages=[str(item) for item in passages],
        verification_method=method,
        verifier_model=str(meta.get("deployment") or meta.get("model") or ""),
        parse_status=parse_status,
        verifier_metadata=meta,
    )
