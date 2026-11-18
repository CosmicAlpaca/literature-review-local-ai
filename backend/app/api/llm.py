from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.services.llm_client import (
    LLMUnavailableError,
    compact_payload_for_llm,
    current_prompt_version,
    estimate_llm_usage,
    safe_fallback_reason,
)
from app.services.llm_query import fallback_query_rewrite, rewrite_query
from app.services.llm_reranker import fallback_rerank, rerank_candidates
from app.services.local_reranker import safe_local_reranker_fallback_reason, score_candidates
from app.services.paper_summary import fallback_paper_summary, summarize_paper
from app.services.reranker import reranker_model_loaded

router = APIRouter(prefix="/llm", tags=["llm"])

MAX_PREVIEW_RERANK_CANDIDATES = 10
MAX_LOCAL_RERANK_PREVIEW_CANDIDATES = 20


class EstimatedUsageResponse(BaseModel):
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_call_count: int
    input_truncated: bool = False


class QueryPreviewRequest(BaseModel):
    topic: str | None = None
    raw_query: str = Field(min_length=1, max_length=1000)


class QueryPreviewResponse(BaseModel):
    original_query: str
    optimized_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    notes: str
    fallback_reason: str | None = None
    provider: str
    model: str
    prompt_version: str | None = None
    estimated_usage: EstimatedUsageResponse | None = None


class RerankPreviewCandidate(BaseModel):
    id: str | int | None = None
    title: str = Field(min_length=1, max_length=1000)
    abstract: str | None = Field(default=None, max_length=8000)
    source_title: str | None = Field(default=None, max_length=1000)
    year: int | None = None
    citation_count: int | None = None


class RerankPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    candidates: list[RerankPreviewCandidate] = Field(default_factory=list)


class RerankPreviewResultResponse(BaseModel):
    candidate_index: int
    candidate_id: str | int | None = None
    relevance_score: float
    reason: str


class RerankPreviewResponse(BaseModel):
    results: list[RerankPreviewResultResponse] = Field(default_factory=list)
    fallback_reason: str | None = None
    provider: str
    model: str
    prompt_version: str | None = None
    estimated_usage: EstimatedUsageResponse | None = None


class SummaryPreviewPaper(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    abstract: str | None = Field(default=None, max_length=8000)
    source_title: str | None = Field(default=None, max_length=1000)
    year: int | None = None


class SummaryPreviewRequest(BaseModel):
    query: str | None = Field(default=None, max_length=1000)
    paper: SummaryPreviewPaper


class SummaryPreviewResponse(BaseModel):
    one_sentence_summary: str
    contribution: str
    method_or_data: str
    relevance_to_query: str
    limitations_or_caution: str
    fallback_reason: str | None = None
    provider: str
    model: str
    prompt_version: str | None = None
    estimated_usage: EstimatedUsageResponse | None = None


class LocalRerankPreviewCandidate(BaseModel):
    candidate_id: str | int
    title: str | None = Field(default=None, max_length=1000)
    abstract: str | None = Field(default=None, max_length=8000)
    source_title: str | None = Field(default=None, max_length=1000)
    year: int | None = None
    citation_count: int | None = None
    doi: str | None = Field(default=None, max_length=255)
    provider: str | None = Field(default=None, max_length=100)


class LocalRerankPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    candidates: list[LocalRerankPreviewCandidate] = Field(default_factory=list)


class LocalRerankPreviewResultResponse(BaseModel):
    candidate_id: str | int | None = None
    local_relevance_score: float
    reason: str
    fallback_reason: str | None = None


class LocalRerankPreviewResponse(BaseModel):
    enabled: bool
    model_version: str
    model_provider: str
    results: list[LocalRerankPreviewResultResponse] = Field(default_factory=list)
    fallback_reason: str | None = None


def _api_key_configured() -> bool:
    return bool(settings.llm_api_key)


def _provider_usable() -> bool:
    provider = settings.llm_provider
    if provider == "mock":
        return True
    if provider == "openai_compatible":
        return _api_key_configured()
    return False


def _any_llm_step_enabled() -> bool:
    return any(
        [
            settings.llm_query_rewrite_enabled,
            settings.llm_rerank_enabled,
            settings.llm_summary_enabled,
        ]
    )


def _enabled_steps() -> list[str]:
    steps = []
    if settings.llm_query_rewrite_enabled:
        steps.append("query_rewrite")
    if settings.llm_rerank_enabled:
        steps.append("rerank")
    if settings.llm_summary_enabled:
        steps.append("summary")
    return steps


def _readiness_reason() -> str | None:
    if not settings.llm_enabled:
        return "llm_disabled"
    if settings.llm_provider == "openai_compatible" and not _api_key_configured():
        return "missing_api_key"
    if not _provider_usable():
        return "provider_unavailable"
    if not _any_llm_step_enabled():
        return "step_disabled"
    return None


def _disabled_reason() -> str | None:
    if not settings.llm_enabled:
        return "AI assistance is disabled on this deployment."
    if settings.llm_provider == "openai_compatible" and not _api_key_configured():
        return "OpenAI-compatible AI assistance is configured without an API key."
    if not _provider_usable():
        return "Configured AI provider is not supported."
    if not _any_llm_step_enabled():
        return "AI assistance is configured, but no AI search steps are enabled."
    return None


def _preview_rerank_candidate_limit() -> int:
    configured_limit = int(getattr(settings, "llm_max_candidates_for_rerank", 10) or 10)
    return max(1, min(configured_limit, MAX_PREVIEW_RERANK_CANDIDATES))


def _preview_local_rerank_candidate_limit() -> int:
    configured_limit = int(getattr(settings, "local_reranker_top_k", 20) or 20)
    return max(1, min(configured_limit, MAX_LOCAL_RERANK_PREVIEW_CANDIDATES))


def _step_unavailable_reason(step_enabled: bool) -> str | None:
    if not settings.llm_enabled:
        return "llm_disabled"
    if not step_enabled:
        return "step_disabled"
    return None


def _usage_response(metadata: dict[str, Any] | None) -> EstimatedUsageResponse | None:
    if not metadata:
        return None

    usage_keys = {
        "estimated_input_tokens",
        "estimated_output_tokens",
        "estimated_total_tokens",
        "estimated_call_count",
        "input_truncated",
    }
    if not any(metadata.get(key) for key in usage_keys):
        return None

    return EstimatedUsageResponse(
        estimated_input_tokens=int(metadata.get("estimated_input_tokens") or 0),
        estimated_output_tokens=int(metadata.get("estimated_output_tokens") or 0),
        estimated_total_tokens=int(metadata.get("estimated_total_tokens") or 0),
        estimated_call_count=int(metadata.get("estimated_call_count") or 0),
        input_truncated=bool(metadata.get("input_truncated")),
    )


def _budget_guard(
    payload: dict[str, Any],
    *,
    default_output_tokens: int = 512,
) -> tuple[str | None, EstimatedUsageResponse | None]:
    if not settings.llm_enabled:
        return None, None

    try:
        compacted_payload, input_truncated = compact_payload_for_llm(
            payload,
            max_chars=getattr(settings, "llm_max_input_chars_per_call", 12000),
        )
    except LLMUnavailableError as exc:
        return safe_fallback_reason(exc.reason), None
    except Exception:
        return "input_too_large", None

    usage = estimate_llm_usage(
        input_payload=compacted_payload,
        default_output_tokens=default_output_tokens,
        input_truncated=input_truncated,
    )
    usage_response = _usage_response(usage.to_metadata())

    if usage.estimated_call_count > max(int(getattr(settings, "llm_max_calls_per_search", 20) or 0), 0):
        return "max_calls_exceeded", usage_response

    if (
        getattr(settings, "llm_usage_estimation_enabled", True)
        and usage.estimated_total_tokens
        > max(int(getattr(settings, "llm_max_estimated_tokens_per_search", 20000) or 0), 0)
    ):
        return "estimated_tokens_exceeded", usage_response

    return None, usage_response


def _first_usage_response(items: list[Any]) -> EstimatedUsageResponse | None:
    for item in items:
        usage = _usage_response(item.to_metadata())
        if usage:
            return usage
    return None


def _query_preview_response(
    result,
    *,
    estimated_usage: EstimatedUsageResponse | None = None,
) -> QueryPreviewResponse:
    return QueryPreviewResponse(
        original_query=result.original_query,
        optimized_query=result.optimized_query,
        expanded_queries=result.expanded_queries,
        key_terms=result.key_terms,
        notes=result.notes,
        fallback_reason=safe_fallback_reason(result.fallback_reason) if result.fallback_reason else None,
        provider=result.provider,
        model=result.model,
        prompt_version=result.prompt_version,
        estimated_usage=estimated_usage or _usage_response(result.to_metadata()),
    )


def _rerank_preview_response(
    results,
    *,
    fallback_reason: str | None = None,
    estimated_usage: EstimatedUsageResponse | None = None,
) -> RerankPreviewResponse:
    first = results[0] if results else None
    result_fallback_reason = fallback_reason or (first.fallback_reason if first else None)
    return RerankPreviewResponse(
        results=[
            RerankPreviewResultResponse(
                candidate_index=result.candidate_index,
                candidate_id=result.candidate_id,
                relevance_score=max(0.0, min(float(result.llm_relevance_score), 1.0)),
                reason=result.reason,
            )
            for result in results
        ],
        fallback_reason=safe_fallback_reason(result_fallback_reason) if result_fallback_reason else None,
        provider=first.provider if first else "fallback",
        model=first.model if first else "none",
        prompt_version=first.prompt_version if first else current_prompt_version(),
        estimated_usage=estimated_usage or _first_usage_response(results),
    )


def _summary_preview_response(
    result,
    *,
    estimated_usage: EstimatedUsageResponse | None = None,
) -> SummaryPreviewResponse:
    return SummaryPreviewResponse(
        one_sentence_summary=result.one_sentence_summary,
        contribution=result.contribution,
        method_or_data=result.method_or_data,
        relevance_to_query=result.relevance_to_query,
        limitations_or_caution=result.limitations_or_caution,
        fallback_reason=safe_fallback_reason(result.fallback_reason) if result.fallback_reason else None,
        provider=result.provider,
        model=result.model,
        prompt_version=result.prompt_version,
        estimated_usage=estimated_usage or _usage_response(result.to_metadata()),
    )


def _local_rerank_preview_response(result) -> LocalRerankPreviewResponse:
    return LocalRerankPreviewResponse(
        enabled=bool(result.enabled),
        model_version=result.model_version,
        model_provider=result.model_provider,
        results=[
            LocalRerankPreviewResultResponse(
                candidate_id=item.candidate_id,
                local_relevance_score=max(0.0, min(float(item.local_relevance_score), 1.0)),
                reason=item.reason,
                fallback_reason=(
                    safe_local_reranker_fallback_reason(item.fallback_reason)
                    if item.fallback_reason
                    else None
                ),
            )
            for item in result.results
        ],
        fallback_reason=(
            safe_local_reranker_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
    )


@router.get("/status")
def llm_status():
    provider_usable = _provider_usable()
    steps_enabled = _any_llm_step_enabled()
    available = bool(settings.llm_enabled and provider_usable and steps_enabled)
    reason = _readiness_reason()

    return {
        "llm_enabled": settings.llm_enabled,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "query_rewrite_enabled": settings.llm_query_rewrite_enabled,
        "rerank_enabled": settings.llm_rerank_enabled,
        "summary_enabled": settings.llm_summary_enabled,
        "enabled_steps": _enabled_steps(),
        "api_key_configured": _api_key_configured(),
        "available": available,
        "ready_for_ai_search": available,
        "reason": safe_fallback_reason(reason) if reason else None,
        "disabled_reason": _disabled_reason(),
        "reranker_enabled": bool(settings.reranker_enabled),
        "reranker_model_name": settings.reranker_model_name,
        "reranker_max_candidates": settings.reranker_max_candidates,
        "reranker_available": (
            bool(settings.reranker_enabled)
            and reranker_model_loaded(settings.reranker_model_name)
        ),
        "reranker_ready": reranker_model_loaded(settings.reranker_model_name),
    }


@router.post(
    "/query-preview",
    response_model=QueryPreviewResponse,
    response_model_exclude_none=True,
)
def query_preview(
    payload: QueryPreviewRequest,
    _current_user: Any = Depends(get_current_user),
):
    step_reason = _step_unavailable_reason(settings.llm_query_rewrite_enabled)
    if step_reason:
        result = fallback_query_rewrite(
            raw_query=payload.raw_query,
            topic=payload.topic,
            attempted=True,
            fallback_reason=step_reason,
        )
        return _query_preview_response(result)

    budget_reason, estimated_usage = _budget_guard(
        {"raw_query": payload.raw_query, "topic": payload.topic},
    )
    if budget_reason:
        result = fallback_query_rewrite(
            raw_query=payload.raw_query,
            topic=payload.topic,
            attempted=True,
            fallback_reason=budget_reason,
        )
        return _query_preview_response(result, estimated_usage=estimated_usage)

    try:
        result = rewrite_query(raw_query=payload.raw_query, topic=payload.topic)
    except Exception:
        result = fallback_query_rewrite(
            raw_query=payload.raw_query,
            topic=payload.topic,
            attempted=True,
            fallback_reason="provider_unavailable",
        )
    return _query_preview_response(result)


@router.post(
    "/rerank-preview",
    response_model=RerankPreviewResponse,
    response_model_exclude_none=True,
)
def rerank_preview(
    payload: RerankPreviewRequest,
    _current_user: Any = Depends(get_current_user),
):
    candidate_limit = _preview_rerank_candidate_limit()
    if len(payload.candidates) > candidate_limit:
        raise HTTPException(
            status_code=422,
            detail=f"Preview rerank supports at most {candidate_limit} candidates.",
        )

    candidates = [candidate.model_dump() for candidate in payload.candidates]
    if not candidates:
        return _rerank_preview_response(
            [],
            fallback_reason="no_candidates",
        )

    step_reason = _step_unavailable_reason(settings.llm_rerank_enabled)
    if step_reason:
        results = fallback_rerank(
            query=payload.query,
            candidates=candidates,
            attempted=True,
            fallback_reason=step_reason,
        )
        return _rerank_preview_response(results)

    budget_reason, estimated_usage = _budget_guard(
        {
            "query": payload.query,
            "candidates": [
                {
                    "candidate_index": index,
                    "candidate_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "abstract": candidate.get("abstract"),
                    "source_title": candidate.get("source_title"),
                    "year": candidate.get("year"),
                }
                for index, candidate in enumerate(candidates)
            ],
        },
        default_output_tokens=768,
    )
    if budget_reason:
        results = fallback_rerank(
            query=payload.query,
            candidates=candidates,
            attempted=True,
            fallback_reason=budget_reason,
        )
        return _rerank_preview_response(results, estimated_usage=estimated_usage)

    try:
        results = rerank_candidates(query=payload.query, candidates=candidates)
    except Exception:
        results = fallback_rerank(
            query=payload.query,
            candidates=candidates,
            attempted=True,
            fallback_reason="provider_unavailable",
        )
    return _rerank_preview_response(results)


@router.post(
    "/summary-preview",
    response_model=SummaryPreviewResponse,
    response_model_exclude_none=True,
)
def summary_preview(
    payload: SummaryPreviewRequest,
    _current_user: Any = Depends(get_current_user),
):
    paper = payload.paper.model_dump()
    step_reason = _step_unavailable_reason(settings.llm_summary_enabled)
    if step_reason:
        result = fallback_paper_summary(
            candidate=paper,
            query=payload.query,
            attempted=True,
            fallback_reason=step_reason,
        )
        return _summary_preview_response(result)

    budget_reason, estimated_usage = _budget_guard(
        {
            "query": payload.query,
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "source_title": paper.get("source_title"),
            "year": paper.get("year"),
        },
    )
    if budget_reason:
        result = fallback_paper_summary(
            candidate=paper,
            query=payload.query,
            attempted=True,
            fallback_reason=budget_reason,
        )
        return _summary_preview_response(result, estimated_usage=estimated_usage)

    try:
        result = summarize_paper(candidate=paper, query=payload.query)
    except Exception:
        result = fallback_paper_summary(
            candidate=paper,
            query=payload.query,
            attempted=True,
            fallback_reason="provider_unavailable",
        )
    return _summary_preview_response(result)


@router.post(
    "/local-rerank-preview",
    response_model=LocalRerankPreviewResponse,
    response_model_exclude_none=True,
)
def local_rerank_preview(
    payload: LocalRerankPreviewRequest,
    _current_user: Any = Depends(get_current_user),
):
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required.")

    candidate_limit = _preview_local_rerank_candidate_limit()
    if len(payload.candidates) > candidate_limit:
        raise HTTPException(
            status_code=422,
            detail=f"Local rerank preview supports at most {candidate_limit} candidates.",
        )

    candidates = [candidate.model_dump() for candidate in payload.candidates]
    try:
        result = score_candidates(payload.query, candidates)
    except Exception:
        result = score_candidates("", candidates)
    return _local_rerank_preview_response(result)
