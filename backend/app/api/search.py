from datetime import datetime
import inspect
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_db, get_current_user
from app.core.guardrails import (
    MAX_QUERY_LENGTH,
    check_rate_limit,
    sanitize_query,
    sanitize_source_filter,
)
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.db.models.search_query_context import SearchQueryContext
from app.db.models.user import User
from app.schemas.session import (
    CandidateResponse,
    CandidateSummaryResponse,
    LLMSummaryResponse,
    ReviewDecisionRequest,
    SearchQueryContextResponse,
    SearchQueryContextStoredResponse,
    SearchResponse,
)
from app.services.llm_client import fallback_reason_from_exception, get_llm_client, safe_fallback_reason
from app.services.orchestrator import (
    SearchFilterError,
    SearchFilters,
    SearchLLMOptions,
    normalize_search_filters,
    run_search_pipeline,
)
from app.services.paper_summary import (
    cached_paper_summary,
    fallback_paper_summary,
    local_extractive_summary,
    store_paper_summary_metadata,
    summarize_paper,
)
from app.services.paper_enrichment import enrich_candidate_abstract, has_summarizable_abstract

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)

_REVIEW_STATUSES = {"shortlisted", "dismissed"}
_QUERY_CONTEXT_FIELDS = {
    "status",
    "original_query",
    "optimized_query",
    "effective_query",
    "steps",
    "expanded_queries",
    "key_terms",
    "query_rewrite_notes",
    "llm_enabled",
    "llm_requested",
    "use_llm_requested",
    "llm_available",
    "llm_used",
    "llm_steps",
    "llm_query_rewrite_used",
    "llm_disabled_reason",
    "warnings",
    "errors",
    "empty_result_retry_used",
    "retry_queries",
    "final_search_query",
    "applied_filters",
}


def _get_user_session_or_404(db: Session, session_id: int, user_id: int) -> ReviewSession:
    session_obj = (
        db.query(ReviewSession)
        .filter(ReviewSession.id == session_id, ReviewSession.user_id == user_id)
        .first()
    )
    if not session_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session_obj


def _candidate_metadata(candidate: PaperCandidate) -> dict:
    metadata = getattr(candidate, "raw_metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def _review_status(candidate: PaperCandidate) -> str | None:
    review = _candidate_metadata(candidate).get("_review")
    if not isinstance(review, dict):
        return None
    status_value = review.get("status")
    return status_value if status_value in _REVIEW_STATUSES else None


def _set_review_status(candidate: PaperCandidate, review_status: str) -> None:
    metadata = dict(_candidate_metadata(candidate))
    metadata["_review"] = {
        "status": review_status,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    candidate.raw_metadata_json = metadata


def _clear_review_status(candidate: PaperCandidate) -> None:
    metadata = dict(_candidate_metadata(candidate))
    metadata.pop("_review", None)
    candidate.raw_metadata_json = metadata


def _safe_string(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _safe_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return None


def _safe_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        safe_item = _safe_string(item)
        if safe_item and safe_item not in cleaned:
            cleaned.append(safe_item)
    return cleaned


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _safe_query_context_payload(query_context: SearchQueryContextResponse | None) -> dict:
    if query_context is None:
        return {}
    raw_payload = _model_dump(query_context)
    return {
        key: _json_safe_value(value)
        for key, value in raw_payload.items()
        if key in _QUERY_CONTEXT_FIELDS
    }


def _persist_query_context(
    db: Session,
    session_id: int,
    query_context: SearchQueryContextResponse | None,
) -> None:
    payload = _safe_query_context_payload(query_context)
    if not payload:
        return
    if not all(hasattr(db, attr) for attr in ("query", "add", "commit")):
        return

    try:
        existing = (
            db.query(SearchQueryContext)
            .filter(SearchQueryContext.session_id == session_id)
            .first()
        )
        now = datetime.utcnow()
        if existing:
            existing.payload_json = payload
            existing.updated_at = now
            db.add(existing)
        else:
            db.add(
                SearchQueryContext(
                    session_id=session_id,
                    payload_json=payload,
                    created_at=now,
                    updated_at=now,
                )
            )
        db.commit()
    except Exception as exc:  # pragma: no cover - defensive persistence boundary
        rollback = getattr(db, "rollback", None)
        if callable(rollback):
            rollback()
        logger.warning(
            "Skipping query_context persistence after search: session_id=%s reason=%s",
            session_id,
            exc,
        )


def _search_error_response(request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "search_failed",
                "message": "Search failed. Please try again.",
                "request_id": getattr(request.state, "request_id", None),
                "retryable": True,
            }
        },
    )


def _normalized_query_value(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _llm_query_rewrite_used(llm_query: dict, fallback_query: str) -> bool:
    if llm_query.get("used_llm") is False:
        return False
    original = _normalized_query_value(llm_query.get("original_query") or fallback_query)
    optimized = _normalized_query_value(llm_query.get("optimized_query"))
    if optimized and optimized != original:
        return True
    for value in _safe_string_list(llm_query.get("expanded_queries")):
        if _normalized_query_value(value) and _normalized_query_value(value) != original:
            return True
    return False


def _query_context_from_metadata(
    metadata: dict,
    *,
    query: str,
    llm_options: SearchLLMOptions,
    applied_filters: dict | None,
    llm_available: bool,
    disabled_reason: str | None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    steps: list[str] | None = None,
) -> SearchQueryContextResponse:
    rewrite_used = bool(metadata.get("llm_query_rewrite_used"))
    if not rewrite_used:
        rewrite_used = _llm_query_rewrite_used(metadata, query)
    effective_query = _effective_query_from_metadata(metadata, query)
    llm_steps = ["query_rewrite"] if rewrite_used else []
    public_steps = _safe_context_string_list(metadata.get("steps")) or (steps or [])
    fallback_status = "ok"
    if errors:
        fallback_status = "partial"
    elif warnings == ["no_candidates"]:
        fallback_status = "empty"
    elif warnings:
        fallback_status = "partial"

    return SearchQueryContextResponse(
        status=_safe_context_status(metadata.get("status"), fallback=fallback_status),
        original_query=_safe_string(metadata.get("original_query")) or query,
        optimized_query=_safe_string(metadata.get("optimized_query")) or query,
        effective_query=effective_query,
        steps=public_steps,
        expanded_queries=_safe_string_list(metadata.get("expanded_queries")),
        key_terms=_safe_string_list(metadata.get("key_terms")),
        query_rewrite_notes=_safe_string(metadata.get("query_rewrite_notes") or metadata.get("notes")),
        llm_enabled=bool(settings.llm_enabled),
        llm_requested=_llm_requested(llm_options),
        use_llm_requested=_use_llm_requested(llm_options),
        llm_available=llm_available,
        llm_used=bool(llm_steps),
        llm_steps=llm_steps,
        llm_query_rewrite_used=rewrite_used,
        llm_disabled_reason=None if rewrite_used else disabled_reason,
        warnings=warnings or [],
        errors=errors or [],
        applied_filters=applied_filters,
        empty_result_retry_used=True if metadata.get("empty_result_retry_used") is True else None,
        retry_queries=(
            _safe_string_list(metadata.get("retry_queries"))
            if metadata.get("empty_result_retry_used") is True
            else None
        ),
        final_search_query=(
            _safe_string(metadata.get("final_search_query"))
            if metadata.get("empty_result_retry_used") is True
            else None
        ),
    )


def _get_candidate_for_session_or_404(
    db: Session,
    session_id: int,
    candidate_id: int,
) -> PaperCandidate:
    candidate = (
        db.query(PaperCandidate)
        .filter(PaperCandidate.session_id == session_id, PaperCandidate.id == candidate_id)
        .first()
    )
    if not candidate:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    return candidate


def _summary_fields_response(result) -> LLMSummaryResponse:
    return LLMSummaryResponse(
        one_sentence_summary=result.one_sentence_summary,
        contribution=result.contribution,
        method_or_data=result.method_or_data,
        relevance_to_query=result.relevance_to_query,
        limitations_or_caution=result.limitations_or_caution,
    )


def _extract_llm_summary(metadata: dict) -> LLMSummaryResponse | None:
    summary = metadata.get("_llm_summary")
    if not isinstance(summary, dict):
        return None

    safe_summary = LLMSummaryResponse(
        one_sentence_summary=_safe_string(summary.get("one_sentence_summary")),
        contribution=_safe_string(summary.get("contribution")) or _safe_string(summary.get("key_findings")),
        method_or_data=_safe_string(summary.get("method_or_data")) or _safe_string(summary.get("main_methodology")),
        relevance_to_query=_safe_string(summary.get("relevance_to_query")),
        limitations_or_caution=(
            _safe_string(summary.get("limitations_or_caution"))
            or _safe_string(summary.get("research_gap"))
            or _safe_string(summary.get("potential_contributions"))
        ),
    )
    if not any(
        [
            safe_summary.one_sentence_summary,
            safe_summary.contribution,
            safe_summary.method_or_data,
            safe_summary.relevance_to_query,
            safe_summary.limitations_or_caution,
        ]
    ):
        return None
    return safe_summary


def _candidate_summary_response(
    session_id: int,
    candidate_id: int,
    result,
    *,
    cached: bool,
) -> CandidateSummaryResponse:
    return CandidateSummaryResponse(
        session_id=session_id,
        candidate_id=candidate_id,
        summary=_summary_fields_response(result),
        cached=cached,
        llm_used=bool(result.used_llm),
        fallback_reason=(
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        prompt_version=result.prompt_version,
    )


def _source_match_debug(metadata: dict) -> dict:
    source_match = metadata.get("_source_match")
    return source_match if isinstance(source_match, dict) else {}


def _source_match_registry_metadata(source_match: dict) -> dict:
    registry_metadata = source_match.get("registry_metadata")
    return registry_metadata if isinstance(registry_metadata, dict) else {}


def _llm_requested(llm_options: SearchLLMOptions) -> bool:
    if llm_options.use_llm is False:
        return False
    return any(
        value is True
        for value in (
            llm_options.use_llm,
            llm_options.use_query_rewrite,
            llm_options.use_llm_rerank,
            llm_options.use_llm_summary,
        )
    )


def _use_llm_requested(llm_options: SearchLLMOptions) -> bool:
    return llm_options.use_llm is True


def _effective_query_from_metadata(metadata: dict | None, fallback_query: str) -> str:
    if not isinstance(metadata, dict):
        return fallback_query
    return (
        _safe_string(metadata.get("final_search_query"))
        or _safe_string(metadata.get("optimized_query"))
        or fallback_query
    )


def _safe_context_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        safe_item = _safe_string(item)
        if safe_item and safe_item not in cleaned:
            cleaned.append(safe_item)
    return cleaned


def _safe_context_status(value: Any, *, fallback: str = "ok") -> str:
    status_value = _safe_string(value)
    if status_value in {"ok", "partial", "empty", "running"}:
        return status_value
    return fallback


def _base_query_context(
    *,
    query: str,
    llm_options: SearchLLMOptions,
    status: str = "running",
    search_filters: SearchFilters | None = None,
) -> SearchQueryContextResponse:
    return SearchQueryContextResponse(
        status=_safe_context_status(status, fallback="running"),
        original_query=query,
        optimized_query=query,
        effective_query=query,
        steps=[],
        llm_enabled=bool(settings.llm_enabled),
        llm_requested=_llm_requested(llm_options),
        use_llm_requested=_use_llm_requested(llm_options),
        llm_available=False,
        llm_used=False,
        llm_steps=[],
        llm_query_rewrite_used=False,
        warnings=[],
        errors=[],
        applied_filters=search_filters.to_metadata() if search_filters else None,
    )


def _merge_context_metadata(
    search_filters: SearchFilters | None,
    base_context: SearchQueryContextResponse,
) -> dict:
    payload = _safe_query_context_payload(base_context)
    if search_filters is not None and isinstance(search_filters.query_context_metadata, dict):
        payload.update(
            {
                key: value
                for key, value in search_filters.query_context_metadata.items()
                if key in _QUERY_CONTEXT_FIELDS
            }
        )
    return payload


def _llm_steps_from_candidates(
    candidates: list[PaperCandidate],
    *,
    rewrite_used: bool,
) -> list[str]:
    steps: list[str] = []
    if rewrite_used:
        steps.append("query_rewrite")
    for candidate in candidates:
        metadata = _candidate_metadata(candidate)
        rerank = metadata.get("_llm_rerank")
        if isinstance(rerank, dict) and rerank.get("used_llm") is True and "rerank" not in steps:
            steps.append("rerank")
        summary = metadata.get("_llm_summary")
        if isinstance(summary, dict) and summary.get("used_llm") is True and "summary" not in steps:
            steps.append("summary")
    return steps


def _query_context_warnings(candidates: list[PaperCandidate]) -> list[str]:
    if not candidates:
        return ["no_candidates"]
    return []


def _query_context_diagnostic_signals(candidates: list[PaperCandidate]) -> tuple[list[str], list[str], list[str]]:
    steps: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    for candidate in candidates:
        diagnostics = _candidate_metadata(candidate).get("_llm_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        for step, diagnostic in diagnostics.items():
            step_name = _safe_string(step)
            if not step_name or not isinstance(diagnostic, dict):
                continue
            if step_name not in steps:
                steps.append(step_name)
            fallback_reason = _safe_string(diagnostic.get("fallback_reason"))
            attempted = diagnostic.get("attempted") is True
            used = diagnostic.get("used") is True
            if fallback_reason and attempted and not used:
                warning = f"{step_name}_{fallback_reason}"
                error = f"{step_name}:{fallback_reason}"
                if warning not in warnings:
                    warnings.append(warning)
                if error not in errors:
                    errors.append(error)
    return steps, warnings, errors


def _llm_available() -> bool:
    return bool(settings.llm_enabled and get_llm_client().is_available())


def _llm_disabled_reason(llm_options: SearchLLMOptions) -> str | None:
    if not _llm_requested(llm_options):
        return None
    if not settings.llm_enabled:
        return "AI assistance is disabled on this deployment."
    if not get_llm_client().is_available():
        return "AI assistance is not available with the current provider settings."
    if not any(
        [
            llm_options.query_rewrite_enabled(),
            llm_options.rerank_enabled(),
            llm_options.summary_enabled(),
        ]
    ):
        return "AI assistance is enabled, but no requested LLM steps are enabled."
    return None


def _candidate_summary_unavailable_reason(candidate: PaperCandidate) -> str | None:
    if not settings.llm_enabled or not settings.llm_summary_enabled:
        return "step_disabled"

    client = get_llm_client()
    availability_reason = client.availability_reason()
    if availability_reason:
        return availability_reason

    return None


def _extract_query_context(
    candidates: list[PaperCandidate],
    query: str,
    llm_options: SearchLLMOptions,
    search_filters: SearchFilters | None = None,
    base_context: SearchQueryContextResponse | None = None,
) -> SearchQueryContextResponse:
    llm_requested = _llm_requested(llm_options)
    llm_available = _llm_available() if llm_requested else False
    disabled_reason = _llm_disabled_reason(llm_options) if llm_requested else None
    applied_filters = search_filters.to_metadata() if search_filters else None
    context_metadata = _merge_context_metadata(
        search_filters,
        base_context or _base_query_context(query=query, llm_options=llm_options, status="ok", search_filters=search_filters),
    )
    if context_metadata.get("status") == "running":
        context_metadata.pop("status", None)
    diagnostic_steps, diagnostic_warnings, diagnostic_errors = _query_context_diagnostic_signals(candidates)
    warnings = _safe_context_string_list(context_metadata.get("warnings"))
    for warning in [*_query_context_warnings(candidates), *diagnostic_warnings]:
        if warning not in warnings:
            warnings.append(warning)
    errors = _safe_context_string_list(context_metadata.get("errors"))
    for error in diagnostic_errors:
        if error not in errors:
            errors.append(error)
    steps = _safe_context_string_list(context_metadata.get("steps"))
    for step in diagnostic_steps:
        if step not in steps:
            steps.append(step)

    for candidate in candidates:
        metadata = _candidate_metadata(candidate)
        llm_query = metadata.get("_llm_query")
        if not isinstance(llm_query, dict):
            continue
        rewrite_used = _llm_query_rewrite_used(llm_query, query)
        llm_steps = _llm_steps_from_candidates(candidates, rewrite_used=rewrite_used)
        status = "partial" if errors else "ok"
        merged_metadata = {**context_metadata, **llm_query}
        effective_query = _effective_query_from_metadata(merged_metadata, query)

        return SearchQueryContextResponse(
            status=_safe_context_status(context_metadata.get("status"), fallback=status),
            original_query=_safe_string(merged_metadata.get("original_query")) or query,
            optimized_query=_safe_string(merged_metadata.get("optimized_query")) or query,
            effective_query=effective_query,
            steps=steps,
            expanded_queries=_safe_string_list(llm_query.get("expanded_queries")),
            key_terms=_safe_string_list(llm_query.get("key_terms")),
            query_rewrite_notes=_safe_string(llm_query.get("notes")),
            llm_enabled=bool(settings.llm_enabled),
            llm_requested=llm_requested,
            use_llm_requested=_use_llm_requested(llm_options),
            llm_available=llm_available,
            llm_used=bool(llm_steps),
            llm_steps=llm_steps,
            llm_query_rewrite_used=rewrite_used,
            llm_disabled_reason=None if rewrite_used else disabled_reason,
            warnings=warnings,
            errors=errors,
            applied_filters=applied_filters,
            empty_result_retry_used=True if llm_query.get("empty_result_retry_used") is True else None,
            retry_queries=(
                _safe_string_list(llm_query.get("retry_queries"))
                if llm_query.get("empty_result_retry_used") is True
                else None
            ),
            final_search_query=(
                _safe_string(llm_query.get("final_search_query"))
                if llm_query.get("empty_result_retry_used") is True
                else None
            ),
        )
    if llm_requested and search_filters and isinstance(search_filters.query_context_metadata, dict):
        return _query_context_from_metadata(
            context_metadata,
            query=query,
            llm_options=llm_options,
            applied_filters=applied_filters,
            llm_available=llm_available,
            disabled_reason=disabled_reason,
            warnings=warnings,
            errors=errors,
            steps=steps,
        )
    if llm_requested:
        status = "partial" if errors else ("empty" if not candidates else "ok")
        return SearchQueryContextResponse(
            status=_safe_context_status(context_metadata.get("status"), fallback=status),
            original_query=query,
            optimized_query=query,
            effective_query=query,
            steps=steps,
            llm_enabled=bool(settings.llm_enabled),
            llm_requested=True,
            use_llm_requested=_use_llm_requested(llm_options),
            llm_available=llm_available,
            llm_used=False,
            llm_steps=[],
            llm_query_rewrite_used=False,
            llm_disabled_reason=disabled_reason,
            warnings=warnings,
            errors=errors,
            applied_filters=applied_filters,
        )
    effective_query = _effective_query_from_metadata(context_metadata, query)
    status = "empty" if not candidates else ("partial" if errors else "ok")
    return SearchQueryContextResponse(
        status=_safe_context_status(context_metadata.get("status"), fallback=status),
        original_query=query,
        optimized_query=effective_query,
        effective_query=effective_query,
        steps=steps,
        llm_enabled=bool(settings.llm_enabled),
        llm_requested=False,
        use_llm_requested=False,
        llm_available=False,
        llm_used=False,
        llm_steps=[],
        llm_query_rewrite_used=False,
        warnings=warnings,
        errors=errors,
        applied_filters=applied_filters,
    )


def _run_search_pipeline_compat(
    db: Session,
    session_id: int,
    query: str,
    *,
    llm_options: SearchLLMOptions,
    search_filters: SearchFilters | None,
):
    params = inspect.signature(run_search_pipeline).parameters
    if "search_filters" in params:
        return run_search_pipeline(
            db,
            session_id,
            query,
            llm_options=llm_options,
            search_filters=search_filters,
        )
    if search_filters and search_filters.active:
        if "year_filter" in params:
            return run_search_pipeline(
                db,
                session_id,
                query,
                llm_options=llm_options,
                year_filter=search_filters.year_filter,
            )
    return run_search_pipeline(db, session_id, query, llm_options=llm_options)


def _normalize_search_pipeline_result(result) -> tuple[list[PaperCandidate], list[PaperCandidate]]:
    if result is None:
        return [], []
    candidates, top5 = result
    return list(candidates or []), list(top5 or [])


def _to_candidate_response(candidate: PaperCandidate) -> CandidateResponse:
    metadata = _candidate_metadata(candidate)
    source_match = _source_match_debug(metadata)
    registry_metadata = _source_match_registry_metadata(source_match)
    llm_rerank = metadata.get("_llm_rerank")
    if not isinstance(llm_rerank, dict):
        llm_rerank = {}
    abstract = getattr(candidate, "abstract", None)
    abstract_length = len(abstract.strip()) if isinstance(abstract, str) else 0

    return CandidateResponse(
        id=candidate.id,
        title=candidate.title,
        abstract=None,
        abstract_present=abstract_length > 0,
        abstract_length=abstract_length,
        year=candidate.year,
        source_title=candidate.source_title,
        doi=getattr(candidate, "doi", None),
        url=getattr(candidate, "url", None),
        citation_count=candidate.citation_count,
        provider=candidate.provider,
        scopus_matched=candidate.scopus_matched,
        source_match_reason=_safe_string(source_match.get("reason")),
        source_match_confidence=_safe_float(source_match.get("confidence")),
        matched_source_title=_safe_string(source_match.get("source_title")),
        matched_source_issn=_safe_string(
            source_match.get("source_issn") or registry_metadata.get("issn")
        ),
        matched_source_eissn=_safe_string(
            source_match.get("source_eissn") or registry_metadata.get("eissn")
        ),
        final_score=candidate.final_score,
        reranker_status=("applied" if metadata.get("_reranker", {}).get("reranker_applied") is True else
                         "unavailable" if metadata.get("_reranker", {}).get("enabled") is True else "disabled"),
        reranker_relative_score=_safe_float(metadata.get("_reranker", {}).get("reranker_score")),
        created_at=candidate.created_at,
        review_status=_review_status(candidate),
        llm_relevance_score=_safe_float(llm_rerank.get("relevance_score")),
        llm_relevance_reason=_safe_string(llm_rerank.get("reason")),
        llm_summary=_extract_llm_summary(metadata),
    )


def _shortlisted_candidates(db: Session, session_id: int) -> list[PaperCandidate]:
    candidates = (
        db.query(PaperCandidate)
        .filter(PaperCandidate.session_id == session_id)
        .order_by(PaperCandidate.final_score.desc(), PaperCandidate.id.asc())
        .all()
    )
    return [candidate for candidate in candidates if _review_status(candidate) == "shortlisted"]


def _format_markdown_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value).strip()
    return text or "N/A"


def _shortlist_markdown(session_obj: ReviewSession, candidates: list[PaperCandidate]) -> str:
    lines = [
        f"# Shortlisted papers for {session_obj.topic}",
        "",
        f"Session ID: {session_obj.id}",
        f"Raw query: {session_obj.raw_query}",
        "",
    ]

    if not candidates:
        lines.append("No shortlisted papers yet.")
        lines.append("")
        return "\n".join(lines)

    for index, candidate in enumerate(candidates, start=1):
        summary = _extract_llm_summary(_candidate_metadata(candidate))
        doi_or_url = candidate.doi or candidate.url
        lines.extend(
            [
                f"## {index}. {candidate.title}",
                "",
                f"- Year: {_format_markdown_value(candidate.year)}",
                f"- Source: {_format_markdown_value(candidate.source_title)}",
                f"- DOI/URL: {_format_markdown_value(doi_or_url)}",
                f"- Provider: {_format_markdown_value(candidate.provider)}",
                f"- Score: {_format_markdown_value(candidate.final_score)}",
            ]
        )
        if summary and summary.one_sentence_summary:
            lines.append(f"- AI summary: {summary.one_sentence_summary}")
        lines.append("")

    return "\n".join(lines)


@router.post("/{session_id}", response_model=SearchResponse, response_model_exclude_none=True)
def run_search(
    session_id: int,
    query: str,
    request: Request,
    use_llm: bool | None = Query(default=None),
    use_query_rewrite: bool | None = Query(default=None),
    use_llm_rerank: bool | None = Query(default=None),
    use_llm_summary: bool | None = Query(default=None),
    limit: int | None = Query(default=None),
    recent_years: int | None = Query(default=None),
    year_from: int | None = Query(default=None),
    year_to: int | None = Query(default=None),
    source: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    scopus_only: bool | None = Query(default=None),
    min_citations: int | None = Query(default=None),
    access_type: str | None = Query(default=None),
    years_back: int | None = Query(default=None),
    publication_year_from: int | None = Query(default=None),
    publication_year_to: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # ── Guardrails ─────────────────────────────────────────────────────────
    # Rate limit: max 20 searches per minute per user
    check_rate_limit(
        f"search:{user.id}",
        max_calls=20,
        window_seconds=60,
        action="search",
    )
    # Sanitize query: length, strip dangerous chars, injection detection
    safe_query = sanitize_query(query)
    # Sanitize source filter string
    safe_source = sanitize_source_filter(source)
    # Clamp min_citations to a sane maximum
    if min_citations is not None and min_citations > 100_000:
        min_citations = 100_000
    # ── End guardrails ─────────────────────────────────────────────────────

    session_obj = _get_user_session_or_404(db, session_id, user.id)
    llm_options = SearchLLMOptions(
        use_llm=use_llm,
        use_query_rewrite=use_query_rewrite,
        use_llm_rerank=use_llm_rerank,
        use_llm_summary=use_llm_summary,
    )
    try:
        search_filters = normalize_search_filters(
            query=safe_query,
            session_filters=getattr(session_obj, "filters_json", None),
            limit=limit,
            recent_years=recent_years,
            year_from=year_from,
            year_to=year_to,
            source=safe_source,
            provider=provider,
            scopus_only=scopus_only,
            min_citations=min_citations,
            access_type=access_type,
            years_back=years_back,
            publication_year_from=publication_year_from,
            publication_year_to=publication_year_to,
        )
    except SearchFilterError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    base_query_context = _base_query_context(
        query=safe_query,
        llm_options=llm_options,
        status="running",
        search_filters=search_filters,
    )
    if search_filters is not None:
        search_filters.query_context_metadata = _safe_query_context_payload(base_query_context)
    _persist_query_context(db, session_id, base_query_context)
        
    try:
        candidates, top5 = _normalize_search_pipeline_result(
            _run_search_pipeline_compat(
                db,
                session_id,
                safe_query,
                llm_options=llm_options,
                search_filters=search_filters,
            )
        )
        query_context = _extract_query_context(
            candidates,
            query=safe_query,
            llm_options=llm_options,
            search_filters=search_filters,
            base_context=base_query_context,
        )
        _persist_query_context(db, session_id, query_context)
        return SearchResponse(
            session_id=session_id,
            candidates=[_to_candidate_response(candidate) for candidate in candidates],
            top5=[_to_candidate_response(candidate) for candidate in top5],
            selected_papers=[_to_candidate_response(candidate) for candidate in top5],
            query_context=query_context,
        )
    except HTTPException:
        raise
    except Exception:
        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
        logger.exception(
            "Search failed: request_id=%s session_id=%s",
            request_id,
            session_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "search_failed",
                    "message": "Search failed. Please try again.",
                    "request_id": request_id,
                    "retryable": True,
                }
            },
            headers={"X-Request-ID": request_id},
        )


@router.get(
    "/{session_id}/query-context",
    response_model=SearchQueryContextStoredResponse,
)
def get_query_context(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_user_session_or_404(db, session_id, user.id)
    stored = (
        db.query(SearchQueryContext)
        .filter(SearchQueryContext.session_id == session_id)
        .first()
    )
    if not stored:
        return SearchQueryContextStoredResponse(
            session_id=session_id,
            exists=False,
            query_context={},
            updated_at=None,
        )

    payload = stored.payload_json if isinstance(stored.payload_json, dict) else {}
    safe_payload = {
        key: value
        for key, value in payload.items()
        if key in _QUERY_CONTEXT_FIELDS
    }
    return SearchQueryContextStoredResponse(
        session_id=session_id,
        exists=True,
        query_context=_safe_query_context_payload(SearchQueryContextResponse(**safe_payload)),
        updated_at=stored.updated_at,
    )


@router.get("/{session_id}/candidates", response_model=list[CandidateResponse], response_model_exclude_none=True)
def list_candidates(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_user_session_or_404(db, session_id, user.id)
    candidates = (
        db.query(PaperCandidate)
        .filter(PaperCandidate.session_id == session_id)
        .order_by(PaperCandidate.final_score.desc(), PaperCandidate.id.asc())
        .all()
    )
    return [_to_candidate_response(candidate) for candidate in candidates]


@router.post(
    "/{session_id}/candidates/{candidate_id}/summary",
    response_model=CandidateSummaryResponse,
    response_model_exclude_none=True,
)
def generate_candidate_summary(
    session_id: int,
    candidate_id: int,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, user.id)
    candidate = _get_candidate_for_session_or_404(db, session_id, candidate_id)
    query = session_obj.raw_query or session_obj.topic

    cached_result = cached_paper_summary(
        _candidate_metadata(candidate),
        candidate=candidate,
        query=query,
    )
    if cached_result and not force:
        return _candidate_summary_response(
            session_id,
            candidate_id,
            cached_result,
            cached=True,
    )

    unavailable_reason = _candidate_summary_unavailable_reason(candidate)
    if unavailable_reason and settings.local_analysis_enabled:
        try:
            result = local_extractive_summary(candidate, query=query)
        except Exception:
            logger.exception("Local extractive analysis failed")
            result = fallback_paper_summary(candidate, query=query, attempted=True, fallback_reason="model_unavailable")
    elif unavailable_reason:
        result = fallback_paper_summary(
            candidate,
            query=query,
            attempted=True,
            fallback_reason=unavailable_reason,
        )
    else:
        if (
            getattr(settings, "llm_skip_summary_if_abstract_missing", True)
            and not has_summarizable_abstract(getattr(candidate, "abstract", None))
        ):
            enrich_candidate_abstract(candidate)
        if (
            getattr(settings, "llm_skip_summary_if_abstract_missing", True)
            and not has_summarizable_abstract(getattr(candidate, "abstract", None))
        ):
            result = fallback_paper_summary(
                candidate,
                query=query,
                attempted=True,
                fallback_reason="no_abstract",
            )
        else:
            try:
                result = summarize_paper(candidate=candidate, query=query)
            except Exception as exc:  # pragma: no cover - defensive boundary for optional LLMs
                result = fallback_paper_summary(
                    candidate,
                    query=query,
                    attempted=True,
                    fallback_reason=fallback_reason_from_exception(exc),
                )

    store_paper_summary_metadata(candidate, result)
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return _candidate_summary_response(
        session_id,
        candidate_id,
        result,
        cached=False,
    )


@router.get("/{session_id}/shortlist", response_model=list[CandidateResponse], response_model_exclude_none=True)
def list_shortlisted_candidates(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_user_session_or_404(db, session_id, user.id)
    return [_to_candidate_response(candidate) for candidate in _shortlisted_candidates(db, session_id)]


@router.post(
    "/{session_id}/candidates/{candidate_id}/review",
    response_model=CandidateResponse,
    response_model_exclude_none=True,
)
def set_candidate_review_status(
    session_id: int,
    candidate_id: int,
    decision: ReviewDecisionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_user_session_or_404(db, session_id, user.id)
    candidate = _get_candidate_for_session_or_404(db, session_id, candidate_id)
    _set_review_status(candidate, decision.status)
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return _to_candidate_response(candidate)


@router.post(
    "/{session_id}/candidates/{candidate_id}/shortlist",
    response_model=CandidateResponse,
    response_model_exclude_none=True,
)
def shortlist_candidate(
    session_id: int,
    candidate_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return set_candidate_review_status(
        session_id=session_id,
        candidate_id=candidate_id,
        decision=ReviewDecisionRequest(status="shortlisted"),
        db=db,
        user=user,
    )


@router.post(
    "/{session_id}/candidates/{candidate_id}/dismiss",
    response_model=CandidateResponse,
    response_model_exclude_none=True,
)
def dismiss_candidate(
    session_id: int,
    candidate_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return set_candidate_review_status(
        session_id=session_id,
        candidate_id=candidate_id,
        decision=ReviewDecisionRequest(status="dismissed"),
        db=db,
        user=user,
    )


@router.delete(
    "/{session_id}/candidates/{candidate_id}/review",
    response_model=CandidateResponse,
    response_model_exclude_none=True,
)
def clear_candidate_review_status(
    session_id: int,
    candidate_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_user_session_or_404(db, session_id, user.id)
    candidate = _get_candidate_for_session_or_404(db, session_id, candidate_id)
    _clear_review_status(candidate)
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return _to_candidate_response(candidate)


@router.get("/{session_id}/shortlist/export")
def export_shortlist_markdown(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, user.id)
    content = _shortlist_markdown(session_obj, _shortlisted_candidates(db, session_id))
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="session-{session_id}-shortlist.md"',
        },
    )
