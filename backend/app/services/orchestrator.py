import logging
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.providers.provider_factory import get_search_provider
from app.services.source_matcher import resolve_scopus_match
from app.services.ranking import build_authoring_trust_summary, build_selection_debug, compute_score, score_paper, score_trust
from app.services.llm_client import (
    candidate_fingerprint,
    compact_payload_for_llm,
    current_prompt_version,
    estimate_llm_usage,
    fallback_reason_from_exception,
    query_fingerprint,
    safe_cache_miss_reason,
    safe_fallback_reason,
)
from app.services.llm_query import fallback_query_rewrite, rewrite_query
from app.services.llm_reranker import rerank_candidates
from app.services.reranker import rerank_candidates as production_rerank_candidates
from app.services.paper_summary import build_summary_payload, summarize_paper
from app.services.paper_enrichment import enrich_candidate_abstract, has_summarizable_abstract
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.selected_paper import SelectedPaper
from app.db.models.review_session import ReviewSession


logger = logging.getLogger(__name__)

PUBLICATION_YEAR_PRESETS = {1, 3, 5, 10, 15}
PUBLICATION_YEAR_MIN = 1900
PUBLICATION_YEAR_MAX_SPAN = 50
PUBLICATION_YEAR_MAX_YEARS_BACK = 50
SEARCH_DEFAULT_LIMIT = 20
SEARCH_MAX_LIMIT = 100
SEARCH_SUMMARY_ENRICHMENT_LIMIT = 5
SEARCH_INLINE_SUMMARY_HARD_CAP = 3
SEARCH_EMPTY_RESULT_MAX_RETRIES = 2
SEARCH_OPTIONAL_LLM_SOFT_TIME_BUDGET_SECONDS = 20.0
SEARCH_KNOWN_PROVIDERS = {"mock", "semantic_scholar", "openalex_strict"}
ACCESS_TYPES = {"open", "closed", "any"}
MISSING_YEAR_POLICY = "excluded_when_year_filter_active"
MISSING_ACCESS_POLICY = "excluded_when_access_filter_active"


class PublicationYearFilterError(ValueError):
    """Raised for invalid publication-year filter requests."""


class SearchFilterError(ValueError):
    """Raised for invalid search filter requests."""


@dataclass(frozen=True)
class SearchYearFilter:
    publication_year_from: int
    publication_year_to: int
    recent_years: int | None = None
    missing_year_policy: str = MISSING_YEAR_POLICY

    @property
    def years_back(self) -> int | None:
        return self.recent_years

    def includes(self, year: int | None) -> bool:
        if year is None:
            return False
        return self.publication_year_from <= year <= self.publication_year_to

    def to_metadata(self) -> dict[str, int | str]:
        metadata: dict[str, int | str] = {
            "publication_year_from": self.publication_year_from,
            "publication_year_to": self.publication_year_to,
            "missing_year_policy": self.missing_year_policy,
        }
        if self.recent_years is not None:
            metadata["years_back"] = self.recent_years
        return metadata


@dataclass
class SearchFilters:
    query: str
    limit: int | None = None
    year_filter: SearchYearFilter | None = None
    source: str | None = None
    provider: str | None = None
    scopus_only: bool | None = None
    min_citations: int | None = None
    access_type: str | None = None
    pre_filter_count: int | None = None
    post_filter_count: int | None = None
    query_context_metadata: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return any(
            [
                self.limit is not None,
                self.year_filter is not None,
                self.source is not None,
                self.provider is not None,
                self.scopus_only is True,
                self.min_citations is not None,
                self.access_type in {"open", "closed"},
            ]
        )

    def fetch_limit(self) -> int:
        return self.limit or SEARCH_DEFAULT_LIMIT

    def to_metadata(self) -> dict[str, Any] | None:
        if not self.active:
            return None

        metadata: dict[str, Any] = {"query": self.query}
        if self.limit is not None:
            metadata["limit"] = self.limit
        if self.year_filter:
            if self.year_filter.recent_years is not None:
                metadata["recent_years"] = self.year_filter.recent_years
                metadata["years_back"] = self.year_filter.recent_years
            metadata["year_from"] = self.year_filter.publication_year_from
            metadata["year_to"] = self.year_filter.publication_year_to
            metadata["publication_year_from"] = self.year_filter.publication_year_from
            metadata["publication_year_to"] = self.year_filter.publication_year_to
            metadata["missing_year_policy"] = self.year_filter.missing_year_policy
        if self.source is not None:
            metadata["source"] = self.source
        if self.provider is not None:
            metadata["provider"] = self.provider
        if self.scopus_only is not None:
            metadata["scopus_only"] = self.scopus_only
        if self.min_citations is not None:
            metadata["min_citations"] = self.min_citations
        if self.access_type is not None:
            metadata["access_type"] = self.access_type
            if self.access_type in {"open", "closed"}:
                metadata["missing_access_policy"] = MISSING_ACCESS_POLICY
        if self.pre_filter_count is not None:
            metadata["pre_filter_count"] = self.pre_filter_count
        if self.post_filter_count is not None:
            metadata["post_filter_count"] = self.post_filter_count
            if self.pre_filter_count is not None:
                metadata["filtered_out_count"] = max(self.pre_filter_count - self.post_filter_count, 0)
        return metadata


def _normalized_query_value(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _dedupe_queries(values: list[Any]) -> list[str]:
    queries = []
    seen = set()
    for value in values:
        query = str(value or "").strip()
        key = _normalized_query_value(query)
        if query and key not in seen:
            queries.append(query)
            seen.add(key)
    return queries


def _query_rewrite_used(metadata: dict[str, Any] | None, original_query: str) -> bool:
    if not isinstance(metadata, dict):
        return False
    if metadata.get("used_llm") is False:
        return False
    original = _normalized_query_value(metadata.get("original_query") or original_query)
    optimized = _normalized_query_value(metadata.get("optimized_query"))
    if optimized and optimized != original:
        return True
    for value in metadata.get("expanded_queries") or []:
        if _normalized_query_value(value) and _normalized_query_value(value) != original:
            return True
    return False


def _retry_queries_from_rewrite(
    *,
    original_query: str,
    first_query: str,
    query_rewrite_metadata: dict[str, Any] | None,
) -> list[str]:
    if not _query_rewrite_used(query_rewrite_metadata, original_query):
        return []
    if not isinstance(query_rewrite_metadata, dict):
        return []
    candidates = [query_rewrite_metadata.get("optimized_query")]
    candidates.extend(query_rewrite_metadata.get("expanded_queries") or [])
    return [
        query
        for query in _dedupe_queries(candidates)
        if _normalized_query_value(query) not in {
            _normalized_query_value(original_query),
            _normalized_query_value(first_query),
        }
    ][:SEARCH_EMPTY_RESULT_MAX_RETRIES]


@dataclass(frozen=True)
class SearchLLMOptions:
    use_llm: bool | None = None
    use_query_rewrite: bool | None = None
    use_llm_rerank: bool | None = None
    use_llm_summary: bool | None = None

    def query_rewrite_enabled(self) -> bool:
        return _llm_step_enabled(
            global_step_enabled=settings.llm_query_rewrite_enabled,
            request_llm_enabled=self.use_llm,
            request_step_enabled=self.use_query_rewrite,
        )

    def rerank_enabled(self) -> bool:
        return _llm_step_enabled(
            global_step_enabled=settings.llm_rerank_enabled,
            request_llm_enabled=self.use_llm,
            request_step_enabled=self.use_llm_rerank,
        )

    def summary_enabled(self) -> bool:
        return _llm_step_enabled(
            global_step_enabled=settings.llm_summary_enabled,
            request_llm_enabled=self.use_llm,
            request_step_enabled=self.use_llm_summary,
        )


@dataclass
class LLMSearchBudget:
    call_count: int = 0
    estimated_tokens: int = 0

    @property
    def max_calls(self) -> int:
        return max(int(getattr(settings, "llm_max_calls_per_search", 20)), 0)

    @property
    def max_estimated_tokens(self) -> int:
        return max(int(getattr(settings, "llm_max_estimated_tokens_per_search", 20000)), 0)

    def remaining(self) -> dict[str, int]:
        return {
            "calls": max(self.max_calls - self.call_count, 0),
            "estimated_tokens": max(self.max_estimated_tokens - self.estimated_tokens, 0),
        }

    def reserve(self, payload: dict, *, default_output_tokens: int = 512) -> tuple[dict | None, str | None]:
        try:
            compacted_payload, input_truncated = compact_payload_for_llm(
                payload,
                max_chars=getattr(settings, "llm_max_input_chars_per_call", 12000),
            )
        except Exception:
            return None, "input_too_large"

        usage = estimate_llm_usage(
            input_payload=compacted_payload,
            default_output_tokens=default_output_tokens,
            input_truncated=input_truncated,
        )
        usage_metadata = usage.to_metadata()

        if self.call_count + usage.estimated_call_count > self.max_calls:
            return usage_metadata, "max_calls_exceeded"

        if (
            getattr(settings, "llm_usage_estimation_enabled", True)
            and self.estimated_tokens + usage.estimated_total_tokens > self.max_estimated_tokens
        ):
            return usage_metadata, "estimated_tokens_exceeded"

        self.call_count += usage.estimated_call_count
        if getattr(settings, "llm_usage_estimation_enabled", True):
            self.estimated_tokens += usage.estimated_total_tokens
        return usage_metadata, None


def _llm_step_enabled(
    *,
    global_step_enabled: bool,
    request_llm_enabled: bool | None,
    request_step_enabled: bool | None,
) -> bool:
    if not settings.llm_enabled:
        return False
    if not global_step_enabled:
        return False
    if request_llm_enabled is False:
        return False
    if request_step_enabled is True:
        return True
    if request_step_enabled is False:
        return False
    return request_llm_enabled is True


def _current_publication_year() -> int:
    return datetime.utcnow().year


def _safe_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PublicationYearFilterError(f"{field_name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError as exc:
            raise PublicationYearFilterError(f"{field_name} must be an integer.") from exc
    raise PublicationYearFilterError(f"{field_name} must be an integer.")


def _safe_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise SearchFilterError(f"{field_name} must be a boolean.")


def _safe_filter_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SearchFilterError("Filter string values must be strings.")
    text = value.strip()
    return text or None


def _session_year_filter_values(session_filters: dict | None) -> dict[str, Any]:
    if not isinstance(session_filters, dict):
        return {}
    return {
        key: session_filters.get(key)
        for key in ("years_back", "publication_year_from", "publication_year_to")
        if session_filters.get(key) is not None
    }


def _session_filter_values(session_filters: dict | None) -> dict[str, Any]:
    if not isinstance(session_filters, dict):
        return {}
    keys = (
        "limit",
        "recent_years",
        "years_back",
        "year_from",
        "year_to",
        "publication_year_from",
        "publication_year_to",
        "source",
        "provider",
        "scopus_only",
        "min_citations",
        "access_type",
    )
    return {key: session_filters.get(key) for key in keys if session_filters.get(key) is not None}


def _canonical_filter_values(values: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(values)
    if canonical.get("recent_years") is None and canonical.get("years_back") is not None:
        canonical["recent_years"] = canonical.get("years_back")
    if canonical.get("year_from") is None and canonical.get("publication_year_from") is not None:
        canonical["year_from"] = canonical.get("publication_year_from")
    if canonical.get("year_to") is None and canonical.get("publication_year_to") is not None:
        canonical["year_to"] = canonical.get("publication_year_to")
    return canonical


def normalize_search_year_filter(
    *,
    session_filters: dict | None = None,
    years_back: int | str | None = None,
    publication_year_from: int | str | None = None,
    publication_year_to: int | str | None = None,
    current_year: int | None = None,
) -> SearchYearFilter | None:
    request_has_filter = any(
        value is not None
        for value in (years_back, publication_year_from, publication_year_to)
    )
    values = (
        {
            "years_back": years_back,
            "publication_year_from": publication_year_from,
            "publication_year_to": publication_year_to,
        }
        if request_has_filter
        else _session_year_filter_values(session_filters)
    )
    if not values:
        return None

    year_now = current_year or _current_publication_year()
    parsed_years_back = _safe_int(values.get("years_back"), "years_back")
    parsed_from = _safe_int(values.get("publication_year_from"), "publication_year_from")
    parsed_to = _safe_int(values.get("publication_year_to"), "publication_year_to")

    if parsed_years_back is not None:
        if parsed_from is not None or parsed_to is not None:
            raise PublicationYearFilterError(
                "Use either years_back or publication_year_from/publication_year_to, not both."
            )
        if parsed_years_back <= 0:
            raise PublicationYearFilterError("years_back must be greater than zero.")
        if parsed_years_back > PUBLICATION_YEAR_MAX_YEARS_BACK:
            raise PublicationYearFilterError(
                f"years_back must be at most {PUBLICATION_YEAR_MAX_YEARS_BACK}."
            )
        if (
            parsed_years_back not in PUBLICATION_YEAR_PRESETS
            and parsed_years_back > PUBLICATION_YEAR_MAX_YEARS_BACK
        ):
            raise PublicationYearFilterError("years_back exceeds the supported maximum.")
        if parsed_years_back not in PUBLICATION_YEAR_PRESETS:
            raise PublicationYearFilterError(
                "years_back must be one of 1, 3, 5, 10, or 15."
            )
        return SearchYearFilter(
            recent_years=parsed_years_back,
            publication_year_from=year_now - parsed_years_back + 1,
            publication_year_to=year_now,
        )

    if parsed_from is None and parsed_to is None:
        return None
    if parsed_from is None or parsed_to is None:
        raise PublicationYearFilterError(
            "publication_year_from and publication_year_to must be provided together."
        )
    if parsed_from < PUBLICATION_YEAR_MIN:
        raise PublicationYearFilterError(
            f"publication_year_from must be {PUBLICATION_YEAR_MIN} or later."
        )
    if parsed_to > year_now + 1:
        raise PublicationYearFilterError(
            "publication_year_to must not be more than one year in the future."
        )
    if parsed_from > parsed_to:
        raise PublicationYearFilterError("publication_year_from must be <= publication_year_to.")
    if parsed_to - parsed_from + 1 > PUBLICATION_YEAR_MAX_SPAN:
        raise PublicationYearFilterError(
            f"Publication year range must not exceed {PUBLICATION_YEAR_MAX_SPAN} years."
        )

    return SearchYearFilter(
        publication_year_from=parsed_from,
        publication_year_to=parsed_to,
    )


def normalize_search_filters(
    *,
    query: str,
    session_filters: dict | None = None,
    limit: int | str | None = None,
    recent_years: int | str | None = None,
    year_from: int | str | None = None,
    year_to: int | str | None = None,
    source: str | None = None,
    provider: str | None = None,
    scopus_only: bool | str | None = None,
    min_citations: int | str | None = None,
    access_type: str | None = None,
    years_back: int | str | None = None,
    publication_year_from: int | str | None = None,
    publication_year_to: int | str | None = None,
    current_year: int | None = None,
) -> SearchFilters:
    values = _canonical_filter_values(_session_filter_values(session_filters))
    request_values = _canonical_filter_values(
        {
            "limit": limit,
            "recent_years": recent_years,
            "year_from": year_from,
            "year_to": year_to,
            "source": source,
            "provider": provider,
            "scopus_only": scopus_only,
            "min_citations": min_citations,
            "access_type": access_type,
            "years_back": years_back,
            "publication_year_from": publication_year_from,
            "publication_year_to": publication_year_to,
        }
    )
    request_recent_years_provided = recent_years is not None or years_back is not None
    request_year_range_provided = any(
        value is not None
        for value in (year_from, year_to, publication_year_from, publication_year_to)
    )
    if request_recent_years_provided:
        for key in ("year_from", "year_to", "publication_year_from", "publication_year_to"):
            values.pop(key, None)
    elif request_year_range_provided:
        for key in ("recent_years", "years_back"):
            values.pop(key, None)

    for key, value in request_values.items():
        if value is not None:
            values[key] = value

    parsed_limit = _safe_int(values.get("limit"), "limit")
    if parsed_limit is not None:
        if parsed_limit <= 0:
            raise SearchFilterError("limit must be greater than zero.")
        parsed_limit = min(parsed_limit, SEARCH_MAX_LIMIT)

    parsed_min_citations = _safe_int(values.get("min_citations"), "min_citations")
    if parsed_min_citations is not None and parsed_min_citations < 0:
        raise SearchFilterError("min_citations must be greater than or equal to zero.")

    parsed_source = _safe_filter_string(values.get("source"))
    parsed_provider = _safe_filter_string(values.get("provider"))
    if parsed_provider is not None:
        parsed_provider = parsed_provider.lower()
        if parsed_provider not in SEARCH_KNOWN_PROVIDERS:
            raise SearchFilterError(
                "provider must be one of mock, semantic_scholar, or openalex_strict."
            )

    parsed_access_type = _safe_filter_string(values.get("access_type"))
    if parsed_access_type is not None:
        parsed_access_type = parsed_access_type.lower()
        if parsed_access_type not in ACCESS_TYPES:
            raise SearchFilterError("access_type must be one of open, closed, or any.")
        if parsed_access_type == "any":
            parsed_access_type = "any"

    try:
        year_filter = normalize_search_year_filter(
            years_back=values.get("recent_years"),
            publication_year_from=values.get("year_from"),
            publication_year_to=values.get("year_to"),
            current_year=current_year,
        )
    except PublicationYearFilterError as exc:
        raise SearchFilterError(str(exc)) from exc

    return SearchFilters(
        query=query,
        limit=parsed_limit,
        year_filter=year_filter,
        source=parsed_source,
        provider=parsed_provider,
        scopus_only=_safe_bool(values.get("scopus_only"), "scopus_only"),
        min_citations=parsed_min_citations,
        access_type=parsed_access_type,
    )


def apply_publication_year_filter(results: list[Any], year_filter: SearchYearFilter | None) -> list[Any]:
    if year_filter is None:
        return results
    return [result for result in results if year_filter.includes(getattr(result, "year", None))]


def _contains_normalized(value: str | None, needle: str) -> bool:
    if not value:
        return False
    return needle.lower() in value.lower()


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _open_access_state(raw_metadata: Any) -> str | None:
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    boolean_keys = (
        "open_access.is_oa",
        "is_oa",
        "is_open_access",
    )
    for key in boolean_keys:
        value = _metadata_value(metadata, key)
        if isinstance(value, bool):
            return "open" if value else "closed"

    if _metadata_value(metadata, "is_in_doaj") is True:
        return "open"

    status_value = _metadata_value(metadata, "open_access_status")
    if isinstance(status_value, str):
        normalized = status_value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"open", "oa", "open_access", "gold", "green", "hybrid", "bronze", "free"}:
            return "open"
        if normalized in {"closed", "closed_access", "subscription", "not_oa", "not_open"}:
            return "closed"

    nested_status = _metadata_value(metadata, "open_access.oa_status")
    if isinstance(nested_status, str):
        normalized = nested_status.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"open", "oa", "open_access", "gold", "green", "hybrid", "bronze", "free"}:
            return "open"
        if normalized in {"closed", "closed_access", "subscription", "not_oa", "not_open"}:
            return "closed"

    return None


def _result_matches_pre_source_filters(result: Any, filters: SearchFilters | None) -> bool:
    if not filters or not filters.active:
        return True

    if filters.year_filter and not filters.year_filter.includes(getattr(result, "year", None)):
        return False

    if filters.source:
        raw_metadata = getattr(result, "raw_metadata", None)
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        venue = metadata.get("venue") if isinstance(metadata.get("venue"), str) else None
        if not (
            _contains_normalized(getattr(result, "source_title", None), filters.source)
            or _contains_normalized(venue, filters.source)
        ):
            return False

    if filters.provider and str(getattr(result, "provider", "") or "").lower() != filters.provider:
        return False

    if filters.min_citations is not None:
        citation_count = getattr(result, "citation_count", None)
        if not isinstance(citation_count, int):
            citation_count = 0
        if citation_count < filters.min_citations:
            return False

    if filters.access_type in {"open", "closed"}:
        state = _open_access_state(getattr(result, "raw_metadata", None))
        if state != filters.access_type:
            return False

    return True


def apply_pre_source_search_filters(results: list[Any], filters: SearchFilters | None) -> list[Any]:
    if filters is None or not filters.active:
        return results
    return [result for result in results if _result_matches_pre_source_filters(result, filters)]


def _phase1_selection_score(candidate: PaperCandidate) -> float:
    metadata = candidate.raw_metadata_json or {}
    selection = metadata.get("_selection") if isinstance(metadata, dict) else {}
    score = selection.get("combined_score") if isinstance(selection, dict) else None
    return float(score) if isinstance(score, (int, float)) else 0.0


def _trust_total(candidate: PaperCandidate) -> float:
    metadata = candidate.raw_metadata_json or {}
    trust = metadata.get("_trust") if isinstance(metadata, dict) else {}
    score = trust.get("total") if isinstance(trust, dict) else None
    if not isinstance(score, (int, float)):
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _active_selection_score(candidate: PaperCandidate) -> float:
    metadata = candidate.raw_metadata_json or {}
    llm_selection = metadata.get("_llm_selection") if isinstance(metadata, dict) else {}
    if settings.llm_enabled and settings.llm_rerank_enabled and isinstance(llm_selection, dict):
        score = llm_selection.get("combined_score")
        if isinstance(score, (int, float)):
            return float(score)
    score = _production_reranker_selection_score(candidate)
    if score is not None:
        return score
    return _phase1_selection_score(candidate)


def _production_reranker_selection_score(candidate: PaperCandidate) -> float | None:
    if not getattr(settings, "reranker_enabled", False):
        return None
    metadata = candidate.raw_metadata_json or {}
    reranker_selection = metadata.get("_reranker_selection") if isinstance(metadata, dict) else {}
    if not isinstance(reranker_selection, dict):
        return None
    score = reranker_selection.get("combined_score")
    return float(score) if isinstance(score, (int, float)) else None


def _candidate_sort_key(candidate: PaperCandidate):
    return (
        _active_selection_score(candidate),
        candidate.final_score or 0.0,
        _trust_total(candidate),
    )


def _select_top_candidates(candidates: list[PaperCandidate], limit: int = 5) -> list[PaperCandidate]:
    sorted_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    if limit <= 0:
        return []

    best_score = _active_selection_score(sorted_candidates[0]) if sorted_candidates else 0.0
    abstract_quota_threshold = max(0.2, best_score * 0.6) if best_score > 0 else 0.0
    with_abstract = [
        candidate
        for candidate in sorted_candidates
        if has_summarizable_abstract(getattr(candidate, "abstract", None))
        and _active_selection_score(candidate) >= abstract_quota_threshold
    ]
    quota = min(3, limit, len(with_abstract))
    selected: list[PaperCandidate] = []
    seen_keys: set[int] = set()

    for candidate in with_abstract[:quota]:
        selected.append(candidate)
        seen_keys.add(candidate.id if candidate.id is not None else id(candidate))

    for candidate in sorted_candidates:
        if len(selected) >= limit:
            break
        candidate_key = candidate.id if candidate.id is not None else id(candidate)
        if candidate_key in seen_keys:
            continue
        selected.append(candidate)
        seen_keys.add(candidate_key)

    return selected


def _llm_step_diagnostic(
    *,
    attempted: bool,
    used: bool,
    fallback_reason: str | None = None,
    cache_hit: bool = False,
    cache_miss_reason: str | None = None,
    estimated_usage: dict | None = None,
    budget_guard: dict | None = None,
    input_truncated: bool = False,
) -> dict:
    return {
        "attempted": bool(attempted),
        "used": bool(used),
        "fallback_reason": safe_fallback_reason(fallback_reason) if fallback_reason else None,
        "cache_hit": bool(cache_hit),
        "cache_miss_reason": safe_cache_miss_reason(cache_miss_reason) if cache_miss_reason else None,
        "estimated_usage": estimated_usage,
        "budget_guard": budget_guard,
        "input_truncated": bool(input_truncated),
    }


def _optional_llm_time_budget_exceeded(search_started_at: float | None) -> bool:
    if search_started_at is None:
        return False
    return (time.monotonic() - search_started_at) >= SEARCH_OPTIONAL_LLM_SOFT_TIME_BUDGET_SECONDS


def _append_query_context_signal(
    search_filters: SearchFilters | None,
    *,
    step: str | None = None,
    warning: str | None = None,
    error: str | None = None,
    status: str | None = None,
    effective_query: str | None = None,
    llm_used: bool | None = None,
) -> None:
    if search_filters is None:
        return
    metadata = search_filters.query_context_metadata
    if not isinstance(metadata, dict):
        metadata = {}
    if status:
        metadata["status"] = status
    if effective_query:
        metadata["effective_query"] = effective_query
        metadata["final_search_query"] = effective_query
    if llm_used is not None:
        metadata["llm_used"] = bool(llm_used)
    if step:
        steps = metadata.get("steps")
        if not isinstance(steps, list):
            steps = []
        if step not in steps:
            steps.append(step)
        metadata["steps"] = steps
    if warning:
        warnings = metadata.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        if warning not in warnings:
            warnings.append(warning)
        metadata["warnings"] = warnings
    if error:
        errors = metadata.get("errors")
        if not isinstance(errors, list):
            errors = []
        if error not in errors:
            errors.append(error)
        metadata["errors"] = errors
        metadata["status"] = "partial"
    search_filters.query_context_metadata = metadata


def _store_llm_diagnostic(
    candidate: PaperCandidate,
    step: str,
    *,
    attempted: bool,
    used: bool,
    fallback_reason: str | None = None,
    cache_hit: bool = False,
    cache_miss_reason: str | None = None,
    estimated_usage: dict | None = None,
    budget_guard: dict | None = None,
    input_truncated: bool = False,
) -> None:
    metadata = dict(candidate.raw_metadata_json or {})
    diagnostics = metadata.get("_llm_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics[step] = _llm_step_diagnostic(
        attempted=attempted,
        used=used,
        fallback_reason=fallback_reason,
        cache_hit=cache_hit,
        cache_miss_reason=cache_miss_reason,
        estimated_usage=estimated_usage,
        budget_guard=budget_guard,
        input_truncated=input_truncated,
    )
    metadata["_llm_diagnostics"] = diagnostics
    candidate.raw_metadata_json = metadata


def _metadata(candidate: PaperCandidate) -> dict:
    metadata = candidate.raw_metadata_json
    return metadata if isinstance(metadata, dict) else {}


def _cache_enabled() -> bool:
    return bool(getattr(settings, "llm_cache_enabled", True))


def _cache_artifact_valid(
    artifact: object,
    *,
    expected_prompt_version: str,
    expected_query_fingerprint: str | None = None,
    expected_candidate_fingerprint: str | None = None,
) -> tuple[dict | None, str | None]:
    if not _cache_enabled():
        return None, "cache_disabled"
    if not isinstance(artifact, dict):
        return None, "no_cached_artifact" if artifact is None else "malformed_cached_artifact"
    if artifact.get("prompt_version") != expected_prompt_version:
        return None, "prompt_version_changed"
    if expected_query_fingerprint and artifact.get("query_fingerprint") != expected_query_fingerprint:
        return None, "query_changed"
    if expected_candidate_fingerprint and artifact.get("candidate_fingerprint") != expected_candidate_fingerprint:
        return None, "candidate_changed"
    if artifact.get("used_llm") is not True:
        return None, "malformed_cached_artifact"
    return dict(artifact), None


def _llm_cache_snapshot(candidates: list[PaperCandidate]) -> dict[str, dict]:
    snapshot: dict[str, dict] = {
        "query_rewrite": {},
        "rerank": {},
        "summary": {},
    }
    for candidate in candidates:
        metadata = _metadata(candidate)
        candidate_fp = candidate_fingerprint(candidate)

        llm_query = metadata.get("_llm_query")
        if isinstance(llm_query, dict):
            query_fp = llm_query.get("query_fingerprint")
            if isinstance(query_fp, str):
                snapshot["query_rewrite"][query_fp] = dict(llm_query)

        llm_rerank = metadata.get("_llm_rerank")
        if isinstance(llm_rerank, dict):
            query_fp = llm_rerank.get("query_fingerprint")
            cached_candidate_fp = llm_rerank.get("candidate_fingerprint") or candidate_fp
            if isinstance(query_fp, str) and isinstance(cached_candidate_fp, str):
                snapshot["rerank"][f"{cached_candidate_fp}:{query_fp}"] = dict(llm_rerank)

        llm_summary = metadata.get("_llm_summary")
        if isinstance(llm_summary, dict):
            query_fp = llm_summary.get("query_fingerprint")
            cached_candidate_fp = llm_summary.get("candidate_fingerprint") or candidate_fp
            if isinstance(query_fp, str) and isinstance(cached_candidate_fp, str):
                snapshot["summary"][f"{cached_candidate_fp}:{query_fp}"] = dict(llm_summary)
    return snapshot


def _cached_artifact_for_candidate(
    candidate: PaperCandidate,
    cache_snapshot: dict[str, dict],
    artifact_name: str,
    query_fp: str,
) -> tuple[dict | None, str | None]:
    candidate_fp = candidate_fingerprint(candidate)
    own_artifact = _metadata(candidate).get(f"_llm_{artifact_name}")
    snapshot_artifact = cache_snapshot.get(artifact_name, {}).get(f"{candidate_fp}:{query_fp}")
    return _cache_artifact_valid(
        own_artifact or snapshot_artifact,
        expected_prompt_version=current_prompt_version(),
        expected_query_fingerprint=query_fp,
        expected_candidate_fingerprint=candidate_fp,
    )


def _candidate_payload(candidate: PaperCandidate, index: int = 0) -> dict:
    return {
        "candidate_index": index,
        "candidate_id": getattr(candidate, "id", None),
        "title": getattr(candidate, "title", None),
        "abstract": getattr(candidate, "abstract", None),
        "source_title": getattr(candidate, "source_title", None),
        "year": getattr(candidate, "year", None),
    }


def _rerank_payload(query: str, candidates: list[PaperCandidate]) -> dict:
    return {
        "query": query,
        "candidates": [
            _candidate_payload(candidate, index=index)
            for index, candidate in enumerate(candidates)
        ],
    }


def _summary_payload(candidate: PaperCandidate, query: str) -> dict:
    return build_summary_payload(candidate, query=query)


def _budget_guard(
    budget: LLMSearchBudget,
    *,
    payload: dict,
    default_output_tokens: int = 512,
) -> tuple[dict | None, dict, str | None]:
    usage_metadata, reason = budget.reserve(payload, default_output_tokens=default_output_tokens)
    return (
        usage_metadata,
        {
            "allowed": reason is None,
            "reason": reason,
            "call_count": budget.call_count,
            "budget_remaining": budget.remaining(),
        },
        reason,
    )


def _apply_rerank_artifact(candidate: PaperCandidate, rerank_metadata: dict) -> None:
    metadata = dict(candidate.raw_metadata_json or {})
    relevance_score = max(0.0, min(float(rerank_metadata.get("relevance_score") or 0.0), 1.0))
    trust_score = _trust_total(candidate)
    if trust_score >= 0.75:
        trust_gate = 1.0
    elif trust_score >= 0.45:
        trust_gate = 0.5
    else:
        trust_gate = 0.0

    base_score = _production_reranker_selection_score(candidate)
    if base_score is None:
        base_score = _phase1_selection_score(candidate)
    llm_adjustment = round(relevance_score * trust_gate * 0.05, 4)
    metadata["_llm_rerank"] = rerank_metadata
    metadata["_llm_selection"] = {
        "base_combined_score": round(base_score, 4),
        "llm_relevance_score": round(relevance_score, 4),
        "trust_score": round(trust_score, 4),
        "trust_gate": trust_gate,
        "llm_adjustment": llm_adjustment,
        "combined_score": round(base_score + llm_adjustment, 4),
        "policy": "phase1_selection_plus_trust_gated_llm_rerank_v1",
    }
    candidate.raw_metadata_json = metadata


def _apply_production_reranker(
    candidates: list[PaperCandidate],
    query: str,
) -> dict[str, Any]:
    phase1_sorted = sorted(candidates, key=_candidate_sort_key, reverse=True)
    reranked_candidates, diagnostics = production_rerank_candidates(query, phase1_sorted)
    candidates[:] = reranked_candidates
    if diagnostics.get("enabled") and (
        diagnostics.get("applied") or diagnostics.get("skipped_reason") or diagnostics.get("error")
    ):
        logger.info("Production reranker diagnostics: %s", diagnostics)
    return diagnostics


def _apply_llm_rerank(
    candidates: list[PaperCandidate],
    query: str,
    llm_options: SearchLLMOptions,
    cache_snapshot: dict[str, dict],
    budget: LLMSearchBudget,
    search_filters: SearchFilters | None = None,
    search_started_at: float | None = None,
) -> None:
    if not llm_options.rerank_enabled():
        return

    limit = max(int(settings.llm_max_candidates_for_rerank), 0)
    if limit <= 0:
        for candidate in candidates:
            _store_llm_diagnostic(
                candidate,
                "rerank",
                attempted=False,
                used=False,
                fallback_reason="step_disabled",
                cache_hit=False,
                cache_miss_reason="cache_disabled" if not _cache_enabled() else None,
            )
        return

    bounded_candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)[:limit]
    if not bounded_candidates:
        logger.info("LLM rerank skipped: reason=no_candidates")
        return

    if _optional_llm_time_budget_exceeded(search_started_at):
        _append_query_context_signal(
            search_filters,
            step="llm_rerank",
            warning="llm_rerank_budget_exceeded",
            error="llm_rerank:budget_exceeded",
            status="partial",
        )
        for candidate in bounded_candidates:
            _store_llm_diagnostic(
                candidate,
                "rerank",
                attempted=False,
                used=False,
                fallback_reason="budget_exceeded",
            )
        return

    query_fp = query_fingerprint(query)
    missing_candidates = []
    cache_miss_by_candidate_id = {}
    for candidate in bounded_candidates:
        cached_rerank, cache_miss_reason = _cached_artifact_for_candidate(
            candidate,
            cache_snapshot,
            "rerank",
            query_fp,
        )
        if cached_rerank:
            _apply_rerank_artifact(candidate, cached_rerank)
            _store_llm_diagnostic(
                candidate,
                "rerank",
                attempted=True,
                used=True,
                cache_hit=True,
                budget_guard={
                    "allowed": True,
                    "reason": None,
                    "call_count": budget.call_count,
                    "budget_remaining": budget.remaining(),
                },
            )
            continue

        _store_llm_diagnostic(
            candidate,
            "rerank",
            attempted=True,
            used=False,
            cache_hit=False,
            cache_miss_reason=cache_miss_reason,
        )
        cache_miss_by_candidate_id[id(candidate)] = cache_miss_reason
        missing_candidates.append(candidate)

    if not missing_candidates:
        return

    usage_metadata, budget_guard, budget_reason = _budget_guard(
        budget,
        payload=_rerank_payload(query, missing_candidates),
        default_output_tokens=512,
    )
    if budget_reason:
        _append_query_context_signal(
            search_filters,
            step="llm_rerank",
            warning=f"llm_rerank_{budget_reason}",
            error=f"llm_rerank:{budget_reason}",
            status="partial",
        )
        for candidate in missing_candidates:
            _store_llm_diagnostic(
                candidate,
                "rerank",
                attempted=False,
                used=False,
                fallback_reason=budget_reason,
                cache_hit=False,
                cache_miss_reason=cache_miss_by_candidate_id.get(id(candidate)),
                estimated_usage=usage_metadata,
                budget_guard=budget_guard,
                input_truncated=bool((usage_metadata or {}).get("input_truncated")),
            )
        return

    try:
        rerank_results = rerank_candidates(query=query, candidates=missing_candidates)
    except Exception as exc:  # pragma: no cover - defensive boundary for optional LLMs
        reason = fallback_reason_from_exception(exc)
        logger.warning("LLM rerank failed; retaining Phase 1 ordering: reason=%s", reason)
        _append_query_context_signal(
            search_filters,
            step="llm_rerank",
            warning=f"llm_rerank_{reason}",
            error=f"llm_rerank:{reason}",
            status="partial",
        )
        for candidate in missing_candidates:
            _store_llm_diagnostic(
                candidate,
                "rerank",
                attempted=True,
                used=False,
                fallback_reason=reason,
                cache_hit=False,
                cache_miss_reason=cache_miss_by_candidate_id.get(id(candidate)),
                estimated_usage=usage_metadata,
                budget_guard=budget_guard,
                input_truncated=bool((usage_metadata or {}).get("input_truncated")),
            )
        return

    for result in rerank_results:
        if result.candidate_index < 0 or result.candidate_index >= len(missing_candidates):
            continue

        candidate = missing_candidates[result.candidate_index]
        _store_llm_diagnostic(
            candidate,
            "rerank",
            attempted=result.attempted,
            used=result.used_llm,
            fallback_reason=result.fallback_reason,
            cache_hit=False,
            cache_miss_reason=cache_miss_by_candidate_id.get(id(candidate)),
            estimated_usage={
                "estimated_input_tokens": result.estimated_input_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "estimated_total_tokens": result.estimated_total_tokens,
                "estimated_call_count": result.estimated_call_count,
                "input_truncated": result.input_truncated,
            }
            if result.estimated_call_count
            else usage_metadata,
            budget_guard=budget_guard,
            input_truncated=bool(result.input_truncated or (usage_metadata or {}).get("input_truncated")),
        )
        if not result.used_llm:
            if result.fallback_reason:
                _append_query_context_signal(
                    search_filters,
                    step="llm_rerank",
                    warning=f"llm_rerank_{safe_fallback_reason(result.fallback_reason)}",
                    error=f"llm_rerank:{safe_fallback_reason(result.fallback_reason)}",
                    status="partial",
                )
            continue

        _apply_rerank_artifact(candidate, result.to_metadata())


def _apply_llm_summaries(
    candidates: list[PaperCandidate],
    query: str,
    llm_options: SearchLLMOptions,
    cache_snapshot: dict[str, dict],
    budget: LLMSearchBudget,
    db: Session | None = None,
    search_filters: SearchFilters | None = None,
    search_started_at: float | None = None,
) -> None:
    if not llm_options.summary_enabled():
        return

    if not candidates:
        logger.info("LLM summary skipped: reason=no_candidates")
        return

    configured_candidate_limit = max(int(settings.llm_max_candidates_for_summary), 0)
    configured_generation_limit = max(
        int(getattr(settings, "llm_max_summaries_per_search", settings.llm_max_candidates_for_summary)),
        0,
    )
    if configured_candidate_limit <= 0:
        for candidate in candidates:
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=False,
                used=False,
                fallback_reason="step_disabled",
            )
        return

    candidate_limit = min(configured_candidate_limit, SEARCH_INLINE_SUMMARY_HARD_CAP)
    generation_limit = min(configured_generation_limit, SEARCH_INLINE_SUMMARY_HARD_CAP)

    query_fp = query_fingerprint(query)
    generated_count = 0
    enrichment_count = 0
    enrichment_limit = min(
        SEARCH_SUMMARY_ENRICHMENT_LIMIT,
        candidate_limit,
        generation_limit if generation_limit > 0 else SEARCH_SUMMARY_ENRICHMENT_LIMIT,
    )
    for candidate in candidates[:candidate_limit]:
        if _optional_llm_time_budget_exceeded(search_started_at):
            _append_query_context_signal(
                search_filters,
                step="llm_summary",
                warning="llm_summary_budget_exceeded",
                error="llm_summary:budget_exceeded",
                status="partial",
            )
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=False,
                used=False,
                fallback_reason="budget_exceeded",
            )
            continue

        cached_summary, cache_miss_reason = _cached_artifact_for_candidate(
            candidate,
            cache_snapshot,
            "summary",
            query_fp,
        )
        if cached_summary:
            metadata = dict(candidate.raw_metadata_json or {})
            metadata["_llm_summary"] = cached_summary
            candidate.raw_metadata_json = metadata
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=True,
                used=True,
                cache_hit=True,
                budget_guard={
                    "allowed": True,
                    "reason": None,
                    "call_count": budget.call_count,
                    "budget_remaining": budget.remaining(),
                },
            )
            continue

        if generated_count >= generation_limit:
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=False,
                used=False,
                fallback_reason="step_disabled",
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
            )
            continue

        if (
            getattr(settings, "llm_skip_summary_if_abstract_missing", True)
            and not has_summarizable_abstract(getattr(candidate, "abstract", None))
            and enrichment_count < enrichment_limit
        ):
            enrichment_count += 1
            enrich_candidate_abstract(candidate, timeout=3.0)
            if db is not None and has_summarizable_abstract(getattr(candidate, "abstract", None)):
                db.add(candidate)

        if (
            getattr(settings, "llm_skip_summary_if_abstract_missing", True)
            and not has_summarizable_abstract(getattr(candidate, "abstract", None))
        ):
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=False,
                used=False,
                fallback_reason="no_abstract",
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
            )
            continue

        usage_metadata, budget_guard, budget_reason = _budget_guard(
            budget,
            payload=_summary_payload(candidate, query),
            default_output_tokens=512,
        )
        if budget_reason:
            _append_query_context_signal(
                search_filters,
                step="llm_summary",
                warning=f"llm_summary_{budget_reason}",
                error=f"llm_summary:{budget_reason}",
                status="partial",
            )
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=False,
                used=False,
                fallback_reason=budget_reason,
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
                estimated_usage=usage_metadata,
                budget_guard=budget_guard,
                input_truncated=bool((usage_metadata or {}).get("input_truncated")),
            )
            continue

        generated_count += 1
        try:
            summary = summarize_paper(candidate=candidate, query=query)
        except Exception as exc:  # pragma: no cover - defensive boundary for optional LLMs
            reason = fallback_reason_from_exception(exc)
            logger.warning("LLM summary failed for candidate %s: reason=%s", candidate.id, reason)
            _append_query_context_signal(
                search_filters,
                step="llm_summary",
                warning=f"llm_summary_{reason}",
                error=f"llm_summary:{reason}",
                status="partial",
            )
            _store_llm_diagnostic(
                candidate,
                "summary",
                attempted=True,
                used=False,
                fallback_reason=reason,
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
                estimated_usage=usage_metadata,
                budget_guard=budget_guard,
                input_truncated=bool((usage_metadata or {}).get("input_truncated")),
            )
            continue

        _store_llm_diagnostic(
            candidate,
            "summary",
            attempted=summary.attempted,
            used=summary.used_llm,
            fallback_reason=summary.fallback_reason,
            cache_hit=False,
            cache_miss_reason=cache_miss_reason,
            estimated_usage={
                "estimated_input_tokens": summary.estimated_input_tokens,
                "estimated_output_tokens": summary.estimated_output_tokens,
                "estimated_total_tokens": summary.estimated_total_tokens,
                "estimated_call_count": summary.estimated_call_count,
                "input_truncated": summary.input_truncated,
            }
            if summary.estimated_call_count
            else usage_metadata,
            budget_guard=budget_guard,
            input_truncated=bool(summary.input_truncated or (usage_metadata or {}).get("input_truncated")),
        )
        if not summary.used_llm:
            if summary.fallback_reason:
                reason = safe_fallback_reason(summary.fallback_reason)
                _append_query_context_signal(
                    search_filters,
                    step="llm_summary",
                    warning=f"llm_summary_{reason}",
                    error=f"llm_summary:{reason}",
                    status="partial",
                )
            continue

        metadata = dict(candidate.raw_metadata_json or {})
        metadata["_llm_summary"] = summary.to_metadata()
        candidate.raw_metadata_json = metadata

    for candidate in candidates[candidate_limit:]:
        _store_llm_diagnostic(
            candidate,
            "summary",
            attempted=False,
            used=False,
            fallback_reason="summary_cap_exceeded",
        )


def run_search_pipeline(
    db: Session,
    session_id: int,
    query: str,
    llm_options: SearchLLMOptions | None = None,
    year_filter: SearchYearFilter | None = None,
    search_filters: SearchFilters | None = None,
):
    search_started_at = time.monotonic()
    llm_options = llm_options or SearchLLMOptions()
    if search_filters is None and year_filter is not None:
        search_filters = SearchFilters(query=query, year_filter=year_filter)
    session_obj = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
    topic = session_obj.topic if session_obj and getattr(session_obj, "topic", None) else None
    existing_candidates = (
        db.query(PaperCandidate)
        .filter(PaperCandidate.session_id == session_id)
        .all()
    )
    cache_snapshot = _llm_cache_snapshot(existing_candidates)
    llm_budget = LLMSearchBudget()

    db.query(SelectedPaper).filter(SelectedPaper.session_id == session_id).delete()
    db.query(PaperCandidate).filter(PaperCandidate.session_id == session_id).delete()
    db.flush()

    search_query = query
    query_rewrite_metadata = None
    query_rewrite_diagnostic = None
    if llm_options.query_rewrite_enabled():
        _append_query_context_signal(search_filters, step="query_rewrite", status="running")
        raw_query_fp = query_fingerprint(query, topic)
        cached_query_rewrite, cache_miss_reason = _cache_artifact_valid(
            cache_snapshot.get("query_rewrite", {}).get(raw_query_fp),
            expected_prompt_version=current_prompt_version(),
            expected_query_fingerprint=raw_query_fp,
        )
        if cached_query_rewrite:
            query_rewrite_metadata = cached_query_rewrite
            query_rewrite_diagnostic = _llm_step_diagnostic(
                attempted=True,
                used=True,
                cache_hit=True,
                budget_guard={
                    "allowed": True,
                    "reason": None,
                    "call_count": llm_budget.call_count,
                    "budget_remaining": llm_budget.remaining(),
                },
            )
            search_query = str(cached_query_rewrite.get("optimized_query") or query)
        else:
            query_rewrite_diagnostic = _llm_step_diagnostic(
                attempted=True,
                used=False,
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
            )
            usage_metadata, budget_guard, budget_reason = _budget_guard(
                llm_budget,
                payload={"raw_query": query, "topic": topic},
                default_output_tokens=256,
            )
            if budget_reason:
                _append_query_context_signal(
                    search_filters,
                    step="query_rewrite",
                    warning=f"query_rewrite_{budget_reason}",
                    error=f"query_rewrite:{budget_reason}",
                    status="partial",
                    effective_query=query,
                )
                fallback_rewrite = fallback_query_rewrite(
                    raw_query=query,
                    topic=topic,
                    attempted=True,
                    fallback_reason=budget_reason,
                )
                query_rewrite_metadata = fallback_rewrite.to_metadata()
                query_rewrite_diagnostic = _llm_step_diagnostic(
                    attempted=False,
                    used=False,
                    fallback_reason=budget_reason,
                    cache_hit=False,
                    cache_miss_reason=cache_miss_reason,
                    estimated_usage=usage_metadata,
                    budget_guard=budget_guard,
                    input_truncated=bool((usage_metadata or {}).get("input_truncated")),
                )
            else:
                try:
                    query_rewrite = rewrite_query(raw_query=query, topic=topic)
                    if query_rewrite.attempted:
                        query_rewrite_metadata = query_rewrite.to_metadata()
                        query_usage_metadata = {
                            "estimated_input_tokens": query_rewrite.estimated_input_tokens,
                            "estimated_output_tokens": query_rewrite.estimated_output_tokens,
                            "estimated_total_tokens": query_rewrite.estimated_total_tokens,
                            "estimated_call_count": query_rewrite.estimated_call_count,
                            "input_truncated": query_rewrite.input_truncated,
                        } if query_rewrite.estimated_call_count else usage_metadata
                        query_rewrite_diagnostic = _llm_step_diagnostic(
                            attempted=query_rewrite.attempted,
                            used=query_rewrite.used_llm,
                            fallback_reason=query_rewrite.fallback_reason,
                            cache_hit=False,
                            cache_miss_reason=cache_miss_reason,
                            estimated_usage=query_usage_metadata,
                            budget_guard=budget_guard,
                            input_truncated=bool(query_rewrite.input_truncated or (usage_metadata or {}).get("input_truncated")),
                        )
                    if query_rewrite.used_llm:
                        search_query = query_rewrite.optimized_query or query
                        _append_query_context_signal(
                            search_filters,
                            step="query_rewrite",
                            effective_query=search_query,
                            llm_used=True,
                        )
                    elif query_rewrite.fallback_reason:
                        reason = safe_fallback_reason(query_rewrite.fallback_reason)
                        _append_query_context_signal(
                            search_filters,
                            step="query_rewrite",
                            warning=f"query_rewrite_{reason}",
                            error=f"query_rewrite:{reason}",
                            status="partial",
                            effective_query=query,
                        )
                except Exception as exc:  # pragma: no cover - defensive boundary for optional LLMs
                    reason = fallback_reason_from_exception(exc)
                    logger.warning("LLM query rewrite failed; using raw query: reason=%s", reason)
                    _append_query_context_signal(
                        search_filters,
                        step="query_rewrite",
                        warning=f"query_rewrite_{reason}",
                        error=f"query_rewrite:{reason}",
                        status="partial",
                        effective_query=query,
                    )
                    fallback_rewrite = fallback_query_rewrite(
                        raw_query=query,
                        topic=topic,
                        attempted=True,
                        fallback_reason=reason,
                    )
                    query_rewrite_metadata = fallback_rewrite.to_metadata()
                    query_rewrite_diagnostic = _llm_step_diagnostic(
                        attempted=True,
                        used=False,
                        fallback_reason=reason,
                        cache_hit=False,
                        cache_miss_reason=cache_miss_reason,
                        estimated_usage=usage_metadata,
                        budget_guard=budget_guard,
                        input_truncated=bool((usage_metadata or {}).get("input_truncated")),
                    )

    provider = get_search_provider()

    all_results = []

    # ── Provider-level filter pushdown (OpenAlex native) ─────────────────────
    # OpenAlexStrictProvider supports year, open_access, and min_citations
    # filters at the API level, reducing result count drop from post-filtering.
    # All other providers fall back to post-retrieval filters transparently.
    from app.providers.openalex_strict_provider import OpenAlexStrictProvider as _OAProvider
    _is_openalex = isinstance(provider, _OAProvider)

    _pushdown_year_from: int | None = None
    _pushdown_year_to: int | None = None
    _pushdown_open_access: bool = False
    _pushdown_min_citations: int | None = None

    if _is_openalex and search_filters is not None:
        yf = search_filters.year_filter
        if yf is not None:
            _pushdown_year_from = yf.publication_year_from
            _pushdown_year_to = yf.publication_year_to
        if search_filters.access_type == "open":
            _pushdown_open_access = True
        if search_filters.min_citations is not None and search_filters.min_citations > 0:
            _pushdown_min_citations = search_filters.min_citations

    def _fetch_provider_query(q: str) -> list[Any]:
        try:
            if _is_openalex:
                papers = provider.fetch_papers(
                    query=q,
                    limit=search_filters.fetch_limit() if search_filters else SEARCH_DEFAULT_LIMIT,
                    year_from=_pushdown_year_from,
                    year_to=_pushdown_year_to,
                    open_access_only=_pushdown_open_access,
                    min_citations=_pushdown_min_citations,
                )
            else:
                papers = provider.fetch_papers(
                    query=q,
                    limit=search_filters.fetch_limit() if search_filters else SEARCH_DEFAULT_LIMIT,
                )
            for p in papers:
                if not isinstance(p.raw_metadata, dict):
                    p.raw_metadata = {"upstream_raw_metadata": p.raw_metadata}
                p.raw_metadata["_query_provenance"] = q
            return list(papers)
        except Exception as e:
            logger.warning(f"Query '{q}' failed: {e}")
            return []

    empty_result_retry_used = False
    retry_queries: list[str] = []
    attempted_retry_queries: list[str] = []
    final_search_query = search_query

    papers = _fetch_provider_query(search_query)
    all_results.extend(papers)
    _append_query_context_signal(search_filters, step="provider_search", effective_query=search_query)
    if not all_results and llm_options.query_rewrite_enabled():
        retry_queries = _retry_queries_from_rewrite(
            original_query=query,
            first_query=search_query,
            query_rewrite_metadata=query_rewrite_metadata,
        )
        for retry_query in retry_queries:
            empty_result_retry_used = True
            attempted_retry_queries.append(retry_query)
            final_search_query = retry_query
            papers = _fetch_provider_query(retry_query)
            all_results.extend(papers)
            if all_results:
                break
    if not all_results:
        _append_query_context_signal(
            search_filters,
            step="provider_search",
            warning="no_candidates",
            status="empty",
            effective_query=final_search_query,
        )

    existing_query_context_metadata = (
        search_filters.query_context_metadata
        if search_filters is not None and isinstance(search_filters.query_context_metadata, dict)
        else {}
    )
    query_context_metadata: dict[str, Any] = {
        **existing_query_context_metadata,
        "original_query": query,
        "optimized_query": search_query,
        "effective_query": final_search_query,
        "expanded_queries": [],
        "key_terms": [],
        "llm_query_rewrite_used": _query_rewrite_used(query_rewrite_metadata, query),
        "empty_result_retry_used": empty_result_retry_used,
        "retry_queries": attempted_retry_queries if empty_result_retry_used else [],
        "final_search_query": final_search_query,
    }
    if isinstance(query_rewrite_metadata, dict):
        query_context_metadata.update(
            {
                "original_query": query_rewrite_metadata.get("original_query") or query,
                "optimized_query": query_rewrite_metadata.get("optimized_query") or search_query,
                "expanded_queries": query_rewrite_metadata.get("expanded_queries") or [],
                "key_terms": query_rewrite_metadata.get("key_terms") or [],
                "query_rewrite_notes": query_rewrite_metadata.get("notes"),
                "llm_query_rewrite_used": _query_rewrite_used(query_rewrite_metadata, query),
            }
        )
        if empty_result_retry_used:
            query_rewrite_metadata = {
                **query_rewrite_metadata,
                "empty_result_retry_used": empty_result_retry_used,
                "retry_queries": attempted_retry_queries,
                "final_search_query": final_search_query,
            }
    if search_filters is not None:
        search_filters.query_context_metadata = query_context_metadata

    from app.utils.normalize import normalize_text
    from difflib import SequenceMatcher

    deduped_results = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    seen_title_texts: set[str] = set()
    seen_openalex_ids: set[str] = set()
    seen_provider_ids: set[str] = set()  # S2 paperId, Elsevier eid
    dedup_stats = {"total_raw": len(all_results), "by_doi": 0, "by_title": 0, "by_provider_id": 0, "by_fuzzy_title": 0}

    for r in all_results:
        doi = (r.doi or "").strip().lower()
        title = normalize_text(r.title)
        title_year = f"{title}:{r.year or ''}" if title else ""

        # Extract provider-specific IDs
        provider_id = None
        openalex_id = None
        if isinstance(r.raw_metadata, dict):
            openalex_id = r.raw_metadata.get("openalex_id")
            openalex_id = openalex_id.strip() if isinstance(openalex_id, str) else None
            provider_id = (
                openalex_id
                or r.raw_metadata.get("paper_id")    # Semantic Scholar
                or r.raw_metadata.get("paperId")
                or r.raw_metadata.get("semantic_scholar_id")
                or r.raw_metadata.get("eid")          # Elsevier
            )
            provider_id = provider_id.strip() if isinstance(provider_id, str) else None

        # Exact dedup: DOI
        if doi and doi in seen_dois:
            dedup_stats["by_doi"] += 1
            continue
        # Exact dedup: normalized title
        if openalex_id and openalex_id in seen_openalex_ids:
            dedup_stats["by_provider_id"] += 1
            continue
        if title_year and title_year in seen_titles:
            dedup_stats["by_title"] += 1
            continue
        # Exact dedup: provider ID (OpenAlex, S2, Elsevier)
        if provider_id and provider_id in seen_provider_ids:
            dedup_stats["by_provider_id"] += 1
            continue

        # Fuzzy near-duplicate title detection (ratio >= 0.92)
        is_fuzzy_dup = False
        if title and not doi:
            for existing_title in seen_title_texts:
                if SequenceMatcher(None, title, existing_title).ratio() >= 0.92:
                    is_fuzzy_dup = True
                    break
        if is_fuzzy_dup:
            dedup_stats["by_fuzzy_title"] += 1
            continue

        if doi:
            seen_dois.add(doi)
        if title_year:
            seen_titles.add(title_year)
        if title:
            seen_title_texts.add(title)
        if openalex_id:
            seen_openalex_ids.add(openalex_id)
        if provider_id:
            seen_provider_ids.add(provider_id)

        deduped_results.append(r)

    dedup_stats["after_dedup"] = len(deduped_results)
    dedup_stats["duplicates_removed"] = dedup_stats["total_raw"] - len(deduped_results)
    logger.info("Dedup: %s", dedup_stats)

    results = deduped_results
    if search_filters and search_filters.active:
        search_filters.pre_filter_count = len(results)
        results = apply_pre_source_search_filters(results, search_filters)
    candidates = []

    # Instantiate Elsevier verifier once per pipeline run (saves overhead)
    try:
        from app.providers.elsevier_provider import ElsevierProvider
        _elsevier_verifier = ElsevierProvider()
        _elsevier_available = bool(_elsevier_verifier.api_key)
    except Exception:
        _elsevier_verifier = None  # type: ignore[assignment]
        _elsevier_available = False

    for r in results:
        scopus_url: str | None = None
        canonical_url: str | None = r.url  # default to whatever provider returned

        # ============================================================
        # STEP 1: Confirm in Scopus via Elsevier API (by title + DOI)
        # ============================================================
        elsevier_confirm_meta: dict = {"attempted": False}
        elsevier_scopus_matched = False

        if _elsevier_available and _elsevier_verifier is not None:
            try:
                confirm = _elsevier_verifier.verify_paper_in_scopus(
                    title=r.title,
                    doi=r.doi,
                )
                elsevier_confirm_meta = {
                    "attempted": True,
                    "confirmed": confirm.confirmed,
                    "reason": confirm.reason,
                    "title_similarity": confirm.title_similarity,
                    "scopus_eid": confirm.scopus_eid,
                    "scopus_source_title": confirm.source_title,
                }
                if confirm.confirmed:
                    elsevier_scopus_matched = True
                    if confirm.scopus_url:
                        scopus_url = confirm.scopus_url
                        canonical_url = confirm.scopus_url
                    if confirm.scopus_doi and not r.doi:
                        r.doi = confirm.scopus_doi
            except Exception as exc:
                elsevier_confirm_meta = {"attempted": True, "error": str(exc)}
                logger.warning("Elsevier paper verification failed for '%s': %s", r.title[:60], exc)

        # ============================================================
        # STEP 2: Fallback — match journal/source against .xlsx registry
        # ============================================================
        source_match = resolve_scopus_match(db, r.source_title, raw_metadata=r.raw_metadata)
        xlsx_matched = source_match.source is not None

        # Determine final matched status
        # Priority: Elsevier API confirmation > .xlsx registry match
        if elsevier_scopus_matched:
            matched = True  # confirmed directly in Scopus
            final_match_reason = f"elsevier_api:{elsevier_confirm_meta.get('reason', 'confirmed')}"
            final_confidence = 0.97 if elsevier_confirm_meta.get("title_similarity", 0) >= 0.95 else 0.90
        elif xlsx_matched:
            matched = source_match.source
            final_match_reason = f"xlsx_registry:{source_match.reason}"
            final_confidence = source_match.confidence
        else:
            matched = None
            final_match_reason = "no_match"
            final_confidence = 0.0

        # Build canonical URL: prefer Scopus > DOI > provider URL
        if not canonical_url:
            if r.doi:
                canonical_url = f"https://doi.org/{r.doi}"
            else:
                canonical_url = r.url

        source_match_debug = {
            "matched": matched is not None,
            "match_method": "elsevier_api" if elsevier_scopus_matched else ("xlsx_registry" if xlsx_matched else "none"),
            "source_id": matched.id if getattr(matched, "id", None) else None,
            "source_title": matched.source_title if getattr(matched, "source_title", None) else None,
            "source_issn": matched.issn if getattr(matched, "issn", None) else None,
            "source_eissn": matched.eissn if getattr(matched, "eissn", None) else None,
            "reason": final_match_reason,
            "confidence": final_confidence,
            "scopus_url": scopus_url,
            "canonical_url": canonical_url,
            "registry_metadata": source_match.registry_metadata,
            # --- detailed sub-results ---
            "elsevier_confirm": elsevier_confirm_meta,
            "xlsx_match": {
                "matched": xlsx_matched,
                "reason": source_match.reason,
                "confidence": source_match.confidence,
                "matched_value": source_match.matched_value,
                "normalized_input_title": source_match.normalized_input_title,
                "input_identifiers": source_match.input_identifiers,
                "registry_metadata": source_match.registry_metadata,
            },
        }

        if search_filters and search_filters.scopus_only is True and matched is None:
            continue

        if search_filters and search_filters.limit is not None and len(candidates) >= search_filters.limit:
            continue

        ranking_debug = score_paper(
            query=query,
            paper=r,
            scopus_valid=matched is not None,
            source_match=source_match_debug,
        )
        score = compute_score(
            query=query,
            paper=r,
            scopus_valid=matched is not None,
            source_match=source_match_debug,
        )
        trust_debug = score_trust(
            paper=r,
            source_match=source_match_debug,
        )
        selection_debug = build_selection_debug(
            ranking_score=score,
            trust_debug=trust_debug,
        )

        # Build authoring trust summary for Phase 3
        authoring_trust = build_authoring_trust_summary(
            paper=r,
            source_match=source_match_debug,
            trust_debug=trust_debug,
        )

        raw_metadata = r.raw_metadata if isinstance(r.raw_metadata, dict) else {}
        if not isinstance(r.raw_metadata, dict) and r.raw_metadata is not None:
            raw_metadata = {"upstream_raw_metadata": r.raw_metadata}
        raw_metadata = {
            **raw_metadata,
            "_source_match": source_match_debug,
            "_ranking": ranking_debug,
            "_trust": trust_debug,
            "_selection": selection_debug,
            "_authoring_trust": authoring_trust,
            "_dedup_stats": dedup_stats,
        }
        if query_rewrite_metadata:
            raw_metadata["_llm_query"] = query_rewrite_metadata
        if query_rewrite_diagnostic:
            raw_metadata["_llm_diagnostics"] = {
                "query_rewrite": query_rewrite_diagnostic,
            }

        candidate = PaperCandidate(
            session_id=session_id,
            title=r.title,
            abstract=r.abstract,
            year=r.year,
            authors_json=r.authors,
            doi=r.doi,
            url=canonical_url,  # Use canonical (Scopus/DOI) URL
            source_title=r.source_title,
            normalized_source_title=r.source_title.lower() if r.source_title else None,
            citation_count=r.citation_count,
            provider=r.provider,
            raw_metadata_json=raw_metadata,
            scopus_matched=matched is not None,
            scopus_source_id=matched.id if getattr(matched, "id", None) else None,
            final_score=score,
        )

        db.add(candidate)
        db.flush()

        candidates.append(candidate)

    if search_filters and search_filters.active:
        search_filters.post_filter_count = len(candidates)
    _append_query_context_signal(
        search_filters,
        step="ranking",
        warning="no_candidates" if not candidates else None,
        status="empty" if not candidates else None,
    )

    db.commit()

    _apply_production_reranker(candidates=candidates, query=query)

    _apply_llm_rerank(
        candidates=candidates,
        query=query,
        llm_options=llm_options,
        cache_snapshot=cache_snapshot,
        budget=llm_budget,
        search_filters=search_filters,
        search_started_at=search_started_at,
    )

    # sort top 5 using trust-aware selection policy, then fall back to ranking score.
    candidates.sort(key=_candidate_sort_key, reverse=True)
    top5 = _select_top_candidates(candidates, limit=5)
    _apply_llm_summaries(
        candidates=top5,
        query=query,
        llm_options=llm_options,
        cache_snapshot=cache_snapshot,
        budget=llm_budget,
        db=db,
        search_filters=search_filters,
        search_started_at=search_started_at,
    )

    for idx, c in enumerate(top5):
        db.add(
            SelectedPaper(
                session_id=session_id,
                paper_candidate_id=c.id,
                rank_order=idx + 1,
            )
        )

    if session_obj:
        session_obj.status = "searched"

    db.commit()

    return candidates, top5
