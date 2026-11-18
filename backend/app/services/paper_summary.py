from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.services.llm_client import (
    LLMClient,
    LLMUnavailableError,
    candidate_fingerprint,
    current_prompt_version,
    get_llm_client,
    query_fingerprint,
    safe_fallback_reason,
    utc_timestamp,
)
from app.services.input_guardrails import sanitize_public_text, truncate_for_llm
from app.services.llm_prompts import PAPER_SUMMARY_FIELDS


SENSITIVE_PAYLOAD_KEY_FRAGMENTS = (
    "api_key",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "cookie",
    "_llm",
    "_ranking",
    "_trust",
    "_source_match",
    "raw_prompt",
    "raw_model_response",
    "raw_response",
    "raw_metadata_json",
    "internal",
    "debug",
    "diagnostics",
)


@dataclass(frozen=True)
class PaperSummaryResult:
    one_sentence_summary: str
    contribution: str
    method_or_data: str
    relevance_to_query: str
    limitations_or_caution: str
    provider: str = "fallback"
    model: str = "none"
    used_llm: bool = False
    attempted: bool = False
    fallback_reason: str | None = None
    prompt_version: str | None = None
    generated_at: str | None = None
    query_fingerprint: str | None = None
    candidate_fingerprint: str | None = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_call_count: int = 0
    input_truncated: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _safe_summary_field_map(summary: dict[str, Any]) -> dict[str, str] | None:
    one_sentence_summary = _safe_string(summary.get("one_sentence_summary"))
    contribution = _safe_string(summary.get("contribution")) or _safe_string(summary.get("key_findings"))
    method_or_data = _safe_string(summary.get("method_or_data")) or _safe_string(summary.get("main_methodology"))
    relevance_to_query = _safe_string(summary.get("relevance_to_query"))
    limitations_or_caution = (
        _safe_string(summary.get("limitations_or_caution"))
        or _safe_string(summary.get("research_gap"))
        or _safe_string(summary.get("potential_contributions"))
    )
    if not all(
        [
            one_sentence_summary,
            contribution,
            method_or_data,
            relevance_to_query,
            limitations_or_caution,
        ]
    ):
        return None
    return {
        "one_sentence_summary": one_sentence_summary,
        "contribution": contribution,
        "method_or_data": method_or_data,
        "relevance_to_query": relevance_to_query,
        "limitations_or_caution": limitations_or_caution,
    }


def cached_paper_summary(
    metadata: dict[str, Any] | None,
    *,
    candidate: Any,
    query: str | None = None,
) -> PaperSummaryResult | None:
    """Return a validated cached paper summary, or None if it is stale/malformed."""
    if not isinstance(metadata, dict):
        return None
    summary = metadata.get("_llm_summary")
    if not isinstance(summary, dict):
        return None

    fields = _safe_summary_field_map(summary)
    if not fields:
        return None

    prompt_version = _safe_string(summary.get("prompt_version"))
    if prompt_version and prompt_version != current_prompt_version():
        return None

    query_fp = _safe_string(summary.get("query_fingerprint"))
    if query_fp and query_fp != query_fingerprint(query):
        return None

    candidate_fp = _safe_string(summary.get("candidate_fingerprint"))
    if candidate_fp and candidate_fp != candidate_fingerprint(candidate):
        return None

    fallback_reason = _safe_string(summary.get("fallback_reason"))
    return PaperSummaryResult(
        one_sentence_summary=fields["one_sentence_summary"],
        contribution=fields["contribution"],
        method_or_data=fields["method_or_data"],
        relevance_to_query=fields["relevance_to_query"],
        limitations_or_caution=fields["limitations_or_caution"],
        provider=_safe_string(summary.get("provider")) or "fallback",
        model=_safe_string(summary.get("model")) or "none",
        used_llm=summary.get("used_llm") is True,
        attempted=summary.get("attempted") is not False,
        fallback_reason=safe_fallback_reason(fallback_reason) if fallback_reason else None,
        prompt_version=prompt_version or current_prompt_version(),
        generated_at=_safe_string(summary.get("generated_at")),
        query_fingerprint=query_fp,
        candidate_fingerprint=candidate_fp,
        estimated_input_tokens=_safe_int(summary.get("estimated_input_tokens")) or 0,
        estimated_output_tokens=_safe_int(summary.get("estimated_output_tokens")) or 0,
        estimated_total_tokens=_safe_int(summary.get("estimated_total_tokens")) or 0,
        estimated_call_count=_safe_int(summary.get("estimated_call_count")) or 0,
        input_truncated=summary.get("input_truncated") is True,
    )


def store_paper_summary_metadata(candidate: Any, result: PaperSummaryResult) -> None:
    metadata = _value(candidate, "raw_metadata_json")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["_llm_summary"] = result.to_metadata()
    setattr(candidate, "raw_metadata_json", metadata)


def _value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _sensitive_key(name: str) -> bool:
    normalized = name.lower()
    return any(fragment in normalized for fragment in SENSITIVE_PAYLOAD_KEY_FRAGMENTS)


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        text = sanitize_public_text(value.strip())
        return text or None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_authors(value: Any) -> list[str] | None:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if not isinstance(value, list):
        return None

    authors: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        author = item.strip()
        if author and author not in authors:
            authors.append(author)
    return authors[:12] or None


def build_summary_payload(candidate: Any, query: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    safe_string_fields = (
        "query",
        "title",
        "abstract",
        "source_title",
        "venue",
        "journal",
        "doi",
        "url",
        "provider",
    )
    values = {
        "query": query,
        "title": _value(candidate, "title"),
        "abstract": _value(candidate, "abstract"),
        "source_title": _value(candidate, "source_title"),
        "venue": _value(candidate, "venue"),
        "journal": _value(candidate, "journal"),
        "doi": _value(candidate, "doi"),
        "url": _value(candidate, "url"),
        "provider": _value(candidate, "provider"),
    }

    for field in safe_string_fields:
        if _sensitive_key(field):
            continue
        text = _safe_string(values.get(field))
        if text:
            if field == "abstract":
                text, _ = truncate_for_llm(text, 8000)
            elif field in {"query", "title"}:
                text, _ = truncate_for_llm(text, 1000)
            else:
                text, _ = truncate_for_llm(text, 1500)
            payload[field] = text

    for field in ("year", "citation_count"):
        if _sensitive_key(field):
            continue
        parsed = _safe_int(_value(candidate, field))
        if parsed is not None:
            payload[field] = parsed

    authors = _safe_authors(_value(candidate, "authors_json", _value(candidate, "authors", None)))
    if authors:
        payload["authors"] = authors

    return payload


def fallback_paper_summary(
    candidate: Any,
    query: str | None = None,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> PaperSummaryResult:
    title = str(_value(candidate, "title", "Untitled paper") or "Untitled paper").strip()
    title = sanitize_public_text(title) or "Untitled paper"
    year = _value(candidate, "year")
    source_title = sanitize_public_text(str(_value(candidate, "source_title") or "")) or "unknown source"
    abstract = sanitize_public_text(str(_value(candidate, "abstract", "") or "").strip())

    year_text = f" ({year})" if year else ""
    contribution = abstract[:180].strip()
    if not contribution:
        contribution = "No abstract was available, so contribution cannot be inferred safely."

    return PaperSummaryResult(
        one_sentence_summary=f"{title}{year_text} is a candidate paper from {source_title}.",
        contribution=contribution,
        method_or_data="Not specified in available metadata.",
        relevance_to_query=(
            f"Potentially relevant to '{query}' based on available metadata."
            if query
            else "Potentially relevant based on available metadata."
        ),
        limitations_or_caution="Verify metadata-derived claims against the full paper before citation.",
        attempted=attempted,
        fallback_reason=fallback_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
        query_fingerprint=query_fingerprint(query),
        candidate_fingerprint=candidate_fingerprint(candidate),
    )


def local_extractive_summary(candidate: Any, query: str | None = None) -> PaperSummaryResult:
    """Select an actual abstract sentence with the local query-document model."""
    import re
    from app.core.config import settings
    from app.services.reranker import _load_cross_encoder, _predict_with_timeout

    abstract = sanitize_public_text(str(_value(candidate, "abstract", "") or "").strip())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", abstract) if part.strip()]
    if not sentences:
        return fallback_paper_summary(candidate, query, attempted=True, fallback_reason="no_abstract")
    sentences = sentences[:30]
    model = _load_cross_encoder(settings.reranker_model_name)
    scores = _predict_with_timeout(model, [(query or str(_value(candidate, "title", "")), s) for s in sentences],
                                   batch_size=settings.reranker_batch_size,
                                   timeout_seconds=settings.reranker_timeout_seconds)
    import math
    if len(scores) != len(sentences) or not all(math.isfinite(x) for x in scores):
        raise ValueError("Invalid local analysis scores")
    chosen = sentences[max(range(len(sentences)), key=lambda i: scores[i])]
    return PaperSummaryResult(
        one_sentence_summary="Query-relevant sentence selected from the abstract:",
        contribution="Not independently determined; read the full paper.",
        method_or_data="Not independently extracted; read the paper to identify its methods and data.",
        relevance_to_query=chosen,
        limitations_or_caution="This is an extract from the abstract, not a full-paper assessment. Verify context before citation.",
        provider="local_extractive", model=settings.reranker_model_name, used_llm=False, attempted=True,
        prompt_version=current_prompt_version(), generated_at=utc_timestamp(),
        query_fingerprint=query_fingerprint(query), candidate_fingerprint=candidate_fingerprint(candidate),
    )


def summarize_paper(
    candidate: Any,
    query: str | None = None,
    client: LLMClient | None = None,
) -> PaperSummaryResult:
    client = client or get_llm_client()
    fallback = fallback_paper_summary(candidate=candidate, query=query)

    try:
        generation = client.generate_structured_text(
            task="paper_summary",
            payload=build_summary_payload(candidate, query=query),
        )
    except LLMUnavailableError as exc:
        return fallback_paper_summary(
            candidate=candidate,
            query=query,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        title = sanitize_public_text(str(_value(candidate, "title", "Untitled paper") or "Untitled paper").strip())
        abstract = sanitize_public_text(str(_value(candidate, "abstract", "") or "").strip())
        source_title = sanitize_public_text(str(_value(candidate, "source_title") or "")) or "unknown source"
        method_or_data = "Available abstract metadata" if abstract else "Not specified in available metadata."

        return PaperSummaryResult(
            one_sentence_summary=f"{title} is a relevant paper from {source_title}.",
            contribution=abstract[:200].strip() or "No abstract was available.",
            method_or_data=method_or_data,
            relevance_to_query=(
                f"Mock summary links this paper to '{query}'."
                if query
                else "Mock summary based on title and abstract metadata."
            ),
            limitations_or_caution="Mock analysis: verify against the full paper before citation.",
            provider=generation.provider,
            model=generation.model,
            used_llm=True,
            attempted=True,
            prompt_version=generation.prompt_version or current_prompt_version(),
            generated_at=utc_timestamp(),
            query_fingerprint=query_fingerprint(query),
            candidate_fingerprint=candidate_fingerprint(candidate),
            estimated_input_tokens=generation.estimated_input_tokens,
            estimated_output_tokens=generation.estimated_output_tokens,
            estimated_total_tokens=generation.estimated_total_tokens,
            estimated_call_count=generation.estimated_call_count,
            input_truncated=generation.input_truncated,
        )

    try:
        if not isinstance(generation.content, dict):
            raise ValueError("structured LLM content must be an object")
        content = generation.content
        if not any(field in content for field in PAPER_SUMMARY_FIELDS):
            raise ValueError("structured LLM content does not match paper summary contract")
        return PaperSummaryResult(
            one_sentence_summary=_string_or_default(
                content.get("one_sentence_summary"),
                fallback.one_sentence_summary,
            ),
            contribution=_string_or_default(content.get("contribution"), fallback.contribution),
            method_or_data=_string_or_default(content.get("method_or_data"), fallback.method_or_data),
            relevance_to_query=_string_or_default(
                content.get("relevance_to_query"),
                fallback.relevance_to_query,
            ),
            limitations_or_caution=_string_or_default(
                content.get("limitations_or_caution"),
                fallback.limitations_or_caution,
            ),
            provider=generation.provider,
            model=generation.model,
            used_llm=True,
            attempted=True,
            prompt_version=generation.prompt_version or current_prompt_version(),
            generated_at=utc_timestamp(),
            query_fingerprint=query_fingerprint(query),
            candidate_fingerprint=candidate_fingerprint(candidate),
            estimated_input_tokens=generation.estimated_input_tokens,
            estimated_output_tokens=generation.estimated_output_tokens,
            estimated_total_tokens=generation.estimated_total_tokens,
            estimated_call_count=generation.estimated_call_count,
            input_truncated=generation.input_truncated,
        )
    except (AttributeError, TypeError, ValueError):
        return fallback_paper_summary(
            candidate=candidate,
            query=query,
            attempted=True,
            fallback_reason="invalid_json",
        )


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return sanitize_public_text(value.strip()) or default
    return default
