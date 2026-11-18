from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from app.services.llm_client import (
    LLMClient,
    LLMUnavailableError,
    candidate_fingerprint,
    current_prompt_version,
    get_llm_client,
    query_fingerprint,
    utc_timestamp,
)
from app.services.input_guardrails import sanitize_public_text, truncate_for_llm
from app.services.llm_prompts import RERANK_RESULT_FIELDS


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}


@dataclass(frozen=True)
class RerankResult:
    candidate_id: int | None
    candidate_index: int
    llm_relevance_score: float
    reason: str
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
        data = asdict(self)
        data["relevance_score"] = data.pop("llm_relevance_score")
        return data


def _value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _safe_payload_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = sanitize_public_text(str(value).strip())
    if not text:
        return None
    text, _ = truncate_for_llm(text, max_chars)
    return text


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _overlap_score(query: str, candidate: Any) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokens(
        " ".join(
            [
                str(_value(candidate, "title", "") or ""),
                str(_value(candidate, "abstract", "") or ""),
                str(_value(candidate, "source_title", "") or ""),
            ]
        )
    )
    if not candidate_tokens:
        return 0.0
    return round(len(query_tokens & candidate_tokens) / len(query_tokens), 4)


def fallback_rerank(
    query: str,
    candidates: list[Any],
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> list[RerankResult]:
    return [
        RerankResult(
            candidate_id=_value(candidate, "id"),
            candidate_index=index,
            llm_relevance_score=0.0,
            reason="LLM rerank unavailable; Phase 1 ranking retained.",
            attempted=attempted,
            fallback_reason=fallback_reason,
            prompt_version=current_prompt_version(),
            generated_at=utc_timestamp() if attempted else None,
            query_fingerprint=query_fingerprint(query),
            candidate_fingerprint=candidate_fingerprint(candidate),
        )
        for index, candidate in enumerate(candidates)
    ]


def rerank_candidates(
    query: str,
    candidates: list[Any],
    client: LLMClient | None = None,
) -> list[RerankResult]:
    client = client or get_llm_client()

    try:
        generation = client.generate_structured_text(
            task="semantic_rerank",
            payload={
                "query": query,
                "candidates": [
                    {
                        "candidate_index": index,
                        "candidate_id": _value(candidate, "id"),
                        "title": _safe_payload_text(_value(candidate, "title"), max_chars=1000),
                        "abstract": _safe_payload_text(_value(candidate, "abstract"), max_chars=4000),
                        "source_title": _safe_payload_text(_value(candidate, "source_title"), max_chars=1000),
                        "year": _value(candidate, "year"),
                    }
                    for index, candidate in enumerate(candidates)
                ],
            },
        )
    except LLMUnavailableError as exc:
        return fallback_rerank(
            query=query,
            candidates=candidates,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        return [
            RerankResult(
                candidate_id=_value(candidate, "id"),
                candidate_index=index,
                llm_relevance_score=_overlap_score(query, candidate),
                reason="Mock LLM relevance from deterministic query/title/abstract overlap.",
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
            for index, candidate in enumerate(candidates)
        ]

    try:
        raw_results = generation.content.get("results", [])
        if not isinstance(raw_results, list):
            return fallback_rerank(
                query=query,
                candidates=candidates,
                attempted=True,
                fallback_reason="invalid_json",
            )

        by_index = {}
        by_id = {}
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            if not any(field in item for field in RERANK_RESULT_FIELDS):
                continue
            candidate_index = item.get("candidate_index")
            candidate_id = item.get("candidate_id")
            if isinstance(candidate_index, int) and 0 <= candidate_index < len(candidates):
                by_index[candidate_index] = item
            if candidate_id is not None:
                by_id[str(candidate_id)] = item
        if raw_results and not by_index and not by_id:
            return fallback_rerank(
                query=query,
                candidates=candidates,
                attempted=True,
                fallback_reason="invalid_json",
            )

        from difflib import SequenceMatcher
        parsed_results = []
        for index, candidate in enumerate(candidates):
            raw = by_index.get(index) or by_id.get(str(_value(candidate, "id"))) or {}
            
            llm_title = raw.get("title")
            actual_title = _value(candidate, "title")
            hallucinated = False
            if llm_title and actual_title:
                if SequenceMatcher(None, llm_title.lower(), actual_title.lower()).ratio() < 0.5:
                    hallucinated = True

            score = _bounded_score(raw.get("relevance_score"), 0.0) if not hallucinated else 0.0
            reason = _string_or_default(raw.get("reason"), "LLM rerank returned no reason.")
            if hallucinated:
                reason = "[HALLUCINATION DETECTED] Title mismatch in LLM rerank output."

            parsed_results.append(
                RerankResult(
                    candidate_id=_value(candidate, "id"),
                    candidate_index=index,
                    llm_relevance_score=score,
                    reason=reason,
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
            )
        return parsed_results
    except (AttributeError, TypeError, ValueError):
        return fallback_rerank(
            query=query,
            candidates=candidates,
            attempted=True,
            fallback_reason="invalid_json",
        )


def _bounded_score(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(parsed, 1.0)), 4)


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return sanitize_public_text(value.strip()) or default
    return default
