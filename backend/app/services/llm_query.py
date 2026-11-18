from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any

from app.services.llm_client import (
    LLMClient,
    LLMUnavailableError,
    current_prompt_version,
    get_llm_client,
    query_fingerprint,
    utc_timestamp,
)
from app.services.input_guardrails import sanitize_public_text
from app.services.llm_prompts import QUERY_REWRITE_FIELDS


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


_VIETNAMESE_ACADEMIC_HINTS = (
    ("mat ngu", ["insomnia", "sleep disorder"]),
    ("roi loan giac ngu", ["sleep disorder"]),
    ("giac ngu", ["sleep"]),
    ("thuoc", ["pharmacological treatment", "medication"]),
    ("dieu tri", ["treatment", "therapy"]),
    ("chua", ["treatment", "therapy"]),
    ("tri", ["treatment", "therapy"]),
    ("benzodiazepine", ["benzodiazepines"]),
    ("melatonin", ["melatonin"]),
)


def is_likely_non_english_query(text: str | None) -> bool:
    query = (text or "").strip()
    if not query:
        return False
    non_space_chars = [char for char in query if not char.isspace()]
    if not non_space_chars:
        return False
    ascii_chars = [char for char in non_space_chars if ord(char) < 128]
    ascii_ratio = len(ascii_chars) / len(non_space_chars)
    return ascii_ratio < 0.85 or any(ord(char) > 127 for char in query)


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    optimized_query: str
    expanded_queries: list[str]
    key_terms: list[str]
    notes: str
    provider: str = "fallback"
    model: str = "none"
    used_llm: bool = False
    attempted: bool = False
    fallback_reason: str | None = None
    prompt_version: str | None = None
    generated_at: str | None = None
    query_fingerprint: str | None = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_call_count: int = 0
    input_truncated: bool = False

    def to_metadata(self) -> dict:
        return asdict(self)


def _key_terms(raw_query: str, topic: str | None = None) -> list[str]:
    text = f"{topic or ''} {raw_query}".strip().lower()
    terms = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token not in _STOPWORDS and len(token) > 1 and token not in terms:
            terms.append(token)
    return terms[:8]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _dedupe_strings(values: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for value in values:
        text = value.strip() if isinstance(value, str) else ""
        key = re.sub(r"\s+", " ", text.lower())
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def _normalized_query(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _english_hint_rewrite(raw_query: str, topic: str | None = None) -> dict[str, list[str] | str] | None:
    normalized = _strip_accents((raw_query or "").lower())
    terms: list[str] = []
    for needle, hints in _VIETNAMESE_ACADEMIC_HINTS:
        if needle in normalized:
            terms.extend(hints)

    if not terms:
        return None

    terms = _dedupe_strings(terms)
    if "insomnia" in terms and "pharmacological treatment" in terms:
        optimized_terms = [
            "pharmacological treatment",
            "insomnia",
            "sleep disorder",
            "medication",
        ]
    else:
        optimized_terms = terms
    if topic and not is_likely_non_english_query(topic):
        optimized_terms.append(topic.strip())

    optimized_query = " ".join(_dedupe_strings(optimized_terms))
    expanded_queries = [optimized_query]
    if "insomnia" in terms:
        expanded_queries.extend(
            [
                "insomnia pharmacological therapy",
                "sleep disorder treatment medication",
                "hypnotics melatonin benzodiazepines insomnia treatment",
            ]
        )
    key_terms = [
        term
        for term in terms
        if re.fullmatch(r"[a-z0-9][a-z0-9\s-]*", term.lower())
    ]
    return {
        "optimized_query": optimized_query,
        "expanded_queries": _dedupe_strings(expanded_queries),
        "key_terms": _dedupe_strings(key_terms),
    }


def fallback_query_rewrite(
    raw_query: str,
    topic: str | None = None,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> QueryRewriteResult:
    query = (raw_query or "").strip()
    topic_text = (topic or "").strip()
    terms = _key_terms(query, topic_text)

    expanded_queries = [query] if query else []
    if topic_text and topic_text.lower() not in query.lower():
        expanded_queries.append(f"{query} {topic_text}".strip())
    if terms:
        expanded_terms_query = " ".join(terms)
        if expanded_terms_query not in expanded_queries:
            expanded_queries.append(expanded_terms_query)

    return QueryRewriteResult(
        original_query=query,
        optimized_query=query,
        expanded_queries=expanded_queries[:3],
        key_terms=terms,
        notes=(
            f"Deterministic fallback; raw query preserved ({fallback_reason})."
            if fallback_reason
            else "Deterministic fallback; raw query preserved."
        ),
        attempted=attempted,
        fallback_reason=fallback_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
        query_fingerprint=query_fingerprint(query, topic_text),
    )


def rewrite_query(
    raw_query: str,
    topic: str | None = None,
    client: LLMClient | None = None,
) -> QueryRewriteResult:
    client = client or get_llm_client()
    fallback = fallback_query_rewrite(raw_query=raw_query, topic=topic)

    try:
        generation = client.generate_structured_text(
            task="query_rewrite",
            payload={"raw_query": raw_query, "topic": topic},
        )
    except LLMUnavailableError as exc:
        return fallback_query_rewrite(
            raw_query=raw_query,
            topic=topic,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        hint_rewrite = _english_hint_rewrite(fallback.original_query, topic)
        if hint_rewrite and is_likely_non_english_query(fallback.original_query):
            return QueryRewriteResult(
                original_query=fallback.original_query,
                optimized_query=str(hint_rewrite["optimized_query"]),
                expanded_queries=list(hint_rewrite["expanded_queries"])[:3],
                key_terms=list(hint_rewrite["key_terms"])[:12],
                notes="Mock LLM rewrite translated non-English query deterministically for tests.",
                provider=generation.provider,
                model=generation.model,
                used_llm=True,
                attempted=True,
                prompt_version=generation.prompt_version or current_prompt_version(),
                generated_at=utc_timestamp(),
                query_fingerprint=query_fingerprint(fallback.original_query, topic),
                estimated_input_tokens=generation.estimated_input_tokens,
                estimated_output_tokens=generation.estimated_output_tokens,
                estimated_total_tokens=generation.estimated_total_tokens,
                estimated_call_count=generation.estimated_call_count,
                input_truncated=generation.input_truncated,
            )

        terms = fallback.key_terms
        optimized_terms = " ".join(terms[:6])
        optimized_query = optimized_terms or fallback.optimized_query
        expanded_queries = [fallback.original_query]
        if optimized_query and optimized_query not in expanded_queries:
            expanded_queries.append(optimized_query)
        if topic:
            topic_query = f"{optimized_query} {topic}".strip()
            if topic_query and topic_query not in expanded_queries:
                expanded_queries.append(topic_query)

        return QueryRewriteResult(
            original_query=fallback.original_query,
            optimized_query=optimized_query,
            expanded_queries=expanded_queries[:3],
            key_terms=terms,
            notes="Mock LLM rewrite generated deterministically for tests.",
            provider=generation.provider,
            model=generation.model,
            used_llm=True,
            attempted=True,
            prompt_version=generation.prompt_version or current_prompt_version(),
            generated_at=utc_timestamp(),
            query_fingerprint=query_fingerprint(fallback.original_query, topic),
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
        if not any(field in content for field in QUERY_REWRITE_FIELDS):
            raise ValueError("structured LLM content does not match query rewrite contract")
        optimized_query = _string_or_default(content.get("optimized_query"), fallback.optimized_query)
        expanded_queries = _string_list_or_default(content.get("expanded_queries"), fallback.expanded_queries)
        key_terms = _string_list_or_default(content.get("key_terms"), fallback.key_terms)
        notes = _string_or_default(content.get("notes"), "LLM rewrite parsed with defaults for missing fields.")
    except (AttributeError, TypeError, ValueError):
        return fallback_query_rewrite(
            raw_query=raw_query,
            topic=topic,
            attempted=True,
            fallback_reason="invalid_json",
        )

    hint_rewrite = _english_hint_rewrite(fallback.original_query, topic)
    if hint_rewrite and is_likely_non_english_query(fallback.original_query):
        optimized_same_as_original = _normalized_query(optimized_query) == _normalized_query(fallback.original_query)
        has_english_alternative = any(not is_likely_non_english_query(item) for item in expanded_queries + key_terms)
        if optimized_same_as_original or not has_english_alternative:
            optimized_query = str(hint_rewrite["optimized_query"])
            expanded_queries = _dedupe_strings(
                list(hint_rewrite["expanded_queries"]) + [
                    item for item in expanded_queries if not is_likely_non_english_query(item)
                ]
            )
            key_terms = _dedupe_strings(
                list(hint_rewrite["key_terms"]) + [
                    item for item in key_terms if not is_likely_non_english_query(item)
                ]
            )
            notes = f"{notes} Non-English query repaired with English academic terms.".strip()

    return QueryRewriteResult(
        original_query=fallback.original_query,
        optimized_query=optimized_query,
        expanded_queries=expanded_queries[:5],
        key_terms=key_terms[:12],
        notes=notes,
        provider=generation.provider,
        model=generation.model,
        used_llm=True,
        attempted=True,
        prompt_version=generation.prompt_version or current_prompt_version(),
        generated_at=utc_timestamp(),
        query_fingerprint=query_fingerprint(fallback.original_query, topic),
        estimated_input_tokens=generation.estimated_input_tokens,
        estimated_output_tokens=generation.estimated_output_tokens,
        estimated_total_tokens=generation.estimated_total_tokens,
        estimated_call_count=generation.estimated_call_count,
        input_truncated=generation.input_truncated,
    )


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return sanitize_public_text(value.strip()) or default
    return default


def _string_list_or_default(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    cleaned = []
    for item in value:
        if isinstance(item, str) and item.strip():
            safe_item = sanitize_public_text(item.strip())
            if safe_item and safe_item not in cleaned:
                cleaned.append(safe_item)
    return cleaned or default
