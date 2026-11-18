import math
import re
from datetime import datetime
from typing import Any

from app.providers.base import PaperResult

RANKING_WEIGHTS = {
    "title_relevance": 0.25,      # reduced from 0.35
    "abstract_relevance": 0.15,   # reduced from 0.20
    "citation_score": 0.17,
    "recency_score": 0.12,        # increased from 0.10
    "source_match_strength": 0.10, # increased from 0.08
    "source_match_reason_bonus": 0.03, # increased from 0.02
    "abstract_present": 0.08,
    "source_present": 0.05,       # increased from 0.03
    "provider_trust": 0.05,
}

SOURCE_MATCH_REASON_SCORES = {
    "identifier_match": 1.0,
    "registry_identifier_match": 0.98,
    "exact_normalized_title": 0.95,
    "related_title_match": 0.88,
    "exact_title_ignoring_leading_article": 0.85,
    "exact_title_after_light_singularization": 0.8,
    "high_confidence_fuzzy_title": 0.7,
    "no_match": 0.0,
    "no_source_metadata": 0.0,
}

TRUST_WEIGHTS = {
    "source_match_confidence": 0.22,
    "source_match_reason_strength": 0.10,
    "registry_category_strength": 0.14,
    "journal_appropriateness": 0.10,
    "discontinued_safety": 0.10,
    "strict_provider_evidence": 0.06,
    "medline_bonus": 0.04,
    "doi_present": 0.08,
    "abstract_present": 0.06,
    "source_present": 0.03,
    "year_present": 0.03,
    "recency_sanity": 0.02,
    "citation_sanity": 0.02,
}

SELECTION_WEIGHTS = {
    "ranking_score": 0.75,
    "trust_score": 0.25,
}

TRUST_BUCKET_BONUSES = {
    "high": 0.03,
    "medium": 0.0,
    "low": -0.06,
}

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
    "with",
}


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _overlap_ratio(query_tokens: set[str], text: str | None) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _citation_score(citation_count: int | None) -> float:
    if not citation_count or citation_count <= 0:
        return 0.0
    return min(math.log1p(citation_count) / math.log1p(1000), 1.0)


def _recency_score(year: int | None) -> float:
    if not year:
        return 0.0
    current_year = datetime.utcnow().year
    age = max(current_year - year, 0)
    return max(0.0, 1.0 - (age / 15.0))


def _provider_trust(provider: str | None) -> float:
    normalized = (provider or "").strip().lower()
    if normalized == "elsevier":
        return 1.5  # Top priority for Scopus native results
    if normalized == "semantic_scholar":
        return 1.0
    if normalized == "mock":
        return 0.6
    if normalized:
        return 0.5
    return 0.0


def _year_present(year: int | None) -> float:
    return 1.0 if year else 0.0


def _recency_sanity(year: int | None) -> float:
    if not year:
        return 0.0
    current_year = datetime.utcnow().year
    if year < 1900 or year > current_year + 1:
        return 0.0
    age = max(current_year - year, 0)
    if age <= 15:
        return 1.0
    if age <= 30:
        return 0.7
    if age <= 50:
        return 0.4
    return 0.2


def _citation_sanity(citation_count: int | None) -> float:
    if citation_count is None:
        return 0.0
    if citation_count < 0:
        return 0.0
    if citation_count == 0:
        return 0.4
    if citation_count <= 5000:
        return 1.0
    if citation_count <= 20000:
        return 0.7
    return 0.4


def _registry_metadata(source_match: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source_match, dict):
        return {}
    metadata = source_match.get("registry_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _provider_metadata(paper: PaperResult) -> dict[str, Any]:
    return paper.raw_metadata if isinstance(paper.raw_metadata, dict) else {}


def _registry_category_strength(source_match: dict[str, Any] | None) -> float:
    category = str(_registry_metadata(source_match).get("registry_category") or "").strip().lower()
    if category == "active_indexed_journal":
        return 1.0
    if category == "accepted_title":
        return 0.58
    if category == "inactive_indexed_source":
        return 0.55
    if category == "conference_proceeding":
        return 0.25
    if category == "non_journal_source":
        return 0.3
    if category == "discontinued_title":
        return 0.1
    if category:
        return 0.45
    return 0.0


def _journal_appropriateness(
    paper: PaperResult,
    source_match: dict[str, Any] | None,
) -> float:
    registry_metadata = _registry_metadata(source_match)
    registry_source_type = str(registry_metadata.get("source_type") or "").strip().lower()
    provider_source_type = str(_provider_metadata(paper).get("source_type") or "").strip().lower()

    source_type = registry_source_type or provider_source_type
    if not source_type:
        return 0.5 if paper.source_title else 0.0
    if "conference" in source_type or "proceeding" in source_type:
        return 0.15
    if "journal" in source_type:
        return 1.0
    return 0.35


def _discontinued_safety(source_match: dict[str, Any] | None) -> float:
    metadata = _registry_metadata(source_match)
    if not metadata:
        return 0.5
    if metadata.get("discontinued_due_to_quality"):
        return 0.0
    if metadata.get("is_discontinued"):
        return 0.2
    return 1.0


def _medline_bonus(source_match: dict[str, Any] | None) -> float:
    return 1.0 if _registry_metadata(source_match).get("medline_sourced") else 0.0


def _strict_provider_evidence(paper: PaperResult) -> float:
    raw_metadata = _provider_metadata(paper)
    provenance = raw_metadata.get("filter_provenance")
    if not isinstance(provenance, dict):
        return 0.0

    provider_mode = str(provenance.get("provider_mode") or "").strip().lower()
    if provider_mode != "openalex_strict":
        return 0.0

    evidence = 0.0
    if raw_metadata.get("is_core"):
        evidence += 0.35
    if raw_metadata.get("is_in_doaj"):
        evidence += 0.35
    if raw_metadata.get("is_retracted") is False:
        evidence += 0.2
    if provenance.get("filter"):
        evidence += 0.1
    return round(min(evidence, 1.0), 4)


def _source_match_strength(
    source_match: dict[str, Any] | None,
    scopus_valid: bool,
) -> float:
    if not scopus_valid or not isinstance(source_match, dict):
        return 0.0
    confidence = source_match.get("confidence")
    if isinstance(confidence, (int, float)):
        return max(0.0, min(float(confidence), 1.0))
    return 1.0 if scopus_valid else 0.0


def _source_match_reason_score(
    source_match: dict[str, Any] | None,
    scopus_valid: bool,
) -> float:
    if not scopus_valid or not isinstance(source_match, dict):
        return 0.0

    reason = str(source_match.get("reason") or "").strip().lower()
    # Handle prefixed reasons from orchestrator (e.g. "elsevier_api:doi_match")
    if reason.startswith("elsevier_api:"):
        return 1.0  # Scopus-confirmed is top-tier
    if reason.startswith("xlsx_registry:"):
        reason = reason.split(":", 1)[1]
    if reason.startswith("identifier_match:"):
        reason = "identifier_match"
    if reason.startswith("registry_identifier_match:"):
        reason = "registry_identifier_match"

    return SOURCE_MATCH_REASON_SCORES.get(reason, 0.6 if reason else 0.0)


def _trust_bucket(total: float) -> str:
    if total >= 0.75:
        return "high"
    if total >= 0.45:
        return "medium"
    return "low"


def score_paper(
    query: str,
    paper: PaperResult,
    scopus_valid: bool,
    source_match: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    components = {
        "title_relevance": _overlap_ratio(query_tokens, paper.title),
        "abstract_relevance": _overlap_ratio(query_tokens, paper.abstract),
        "citation_score": _citation_score(paper.citation_count),
        "recency_score": _recency_score(paper.year),
        "source_match_strength": _source_match_strength(source_match, scopus_valid),
        "source_match_reason_bonus": _source_match_reason_score(source_match, scopus_valid),
        "abstract_present": 1.0 if paper.abstract else 0.0,
        "source_present": 1.0 if paper.source_title else 0.0,
        "provider_trust": _provider_trust(paper.provider),
    }

    weighted = {
        name: round(value * RANKING_WEIGHTS[name], 4)
        for name, value in components.items()
    }
    total = round(sum(weighted.values()), 4)

    return {
        "weights": RANKING_WEIGHTS.copy(),
        "components": {name: round(value, 4) for name, value in components.items()},
        "weighted": weighted,
        "total": total,
    }


def score_trust(
    paper: PaperResult,
    source_match: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_match_confidence = 0.0
    if isinstance(source_match, dict):
        confidence = source_match.get("confidence")
        if isinstance(confidence, (int, float)):
            source_match_confidence = max(0.0, min(float(confidence), 1.0))

    source_match_reason_strength = _source_match_reason_score(
        source_match=source_match,
        scopus_valid=bool(source_match_confidence > 0.0),
    )

    components = {
        "source_match_confidence": source_match_confidence,
        "source_match_reason_strength": source_match_reason_strength,
        "registry_category_strength": _registry_category_strength(source_match),
        "journal_appropriateness": _journal_appropriateness(paper, source_match),
        "discontinued_safety": _discontinued_safety(source_match),
        "strict_provider_evidence": _strict_provider_evidence(paper),
        "medline_bonus": _medline_bonus(source_match),
        "doi_present": 1.0 if paper.doi else 0.0,
        "abstract_present": 1.0 if paper.abstract else 0.0,
        "source_present": 1.0 if paper.source_title else 0.0,
        "year_present": _year_present(paper.year),
        "recency_sanity": _recency_sanity(paper.year),
        "citation_sanity": _citation_sanity(paper.citation_count),
    }

    weighted = {
        name: round(value * TRUST_WEIGHTS[name], 4)
        for name, value in components.items()
    }
    total = round(sum(weighted.values()), 4)

    return {
        "weights": TRUST_WEIGHTS.copy(),
        "components": {name: round(value, 4) for name, value in components.items()},
        "weighted": weighted,
        "total": total,
        "bucket": _trust_bucket(total),
    }


def build_selection_debug(ranking_score: float, trust_debug: dict[str, Any]) -> dict[str, Any]:
    trust_score = 0.0
    trust_bucket = "low"
    if isinstance(trust_debug, dict):
        raw_total = trust_debug.get("total")
        if isinstance(raw_total, (int, float)):
            trust_score = max(0.0, min(float(raw_total), 1.0))
        trust_bucket = str(trust_debug.get("bucket") or "low").strip().lower()

    bucket_adjustment = TRUST_BUCKET_BONUSES.get(trust_bucket, 0.0)
    combined_score = round(
        (ranking_score * SELECTION_WEIGHTS["ranking_score"])
        + (trust_score * SELECTION_WEIGHTS["trust_score"])
        + bucket_adjustment,
        4,
    )

    return {
        "weights": SELECTION_WEIGHTS.copy(),
        "bucket_adjustment": bucket_adjustment,
        "ranking_score": round(ranking_score, 4),
        "trust_score": round(trust_score, 4),
        "trust_bucket": trust_bucket,
        "combined_score": combined_score,
        "policy": "ranking_weighted_plus_trust_bucket_adjustment_v1",
    }


def compute_score(
    query: str,
    paper: PaperResult,
    scopus_valid: bool,
    source_match: dict[str, Any] | None = None,
) -> float:
    return score_paper(
        query=query,
        paper=paper,
        scopus_valid=scopus_valid,
        source_match=source_match,
    )["total"]


def _metadata_completeness(paper: PaperResult) -> float:
    """Score from 0-1 based on how many key metadata fields are present."""
    fields = [
        bool(paper.title),
        bool(paper.abstract),
        bool(paper.year),
        bool(paper.doi),
        bool(paper.source_title),
        bool(paper.authors),
        paper.citation_count is not None,
    ]
    return round(sum(fields) / len(fields), 2)


def build_authoring_trust_summary(
    paper: PaperResult,
    source_match: dict[str, Any] | None = None,
    trust_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a flat, clean trust summary for Phase 3 authoring.
    The authoring module uses this to decide how to cite a paper.
    """
    trust_bucket = "low"
    trust_total = 0.0
    if isinstance(trust_debug, dict):
        trust_bucket = str(trust_debug.get("bucket") or "low").strip().lower()
        raw_total = trust_debug.get("total")
        if isinstance(raw_total, (int, float)):
            trust_total = max(0.0, min(float(raw_total), 1.0))

    # Extract source match info
    sm = source_match if isinstance(source_match, dict) else {}
    confidence = sm.get("confidence", 0.0)
    reason = sm.get("reason", "no_match")
    match_method = sm.get("match_method", "none")
    registry = sm.get("registry_metadata") or {}

    registry_category = str(registry.get("registry_category") or "unknown")
    source_active = not registry.get("is_discontinued", False)
    source_discontinued = bool(registry.get("is_discontinued"))
    discontinued_quality = bool(registry.get("discontinued_due_to_quality"))

    completeness = _metadata_completeness(paper)

    # Authoring recommendation
    if trust_bucket == "high" and confidence >= 0.85:
        recommendation = "safe_to_cite"
    elif trust_bucket == "low" or confidence < 0.3:
        recommendation = "verify_before_citing"
    elif discontinued_quality:
        recommendation = "verify_before_citing"
    elif source_discontinued:
        recommendation = "cite_with_caution"
    else:
        recommendation = "cite_with_caution"

    return {
        "trust_bucket": trust_bucket,
        "trust_score": round(trust_total, 4),
        "source_match_confidence": round(confidence, 4) if isinstance(confidence, (int, float)) else 0.0,
        "source_match_reason": reason,
        "source_match_method": match_method,
        "registry_category": registry_category,
        "source_active": source_active,
        "source_discontinued": source_discontinued,
        "discontinued_due_to_quality": discontinued_quality,
        "metadata_completeness": completeness,
        "authoring_recommendation": recommendation,
    }

