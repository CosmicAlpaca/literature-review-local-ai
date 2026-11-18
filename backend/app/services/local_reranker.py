from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

SAFE_LOCAL_RERANKER_FALLBACK_REASONS = {
    "local_reranker_disabled",
    "model_path_missing",
    "model_load_failed",
    "invalid_input",
    "no_candidates",
}

# ── Lazy-loaded cross-encoder singleton ──────────────────────────────────
_cross_encoder_cache: dict[str, Any] = {}


def _load_cross_encoder(model_identifier: str) -> Any:
    """Lazy-load CrossEncoder from sentence-transformers.
    Supports both local paths and Hugging Face repo IDs.
    """
    if model_identifier in _cross_encoder_cache:
        return _cross_encoder_cache[model_identifier]

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("sentence-transformers is not installed; local reranker unavailable")
        return None

    # A fine-tuned checkpoint must be loaded from its own directory. Metadata's
    # base_model describes provenance, not the weights to use for inference.
    try:
        logger.info("Loading local cross-encoder: %s", model_identifier)
        model = CrossEncoder(model_identifier, max_length=512)
        _cross_encoder_cache[model_identifier] = model
        return model
    except Exception:
        logger.exception("Failed to load cross-encoder from %s", model_identifier)
        return None


def _read_model_metadata(model_identifier: str) -> dict[str, Any] | None:
    """Read metadata.json from model directory or HF hub (safe for public exposure)."""
    try:
        # If it's a local path
        local_path = Path(model_identifier) / "metadata.json"
        if local_path.exists():
            raw = json.loads(local_path.read_text(encoding="utf-8"))
            return _filter_metadata(raw)
            
        # If it's a HF repo, we could try to fetch metadata.json from the hub
        # but for now, let's keep it simple or return None if not local.
    except Exception:
        pass
    return None


def _filter_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Only expose safe fields from metadata."""
    return {
        "model_name": raw.get("model_name"),
        "model_version": raw.get("model_version"),
        "model_type": raw.get("model_type"),
        "base_model": raw.get("base_model"),
        "trained_at": raw.get("trained_at"),
        "dataset_version": raw.get("dataset_version"),
        "label_schema": raw.get("label_schema"),
        "metrics": raw.get("metrics"),
    }


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LocalRerankCandidate:
    candidate_id: str | int | None
    title: str | None = None
    abstract: str | None = None
    source_title: str | None = None
    year: int | None = None
    citation_count: int | None = None
    doi: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class LocalRerankScore:
    candidate_id: str | int | None
    local_relevance_score: float
    reason: str
    model_version: str
    model_provider: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "local_relevance_score": self.local_relevance_score,
            "reason": self.reason,
            "model_version": self.model_version,
            "model_provider": self.model_provider,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class LocalRerankResult:
    enabled: bool
    model_version: str
    model_provider: str
    results: list[LocalRerankScore] = field(default_factory=list)
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_version": self.model_version,
            "model_provider": self.model_provider,
            "results": [result.to_dict() for result in self.results],
            "fallback_reason": self.fallback_reason,
        }


def safe_local_reranker_fallback_reason(reason: str | None) -> str:
    if reason in SAFE_LOCAL_RERANKER_FALLBACK_REASONS:
        return reason
    return "model_load_failed"


# ── Main entry point ────────────────────────────────────────────────────

def score_candidates(
    query: str,
    candidates: list[Any],
    *,
    settings_obj: Any = settings,
) -> LocalRerankResult:
    model_version = _safe_text(getattr(settings_obj, "local_reranker_model_version", "v1")) or "v1"
    model_provider = _safe_text(getattr(settings_obj, "local_reranker_provider", "mock")) or "mock"
    model_provider = model_provider.lower()
    model_path = _safe_text(getattr(settings_obj, "local_reranker_model_path", None))

    safe_query = _safe_text(query)
    parsed_candidates = [_coerce_candidate(candidate) for candidate in candidates]
    parsed_candidates = [candidate for candidate in parsed_candidates if candidate is not None]

    if not safe_query:
        return _fallback_result(
            parsed_candidates,
            query="",
            model_version=model_version,
            model_provider=model_provider,
            reason="invalid_input",
            enabled=False,
        )

    if not parsed_candidates:
        return LocalRerankResult(
            enabled=False,
            model_version=model_version,
            model_provider=model_provider,
            results=[],
            fallback_reason="no_candidates",
        )

    enabled = bool(getattr(settings_obj, "local_reranker_enabled", False))
    if not enabled:
        return _fallback_result(
            parsed_candidates,
            query=safe_query,
            model_version=model_version,
            model_provider=model_provider,
            reason="local_reranker_disabled",
            enabled=False,
        )

    if model_provider == "mock":
        return LocalRerankResult(
            enabled=True,
            model_version=model_version,
            model_provider=model_provider,
            results=[
                _mock_score(
                    candidate,
                    query=safe_query,
                    model_version=model_version,
                    model_provider=model_provider,
                )
                for candidate in parsed_candidates
            ],
        )

    # ── Real local cross-encoder inference ───────────────────────────
    # Detect placeholder paths or empty paths
    if _model_path_missing(model_path):
        return _fallback_result(
            parsed_candidates,
            query=safe_query,
            model_version=model_version,
            model_provider=model_provider,
            reason="model_path_missing",
            enabled=False,
        )

    try:
        # We don't check Path(model_path).exists() anymore because it could be a HF repo ID
        cross_encoder = _load_cross_encoder(model_path)
        if cross_encoder is None:
            return _fallback_result(
                parsed_candidates,
                query=safe_query,
                model_version=model_version,
                model_provider=model_provider,
                reason="model_load_failed",
                enabled=False,
            )

        # Build (query, candidate_text) pairs for cross-encoder
        pairs: list[tuple[str, str]] = []
        for candidate in parsed_candidates:
            candidate_text = _build_candidate_text(candidate)
            pairs.append((safe_query, candidate_text))

        # Run cross-encoder inference
        raw_scores = cross_encoder.predict(pairs)

        # Normalize scores to [0, 1] using sigmoid if needed
        import numpy as np
        scores_array = np.array(raw_scores, dtype=float)
        # ms-marco cross-encoders output logits; apply sigmoid to normalize
        normalized = 1.0 / (1.0 + np.exp(-scores_array))

        results = []
        for candidate, score in zip(parsed_candidates, normalized):
            relevance = _clamp_score(float(score))
            if relevance >= 0.7:
                reason = "Strong semantic match between query and title/abstract."
            elif relevance >= 0.4:
                reason = "Moderate semantic relevance to query."
            else:
                reason = "Low semantic similarity to query."

            results.append(
                LocalRerankScore(
                    candidate_id=candidate.candidate_id,
                    local_relevance_score=relevance,
                    reason=reason,
                    model_version=model_version,
                    model_provider=model_provider,
                )
            )

        return LocalRerankResult(
            enabled=True,
            model_version=model_version,
            model_provider=model_provider,
            results=results,
        )

    except OSError:
        return _fallback_result(
            parsed_candidates,
            query=safe_query,
            model_version=model_version,
            model_provider=model_provider,
            reason="model_load_failed",
            enabled=False,
        )
    except Exception:
        logger.exception("Unexpected error in local reranker inference")
        return _fallback_result(
            parsed_candidates,
            query=safe_query,
            model_version=model_version,
            model_provider=model_provider,
            reason="model_load_failed",
            enabled=False,
        )


# ── Helper functions ─────────────────────────────────────────────────────

def _build_candidate_text(candidate: LocalRerankCandidate) -> str:
    """Concatenate title + abstract + source for cross-encoder input."""
    parts = []
    if candidate.title:
        parts.append(candidate.title)
    if candidate.abstract:
        parts.append(candidate.abstract)
    if candidate.source_title:
        parts.append(f"Source: {candidate.source_title}")
    return " ".join(parts) if parts else "No content available."


def _model_path_missing(model_path: str | None) -> bool:
    if not model_path:
        return True
    normalized = model_path.strip().lower()
    if normalized in {"path/to/model", "none", "null", ""}:
        return True
    if _looks_like_local_path(model_path) and not Path(model_path).exists():
        return True
    return False


def _looks_like_local_path(model_path: str) -> bool:
    return (
        ":" in model_path
        or model_path.startswith("/")
        or model_path.startswith("\\")
        or model_path.startswith(".")
    )


def _fallback_result(
    candidates: list[LocalRerankCandidate],
    *,
    query: str,
    model_version: str,
    model_provider: str,
    reason: str,
    enabled: bool,
) -> LocalRerankResult:
    fallback_reason = safe_local_reranker_fallback_reason(reason)
    return LocalRerankResult(
        enabled=enabled,
        model_version=model_version,
        model_provider=model_provider,
        results=[
            _mock_score(
                candidate,
                query=query,
                model_version=model_version,
                model_provider=model_provider,
                fallback_reason=fallback_reason,
            )
            for candidate in candidates
        ],
        fallback_reason=fallback_reason,
    )


def _coerce_candidate(candidate: Any) -> LocalRerankCandidate | None:
    candidate_id = _value(candidate, "candidate_id", _value(candidate, "id"))
    if isinstance(candidate_id, bool):
        candidate_id = str(candidate_id)
    if candidate_id is not None and not isinstance(candidate_id, (str, int)):
        candidate_id = str(candidate_id)

    return LocalRerankCandidate(
        candidate_id=candidate_id,
        title=_safe_text(_value(candidate, "title")),
        abstract=_safe_text(_value(candidate, "abstract")),
        source_title=_safe_text(_value(candidate, "source_title")),
        year=_safe_int(_value(candidate, "year")),
        citation_count=_safe_int(_value(candidate, "citation_count")),
        doi=_safe_text(_value(candidate, "doi")),
        provider=_safe_text(_value(candidate, "provider")),
    )


def _mock_score(
    candidate: LocalRerankCandidate,
    *,
    query: str,
    model_version: str,
    model_provider: str,
    fallback_reason: str | None = None,
) -> LocalRerankScore:
    query_terms = _tokens(query)
    title_terms = _tokens(candidate.title)
    abstract_terms = _tokens(candidate.abstract)
    source_terms = _tokens(candidate.source_title)

    if not query_terms:
        score = 0.05  # Base floor for candidates with no query overlap
    else:
        title_overlap = len(query_terms & title_terms) / len(query_terms)
        abstract_overlap = len(query_terms & abstract_terms) / len(query_terms)
        source_overlap = len(query_terms & source_terms) / len(query_terms)
        
        # Scoring logic aligned with 'Old Ways' (Phase 1 Lexical Ranking)
        citation_bonus = min(max(candidate.citation_count or 0, 0), 100) / 1000.0
        year_bonus = 0.04 if candidate.year and candidate.year >= 2021 else 0.0
        
        # Primary score components
        score = 0.10 + (0.50 * title_overlap) + (0.30 * abstract_overlap) + (0.05 * source_overlap)
        score += citation_bonus + year_bonus

    reason = "Legacy Lexical Reranker: relevance based on keyword overlap and metadata."
    if fallback_reason:
        reason = f"Fallback {reason} ({fallback_reason})"

    return LocalRerankScore(
        candidate_id=candidate.candidate_id,
        local_relevance_score=_clamp_score(score),
        reason=reason,
        model_version=model_version,
        model_provider=model_provider,
        fallback_reason=fallback_reason,
    )


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2
    }


def _safe_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 4)
