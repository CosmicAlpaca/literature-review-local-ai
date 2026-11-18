from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import math
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_model_cache: dict[str, Any] = {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_value(candidate: Any, key: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key)
    return getattr(candidate, key, None)


def _set_value(candidate: Any, key: str, value: Any) -> None:
    if isinstance(candidate, dict):
        candidate[key] = value
    else:
        setattr(candidate, key, value)


def _metadata(candidate: Any) -> dict[str, Any]:
    raw = _get_value(candidate, "raw_metadata_json")
    if raw is None:
        raw = _get_value(candidate, "raw_metadata")
    return dict(raw) if isinstance(raw, dict) else {}


def _set_metadata(candidate: Any, metadata: dict[str, Any]) -> None:
    if isinstance(candidate, dict):
        if "raw_metadata_json" in candidate:
            candidate["raw_metadata_json"] = metadata
        elif "raw_metadata" in candidate:
            candidate["raw_metadata"] = metadata
        else:
            candidate["raw_metadata_json"] = metadata
    else:
        _set_value(candidate, "raw_metadata_json", metadata)


def _candidate_text(candidate: Any) -> str:
    title = _safe_text(_get_value(candidate, "title"))
    abstract = _safe_text(_get_value(candidate, "abstract"))
    if title and abstract:
        return f"{title}\n{abstract}"
    return title or abstract


def _selection_score(candidate: Any) -> float:
    metadata = _metadata(candidate)
    selection = metadata.get("_selection")
    if isinstance(selection, dict) and isinstance(selection.get("combined_score"), (int, float)):
        return max(0.0, min(float(selection["combined_score"]), 1.0))
    score = _get_value(candidate, "final_score")
    if isinstance(score, (int, float)):
        return max(0.0, min(float(score), 1.0))
    return 0.0


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [0.5 for _ in scores]
    return [max(0.0, min((score - low) / (high - low), 1.0)) for score in scores]


def _diagnostics(
    *,
    settings_obj: Any,
    applied: bool,
    input_candidates: int,
    reranked_candidates: int = 0,
    error: str | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(settings_obj, "reranker_enabled", False)),
        "applied": applied,
        "model_name": _safe_text(getattr(settings_obj, "reranker_model_name", DEFAULT_MODEL_NAME))
        or DEFAULT_MODEL_NAME,
        "input_candidates": input_candidates,
        "reranked_candidates": reranked_candidates,
        "error": error,
        "skipped_reason": skipped_reason,
    }


def _mark_skipped(candidate: Any, diagnostics: dict[str, Any], reason: str) -> None:
    metadata = _metadata(candidate)
    metadata["_reranker"] = {
        "enabled": bool(diagnostics["enabled"]),
        "reranker_applied": False,
        "reranker_model": diagnostics["model_name"],
        "skipped_reason": reason,
    }
    _set_metadata(candidate, metadata)


def _mark_applied(
    candidate: Any,
    *,
    raw_score: float,
    normalized_score: float,
    base_score: float,
    combined_score: float,
    model_name: str,
    weight: float,
) -> None:
    metadata = _metadata(candidate)
    metadata["_reranker"] = {
        "enabled": True,
        "reranker_applied": True,
        "reranker_score_raw": round(raw_score, 6),
        "reranker_score": round(normalized_score, 6),
        "reranker_model": model_name,
    }
    metadata["_reranker_selection"] = {
        "base_selection_score": round(base_score, 6),
        "reranker_score": round(normalized_score, 6),
        "reranker_score_raw": round(raw_score, 6),
        "reranker_score_weight": round(weight, 6),
        "combined_score": round(combined_score, 6),
        "final_score_after_rerank": round(combined_score, 6),
        "policy": "phase1_selection_weighted_with_cross_encoder_reranker_v1",
    }
    _set_metadata(candidate, metadata)


def _load_cross_encoder(model_name: str) -> Any:
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError("missing_dependency") from exc

    try:
        model = CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        raise RuntimeError("model_load_failed") from exc

    _model_cache[model_name] = model
    return model


def _predict_scores(model: Any, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
    raw_scores = model.predict(pairs, batch_size=batch_size)
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    return [float(score) for score in raw_scores]


def _predict_with_timeout(
    model: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    timeout_seconds: float,
) -> list[float]:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_predict_scores, model, pairs, batch_size=batch_size)
    try:
        scores = future.result(timeout=timeout_seconds)
    except TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("predict_timeout") from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return scores


def reranker_model_loaded(model_name: str | None = None) -> bool:
    safe_model_name = _safe_text(model_name) or _safe_text(settings.reranker_model_name) or DEFAULT_MODEL_NAME
    return safe_model_name in _model_cache


def rerank_candidates(
    query: str,
    candidates: list[Any],
    settings_obj: Any = settings,
) -> tuple[list[Any], dict[str, Any]]:
    safe_query = _safe_text(query)
    candidate_count = len(candidates)
    diagnostics = _diagnostics(
        settings_obj=settings_obj,
        applied=False,
        input_candidates=candidate_count,
    )

    if not bool(getattr(settings_obj, "reranker_enabled", False)):
        diagnostics["skipped_reason"] = "reranker_disabled"
        return candidates, diagnostics

    if not safe_query:
        diagnostics["skipped_reason"] = "empty_query"
        for candidate in candidates:
            _mark_skipped(candidate, diagnostics, "empty_query")
        return candidates, diagnostics

    if not candidates:
        diagnostics["skipped_reason"] = "no_candidates"
        return candidates, diagnostics

    limit = max(int(getattr(settings_obj, "reranker_max_candidates", 50) or 0), 0)
    if limit <= 0:
        diagnostics["skipped_reason"] = "max_candidates_zero"
        for candidate in candidates:
            _mark_skipped(candidate, diagnostics, "max_candidates_zero")
        return candidates, diagnostics

    model_name = diagnostics["model_name"]
    batch_size = max(int(getattr(settings_obj, "reranker_batch_size", 16) or 1), 1)
    timeout_seconds = max(float(getattr(settings_obj, "reranker_timeout_seconds", 20.0) or 20.0), 0.1)
    weight = max(0.0, min(float(getattr(settings_obj, "reranker_score_weight", 0.35) or 0.0), 1.0))

    bounded = candidates[:limit]
    tail = candidates[limit:]
    scored_candidates: list[Any] = []
    pairs: list[tuple[str, str]] = []

    for candidate in bounded:
        text = _candidate_text(candidate)
        if not text:
            _mark_skipped(candidate, diagnostics, "missing_text")
            continue
        scored_candidates.append(candidate)
        pairs.append((safe_query, text))

    for candidate in tail:
        _mark_skipped(candidate, diagnostics, "outside_top_n")

    if not pairs:
        diagnostics["skipped_reason"] = "no_scored_candidates"
        return candidates, diagnostics

    try:
        model = _load_cross_encoder(model_name)
        raw_scores = _predict_with_timeout(
            model,
            pairs,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        diagnostics["error"] = "predict_timeout"
        diagnostics["skipped_reason"] = "predict_timeout"
        for candidate in scored_candidates:
            _mark_skipped(candidate, diagnostics, "predict_timeout")
        return candidates, diagnostics
    except Exception as exc:
        safe_error = str(exc) if str(exc) in {"missing_dependency", "model_load_failed"} else "predict_failed"
        diagnostics["error"] = safe_error
        diagnostics["skipped_reason"] = safe_error
        for candidate in scored_candidates:
            _mark_skipped(candidate, diagnostics, safe_error)
        logger.warning("Production reranker skipped: reason=%s", safe_error)
        return candidates, diagnostics

    if len(raw_scores) != len(scored_candidates) or not all(math.isfinite(score) for score in raw_scores):
        diagnostics["error"] = "invalid_score_count"
        diagnostics["skipped_reason"] = "invalid_score_count"
        for candidate in scored_candidates:
            _mark_skipped(candidate, diagnostics, "invalid_score_count")
        return candidates, diagnostics

    normalized_scores = _normalize_scores(raw_scores)
    scored_rows = []
    for candidate, raw_score, normalized_score in zip(scored_candidates, raw_scores, normalized_scores):
        base_score = _selection_score(candidate)
        combined_score = (base_score * (1.0 - weight)) + (normalized_score * weight)
        _mark_applied(
            candidate,
            raw_score=raw_score,
            normalized_score=normalized_score,
            base_score=base_score,
            combined_score=combined_score,
            model_name=model_name,
            weight=weight,
        )
        scored_rows.append((combined_score, normalized_score, raw_score, candidate))

    scored_rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    reranked_set = {id(candidate) for *_, candidate in scored_rows}
    skipped_bounded = [candidate for candidate in bounded if id(candidate) not in reranked_set]
    reranked = [candidate for *_, candidate in scored_rows] + skipped_bounded + tail

    diagnostics["applied"] = True
    diagnostics["reranked_candidates"] = len(scored_candidates)
    return reranked, diagnostics
