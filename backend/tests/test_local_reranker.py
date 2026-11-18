from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.local_reranker import score_candidates


def _settings(
    *,
    enabled=False,
    provider="mock",
    model_path="",
    version="test-v1",
    top_k=20,
):
    return SimpleNamespace(
        local_reranker_enabled=enabled,
        local_reranker_provider=provider,
        local_reranker_model_path=model_path,
        local_reranker_model_version=version,
        local_reranker_top_k=top_k,
    )


def _candidate(**overrides):
    base = {
        "candidate_id": 1,
        "title": "AI for Early Cancer Detection",
        "abstract": "Artificial intelligence models support cancer screening and early detection.",
        "source_title": "Journal of Medical AI",
        "year": 2023,
        "citation_count": 42,
        "doi": "10.1000/ai",
        "provider": "openalex_strict",
    }
    base.update(overrides)
    return base


def test_local_reranker_disabled_returns_fallback_safe_result():
    result = score_candidates(
        "artificial intelligence cancer detection",
        [_candidate()],
        settings_obj=_settings(enabled=False),
    )

    assert result.enabled is False
    assert result.model_version == "test-v1"
    assert result.model_provider == "mock"
    assert result.fallback_reason == "local_reranker_disabled"
    assert len(result.results) == 1
    assert result.results[0].fallback_reason == "local_reranker_disabled"
    assert 0.0 <= result.results[0].local_relevance_score <= 1.0


def test_mock_local_reranker_returns_deterministic_bounded_scores():
    settings_obj = _settings(enabled=True, provider="mock", version="mock-v2")
    first = score_candidates(
        "artificial intelligence cancer detection",
        [_candidate(), _candidate(candidate_id=2, title="Unrelated Topic", abstract="Soil moisture study.")],
        settings_obj=settings_obj,
    )
    second = score_candidates(
        "artificial intelligence cancer detection",
        [_candidate(), _candidate(candidate_id=2, title="Unrelated Topic", abstract="Soil moisture study.")],
        settings_obj=settings_obj,
    )

    first_scores = [item.local_relevance_score for item in first.results]
    second_scores = [item.local_relevance_score for item in second.results]
    assert first.enabled is True
    assert first.fallback_reason is None
    assert first_scores == second_scores
    assert first_scores[0] > first_scores[1]
    assert all(0.0 <= score <= 1.0 for score in first_scores)
    assert first.results[0].model_version == "mock-v2"


def test_missing_model_path_does_not_crash_or_leak_path():
    result = score_candidates(
        "artificial intelligence cancer detection",
        [_candidate()],
        settings_obj=_settings(
            enabled=True,
            provider="local",
            model_path="D:/private/models/secret-reranker",
        ),
    )

    assert result.enabled is False
    assert result.fallback_reason == "model_path_missing"
    serialized = str(result.to_dict())
    assert "secret-reranker" not in serialized
    assert "D:/private" not in serialized


def test_invalid_input_and_empty_candidates_are_safe():
    invalid = score_candidates("", [_candidate()], settings_obj=_settings(enabled=True))
    empty = score_candidates("cancer detection", [], settings_obj=_settings(enabled=True))

    assert invalid.fallback_reason == "invalid_input"
    assert empty.fallback_reason == "no_candidates"
    assert empty.results == []


def test_loader_uses_checkpoint_not_metadata_base_model(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType
    from app.services import local_reranker
    (tmp_path / "metadata.json").write_text(json.dumps({"base_model": "base-model"}))
    calls = []
    fake = ModuleType("sentence_transformers")
    fake.CrossEncoder = lambda model, **kwargs: calls.append(model) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    local_reranker._cross_encoder_cache.clear()
    assert local_reranker._load_cross_encoder(str(tmp_path)) is not None
    assert calls == [str(tmp_path)]
