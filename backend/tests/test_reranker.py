from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import reranker


class _FakeModel:
    pass


def _settings(**overrides):
    base = {
        "reranker_enabled": True,
        "reranker_model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "reranker_max_candidates": 50,
        "reranker_batch_size": 16,
        "reranker_timeout_seconds": 20.0,
        "reranker_score_weight": 0.35,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _candidate(title, *, abstract="abstract", score=0.5, candidate_id=1):
    return SimpleNamespace(
        id=candidate_id,
        title=title,
        abstract=abstract,
        final_score=score,
        raw_metadata_json={"_selection": {"combined_score": score}},
    )


def test_reranker_disabled_keeps_order_and_does_not_load_model(monkeypatch):
    candidates = [_candidate("A", candidate_id=1), _candidate("B", candidate_id=2)]

    def _fail_load(*args, **kwargs):
        raise AssertionError("disabled reranker must not load model")

    monkeypatch.setattr(reranker, "_load_cross_encoder", _fail_load)

    result, diagnostics = reranker.rerank_candidates(
        "query",
        candidates,
        settings_obj=_settings(reranker_enabled=False),
    )

    assert result == candidates
    assert diagnostics["enabled"] is False
    assert diagnostics["applied"] is False
    assert diagnostics["skipped_reason"] == "reranker_disabled"
    assert all("_reranker" not in candidate.raw_metadata_json for candidate in candidates)


def test_reranker_model_load_failure_falls_back_without_crash(monkeypatch):
    candidates = [_candidate("A", candidate_id=1), _candidate("B", candidate_id=2)]
    monkeypatch.setattr(
        reranker,
        "_load_cross_encoder",
        lambda model_name: (_ for _ in ()).throw(RuntimeError("model_load_failed")),
    )

    result, diagnostics = reranker.rerank_candidates("query", candidates, settings_obj=_settings())

    assert result == candidates
    assert diagnostics["applied"] is False
    assert diagnostics["error"] == "model_load_failed"
    assert candidates[0].raw_metadata_json["_reranker"]["reranker_applied"] is False


def test_reranker_fake_scores_reorder_candidates(monkeypatch):
    candidates = [
        _candidate("weak", score=0.9, candidate_id=1),
        _candidate("strong", score=0.1, candidate_id=2),
    ]
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda model_name: _FakeModel())
    monkeypatch.setattr(reranker, "_predict_with_timeout", lambda *args, **kwargs: [0.1, 0.9])

    result, diagnostics = reranker.rerank_candidates(
        "target query",
        candidates,
        settings_obj=_settings(reranker_score_weight=1.0),
    )

    assert [candidate.title for candidate in result] == ["strong", "weak"]
    assert diagnostics["applied"] is True
    assert result[0].raw_metadata_json["_reranker"]["reranker_score"] == 1.0
    assert result[0].raw_metadata_json["_reranker_selection"]["final_score_after_rerank"] == 1.0


def test_reranker_only_reranks_top_n_and_keeps_tail(monkeypatch):
    candidates = [
        _candidate("first", score=0.2, candidate_id=1),
        _candidate("second", score=0.2, candidate_id=2),
        _candidate("tail", score=0.99, candidate_id=3),
    ]
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda model_name: _FakeModel())
    monkeypatch.setattr(reranker, "_predict_with_timeout", lambda *args, **kwargs: [0.1, 0.9])

    result, diagnostics = reranker.rerank_candidates(
        "target query",
        candidates,
        settings_obj=_settings(reranker_max_candidates=2, reranker_score_weight=1.0),
    )

    assert [candidate.title for candidate in result] == ["second", "first", "tail"]
    assert diagnostics["reranked_candidates"] == 2
    assert candidates[2].raw_metadata_json["_reranker"]["skipped_reason"] == "outside_top_n"


def test_reranker_empty_candidates_or_query_do_not_crash():
    empty_result, empty_diag = reranker.rerank_candidates("query", [], settings_obj=_settings())
    assert empty_result == []
    assert empty_diag["skipped_reason"] == "no_candidates"

    candidates = [_candidate("A")]
    query_result, query_diag = reranker.rerank_candidates("", candidates, settings_obj=_settings())
    assert query_result == candidates
    assert query_diag["skipped_reason"] == "empty_query"
    assert candidates[0].raw_metadata_json["_reranker"]["reranker_applied"] is False


def test_score_normalization_with_different_scores():
    assert reranker._normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_score_normalization_with_equal_scores():
    assert reranker._normalize_scores([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]


def test_nonfinite_predictions_preserve_baseline(monkeypatch):
    candidates = [_candidate("A", candidate_id=1), _candidate("B", candidate_id=2)]
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda name: _FakeModel())
    monkeypatch.setattr(reranker, "_predict_with_timeout", lambda *args, **kwargs: [float("nan"), 0.9])
    result, diagnostics = reranker.rerank_candidates("query", candidates, settings_obj=_settings())
    assert result == candidates
    assert not diagnostics["applied"]
    assert diagnostics["skipped_reason"] == "invalid_score_count"
