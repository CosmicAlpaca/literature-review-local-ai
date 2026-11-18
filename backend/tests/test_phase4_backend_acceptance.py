from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.main import app
from app.services.llm_client import LLMClient
from app.services.paper_summary import PaperSummaryResult
from app.tools import model_compare


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class _Query:
    def __init__(self, db, model):
        self._db = db
        self._model = model
        self._filters = {}

    def filter(self, *expressions, **kwargs):
        for expression in expressions:
            key = getattr(getattr(expression, "left", None), "key", None)
            value = getattr(getattr(expression, "right", None), "value", None)
            if key is not None:
                self._filters[key] = value
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        if self._model is ReviewSession:
            for session_obj in self._db.sessions:
                if self._matches(session_obj):
                    return session_obj
        if self._model is PaperCandidate:
            for candidate in self._db.candidates:
                if self._matches(candidate):
                    return candidate
        return None

    def all(self):
        if self._model is PaperCandidate:
            return [candidate for candidate in self._db.candidates if self._matches(candidate)]
        return []

    def _matches(self, obj):
        return all(getattr(obj, key, None) == value for key, value in self._filters.items())


class _DB:
    def __init__(self, sessions=None, candidates=None):
        self.sessions = sessions or []
        self.candidates = candidates or []
        self.commit_count = 0

    def query(self, model):
        return _Query(self, model)

    def add(self, obj):
        return None

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        return None


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Flood review",
        "raw_query": "flood detection",
        "filters_json": None,
        "status": "created",
        "created_at": datetime.utcnow(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _candidate(**overrides):
    base = {
        "id": 21,
        "session_id": 7,
        "title": "Flood Detection Paper",
        "abstract": "A paper about flood detection from satellite images.",
        "year": 2024,
        "source_title": "Remote Sensing Journal",
        "citation_count": 12,
        "provider": "mock",
        "scopus_matched": True,
        "final_score": 0.82,
        "created_at": datetime.utcnow(),
        "doi": "10.1000/flood",
        "url": "https://example.com/flood",
        "raw_metadata_json": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _override_app(db):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1,
        email="phase4@example.com",
        role="user",
    )

    def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db


def _local_rerank_payload():
    return {
        "query": "flood detection satellite images",
        "candidates": [
            {
                "candidate_id": 21,
                "title": "Flood Detection Paper",
                "abstract": "A paper about flood detection from satellite images.",
                "source_title": "Remote Sensing Journal",
                "year": 2024,
                "citation_count": 12,
                "provider": "mock",
            }
        ],
    }


def _model_compare_settings():
    return SimpleNamespace(
        llm_enabled=True,
        llm_provider="mock",
        llm_model="mock-test-model",
        llm_api_key="sk-secret-acceptance",
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_prompt_version="v1",
        llm_max_input_chars_per_call=12000,
        local_reranker_enabled=True,
        local_reranker_provider="mock",
        local_reranker_model_path="D:/private/models/secret-reranker",
        local_reranker_model_version="local-acceptance-v1",
        local_reranker_top_k=20,
    )


def _llm_settings():
    return SimpleNamespace(
        llm_enabled=True,
        llm_provider="mock",
        llm_model="mock-test-model",
        llm_api_key=None,
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_prompt_version="v1",
        llm_max_input_chars_per_call=12000,
    )


def test_phase4_summary_endpoint_contract_returns_safe_shape(monkeypatch):
    candidate = _candidate()
    db = _DB(sessions=[_session()], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings()),
    )
    monkeypatch.setattr(
        "app.api.search.summarize_paper",
        lambda candidate, query=None: PaperSummaryResult(
            one_sentence_summary="Safe generated summary.",
            contribution="Safe generated contribution.",
            method_or_data="Safe generated method.",
            relevance_to_query="Safe generated relevance.",
            limitations_or_caution="Safe generated caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
        ),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert set(data) >= {"session_id", "candidate_id", "summary", "cached", "llm_used", "prompt_version"}
    assert data["summary"]["one_sentence_summary"] == "Safe generated summary."
    assert data["cached"] is False
    assert data["llm_used"] is True
    assert candidate.raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Safe generated summary."
    for unsafe in ("raw_prompt", "raw_model_response", "api_key", "_ranking", "_trust", "traceback"):
        assert unsafe not in response.text


def test_phase4_local_reranker_preview_requires_auth():
    response = TestClient(app).post("/llm/local-rerank-preview", json=_local_rerank_payload())

    assert response.status_code == 403


def test_phase4_local_reranker_preview_returns_safe_mock_shape(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1,
        email="phase4@example.com",
        role="user",
    )
    monkeypatch.setattr("app.api.llm.settings.local_reranker_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.local_reranker_model_version", "acceptance-v1")
    monkeypatch.setattr("app.api.llm.settings.local_reranker_top_k", 20)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_model_path", "D:/private/models/secret-reranker")

    response = TestClient(app).post("/llm/local-rerank-preview", json=_local_rerank_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["model_provider"] == "mock"
    assert data["model_version"] == "acceptance-v1"
    assert data["results"][0]["candidate_id"] == 21
    assert 0.0 <= data["results"][0]["local_relevance_score"] <= 1.0
    assert "secret-reranker" not in response.text
    assert "traceback" not in response.text.lower()


def test_phase4_model_compare_mock_json_contract_and_safe_output(monkeypatch):
    def _fail_if_provider_called(*args, **kwargs):
        raise AssertionError("model comparison must not call retrieval providers")

    monkeypatch.setattr("app.providers.provider_factory.get_search_provider", _fail_if_provider_called)
    stdout = StringIO()

    code = model_compare.main(
        ["--mock-only", "--json", "--case", "early_cancer_detection"],
        settings_obj=_model_compare_settings(),
        stdout=stdout,
    )

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["methods"] == ["baseline", "local_reranker", "llm_rerank"]
    assert data["summary"]["total_cases"] == 1
    assert data["retrieval_provider_called"] is False
    assert data["database_access_required"] is False
    text = stdout.getvalue().lower()
    for unsafe in ("sk-secret-acceptance", "raw_prompt", "raw_model_response", "secret-reranker", "traceback"):
        assert unsafe not in text


def test_phase4_local_reranker_does_not_change_production_search_ranking(monkeypatch):
    first = _candidate(id=21, title="First ranked paper", final_score=0.9)
    second = _candidate(id=22, title="Second ranked paper", final_score=0.7)
    db = _DB(sessions=[_session()], candidates=[first, second])
    _override_app(db)

    def _fail_if_local_reranker_called(*args, **kwargs):
        raise AssertionError("production search must not call local reranker in Phase 4")

    monkeypatch.setattr("app.services.local_reranker.score_candidates", _fail_if_local_reranker_called)
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None, search_filters=None: ([first, second], [first, second]),
    )

    response = TestClient(app).post("/search/7?query=flood%20detection")

    assert response.status_code == 200
    data = response.json()
    assert [candidate["id"] for candidate in data["candidates"]] == [21, 22]
    assert [candidate["title"] for candidate in data["top5"]] == ["First ranked paper", "Second ranked paper"]
