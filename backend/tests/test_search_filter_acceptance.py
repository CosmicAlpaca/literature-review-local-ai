from datetime import datetime
from pathlib import Path
import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.main import app
from app.services.orchestrator import normalize_search_filters


class _ReviewSessionQuery:
    def __init__(self, session_obj):
        self._session_obj = session_obj

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._session_obj


class _FakeDB:
    def __init__(self, session_obj=None):
        self.session_obj = session_obj or _session()

    def query(self, model):
        if model is ReviewSession:
            return _ReviewSessionQuery(self.session_obj)
        raise AssertionError(f"Unexpected model queried: {model}")


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "AI cancer review",
        "raw_query": "AI cancer detection",
        "filters_json": None,
        "status": "created",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def _candidate():
    return PaperCandidate(
        id=21,
        session_id=7,
        title="Open Scopus Nature AI",
        abstract="AI cancer detection.",
        year=2026,
        source_title="Nature Medicine",
        doi="10.1000/open",
        url="https://example.com/open",
        citation_count=120,
        provider="openalex_strict",
        scopus_matched=True,
        final_score=0.91,
        created_at=datetime.utcnow(),
        raw_metadata_json={
            "paper_id": "open",
            "api_key": "secret",
            "_ranking": {"total": 0.9},
            "_trust": {"bucket": "high"},
            "_source_match": {
                "source_title": "Nature Medicine",
                "reason": "test_match",
                "confidence": 0.99,
                "registry_metadata": {"issn": "1234-5678"},
            },
            "_llm_raw_response": "raw model response",
        },
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_combined_filters_normalize_with_request_overrides():
    filters = normalize_search_filters(
        query="AI cancer",
        session_filters={
            "recent_years": 10,
            "provider": "mock",
            "limit": 5,
            "access_type": "any",
        },
        recent_years=3,
        source="Nature",
        provider="openalex_strict",
        scopus_only=True,
        min_citations=50,
        access_type="open",
        current_year=2026,
    )

    metadata = filters.to_metadata()
    assert metadata["recent_years"] == 3
    assert metadata["years_back"] == 3
    assert metadata["year_from"] == 2024
    assert metadata["publication_year_from"] == 2024
    assert metadata["provider"] == "openalex_strict"
    assert metadata["source"] == "Nature"
    assert metadata["scopus_only"] is True
    assert metadata["min_citations"] == 50
    assert metadata["access_type"] == "open"
    assert metadata["limit"] == 5


def test_search_response_contract_metadata_and_no_raw_metadata_leak(monkeypatch):
    def _pipeline(db, session_id, query, llm_options=None, search_filters=None):
        search_filters.pre_filter_count = 3
        search_filters.post_filter_count = 1
        candidate = _candidate()
        return [candidate], [candidate]

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_FakeDB())
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)

    response = TestClient(app).post(
        "/search/7?query=AI%20cancer&recent_years=3&source=Nature&provider=openalex_strict"
        "&scopus_only=true&min_citations=50&access_type=open&limit=10"
    )

    assert response.status_code == 200
    data = response.json()
    applied = data["query_context"]["applied_filters"]
    assert applied["recent_years"] == 3
    assert applied["years_back"] == 3
    assert applied["year_from"] == 2024
    assert applied["publication_year_from"] == 2024
    assert applied["provider"] == "openalex_strict"
    assert applied["pre_filter_count"] == 3
    assert applied["post_filter_count"] == 1
    assert applied["filtered_out_count"] == 2
    assert data["candidates"][0]["title"] == "Open Scopus Nature AI"

    response_text = json.dumps(data, sort_keys=True)
    for forbidden in (
        "raw_metadata",
        "_ranking",
        "_trust",
        "_llm_diagnostics",
        "_llm_query",
        "_llm_rerank",
        "_llm_summary",
        "_llm_raw_response",
        "raw_prompt",
        "raw_response",
        "api_key",
        "raw model response",
    ):
        assert forbidden not in response_text


def test_invalid_filter_returns_clean_client_error(monkeypatch):
    def _pipeline(*args, **kwargs):
        raise AssertionError("pipeline should not run for invalid filters")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_FakeDB())
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)

    response = TestClient(app).post("/search/7?query=AI%20cancer&access_type=unknown")

    assert response.status_code == 400
    assert "access_type" in response.json()["detail"]


def test_no_filter_search_response_remains_phase1_compatible(monkeypatch):
    def _pipeline(db, session_id, query, llm_options=None):
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_FakeDB())
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)

    response = TestClient(app).post("/search/7?query=AI%20cancer")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["session_id"] == 7
    assert data["query_context"]["original_query"] == "AI cancer"
    assert data["query_context"]["warnings"] == ["no_candidates"]
