from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.db.models.selected_paper import SelectedPaper
from app.main import app
from app.providers.base import PaperResult
from app.services.orchestrator import (
    PublicationYearFilterError,
    SearchYearFilter,
    normalize_search_year_filter,
    run_search_pipeline,
)


class _DeleteQuery:
    def filter(self, *args, **kwargs):
        return self

    def delete(self):
        return 0


class _PaperCandidateQuery:
    def __init__(self, db):
        self._db = db

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return []

    def delete(self):
        return 0


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
        self.added = []
        self.commit_count = 0
        self._next_candidate_id = 1

    def query(self, model):
        if model is SelectedPaper:
            return _DeleteQuery()
        if model is PaperCandidate:
            return _PaperCandidateQuery(self)
        if model is ReviewSession:
            return _ReviewSessionQuery(self.session_obj)
        raise AssertionError(f"Unexpected model queried: {model}")

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        if not self.added:
            return
        obj = self.added[-1]
        if isinstance(obj, PaperCandidate) and obj.id is None:
            obj.id = self._next_candidate_id
            self._next_candidate_id += 1

    def commit(self):
        self.commit_count += 1


class _APIDB:
    pass


class _YearProvider:
    def __init__(self):
        self.calls = []

    def fetch_papers(self, query: str, limit: int = 20):
        self.calls.append({"query": query, "limit": limit})
        return [
            PaperResult(
                title="Current Paper",
                abstract="Current abstract",
                year=2026,
                authors=["Author A"],
                doi="10.1000/current",
                source_title="Journal A",
                citation_count=10,
                provider="mock",
                raw_metadata={"paper_id": "current"},
            ),
            PaperResult(
                title="Old Paper",
                abstract="Old abstract",
                year=2020,
                authors=["Author B"],
                doi="10.1000/old",
                source_title="Journal B",
                citation_count=8,
                provider="mock",
                raw_metadata={"paper_id": "old"},
            ),
            PaperResult(
                title="Missing Year Paper",
                abstract="Missing year abstract",
                year=None,
                authors=["Author C"],
                doi="10.1000/missing",
                source_title="Journal C",
                citation_count=4,
                provider="mock",
                raw_metadata={"paper_id": "missing"},
            ),
        ]


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Flood review",
        "raw_query": "flood detection",
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


def _patch_phase1_dependencies(monkeypatch, provider):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)
    monkeypatch.setattr(
        "app.services.orchestrator.resolve_scopus_match",
        lambda db_session, source_title, raw_metadata=None: SimpleNamespace(
            source=SimpleNamespace(id=99, source_title=source_title),
            reason="exact_normalized_title",
            confidence=0.99,
            matched_value=source_title,
            normalized_input_title=(source_title or "").lower(),
            input_identifiers=None,
            registry_metadata={"registry_category": "active_indexed_journal"},
        ),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.compute_score",
        lambda query, paper, scopus_valid, source_match=None: 0.77,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_paper",
        lambda query, paper, scopus_valid, source_match=None: {"total": 0.77},
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_trust",
        lambda paper, source_match=None: {"total": 0.8, "bucket": "high"},
    )
    monkeypatch.setattr(
        "app.services.orchestrator.build_selection_debug",
        lambda ranking_score, trust_debug: {
            "combined_score": 0.83,
            "trust_score": trust_debug["total"],
            "trust_bucket": trust_debug["bucket"],
        },
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_search_without_year_filter_remains_phase1_compatible(monkeypatch):
    captured = {}

    def _pipeline(db, session_id, query, llm_options=None):
        captured["called"] = True
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())
    monkeypatch.setattr("app.api.search._get_user_session_or_404", lambda db, session_id, user_id: _session())
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)

    response = TestClient(app).post("/search/7?query=flood%20detection")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["session_id"] == 7
    assert data["query_context"]["original_query"] == "flood detection"
    assert data["query_context"]["warnings"] == ["no_candidates"]
    assert captured["called"] is True


def test_years_back_3_computes_expected_range():
    year_filter = normalize_search_year_filter(years_back=3, current_year=2026)

    assert year_filter.to_metadata() == {
        "years_back": 3,
        "publication_year_from": 2024,
        "publication_year_to": 2026,
        "missing_year_policy": "excluded_when_year_filter_active",
    }


def test_years_back_5_computes_expected_range():
    year_filter = normalize_search_year_filter(years_back=5, current_year=2026)

    assert year_filter.publication_year_from == 2022
    assert year_filter.publication_year_to == 2026


def test_invalid_years_back_returns_clean_client_error(monkeypatch):
    def _pipeline(*args, **kwargs):
        raise AssertionError("pipeline should not run with invalid filters")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())
    monkeypatch.setattr("app.api.search._get_user_session_or_404", lambda db, session_id, user_id: _session())
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)

    response = TestClient(app).post("/search/7?query=flood&years_back=2")

    assert response.status_code == 400
    assert "years_back" in response.json()["detail"]


def test_extremely_large_years_back_returns_clean_client_error(monkeypatch):
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())
    monkeypatch.setattr("app.api.search._get_user_session_or_404", lambda db, session_id, user_id: _session())

    response = TestClient(app).post("/search/7?query=flood&years_back=51")

    assert response.status_code == 400
    assert "at most 50" in response.json()["detail"]


def test_custom_valid_from_to_range_works():
    year_filter = normalize_search_year_filter(
        publication_year_from=2021,
        publication_year_to=2026,
        current_year=2026,
    )

    assert year_filter.years_back is None
    assert year_filter.publication_year_from == 2021
    assert year_filter.publication_year_to == 2026


def test_from_greater_than_to_returns_clean_error():
    try:
        normalize_search_year_filter(
            publication_year_from=2026,
            publication_year_to=2021,
            current_year=2026,
        )
    except PublicationYearFilterError as exc:
        assert "<=" in str(exc)
    else:
        raise AssertionError("Expected invalid range to fail")


def test_future_year_too_far_returns_clean_error():
    try:
        normalize_search_year_filter(
            publication_year_from=2025,
            publication_year_to=2028,
            current_year=2026,
        )
    except PublicationYearFilterError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("Expected future range to fail")


def test_candidates_outside_range_and_missing_year_are_excluded(monkeypatch):
    provider = _YearProvider()
    _patch_phase1_dependencies(monkeypatch, provider)

    db = _FakeDB()
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        year_filter=SearchYearFilter(publication_year_from=2024, publication_year_to=2026),
    )

    assert provider.calls == [{"query": "flood detection", "limit": 20}]
    assert [candidate.title for candidate in candidates] == ["Current Paper"]
    assert [candidate.title for candidate in top5] == ["Current Paper"]
    saved_candidates = [item for item in db.added if isinstance(item, PaperCandidate)]
    assert [candidate.title for candidate in saved_candidates] == ["Current Paper"]
    assert all(candidate.year is not None for candidate in saved_candidates)


def test_session_filters_json_can_provide_years_back(monkeypatch):
    captured = {}

    def _pipeline(db, session_id, query, llm_options=None, year_filter=None):
        captured["year_filter"] = year_filter
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: _session(filters_json={"years_back": 5}),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)

    response = TestClient(app).post("/search/7?query=flood")

    assert response.status_code == 200
    assert captured["year_filter"].to_metadata() == {
        "years_back": 5,
        "publication_year_from": 2022,
        "publication_year_to": 2026,
        "missing_year_policy": "excluded_when_year_filter_active",
    }
    assert response.json()["query_context"]["applied_filters"]["years_back"] == 5


def test_request_level_filter_overrides_session_filter(monkeypatch):
    captured = {}

    def _pipeline(db, session_id, query, llm_options=None, year_filter=None):
        captured["year_filter"] = year_filter
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: _session(filters_json={"years_back": 10}),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)

    response = TestClient(app).post("/search/7?query=flood&years_back=3")

    assert response.status_code == 200
    assert captured["year_filter"].to_metadata() == {
        "years_back": 3,
        "publication_year_from": 2024,
        "publication_year_to": 2026,
        "missing_year_policy": "excluded_when_year_filter_active",
    }


def test_session_creation_rejects_invalid_year_filter(monkeypatch):
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_APIDB())

    response = TestClient(app).post(
        "/sessions",
        json={
            "topic": "Flood review",
            "raw_query": "flood detection",
            "filters_json": {"publication_year_from": 1800, "publication_year_to": 2026},
        },
    )

    assert response.status_code == 400
    assert "1900" in response.json()["detail"]
