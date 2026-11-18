from pathlib import Path
import sys
from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.db.models.selected_paper import SelectedPaper
from app.main import app
from app.providers.base import PaperResult
from app.services.orchestrator import normalize_search_filters, run_search_pipeline


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
        if isinstance(obj, PaperCandidate) and obj.created_at is None:
            obj.created_at = datetime.utcnow()

    def commit(self):
        self.commit_count += 1


class _Provider:
    def __init__(self, papers):
        self.papers = papers
        self.calls = []

    def fetch_papers(self, query: str, limit: int = 20):
        self.calls.append({"query": query, "limit": limit})
        return self.papers[:limit]


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


def _paper(**overrides):
    base = {
        "title": "Open Scopus Nature AI",
        "abstract": "AI cancer detection.",
        "year": 2026,
        "authors": ["Author A"],
        "doi": "10.1000/open",
        "url": "https://example.com/open",
        "source_title": "Nature Medicine",
        "citation_count": 100,
        "provider": "openalex_strict",
        "raw_metadata": {"paper_id": "open", "is_oa": True, "venue": "Nature Medicine"},
    }
    base.update(overrides)
    return PaperResult(**base)


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def _patch_pipeline_dependencies(monkeypatch, provider):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)

    class _NoopElsevierProvider:
        api_key = None

    monkeypatch.setattr("app.providers.elsevier_provider.ElsevierProvider", _NoopElsevierProvider)

    def _match(db_session, source_title, raw_metadata=None):
        matched = source_title and "non-scopus" not in source_title.lower()
        return SimpleNamespace(
            source=SimpleNamespace(id=99, source_title=source_title) if matched else None,
            reason="test_match" if matched else "test_no_match",
            confidence=0.99 if matched else 0.0,
            matched_value=source_title,
            normalized_input_title=(source_title or "").lower(),
            input_identifiers=None,
            registry_metadata={},
        )

    monkeypatch.setattr("app.services.orchestrator.resolve_scopus_match", _match)
    monkeypatch.setattr("app.services.orchestrator.compute_score", lambda **kwargs: 0.77)
    monkeypatch.setattr("app.services.orchestrator.score_paper", lambda **kwargs: {"total": 0.77})
    monkeypatch.setattr("app.services.orchestrator.score_trust", lambda **kwargs: {"total": 0.8, "bucket": "high"})
    monkeypatch.setattr(
        "app.services.orchestrator.build_selection_debug",
        lambda ranking_score, trust_debug: {"combined_score": 0.83},
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_normalized_filter_contract_supports_new_and_old_year_names():
    recent = normalize_search_filters(query="q", recent_years=3, current_year=2026)
    old_recent = normalize_search_filters(query="q", years_back=3, current_year=2026)
    custom = normalize_search_filters(query="q", year_from=2021, year_to=2026, current_year=2026)
    old_custom = normalize_search_filters(
        query="q",
        publication_year_from=2021,
        publication_year_to=2026,
        current_year=2026,
    )

    assert recent.to_metadata()["recent_years"] == 3
    assert recent.to_metadata()["year_from"] == 2024
    assert old_recent.to_metadata()["recent_years"] == 3
    assert custom.to_metadata()["year_from"] == 2021
    assert old_custom.to_metadata()["year_to"] == 2026


def test_invalid_filters_return_clean_client_errors(monkeypatch):
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_FakeDB())
    client = TestClient(app)

    bad_urls = [
        "/search/7?query=q&year_from=2026&year_to=2021",
        "/search/7?query=q&limit=0",
        "/search/7?query=q&min_citations=-1",
        "/search/7?query=q&provider=unknown",
        "/search/7?query=q&access_type=bad",
    ]

    for url in bad_urls:
        response = client.post(url)
        assert response.status_code == 400
        assert response.json()["detail"]


def test_limit_is_capped_and_used_for_provider_fetch(monkeypatch):
    provider = _Provider(
        [
            _paper(
                title=f"Paper {index}",
                doi=f"10.1000/{index}",
                raw_metadata={
                    "paper_id": f"paper-{index}",
                    "is_oa": True,
                    "venue": "Nature Medicine",
                },
            )
            for index in range(120)
        ]
    )
    _patch_pipeline_dependencies(monkeypatch, provider)

    filters = normalize_search_filters(query="AI cancer", limit=150)
    candidates, _ = run_search_pipeline(_FakeDB(), 7, "AI cancer", search_filters=filters)

    assert filters.to_metadata()["limit"] == 100
    assert provider.calls == [{"query": "AI cancer", "limit": 100}]
    assert len(candidates) == 100


def test_post_retrieval_filters_apply_before_persistence(monkeypatch):
    provider = _Provider(
        [
            _paper(title="Open Nature High", source_title="Nature Medicine", citation_count=100),
            _paper(
                title="Closed Nature Low",
                doi="10.1000/closed",
                source_title="Nature Medicine",
                citation_count=5,
                raw_metadata={"paper_id": "closed", "is_oa": False},
            ),
            _paper(
                title="Open Other High",
                doi="10.1000/other",
                source_title="Other Journal",
                citation_count=100,
                raw_metadata={"paper_id": "other", "is_oa": True},
            ),
            _paper(
                title="Missing Source",
                doi="10.1000/missing-source",
                source_title=None,
                citation_count=100,
                raw_metadata={"paper_id": "missing-source", "is_oa": True},
            ),
            _paper(
                title="Unknown Access",
                doi="10.1000/unknown-access",
                source_title="Nature Medicine",
                citation_count=100,
                raw_metadata={"paper_id": "unknown-access"},
            ),
        ]
    )
    _patch_pipeline_dependencies(monkeypatch, provider)

    filters = normalize_search_filters(
        query="AI cancer",
        source="nature",
        min_citations=50,
        access_type="open",
    )
    candidates, _ = run_search_pipeline(_FakeDB(), 7, "AI cancer", search_filters=filters)

    assert [candidate.title for candidate in candidates] == ["Open Nature High"]
    assert filters.to_metadata()["pre_filter_count"] == 5
    assert filters.to_metadata()["post_filter_count"] == 1
    assert filters.to_metadata()["filtered_out_count"] == 4


def test_provider_scopus_closed_access_and_missing_access_filters(monkeypatch):
    provider = _Provider(
        [
            _paper(
                title="Closed Scopus Semantic",
                provider="semantic_scholar",
                source_title="Scopus Journal",
                doi="10.1000/semantic",
                raw_metadata={"paper_id": "semantic", "is_open_access": False},
            ),
            _paper(
                title="Closed Non Scopus Semantic",
                provider="semantic_scholar",
                source_title="Non-Scopus Journal",
                doi=None,
                raw_metadata={"paper_id": "non-scopus", "is_open_access": False},
            ),
            _paper(
                title="Open Scopus Mock",
                provider="mock",
                source_title="Scopus Journal",
                doi="10.1000/mock",
                raw_metadata={"paper_id": "mock", "is_open_access": True},
            ),
            _paper(
                title="Unknown Access Semantic",
                provider="semantic_scholar",
                source_title="Scopus Journal",
                doi="10.1000/unknown",
                raw_metadata={"paper_id": "unknown"},
            ),
        ]
    )
    _patch_pipeline_dependencies(monkeypatch, provider)

    filters = normalize_search_filters(
        query="AI cancer",
        provider="semantic_scholar",
        scopus_only=True,
        access_type="closed",
    )
    candidates, _ = run_search_pipeline(_FakeDB(), 7, "AI cancer", search_filters=filters)

    assert [candidate.title for candidate in candidates] == ["Closed Scopus Semantic"]
    assert candidates[0].scopus_matched is True


def test_api_response_exposes_safe_normalized_applied_filters(monkeypatch):
    provider = _Provider([
        _paper(title="Open Nature High", source_title="Nature Medicine", citation_count=100),
        _paper(
            title="Old Paper",
            doi="10.1000/old",
            year=2020,
            source_title="Nature Medicine",
            raw_metadata={"paper_id": "old", "is_oa": True, "venue": "Nature Medicine"},
        ),
    ])
    _patch_pipeline_dependencies(monkeypatch, provider)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_FakeDB())

    response = TestClient(app).post(
        "/search/7?query=AI%20cancer&recent_years=3&source=Nature&min_citations=50&access_type=open&limit=10"
    )

    assert response.status_code == 200
    applied = response.json()["query_context"]["applied_filters"]
    assert applied["query"] == "AI cancer"
    assert applied["limit"] == 10
    assert applied["recent_years"] == 3
    assert applied["year_from"] == 2024
    assert applied["year_to"] == 2026
    assert applied["source"] == "Nature"
    assert applied["min_citations"] == 50
    assert applied["access_type"] == "open"
    assert applied["missing_year_policy"] == "excluded_when_year_filter_active"
    assert applied["missing_access_policy"] == "excluded_when_access_filter_active"
    assert applied["pre_filter_count"] == 2
    assert applied["post_filter_count"] == 1
    assert applied["filtered_out_count"] == 1


def test_session_filters_json_and_request_overrides_work(monkeypatch):
    captured = {}

    def _pipeline(db, session_id, query, llm_options=None, search_filters=None):
        captured["filters"] = search_filters
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(
        _FakeDB(session_obj=_session(filters_json={"recent_years": 10, "provider": "mock", "limit": 5}))
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)

    response = TestClient(app).post("/search/7?query=AI%20cancer&recent_years=3&provider=semantic_scholar")

    assert response.status_code == 200
    metadata = captured["filters"].to_metadata()
    assert metadata["recent_years"] == 3
    assert metadata["provider"] == "semantic_scholar"
    assert metadata["limit"] == 5


def test_request_custom_year_range_overrides_session_recent_years():
    filters = normalize_search_filters(
        query="AI cancer",
        session_filters={"recent_years": 10},
        year_from=2021,
        year_to=2026,
        current_year=2026,
    )

    metadata = filters.to_metadata()
    assert metadata["year_from"] == 2021
    assert metadata["year_to"] == 2026
    assert "recent_years" not in metadata
    assert "years_back" not in metadata
