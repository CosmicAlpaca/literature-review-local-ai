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
from app.services.llm_client import LLMClient


AUTHORING_ENDPOINTS = (
    "/sessions/7/review/outline",
    "/sessions/7/review/themes",
    "/sessions/7/review/draft",
    "/sessions/7/review/export/markdown",
)

FORBIDDEN_RESPONSE_STRINGS = (
    "raw_prompt",
    "raw_model_response",
    "_llm",
    "_ranking",
    "_trust",
    "api_key",
    "stack",
    "traceback",
    "do-not-leak",
    "sk-test-secret",
)


class _FakeQuery:
    def __init__(self, db, model):
        self._db = db
        self._model = model

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        if self._model is ReviewSession:
            return self._db.session_obj
        return None

    def all(self):
        if self._model is PaperCandidate:
            return self._db.candidates
        return []


class _FakeDB:
    def __init__(self, session_obj, candidates):
        self.session_obj = session_obj
        self.candidates = candidates

    def query(self, model):
        return _FakeQuery(self, model)


class _SearchDB:
    pass


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Cancer AI review",
        "raw_query": "artificial intelligence for early cancer detection",
        "filters_json": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _candidate(**overrides):
    base = {
        "id": 21,
        "session_id": 7,
        "title": "Imaging AI for Early Cancer Detection",
        "abstract": "Deep learning models for image-based early cancer diagnosis.",
        "year": 2024,
        "source_title": "Journal of Medical AI",
        "doi": "10.1000/imaging",
        "url": "https://example.com/imaging",
        "provider": "mock",
        "final_score": 0.91,
        "created_at": datetime.utcnow(),
        "raw_metadata_json": {
            "_review": {
                "status": "shortlisted",
                "notes": "Important for imaging section.",
                "tags": ["imaging", "diagnosis"],
                "raw_prompt": "do-not-leak",
            },
            "_llm_summary": {
                "one_sentence_summary": "A paper on imaging AI for diagnosis.",
                "contribution": "Shows an imaging AI method.",
                "raw_model_response": "do-not-leak",
            },
            "_llm_diagnostics": {"raw_prompt": "do-not-leak"},
            "_ranking": {"debug": "do-not-leak"},
            "_trust": {"debug": "do-not-leak"},
            "api_key": "sk-test-secret",
        },
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _llm_settings(*, enabled: bool):
    return SimpleNamespace(
        llm_enabled=enabled,
        llm_provider="mock",
        llm_model="mock-test-model",
        llm_api_key=None,
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_prompt_version="v1",
        llm_max_input_chars_per_call=12000,
    )


_DEFAULT_SESSION = object()


def _install_authoring_db(candidates, *, session_obj=_DEFAULT_SESSION):
    db = _FakeDB(
        session_obj=_session() if session_obj is _DEFAULT_SESSION else session_obj,
        candidates=candidates,
    )
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    return db


def teardown_function():
    app.dependency_overrides.clear()


def test_authoring_endpoints_reject_unauthenticated_access():
    client = TestClient(app)

    for endpoint in AUTHORING_ENDPOINTS:
        response = client.post(endpoint)
        assert response.status_code == 403


def test_authoring_endpoints_protect_session_ownership():
    _install_authoring_db([_candidate()], session_obj=None)
    client = TestClient(app)

    for endpoint in AUTHORING_ENDPOINTS:
        response = client.post(endpoint)
        assert response.status_code == 404
        assert response.json() == {"detail": "Session not found"}


def test_authoring_endpoints_return_clean_error_without_shortlisted_papers(monkeypatch):
    _install_authoring_db([
        _candidate(
            raw_metadata_json={"_review": {"status": "dismissed", "raw_prompt": "do-not-leak"}}
        )
    ])

    def _fail_generation(*args, **kwargs):
        raise AssertionError("authoring generation should not run without shortlisted papers")

    monkeypatch.setattr("app.api.review.generate_review_outline", _fail_generation)
    monkeypatch.setattr("app.api.review.generate_theme_groups", _fail_generation)
    monkeypatch.setattr("app.api.review.generate_review_draft", _fail_generation)

    client = TestClient(app)
    for endpoint in AUTHORING_ENDPOINTS:
        response = client.post(endpoint)
        assert response.status_code == 400
        assert response.json() == {"detail": "No shortlisted papers for this session."}


def test_authoring_flow_returns_safe_artifacts_from_shortlisted_papers(monkeypatch):
    shortlisted = _candidate(id=21)
    second_shortlisted = _candidate(
        id=22,
        title="Clinical AI for Early Detection",
        doi=None,
        url="https://example.com/clinical",
        raw_metadata_json={
            "_review": {"status": "shortlisted", "tags": ["clinical"], "raw_prompt": "do-not-leak"},
            "_llm_summary": {"one_sentence_summary": "A clinical AI detection paper."},
            "_ranking": {"debug": "do-not-leak"},
            "_trust": {"debug": "do-not-leak"},
        },
    )
    dismissed = _candidate(
        id=23,
        title="Dismissed AI Paper",
        raw_metadata_json={"_review": {"status": "dismissed", "tags": ["dismissed"]}},
    )
    _install_authoring_db([shortlisted, second_shortlisted, dismissed])
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    client = TestClient(app)
    outline = client.post("/sessions/7/review/outline").json()
    themes = client.post("/sessions/7/review/themes").json()
    draft = client.post("/sessions/7/review/draft").json()
    export = client.post("/sessions/7/review/export/markdown").json()

    assert outline["paper_count"] == 2
    assert outline["outline"]["sections"]
    assert set(outline["outline"]["sections"][0]["supporting_candidate_ids"]) == {21, 22}

    assert themes["paper_count"] == 2
    assert themes["themes"]
    theme_ids = {
        candidate_id
        for theme in themes["themes"]
        for candidate_id in theme["supporting_candidate_ids"]
    }
    assert theme_ids <= {21, 22}

    assert draft["paper_count"] == 2
    assert draft["draft"]["sections"]
    assert draft["markdown_preview"].startswith("# ")
    draft_ids = {
        candidate_id
        for section in draft["draft"]["sections"]
        for candidate_id in section["supporting_candidate_ids"]
    }
    assert draft_ids <= {21, 22}

    assert export["filename"] == "literature_review_session_7.md"
    assert export["grounding_check"]["valid"] is True
    assert export["markdown"].startswith("# ")
    assert "Research query: artificial intelligence for early cancer detection" in export["markdown"]
    assert "[C21] Imaging AI for Early Cancer Detection" in export["markdown"]
    assert "[C22] Clinical AI for Early Detection" in export["markdown"]
    assert "Dismissed AI Paper" not in json.dumps([outline, themes, draft, export], sort_keys=True)

    response_text = json.dumps([outline, themes, draft, export], sort_keys=True)
    for forbidden in FORBIDDEN_RESPONSE_STRINGS:
        assert forbidden not in response_text


def test_authoring_endpoints_fallback_when_llm_disabled(monkeypatch):
    _install_authoring_db([_candidate()])
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))

    client = TestClient(app)
    for endpoint in AUTHORING_ENDPOINTS:
        response = client.post(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert data["llm_used"] is False
        assert data["fallback_reason"] == "llm_disabled"

    export = client.post("/sessions/7/review/export/markdown").json()
    assert export["grounding_check"]["valid"] is True


def test_search_year_filter_contract_remains_unaffected(monkeypatch):
    captured = {}

    def _pipeline(db, session_id, query, llm_options=None, year_filter=None):
        captured["year_filter"] = year_filter
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchDB())
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: _session(filters_json={"years_back": 10}),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline)
    monkeypatch.setattr("app.services.orchestrator._current_publication_year", lambda: 2026)

    response = TestClient(app).post("/search/7?query=cancer&years_back=3")

    assert response.status_code == 200
    assert captured["year_filter"].to_metadata() == {
        "years_back": 3,
        "publication_year_from": 2024,
        "publication_year_to": 2026,
        "missing_year_policy": "excluded_when_year_filter_active",
    }
    assert response.json()["query_context"]["applied_filters"]["years_back"] == 3
