from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.db.models.search_query_context import SearchQueryContext
from app.main import app
from app.services.llm_client import LLMClient, query_fingerprint
from app.services.paper_summary import PaperSummaryResult


def assert_requires_auth(response):
    assert response.status_code in (401, 403)


class _FakeCandidatesQuery:
    def __init__(self, results):
        self._results = results

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return self._results


class _FakeDB:
    def __init__(self, candidate_results=None):
        self._candidate_results = candidate_results or []

    def query(self, model):
        return _FakeCandidatesQuery(self._candidate_results)


class _SearchAPIQuery:
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
        if self._model is SearchQueryContext:
            for context in self._db.query_contexts:
                if self._matches(context):
                    return context
        return None

    def all(self):
        if self._model is PaperCandidate:
            return [candidate for candidate in self._db.candidates if self._matches(candidate)]
        return []

    def _matches(self, obj):
        return all(getattr(obj, key, None) == value for key, value in self._filters.items())


class _SearchAPIDB:
    def __init__(self, sessions=None, candidates=None, query_contexts=None):
        self.sessions = sessions or []
        self.candidates = candidates or []
        self.query_contexts = query_contexts or []
        self.commit_count = 0
        self.added = []

    def query(self, model):
        return _SearchAPIQuery(self, model)

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, SearchQueryContext) and obj not in self.query_contexts:
            if obj.id is None:
                obj.id = len(self.query_contexts) + 1
            self.query_contexts.append(obj)

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        return None


class _ReviewWorkflowQuery:
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
        if self._model is PaperCandidate:
            return self._db.candidate
        return None

    def all(self):
        if self._model is PaperCandidate:
            return self._db.candidates
        return []


class _ReviewWorkflowDB:
    def __init__(self, session_obj, candidates):
        self.session_obj = session_obj
        self.candidates = candidates
        self.candidate = candidates[0] if candidates else None
        self.commit_count = 0

    def query(self, model):
        return _ReviewWorkflowQuery(self, model)

    def add(self, obj):
        return None

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        return None


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
        "topic": "Flood review",
        "raw_query": "flood detection",
        "filters_json": None,
        "status": "created",
        "created_at": datetime.utcnow(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _review_candidate(**overrides):
    base = {
        "id": 21,
        "session_id": 7,
        "title": "Review Workflow Paper",
        "abstract": "A paper used for review workflow tests.",
        "year": 2024,
        "source_title": "Journal of Review Workflows",
        "citation_count": 12,
        "provider": "mock",
        "scopus_matched": True,
        "final_score": 0.82,
        "created_at": datetime.utcnow(),
        "doi": "10.1000/workflow",
        "url": "https://example.com/workflow",
        "raw_metadata_json": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _llm_settings(*, enabled=True):
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


def _override_app(db):
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)


def test_post_search_requires_authentication():
    client = TestClient(app)

    response = client.post("/search/1?query=test")

    assert_requires_auth(response)


def test_candidate_summary_requires_authentication():
    app.dependency_overrides.clear()
    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert_requires_auth(response)


def test_candidate_summary_requires_session_ownership():
    candidate = _review_candidate()
    db = _SearchAPIDB(sessions=[], candidates=[candidate])
    _override_app(db)

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 404
    assert db.commit_count == 0
    assert candidate.raw_metadata_json == {}

    app.dependency_overrides.clear()


def test_candidate_summary_candidate_not_found_returns_clean_404():
    db = _SearchAPIDB(sessions=[_session()], candidates=[])
    _override_app(db)

    response = TestClient(app).post("/search/7/candidates/999/summary")

    assert response.status_code == 404
    assert response.json()["detail"] == "Candidate not found"
    assert db.commit_count == 0

    app.dependency_overrides.clear()


def test_candidate_summary_generated_in_mock_mode(monkeypatch):
    candidate = _review_candidate(title="Mock Summary Paper")
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.services.paper_summary.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == 7
    assert data["candidate_id"] == 21
    assert data["cached"] is False
    assert data["llm_used"] is True
    assert data["prompt_version"] == "v1"
    assert data["summary"]["one_sentence_summary"].startswith("Mock Summary Paper")
    stored_summary = candidate.raw_metadata_json["_llm_summary"]
    assert stored_summary["used_llm"] is True
    assert stored_summary["provider"] == "mock"
    assert db.commit_count == 1

    app.dependency_overrides.clear()


def test_candidate_summary_reuses_cached_summary_when_force_false(monkeypatch):
    cached = {
        "one_sentence_summary": "Cached summary.",
        "contribution": "Cached contribution.",
        "method_or_data": "Cached method.",
        "relevance_to_query": "Cached relevance.",
        "limitations_or_caution": "Cached caution.",
        "used_llm": True,
        "provider": "mock",
        "model": "mock-test-model",
        "prompt_version": "v1",
        "query_fingerprint": query_fingerprint("flood detection"),
    }
    candidate = _review_candidate(raw_metadata_json={"_llm_summary": cached})
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)

    def _fail_summary(*args, **kwargs):
        raise AssertionError("summary should be served from cache")

    monkeypatch.setattr("app.api.search.summarize_paper", _fail_summary)

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["llm_used"] is True
    assert data["summary"]["one_sentence_summary"] == "Cached summary."
    assert db.commit_count == 0

    app.dependency_overrides.clear()


def test_candidate_summary_force_regenerates_cached_summary(monkeypatch):
    candidate = _review_candidate(
        raw_metadata_json={
            "_llm_summary": {
                "one_sentence_summary": "Old cached summary.",
                "contribution": "Old contribution.",
                "method_or_data": "Old method.",
                "relevance_to_query": "Old relevance.",
                "limitations_or_caution": "Old caution.",
                "used_llm": True,
                "prompt_version": "v1",
            }
        }
    )
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )

    def _summary(candidate, query=None):
        return PaperSummaryResult(
            one_sentence_summary="Fresh summary.",
            contribution="Fresh contribution.",
            method_or_data="Fresh method.",
            relevance_to_query="Fresh relevance.",
            limitations_or_caution="Fresh caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
        )

    monkeypatch.setattr("app.api.search.summarize_paper", _summary)

    response = TestClient(app).post("/search/7/candidates/21/summary?force=true")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["summary"]["one_sentence_summary"] == "Fresh summary."
    assert candidate.raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Fresh summary."
    assert db.commit_count == 1

    app.dependency_overrides.clear()


def test_candidate_summary_malformed_cache_does_not_crash(monkeypatch):
    candidate = _review_candidate(
        raw_metadata_json={
            "_llm_summary": {
                "one_sentence_summary": ["not", "safe"],
                "raw_model_response": "do not leak",
            }
        }
    )
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )
    monkeypatch.setattr(
        "app.api.search.summarize_paper",
        lambda candidate, query=None: PaperSummaryResult(
            one_sentence_summary="Recovered summary.",
            contribution="Recovered contribution.",
            method_or_data="Recovered method.",
            relevance_to_query="Recovered relevance.",
            limitations_or_caution="Recovered caution.",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
        ),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["summary"]["one_sentence_summary"] == "Recovered summary."
    assert "raw_model_response" not in response.text

    app.dependency_overrides.clear()


def test_candidate_summary_step_disabled_returns_safe_fallback(monkeypatch):
    candidate = _review_candidate(title="Disabled LLM Paper")
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.services.paper_summary.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=False)),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "step_disabled"
    assert data["summary"]["one_sentence_summary"].startswith("Disabled LLM Paper")
    assert candidate.raw_metadata_json["_llm_summary"]["fallback_reason"] == "step_disabled"

    app.dependency_overrides.clear()


def test_candidate_summary_llm_failure_returns_safe_fallback(monkeypatch):
    candidate = _review_candidate(title="Unavailable LLM Paper")
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )

    def _fail_summary(*args, **kwargs):
        raise RuntimeError("provider stack trace with secret-key should not leak")

    monkeypatch.setattr("app.api.search.summarize_paper", _fail_summary)

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "provider_unavailable"
    assert data["summary"]["one_sentence_summary"].startswith("Unavailable LLM Paper")
    assert "secret-key" not in response.text
    assert "stack trace" not in response.text

    app.dependency_overrides.clear()


def test_candidate_summary_missing_abstract_handled_safely(monkeypatch):
    candidate = _review_candidate(abstract=None)
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )

    def _fail_summary(*args, **kwargs):
        raise AssertionError("summary should not call LLM without an abstract")

    monkeypatch.setattr("app.api.search.summarize_paper", _fail_summary)
    monkeypatch.setattr(
        "app.api.search.enrich_candidate_abstract",
        lambda candidate: SimpleNamespace(found=False, reason="no_abstract"),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary?force=true")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "no_abstract"
    assert data["fallback_reason"] != "step_disabled"
    assert "No abstract was available" in data["summary"]["contribution"]

    app.dependency_overrides.clear()


def test_candidate_summary_enriches_missing_abstract_before_llm(monkeypatch):
    candidate = _review_candidate(abstract=None)
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )

    def _enrich(candidate):
        candidate.abstract = "Enriched abstract with enough content for generated paper summary."
        return SimpleNamespace(found=True, source="semantic_scholar")

    def _summary(candidate, query=None):
        assert candidate.abstract.startswith("Enriched abstract")
        return PaperSummaryResult(
            one_sentence_summary="Summary from enriched abstract.",
            contribution="Fresh contribution.",
            method_or_data="Fresh method.",
            relevance_to_query="Fresh relevance.",
            limitations_or_caution="Fresh caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
        )

    monkeypatch.setattr("app.api.search.enrich_candidate_abstract", _enrich)
    monkeypatch.setattr("app.api.search.summarize_paper", _summary)

    response = TestClient(app).post("/search/7/candidates/21/summary?force=true")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["llm_used"] is True
    assert "fallback_reason" not in data
    assert candidate.abstract.startswith("Enriched abstract")
    assert candidate.raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Summary from enriched abstract."
    assert db.commit_count == 1

    app.dependency_overrides.clear()


def test_candidate_summary_force_bypasses_stale_step_disabled_cache(monkeypatch):
    candidate = _review_candidate(
        raw_metadata_json={
            "_llm_summary": {
                "one_sentence_summary": "Old fallback summary.",
                "contribution": "Old fallback contribution.",
                "method_or_data": "Old fallback method.",
                "relevance_to_query": "Old fallback relevance.",
                "limitations_or_caution": "Old fallback caution.",
                "used_llm": False,
                "fallback_reason": "step_disabled",
                "prompt_version": "v1",
            }
        }
    )
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.search.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: LLMClient(_llm_settings(enabled=True)),
    )
    monkeypatch.setattr(
        "app.api.search.summarize_paper",
        lambda candidate, query=None: PaperSummaryResult(
            one_sentence_summary="Fresh summary.",
            contribution="Fresh contribution.",
            method_or_data="Fresh method.",
            relevance_to_query="Fresh relevance.",
            limitations_or_caution="Fresh caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
        ),
    )

    response = TestClient(app).post("/search/7/candidates/21/summary?force=true")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["llm_used"] is True
    assert "fallback_reason" not in data
    assert data["summary"]["one_sentence_summary"] == "Fresh summary."
    assert candidate.raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Fresh summary."
    assert candidate.raw_metadata_json["_llm_summary"]["fallback_reason"] is None

    app.dependency_overrides.clear()


def test_candidate_summary_response_does_not_leak_internal_metadata_or_secrets():
    candidate = _review_candidate(
        raw_metadata_json={
            "_llm_summary": {
                "one_sentence_summary": "Safe cached summary.",
                "contribution": "Safe contribution.",
                "method_or_data": "Safe method.",
                "relevance_to_query": "Safe relevance.",
                "limitations_or_caution": "Safe caution.",
                "used_llm": True,
                "prompt_version": "v1",
                "raw_prompt": "do not leak prompt",
                "raw_model_response": "do not leak response",
                "api_key": "secret-key",
            },
            "_llm_diagnostics": {"summary": {"raw_prompt": "do not leak"}},
            "_ranking": {"score": 1},
            "_trust": {"bucket": "high"},
        }
    )
    db = _SearchAPIDB(sessions=[_session(raw_query="flood detection")], candidates=[candidate])
    _override_app(db)

    response = TestClient(app).post("/search/7/candidates/21/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["summary"]["one_sentence_summary"] == "Safe cached summary."
    response_text = response.text
    for unsafe in (
        "raw_prompt",
        "raw_model_response",
        "api_key",
        "secret-key",
        "_llm_diagnostics",
        "_ranking",
        "_trust",
    ):
        assert unsafe not in response_text

    app.dependency_overrides.clear()


def test_get_candidates_missing_session_returns_404():
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[]))

    response = TestClient(app).get("/search/13/candidates")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}

    app.dependency_overrides.clear()


def test_post_search_missing_session_returns_404(monkeypatch):
    def _pipeline_should_not_run(*args, **kwargs):
        raise AssertionError("search pipeline should not run for a missing session")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[]))
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline_should_not_run)

    response = TestClient(app).post("/search/13?query=flood%20detection")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}

    app.dependency_overrides.clear()


def test_search_exception_returns_json_error_with_request_id_and_no_leaks(monkeypatch):
    def _run_pipeline(*args, **kwargs):
        raise RuntimeError("database_url=postgres://secret stacktrace raw_prompt api_key")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=7)]))
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/search/7?query=flood%20detection",
        headers={"X-Request-ID": "req-search-failed"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-search-failed"
    assert response.json() == {
        "error": {
            "code": "search_failed",
            "message": "Search failed. Please try again.",
            "request_id": "req-search-failed",
            "retryable": True,
        }
    }
    for unsafe in ("database_url", "secret", "stacktrace", "raw_prompt", "api_key"):
        assert unsafe not in response.text

    app.dependency_overrides.clear()


def test_unexpected_exception_handler_returns_json_error_with_request_id():
    route_count = len(app.router.routes)

    @app.get("/__test_unexpected_error")
    def _test_unexpected_error():
        raise RuntimeError("secret stack trace api_key")

    try:
        response = TestClient(app, raise_server_exceptions=False).get(
            "/__test_unexpected_error",
            headers={"X-Request-ID": "req-internal-error"},
        )
    finally:
        del app.router.routes[route_count:]

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-internal-error"
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "Internal server error.",
            "request_id": "req-internal-error",
            "retryable": True,
        }
    }
    for unsafe in ("secret", "stack trace", "api_key"):
        assert unsafe not in response.text


def test_http_exception_status_is_preserved_and_request_id_is_returned():
    route_count = len(app.router.routes)

    @app.get("/__test_http_exception")
    def _test_http_exception():
        raise HTTPException(status_code=418, detail="teapot")

    try:
        response = TestClient(app).get(
            "/__test_http_exception",
            headers={"X-Request-ID": "req-http-exception"},
        )
    finally:
        del app.router.routes[route_count:]

    assert response.status_code == 418
    assert response.headers["X-Request-ID"] == "req-http-exception"
    assert response.json() == {"detail": "teapot"}


def test_get_candidates_existing_session_without_candidates_returns_empty_list():
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=14)]))

    response = TestClient(app).get("/search/14/candidates")

    assert response.status_code == 200
    assert response.json() == []

    app.dependency_overrides.clear()


def test_get_candidates_existing_session_with_candidates_returns_data():
    now = datetime.utcnow()
    candidate = _review_candidate(
        id=22,
        title="Stored Session Paper",
        session_id=7,
        created_at=now,
        raw_metadata_json={},
    )
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(
        _SearchAPIDB(sessions=[_session(id=7)], candidates=[candidate])
    )

    response = TestClient(app).get("/search/7/candidates")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == 22
    assert data[0]["title"] == "Stored Session Paper"
    assert data[0]["abstract_present"] is True
    assert data[0]["abstract_length"] == len(candidate.abstract)
    assert "abstract" not in data[0]

    app.dependency_overrides.clear()


def test_get_candidates_for_another_users_session_returns_404():
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=7, user_id=2)]))

    response = TestClient(app).get("/search/7/candidates")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}

    app.dependency_overrides.clear()


def test_get_query_context_requires_authentication():
    app.dependency_overrides.clear()

    response = TestClient(app).get("/search/7/query-context")

    assert_requires_auth(response)


def test_get_query_context_before_search_returns_exists_false():
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=7)]))

    response = TestClient(app).get("/search/7/query-context")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": 7,
        "exists": False,
        "query_context": {},
        "updated_at": None,
    }

    app.dependency_overrides.clear()


def test_get_query_context_enforces_session_ownership():
    stored = SearchQueryContext(
        id=1,
        session_id=7,
        payload_json={"original_query": "hidden"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(
        _SearchAPIDB(sessions=[_session(id=7, user_id=2)], query_contexts=[stored])
    )

    response = TestClient(app).get("/search/7/query-context")

    assert response.status_code == 404

    app.dependency_overrides.clear()


def test_get_query_context_returns_persisted_safe_payload_without_provider_or_llm(monkeypatch):
    stored = SearchQueryContext(
        id=1,
        session_id=7,
        payload_json={
            "original_query": "brain cells when sleeping",
            "optimized_query": "brain cells sleep",
            "expanded_queries": ["sleep neurons"],
            "llm_requested": True,
            "_ranking": {"do": "not expose"},
            "_llm_diagnostics": {"secret": "no"},
            "raw_model_response": "no",
        },
        created_at=datetime.utcnow(),
        updated_at=datetime(2026, 5, 13, 10, 0, 0),
    )

    def _pipeline_should_not_run(*args, **kwargs):
        raise AssertionError("GET query-context must not run search")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(
        _SearchAPIDB(sessions=[_session(id=7)], query_contexts=[stored])
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _pipeline_should_not_run)

    response = TestClient(app).get("/search/7/query-context")

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["session_id"] == 7
    assert data["query_context"]["original_query"] == "brain cells when sleeping"
    assert data["query_context"]["optimized_query"] == "brain cells sleep"
    assert data["query_context"]["expanded_queries"] == ["sleep neurons"]
    assert "raw_model_response" not in response.text
    assert "_llm_diagnostics" not in response.text
    assert "_ranking" not in response.text

    app.dependency_overrides.clear()


def test_post_search_returns_frontend_compatible_shape(monkeypatch):
    now = datetime.utcnow()
    fake_candidates = [
        SimpleNamespace(
            id=11,
            title="Paper A",
            abstract="Abstract A",
            year=2024,
            source_title="Journal A",
            citation_count=12,
            provider="mock",
            scopus_matched=True,
            final_score=0.91,
            created_at=now,
        )
    ]
    captured = {}

    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None):
        captured["llm_options"] = llm_options
        return fake_candidates, fake_candidates

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    client = TestClient(app)
    response = client.post("/search/7?query=flood%20detection")

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"session_id", "candidates", "top5", "selected_papers", "query_context"}
    assert data["session_id"] == 7
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["title"] == "Paper A"
    assert data["candidates"][0]["provider"] == "mock"
    assert data["top5"][0]["final_score"] == 0.91
    assert data["selected_papers"][0]["final_score"] == 0.91
    assert data["query_context"]["original_query"] == "flood detection"
    assert data["query_context"]["effective_query"] == "flood detection"
    assert data["query_context"]["llm_used"] is False
    assert captured["llm_options"].use_llm is None

    app.dependency_overrides.clear()


def test_post_search_use_llm_false_empty_pipeline_returns_json_response(monkeypatch):
    captured = {}

    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None):
        captured["llm_options"] = llm_options
        return None

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app).post(
        "/search/7?query=flood%20detection&use_llm=false",
        json={},
    )

    assert response.status_code == 200
    assert response.text
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["query_context"]["original_query"] == "flood detection"
    assert data["query_context"]["effective_query"] == "flood detection"
    assert data["query_context"]["use_llm_requested"] is False
    assert data["query_context"]["llm_used"] is False
    assert data["query_context"]["llm_steps"] == []
    assert data["query_context"]["warnings"] == ["no_candidates"]
    assert captured["llm_options"].use_llm is False

    app.dependency_overrides.clear()


def test_post_search_insomnia_use_llm_false_returns_200_and_no_llm_client(monkeypatch):
    captured = {}

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        captured["query"] = query
        captured["llm_options"] = llm_options
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=7)]))
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM client should not be called")),
    )

    response = TestClient(app).post(
        "/search/7?query=insomnia&use_llm=false&use_query_rewrite=true&use_llm_rerank=true&use_llm_summary=true"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["selected_papers"] == []
    assert data["query_context"]["original_query"] == "insomnia"
    assert data["query_context"]["effective_query"] == "insomnia"
    assert data["query_context"]["llm_requested"] is False
    assert data["query_context"]["use_llm_requested"] is False
    assert data["query_context"]["llm_used"] is False
    assert data["query_context"]["llm_steps"] == []
    assert data["query_context"]["warnings"] == ["no_candidates"]
    assert captured["query"] == "insomnia"
    assert captured["llm_options"].use_llm is False

    app.dependency_overrides.clear()


def test_post_search_vietnamese_insomnia_use_llm_false_empty_returns_200(monkeypatch):
    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(_SearchAPIDB(sessions=[_session(id=7)]))
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app).post(
        "/search/7?query=thu%E1%BB%91c%20tr%E1%BB%8B%20m%E1%BA%A5t%20ng%E1%BB%A7&use_llm=false"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["query_context"]["original_query"] == "thuốc trị mất ngủ"
    assert data["query_context"]["effective_query"] == "thuốc trị mất ngủ"
    assert data["query_context"]["llm_used"] is False
    assert data["query_context"]["warnings"] == ["no_candidates"]

    app.dependency_overrides.clear()


def test_post_search_use_llm_true_always_returns_query_context(monkeypatch):
    db = _SearchAPIDB(sessions=[_session(id=7)])

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post("/search/7?query=weight%20loss%20pills&use_llm=true")

    assert response.status_code == 200
    data = response.json()
    assert data["query_context"] is not None
    assert data["query_context"]["original_query"] == "weight loss pills"
    assert data["query_context"]["status"] == "empty"
    assert data["query_context"]["warnings"] == ["no_candidates"]
    assert len(db.query_contexts) == 1

    app.dependency_overrides.clear()


def test_post_search_partial_llm_failure_context_is_persisted(monkeypatch):
    db = _SearchAPIDB(sessions=[_session(id=7)])

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        search_filters.query_context_metadata = {
            **(search_filters.query_context_metadata or {}),
            "status": "partial",
            "original_query": query,
            "optimized_query": query,
            "effective_query": query,
            "steps": ["query_rewrite", "provider_search", "ranking"],
            "warnings": ["query_rewrite_timeout"],
            "errors": ["query_rewrite:timeout"],
        }
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post("/search/7?query=weight%20loss%20pills&use_llm=true")

    assert response.status_code == 200
    context = response.json()["query_context"]
    assert context["status"] == "partial"
    assert context["warnings"] == ["query_rewrite_timeout", "no_candidates"]
    assert context["errors"] == ["query_rewrite:timeout"]
    assert db.query_contexts[0].payload_json["status"] == "partial"
    assert db.query_contexts[0].payload_json["errors"] == ["query_rewrite:timeout"]

    app.dependency_overrides.clear()


def test_get_query_context_after_no_llm_search_returns_persisted_empty_state(monkeypatch):
    db = _SearchAPIDB(sessions=[_session(id=7)])

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    post_response = TestClient(app).post("/search/7?query=insomnia&use_llm=false")
    get_response = TestClient(app).get("/search/7/query-context")

    assert post_response.status_code == 200
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["exists"] is True
    assert data["query_context"]["original_query"] == "insomnia"
    assert data["query_context"]["effective_query"] == "insomnia"
    assert data["query_context"]["llm_used"] is False
    assert data["query_context"]["llm_steps"] == []
    assert data["query_context"]["warnings"] == ["no_candidates"]

    app.dependency_overrides.clear()


def test_query_context_persistence_failure_does_not_fail_search(monkeypatch):
    class _FailingContextDB(_SearchAPIDB):
        def add(self, obj):
            if isinstance(obj, SearchQueryContext):
                raise RuntimeError("json persistence failed")
            super().add(obj)

        def rollback(self):
            self.rolled_back = True

    db = _FailingContextDB(sessions=[_session(id=7)])
    db.rolled_back = False

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app).post("/search/7?query=insomnia&use_llm=false")

    assert response.status_code == 200
    assert response.json()["query_context"]["llm_used"] is False
    assert db.rolled_back is True

    app.dependency_overrides.clear()


def test_post_search_zero_candidates_returns_empty_arrays(monkeypatch):
    def _override_db():
        yield _FakeDB()

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None: ([], []),
    )

    response = TestClient(app).post("/search/7?query=flood%20detection&use_llm=false")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["query_context"]["warnings"] == ["no_candidates"]

    app.dependency_overrides.clear()


def test_post_search_filters_remove_all_candidates_returns_filters_metadata(monkeypatch):
    captured = {}

    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        search_filters.pre_filter_count = 2
        search_filters.post_filter_count = 0
        captured["filters"] = search_filters
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(
            id=session_id,
            user_id=user_id,
            filters_json=None,
        ),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app).post("/search/7?query=flood%20detection&min_citations=999")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["query_context"]["applied_filters"]["min_citations"] == 999
    assert data["query_context"]["applied_filters"]["pre_filter_count"] == 2
    assert data["query_context"]["applied_filters"]["post_filter_count"] == 0
    assert data["query_context"]["applied_filters"]["filtered_out_count"] == 2
    assert captured["filters"].active is True

    app.dependency_overrides.clear()


def test_post_search_non_english_zero_results_returns_json_shape(monkeypatch):
    def _override_db():
        yield _FakeDB()

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None: ([], []),
    )

    response = TestClient(app).post(
        "/search/7?query=thu%E1%BB%91c%20ch%E1%BB%91ng%20covid&use_llm=false",
        json={},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["selected_papers"] == []
    assert data["query_context"]["original_query"] == "thuốc chống covid"
    assert data["query_context"]["llm_used"] is False
    assert data["query_context"]["warnings"] == ["no_candidates"]

    app.dependency_overrides.clear()


def test_get_candidates_returns_frontend_compatible_list(monkeypatch):
    now = datetime.utcnow()
    stored_candidates = [
        SimpleNamespace(
            id=21,
            title="Stored Paper",
            abstract="Stored abstract",
            year=2023,
            source_title="Journal Stored",
            citation_count=7,
            provider="semantic_scholar",
            scopus_matched=False,
            final_score=0.55,
            created_at=now,
        )
    ]

    def _override_db():
        yield _FakeDB(candidate_results=stored_candidates)

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )

    client = TestClient(app)
    response = client.get("/search/7/candidates")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "Stored Paper"
    assert data[0]["provider"] == "semantic_scholar"
    assert data[0]["scopus_matched"] is False
    assert data[0]["abstract_present"] is True
    assert data[0]["abstract_length"] == len("Stored abstract")

    app.dependency_overrides.clear()


def test_get_candidates_exposes_minimal_scopus_match_debug(monkeypatch):
    now = datetime.utcnow()
    stored_candidates = [
        SimpleNamespace(
            id=22,
            title="IEEE Access Paper",
            abstract="Stored abstract",
            year=2024,
            source_title="IEEE Access",
            citation_count=7,
            provider="openalex",
            scopus_matched=True,
            final_score=0.88,
            created_at=now,
            raw_metadata_json={
                "_source_match": {
                    "source_title": "IEEE Access",
                    "source_issn": None,
                    "source_eissn": "21693536",
                    "reason": "identifier_match:eissn",
                    "confidence": 1.0,
                }
            },
        )
    ]

    def _override_db():
        yield _FakeDB(candidate_results=stored_candidates)

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )

    client = TestClient(app)
    response = client.get("/search/7/candidates")

    assert response.status_code == 200
    candidate = response.json()[0]
    assert candidate["scopus_matched"] is True
    assert candidate["source_match_reason"] == "identifier_match:eissn"
    assert candidate["source_match_confidence"] == 1.0
    assert candidate["matched_source_title"] == "IEEE Access"
    assert "matched_source_issn" not in candidate
    assert candidate["matched_source_eissn"] == "21693536"

    app.dependency_overrides.clear()


def test_post_search_exposes_safe_llm_candidate_fields_and_query_context(monkeypatch):
    now = datetime.utcnow()
    llm_metadata = {
        "_llm_query": {
            "original_query": "flood detection",
            "optimized_query": "flood detection remote sensing",
            "expanded_queries": ["flood detection", "satellite flood mapping"],
            "key_terms": ["flood", "detection"],
            "notes": "Expanded for scholarly retrieval.",
            "provider": "mock",
            "model": "mock-test-model",
            "used_llm": True,
        },
        "_llm_rerank": {
            "candidate_id": 11,
            "candidate_index": 0,
            "relevance_score": 0.87,
            "reason": "Matches flood detection query.",
            "provider": "mock",
            "model": "mock-test-model",
            "used_llm": True,
        },
        "_llm_summary": {
            "one_sentence_summary": "A flood detection paper.",
            "contribution": "Shows a remote sensing approach.",
            "method_or_data": "Satellite imagery.",
            "relevance_to_query": "Directly relevant.",
            "limitations_or_caution": "Verify against full text.",
            "provider": "mock",
            "model": "mock-test-model",
            "used_llm": True,
        },
        "_ranking": {"private": "debug"},
        "_trust": {"private": "debug"},
    }
    fake_candidates = [
        SimpleNamespace(
            id=11,
            title="Paper A",
            abstract="Abstract A",
            year=2024,
            source_title="Journal A",
            citation_count=12,
            provider="mock",
            scopus_matched=True,
            final_score=0.91,
            created_at=now,
            raw_metadata_json=llm_metadata,
        )
    ]

    def _override_db():
        yield _FakeDB()

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None: (fake_candidates, fake_candidates),
    )
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)

    client = TestClient(app)
    response = client.post("/search/7?query=flood%20detection&use_llm=true")

    assert response.status_code == 200
    data = response.json()
    assert data["query_context"]["original_query"] == "flood detection"
    assert data["query_context"]["optimized_query"] == "flood detection remote sensing"
    assert data["query_context"]["effective_query"] == "flood detection remote sensing"
    assert data["query_context"]["expanded_queries"] == ["flood detection", "satellite flood mapping"]
    assert data["query_context"]["key_terms"] == ["flood", "detection"]
    assert data["query_context"]["query_rewrite_notes"] == "Expanded for scholarly retrieval."
    assert data["query_context"]["llm_enabled"] is True
    assert data["query_context"]["llm_requested"] is True
    assert data["query_context"]["use_llm_requested"] is True
    assert data["query_context"]["llm_available"] is True
    assert data["query_context"]["llm_used"] is True
    assert data["query_context"]["llm_steps"] == ["query_rewrite", "rerank", "summary"]
    assert data["query_context"]["llm_query_rewrite_used"] is True
    candidate = data["candidates"][0]
    assert candidate["llm_relevance_score"] == 0.87
    assert candidate["llm_relevance_reason"] == "Matches flood detection query."
    assert candidate["llm_summary"]["one_sentence_summary"] == "A flood detection paper."
    assert "_llm_rerank" not in candidate
    assert "_ranking" not in candidate

    app.dependency_overrides.clear()


def test_post_search_persists_safe_query_context_when_candidates_exist(monkeypatch):
    now = datetime.utcnow()
    fake_candidates = [
        SimpleNamespace(
            id=11,
            title="Paper A",
            abstract="Abstract A",
            year=2024,
            source_title="Journal A",
            citation_count=12,
            provider="mock",
            scopus_matched=True,
            final_score=0.91,
            created_at=now,
            raw_metadata_json={
                "_llm_query": {
                    "original_query": "brain cells when sleeping",
                    "optimized_query": "neurons sleep",
                    "expanded_queries": ["sleep neurons"],
                    "key_terms": ["neurons", "sleep"],
                    "notes": "Expanded safely.",
                    "used_llm": True,
                    "raw_model_response": "should not persist",
                }
            },
        )
    ]
    db = _SearchAPIDB(sessions=[_session(id=7)])

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None: (fake_candidates, fake_candidates),
    )
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post(
        "/search/7?query=brain%20cells%20when%20sleeping&use_llm=true"
    )

    assert response.status_code == 200
    assert len(db.query_contexts) == 1
    payload = db.query_contexts[0].payload_json
    assert payload["original_query"] == "brain cells when sleeping"
    assert payload["optimized_query"] == "neurons sleep"
    assert payload["effective_query"] == "neurons sleep"
    assert payload["expanded_queries"] == ["sleep neurons"]
    assert payload["key_terms"] == ["neurons", "sleep"]
    assert payload["llm_used"] is True
    assert payload["llm_query_rewrite_used"] is True
    assert "raw_model_response" not in str(payload)
    assert "_llm_diagnostics" not in str(payload)
    assert "_ranking" not in str(payload)

    app.dependency_overrides.clear()


def test_post_search_persists_query_context_when_candidates_empty(monkeypatch):
    db = _SearchAPIDB(sessions=[_session(id=7)])

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        search_filters.query_context_metadata = {
            "original_query": query,
            "optimized_query": "neurons sleep",
            "expanded_queries": ["sleep neurons"],
            "key_terms": ["sleep"],
            "query_rewrite_notes": "Expanded safely.",
            "llm_query_rewrite_used": True,
            "empty_result_retry_used": True,
            "retry_queries": ["sleep neurons"],
            "final_search_query": "sleep neurons",
            "raw_model_response": "should not persist",
        }
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post(
        "/search/7?query=brain%20cells%20when%20sleeping&use_llm=true"
    )

    assert response.status_code == 200
    assert response.json()["query_context"]["retry_queries"] == ["sleep neurons"]
    assert response.json()["query_context"]["warnings"] == ["no_candidates"]
    assert len(db.query_contexts) == 1
    payload = db.query_contexts[0].payload_json
    assert payload["final_search_query"] == "sleep neurons"
    assert payload["empty_result_retry_used"] is True
    assert payload["llm_used"] is True
    assert payload["warnings"] == ["no_candidates"]
    assert "raw_model_response" not in str(payload)

    app.dependency_overrides.clear()


def test_post_search_use_llm_true_with_llm_disabled_falls_back(monkeypatch):
    now = datetime.utcnow()
    fake_candidates = [
        SimpleNamespace(
            id=12,
            title="Paper without LLM",
            abstract="Abstract",
            year=2024,
            source_title="Journal",
            citation_count=3,
            provider="mock",
            scopus_matched=False,
            final_score=0.42,
            created_at=now,
            raw_metadata_json={},
        )
    ]
    captured = {}

    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None):
        captured["llm_options"] = llm_options
        return fake_candidates, fake_candidates

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", False)

    client = TestClient(app)
    response = client.post("/search/7?query=flood%20detection&use_llm=true")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"][0]["title"] == "Paper without LLM"
    assert "llm_relevance_score" not in data["candidates"][0]
    assert data["query_context"]["llm_enabled"] is False
    assert data["query_context"]["llm_requested"] is True
    assert data["query_context"]["llm_available"] is False
    assert data["query_context"]["llm_query_rewrite_used"] is False
    assert "disabled" in data["query_context"]["llm_disabled_reason"]
    assert captured["llm_options"].use_llm is True

    app.dependency_overrides.clear()


def test_post_search_query_context_reports_rewrite_fallback_notes(monkeypatch):
    now = datetime.utcnow()
    fake_candidates = [
        SimpleNamespace(
            id=13,
            title="Paper with rewrite fallback",
            abstract="Abstract",
            year=2024,
            source_title="Journal",
            citation_count=3,
            provider="mock",
            scopus_matched=False,
            final_score=0.42,
            created_at=now,
            raw_metadata_json={
                "_llm_query": {
                    "original_query": "flood detection",
                    "optimized_query": "flood detection",
                    "expanded_queries": ["flood detection"],
                    "key_terms": ["flood", "detection"],
                    "notes": "Deterministic fallback; raw query preserved (invalid_json).",
                    "provider": "fallback",
                    "model": "none",
                    "used_llm": False,
                    "attempted": True,
                    "fallback_reason": "invalid_json",
                },
                "_llm_diagnostics": {
                    "query_rewrite": {
                        "attempted": True,
                        "used": False,
                        "fallback_reason": "invalid_json",
                    }
                },
            },
        )
    ]

    def _override_db():
        yield _FakeDB()

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )
    monkeypatch.setattr(
        "app.api.search.run_search_pipeline",
        lambda db, session_id, query, llm_options=None: (fake_candidates, fake_candidates),
    )
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post("/search/7?query=flood%20detection&use_llm=true")

    assert response.status_code == 200
    data = response.json()
    assert data["query_context"]["llm_query_rewrite_used"] is False
    assert "invalid_json" in data["query_context"]["query_rewrite_notes"]
    assert "_llm_diagnostics" not in data["candidates"][0]

    app.dependency_overrides.clear()


def test_post_search_empty_ai_retry_context_returns_valid_json(monkeypatch):
    captured = {}

    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        captured["filters"] = search_filters
        search_filters.query_context_metadata = {
            "original_query": "thuốc trị mất ngủ",
            "optimized_query": "pharmacological treatment insomnia sleep disorder medication",
            "expanded_queries": [
                "insomnia pharmacological therapy",
                "sleep disorder treatment medication",
            ],
            "key_terms": ["insomnia", "sleep disorder", "medication"],
            "query_rewrite_notes": "Translated for scholarly retrieval.",
            "llm_query_rewrite_used": True,
            "empty_result_retry_used": True,
            "retry_queries": ["insomnia pharmacological therapy"],
            "final_search_query": "insomnia pharmacological therapy",
        }
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(
            id=session_id,
            user_id=user_id,
            filters_json=None,
        ),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)
    monkeypatch.setattr("app.api.search.settings.llm_enabled", True)
    monkeypatch.setattr(
        "app.api.search.get_llm_client",
        lambda: SimpleNamespace(is_available=lambda: True),
    )

    response = TestClient(app).post(
        "/search/7?query=thu%E1%BB%91c%20tr%E1%BB%8B%20m%E1%BA%A5t%20ng%E1%BB%A7&use_llm=true",
        json={},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["query_context"]["original_query"] == "thuốc trị mất ngủ"
    assert data["query_context"]["optimized_query"] == "pharmacological treatment insomnia sleep disorder medication"
    assert data["query_context"]["llm_query_rewrite_used"] is True
    assert data["query_context"]["empty_result_retry_used"] is True
    assert data["query_context"]["retry_queries"] == ["insomnia pharmacological therapy"]
    assert data["query_context"]["final_search_query"] == "insomnia pharmacological therapy"
    assert captured["filters"].query_context_metadata["empty_result_retry_used"] is True

    app.dependency_overrides.clear()


def test_post_search_filters_pre_filter_zero_returns_valid_json(monkeypatch):
    def _override_db():
        yield _FakeDB()

    def _run_pipeline(db, session_id, query, llm_options=None, search_filters=None):
        search_filters.pre_filter_count = 0
        search_filters.post_filter_count = 0
        return [], []

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(
            id=session_id,
            user_id=user_id,
            filters_json=None,
        ),
    )
    monkeypatch.setattr("app.api.search.run_search_pipeline", _run_pipeline)

    response = TestClient(app).post("/search/7?query=flood%20detection&min_citations=999")

    assert response.status_code == 200
    data = response.json()
    assert data["candidates"] == []
    assert data["top5"] == []
    assert data["query_context"]["applied_filters"]["pre_filter_count"] == 0
    assert data["query_context"]["applied_filters"]["post_filter_count"] == 0
    assert data["query_context"]["applied_filters"]["filtered_out_count"] == 0

    app.dependency_overrides.clear()


def test_get_candidates_exposes_safe_llm_fields_for_persisted_candidates(monkeypatch):
    now = datetime.utcnow()
    stored_candidates = [
        SimpleNamespace(
            id=21,
            title="Stored Paper",
            abstract="Stored abstract",
            year=2023,
            source_title="Journal Stored",
            citation_count=7,
            provider="semantic_scholar",
            scopus_matched=False,
            final_score=0.55,
            created_at=now,
            raw_metadata_json={
                "_llm_rerank": {
                    "relevance_score": 0.64,
                    "reason": "Some overlap with query.",
                    "raw_model_response": "do not expose",
                },
                "_llm_summary": {
                    "one_sentence_summary": "Stored paper summary.",
                    "contribution": "Stored contribution.",
                    "method_or_data": "Stored method.",
                    "relevance_to_query": "Stored relevance.",
                    "limitations_or_caution": "Stored caution.",
                },
            },
        )
    ]

    def _override_db():
        yield _FakeDB(candidate_results=stored_candidates)

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )

    client = TestClient(app)
    response = client.get("/search/7/candidates")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["llm_relevance_score"] == 0.64
    assert data[0]["llm_relevance_reason"] == "Some overlap with query."
    assert data[0]["llm_summary"]["contribution"] == "Stored contribution."
    assert "raw_model_response" not in data[0]["llm_summary"]

    app.dependency_overrides.clear()


def test_malformed_llm_metadata_does_not_crash_response_serialization(monkeypatch):
    now = datetime.utcnow()
    stored_candidates = [
        SimpleNamespace(
            id=31,
            title="Malformed Metadata Paper",
            abstract=None,
            year=None,
            source_title=None,
            citation_count=None,
            provider="mock",
            scopus_matched=False,
            final_score=None,
            created_at=now,
            raw_metadata_json={
                "_llm_rerank": "not-a-dict",
                "_llm_summary": {"one_sentence_summary": ["not", "a", "string"]},
            },
        ),
        SimpleNamespace(
            id=32,
            title="Non-dict Metadata Paper",
            abstract=None,
            year=None,
            source_title=None,
            citation_count=None,
            provider="mock",
            scopus_matched=False,
            final_score=None,
            created_at=now,
            raw_metadata_json="not-a-dict",
        ),
    ]

    def _override_db():
        yield _FakeDB(candidate_results=stored_candidates)

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    monkeypatch.setattr(
        "app.api.search._get_user_session_or_404",
        lambda db, session_id, user_id: SimpleNamespace(id=session_id, user_id=user_id),
    )

    client = TestClient(app)
    response = client.get("/search/7/candidates")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["abstract_present"] is False
    assert data[0]["abstract_length"] == 0
    assert "llm_relevance_score" not in data[0]
    assert "llm_summary" not in data[0]
    assert "llm_relevance_score" not in data[1]
    assert "llm_summary" not in data[1]

    app.dependency_overrides.clear()


def test_shortlist_candidate_persists_review_status():
    candidate = _review_candidate()
    db = _ReviewWorkflowDB(
        session_obj=SimpleNamespace(id=7, user_id=1, topic="Flood review", raw_query="flood detection"),
        candidates=[candidate],
    )

    def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    response = TestClient(app).post("/search/7/candidates/21/shortlist")

    assert response.status_code == 200
    data = response.json()
    assert data["review_status"] == "shortlisted"
    assert candidate.raw_metadata_json["_review"]["status"] == "shortlisted"
    assert db.commit_count == 1

    app.dependency_overrides.clear()


def test_dismiss_candidate_and_restore_review_status():
    candidate = _review_candidate()
    db = _ReviewWorkflowDB(
        session_obj=SimpleNamespace(id=7, user_id=1, topic="Flood review", raw_query="flood detection"),
        candidates=[candidate],
    )

    def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    dismiss_response = TestClient(app).post("/search/7/candidates/21/dismiss")
    restore_response = TestClient(app).delete("/search/7/candidates/21/review")

    assert dismiss_response.status_code == 200
    assert dismiss_response.json()["review_status"] == "dismissed"
    assert restore_response.status_code == 200
    assert "review_status" not in restore_response.json()
    assert "_review" not in candidate.raw_metadata_json
    assert db.commit_count == 2

    app.dependency_overrides.clear()


def test_review_decision_requires_session_ownership():
    candidate = _review_candidate()
    db = _ReviewWorkflowDB(session_obj=None, candidates=[candidate])

    def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db

    response = TestClient(app).post("/search/7/candidates/21/shortlist")

    assert response.status_code == 404
    assert candidate.raw_metadata_json == {}
    assert db.commit_count == 0

    app.dependency_overrides.clear()


def test_list_shortlist_and_export_markdown():
    shortlisted = _review_candidate(
        raw_metadata_json={
            "_review": {"status": "shortlisted", "updated_at": "2026-04-28T00:00:00Z"},
            "_llm_summary": {"one_sentence_summary": "A concise AI-assisted summary."},
        }
    )
    dismissed = _review_candidate(
        id=22,
        title="Dismissed Paper",
        raw_metadata_json={"_review": {"status": "dismissed", "updated_at": "2026-04-28T00:00:00Z"}},
    )
    db = _ReviewWorkflowDB(
        session_obj=SimpleNamespace(id=7, user_id=1, topic="Flood review", raw_query="flood detection"),
        candidates=[shortlisted, dismissed],
    )

    def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db
    client = TestClient(app)

    list_response = client.get("/search/7/shortlist")
    export_response = client.get("/search/7/shortlist/export")

    assert list_response.status_code == 200
    data = list_response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Review Workflow Paper"
    assert data[0]["review_status"] == "shortlisted"

    assert export_response.status_code == 200
    assert "text/markdown" in export_response.headers["content-type"]
    assert "# Shortlisted papers for Flood review" in export_response.text
    assert "Review Workflow Paper" in export_response.text
    assert "10.1000/workflow" in export_response.text
    assert "A concise AI-assisted summary." in export_response.text
    assert "Dismissed Paper" not in export_response.text

    app.dependency_overrides.clear()
