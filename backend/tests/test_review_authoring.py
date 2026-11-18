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
from app.services.evidence_pack import build_evidence_pack
from app.services.llm_client import LLMClient, LLMUnavailableError
from app.services.review_writer import generate_review_outline


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


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Flood review",
        "raw_query": "flood detection remote sensing",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _candidate(**overrides):
    base = {
        "id": 21,
        "session_id": 7,
        "title": "Flood Detection with Remote Sensing",
        "abstract": "A CNN approach for flood detection from satellite imagery.",
        "year": 2024,
        "source_title": "Remote Sensing",
        "doi": "10.1000/flood",
        "url": "https://example.com/flood",
        "provider": "mock",
        "final_score": 0.91,
        "created_at": datetime.utcnow(),
        "raw_metadata_json": {
            "_review": {
                "status": "shortlisted",
                "notes": "Use in methods section.",
                "tags": ["remote sensing", "cnn", {"unsafe": "object"}],
                "raw_prompt": "do-not-leak",
            },
            "_llm_summary": {
                "one_sentence_summary": "A paper on satellite flood detection.",
                "contribution": "Shows a remote sensing method.",
                "raw_model_response": "do-not-leak",
            },
            "_llm_diagnostics": {"raw_prompt": "do-not-leak"},
            "_ranking": {"debug": "do-not-leak"},
        },
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def teardown_function():
    app.dependency_overrides.clear()


def test_review_outline_requires_authentication():
    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 403


def test_evidence_pack_includes_only_shortlisted_safe_fields():
    shortlisted = _candidate()
    dismissed = _candidate(
        id=22,
        title="Dismissed Paper",
        raw_metadata_json={"_review": {"status": "dismissed"}},
    )
    pack = build_evidence_pack(
        _FakeDB(session_obj=_session(), candidates=[shortlisted, dismissed]),
        session_id=7,
        research_query="flood detection",
    )

    assert pack.paper_count == 1
    paper = pack.papers[0].to_dict()
    assert paper["candidate_id"] == 21
    assert paper["title"] == "Flood Detection with Remote Sensing"
    assert paper["llm_summary"]["one_sentence_summary"] == "A paper on satellite flood detection."
    assert paper["review_notes"] == "Use in methods section."
    assert paper["review_tags"] == ["remote sensing", "cnn"]

    payload_text = json.dumps(pack.to_llm_payload(), sort_keys=True)
    for forbidden in ("raw_prompt", "raw_model_response", "_llm_diagnostics", "_ranking", "do-not-leak"):
        assert forbidden not in payload_text


def test_review_outline_no_shortlisted_papers_returns_clean_error(monkeypatch):
    db = _FakeDB(
        session_obj=_session(),
        candidates=[_candidate(raw_metadata_json={"_review": {"status": "dismissed"}})],
    )

    def _should_not_call_llm(evidence_pack):
        raise AssertionError("LLM should not be called without shortlisted papers")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.review.generate_review_outline", _should_not_call_llm)

    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 400
    assert response.json() == {"detail": "No shortlisted papers for this session."}


def test_review_outline_mock_mode_is_deterministic(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == 7
    assert data["research_query"] == "flood detection remote sensing"
    assert data["paper_count"] == 1
    assert data["llm_used"] is True
    assert data["prompt_version"] == "v1"
    assert "fallback_reason" not in data
    assert data["outline"]["title"] == "Literature Review Outline: flood detection remote sensing"
    assert data["outline"]["sections"][0]["section_id"] == "introduction"
    assert data["outline"]["sections"][0]["supporting_candidate_ids"] == [21]


def test_review_outline_verifies_session_ownership():
    db = _FakeDB(session_obj=None, candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_review_outline_response_does_not_leak_internal_metadata(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 200
    response_text = json.dumps(response.json(), sort_keys=True)
    for forbidden in (
        "raw_prompt",
        "raw_model_response",
        "_llm",
        "_ranking",
        "api_key",
        "stack",
        "traceback",
        "do-not-leak",
    ):
        assert forbidden not in response_text


def test_review_outline_llm_disabled_returns_safe_fallback(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))

    response = TestClient(app).post("/sessions/7/review/outline")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "llm_disabled"
    assert data["outline"]["sections"]
    assert data["outline"]["sections"][0]["supporting_candidate_ids"] == [21]


class _FailingClient:
    def generate_structured_text(self, task, payload):
        raise LLMUnavailableError("missing_api_key")


def test_review_writer_failing_client_returns_safe_outline():
    pack = build_evidence_pack(
        _FakeDB(session_obj=_session(), candidates=[_candidate()]),
        session_id=7,
        research_query="flood detection",
    )

    result = generate_review_outline(pack, client=_FailingClient())

    assert result.used_llm is False
    assert result.fallback_reason == "missing_api_key"
    assert result.outline.sections[0].supporting_candidate_ids == [21]


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
