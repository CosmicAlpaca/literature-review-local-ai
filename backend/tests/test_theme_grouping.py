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
from app.services.llm_client import LLMClient, LLMGeneration
from app.services.theme_grouping import generate_theme_groups


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


class _FakeStructuredClient:
    provider = "openai_compatible"
    model = "fake-model"

    def __init__(self, content):
        self.content = content
        self.payload = None

    def generate_structured_text(self, task, payload):
        self.payload = payload
        return LLMGeneration(
            provider=self.provider,
            model=self.model,
            task=task,
            content=self.content,
            used_llm=True,
            prompt_version="test-v1",
        )


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Cancer AI review",
        "raw_query": "artificial intelligence for early cancer detection",
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


def teardown_function():
    app.dependency_overrides.clear()


def test_review_themes_requires_authentication():
    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 403


def test_review_themes_no_shortlisted_papers_returns_clean_error(monkeypatch):
    db = _FakeDB(
        session_obj=_session(),
        candidates=[_candidate(raw_metadata_json={"_review": {"status": "dismissed"}})],
    )

    def _should_not_call_llm(evidence_pack):
        raise AssertionError("LLM should not be called without shortlisted papers")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.review.generate_theme_groups", _should_not_call_llm)

    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 400
    assert response.json() == {"detail": "No shortlisted papers for this session."}


def test_review_themes_verifies_session_ownership():
    db = _FakeDB(session_obj=None, candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_review_themes_mock_mode_is_deterministic(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == 7
    assert data["research_query"] == "artificial intelligence for early cancer detection"
    assert data["paper_count"] == 1
    assert data["llm_used"] is True
    assert data["prompt_version"] == "v1"
    assert "fallback_reason" not in data
    assert data["themes"][0]["theme_id"] == "shortlisted_evidence"
    assert data["themes"][0]["supporting_candidate_ids"] == [21]


def test_review_themes_llm_disabled_returns_deterministic_fallback(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))

    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "llm_disabled"
    assert data["themes"][0]["theme_id"] == "shortlisted_evidence"
    assert data["themes"][0]["supporting_candidate_ids"] == [21]


def test_theme_groups_only_reference_shortlisted_candidate_ids(monkeypatch):
    shortlisted = _candidate(id=21)
    dismissed = _candidate(
        id=22,
        title="Dismissed AI Paper",
        raw_metadata_json={"_review": {"status": "dismissed", "tags": ["dismissed"]}},
    )
    db = _FakeDB(session_obj=_session(), candidates=[shortlisted, dismissed])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/themes")

    assert response.status_code == 200
    candidate_ids = {
        candidate_id
        for theme in response.json()["themes"]
        for candidate_id in theme["supporting_candidate_ids"]
    }
    assert candidate_ids == {21}


def test_unknown_candidate_ids_from_llm_output_are_sanitized():
    pack = build_evidence_pack(
        _FakeDB(
            session_obj=_session(),
            candidates=[
                _candidate(id=21),
                _candidate(id=22, title="Clinical AI Detection", raw_metadata_json={
                    "_review": {"status": "shortlisted", "tags": ["clinical"]},
                }),
            ],
        ),
        session_id=7,
        research_query="cancer detection",
    )
    client = _FakeStructuredClient(
        {
            "themes": [
                {
                    "theme_id": "imaging_ai",
                    "name": "Imaging AI",
                    "rationale": "Image-based AI methods.",
                    "keywords": ["imaging", "deep learning"],
                    "supporting_candidate_ids": [21, 999],
                },
                {
                    "theme_id": "invalid_theme",
                    "name": "Invalid Theme",
                    "rationale": "Unknown candidate only.",
                    "keywords": ["invalid"],
                    "supporting_candidate_ids": [999],
                },
            ]
        }
    )

    result = generate_theme_groups(pack, client=client)

    assert result.used_llm is True
    assert len(result.themes) == 1
    assert result.themes[0].theme_id == "imaging_ai"
    assert result.themes[0].supporting_candidate_ids == [21]


def test_review_themes_response_does_not_leak_internal_metadata(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.theme_grouping.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/themes")

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
