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
from app.services.review_writer import (
    ReviewDraft,
    ReviewDraftSection,
    check_draft_grounding,
    generate_review_draft,
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


class _FakeStructuredClient:
    provider = "openai_compatible"
    model = "fake-model"

    def __init__(self, content):
        self.content = content

    def generate_structured_text(self, task, payload):
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
                "contribution": "Shows an imaging AI method.",
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


def test_review_markdown_export_requires_authentication():
    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 403


def test_review_markdown_export_no_shortlisted_papers_returns_clean_error(monkeypatch):
    db = _FakeDB(
        session_obj=_session(),
        candidates=[_candidate(raw_metadata_json={"_review": {"status": "dismissed"}})],
    )

    def _should_not_call_llm(evidence_pack):
        raise AssertionError("LLM should not be called without shortlisted papers")

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.api.review.generate_review_draft", _should_not_call_llm)

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 400
    assert response.json() == {"detail": "No shortlisted papers for this session."}


def test_review_markdown_export_verifies_session_ownership():
    db = _FakeDB(session_obj=None, candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_review_markdown_export_returns_filename_markdown_and_references(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "literature_review_session_7.md"
    assert data["paper_count"] == 1
    assert data["llm_used"] is True
    markdown = data["markdown"]
    assert markdown.startswith("# ")
    assert "Research query: artificial intelligence for early cancer detection" in markdown
    assert "Evidence summary: 1 shortlisted papers." in markdown
    assert "## Overview of Shortlisted Evidence" in markdown
    assert "Key claims:" in markdown
    assert "Limitations or cautions:" in markdown
    assert "## References" in markdown
    assert "[C21] Imaging AI for Early Cancer Detection" in markdown
    assert "10.1000/imaging" in markdown
    assert data["grounding_check"] == {
        "valid": True,
        "unknown_citations": 0,
        "unknown_supporting_candidate_ids": 0,
        "sections_without_evidence": 0,
        "total_sections": 2,
        "total_supported_sections": 2,
    }


def test_review_markdown_export_references_only_shortlisted_candidates(monkeypatch):
    shortlisted = _candidate(id=21)
    dismissed = _candidate(
        id=22,
        title="Dismissed AI Paper",
        raw_metadata_json={"_review": {"status": "dismissed"}},
    )
    db = _FakeDB(session_obj=_session(), candidates=[shortlisted, dismissed])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 200
    markdown = response.json()["markdown"]
    assert "[C21] Imaging AI for Early Cancer Detection" in markdown
    assert "Dismissed AI Paper" not in markdown
    assert "[C22]" not in markdown


def test_review_markdown_export_llm_disabled_fallback_works(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "llm_disabled"
    assert data["grounding_check"]["valid"] is True
    assert "[C21]" in data["markdown"]


def test_grounding_check_catches_unknown_references_directly():
    pack = build_evidence_pack(
        _FakeDB(session_obj=_session(), candidates=[_candidate(id=21)]),
        session_id=7,
        research_query="cancer detection",
    )
    draft = ReviewDraft(
        title="Unsafe Draft",
        sections=[
            ReviewDraftSection(
                section_id="unsafe",
                title="Unsafe",
                content="This section cites one known and one unknown paper [C21] [C999].",
                supporting_candidate_ids=[21, 999],
                citation_placeholders=["[C21]", "[C999]", "bad-placeholder"],
                key_claims=["Unsafe claim."],
                limitations_or_cautions=[],
            ),
            ReviewDraftSection(
                section_id="unsupported",
                title="Unsupported",
                content="No support.",
                supporting_candidate_ids=[],
                citation_placeholders=[],
                key_claims=[],
                limitations_or_cautions=[],
            ),
        ],
    )

    check = check_draft_grounding(pack, draft).to_dict()

    assert check["valid"] is False
    assert check["unknown_citations"] == 3
    assert check["unknown_supporting_candidate_ids"] == 1
    assert check["sections_without_evidence"] == 1
    assert check["total_sections"] == 2
    assert check["total_supported_sections"] == 1


def test_export_sanitizes_unknown_llm_citations_before_grounding_check(monkeypatch):
    content = {
        "draft": {
            "title": "AI Cancer Detection Review",
            "sections": [
                {
                    "section_id": "imaging_ai",
                    "title": "Imaging AI",
                    "content": "Imaging methods support detection [C21] [C999].",
                    "supporting_candidate_ids": [21, 999],
                    "citation_placeholders": ["[C21]", "[C999]", "not-a-citation"],
                    "key_claims": ["AI imaging methods support early detection."],
                    "limitations_or_cautions": ["Evidence varies by dataset."],
                }
            ],
        }
    }
    db = _FakeDB(session_obj=_session(), candidates=[_candidate(id=21)])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr(
        "app.services.review_writer.get_llm_client",
        lambda: _FakeStructuredClient(content),
    )

    response = TestClient(app).post("/sessions/7/review/export/markdown")

    assert response.status_code == 200
    data = response.json()
    assert data["grounding_check"]["valid"] is True
    assert data["grounding_check"]["unknown_citations"] == 0
    assert "[C21]" in data["markdown"]
    assert "[C999]" not in data["markdown"]
    assert "not-a-citation" not in data["markdown"]


def test_review_markdown_export_response_does_not_leak_internal_metadata(monkeypatch):
    db = _FakeDB(session_obj=_session(), candidates=[_candidate()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    monkeypatch.setattr("app.services.review_writer.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/export/markdown")

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
