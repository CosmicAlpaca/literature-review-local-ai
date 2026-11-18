from datetime import datetime
from pathlib import Path
import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_debate_report import ReviewDebateReport
from app.db.models.review_session import ReviewSession
from app.main import app
from app.services.debate_agent import FINAL_RECOMMENDATIONS, generate_debate_report
from app.services.evidence_pack import build_evidence_pack
from app.services.llm_client import LLMClient, LLMGeneration
from app.services.llm_prompts import REVIEW_DEBATE_TASK
from app.services.review_writer import fallback_review_draft


FORBIDDEN_RESPONSE_STRINGS = (
    "raw_prompt",
    "raw_model_response",
    "_llm_diagnostics",
    "_ranking",
    '"_trust"',
    "api_key",
    "traceback",
    "stack trace",
    "do-not-leak",
    "sk-test-secret",
)


class _FakeQuery:
    def __init__(self, db, model):
        self._db = db
        self._model = model
        self._filters = {}

    def filter(self, *args, **kwargs):
        for expression in args:
            key = getattr(getattr(expression, "left", None), "key", None)
            value = getattr(getattr(expression, "right", None), "value", None)
            if key is not None:
                self._filters[key] = value
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        if self._model is ReviewSession:
            if self._db.session_obj and self._matches(self._db.session_obj):
                return self._db.session_obj
            return None
        if self._model is ReviewDebateReport:
            for report in self._db.debate_reports:
                if self._matches(report):
                    return report
        return None

    def all(self):
        if self._model is PaperCandidate:
            return self._db.candidates
        return []

    def _matches(self, obj):
        return all(getattr(obj, key, None) == value for key, value in self._filters.items())


class _FakeDB:
    def __init__(self, session_obj, candidates, debate_reports=None):
        self.session_obj = session_obj
        self.candidates = candidates
        self.debate_reports = debate_reports or []
        self.commit_count = 0

    def query(self, model):
        return _FakeQuery(self, model)

    def add(self, obj):
        if isinstance(obj, ReviewDebateReport) and obj not in self.debate_reports:
            self.debate_reports.append(obj)

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        return None


class _FakeDebateClient:
    provider = "openai_compatible"
    model = "fake-debate-model"

    def __init__(self, content):
        self.content = content
        self.payload = None

    def generate_structured_text(self, task, payload):
        assert task == REVIEW_DEBATE_TASK
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
        "citation_count": 42,
        "provider": "mock",
        "scopus_matched": True,
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


def _install_db(candidates, *, session_obj=_DEFAULT_SESSION, debate_reports=None):
    db = _FakeDB(
        session_obj=_session() if session_obj is _DEFAULT_SESSION else session_obj,
        candidates=candidates,
        debate_reports=debate_reports,
    )
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    return db


class _PayloadPart:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


def _debate_report_payload(summary="Generated debate summary."):
    return {
        "overall_assessment": "The draft is usable with revisions.",
        "strengths": ["Relevant evidence is present."],
        "weaknesses": ["Limitations need more detail."],
        "missing_evidence": ["Add validation evidence."],
        "unsupported_or_weak_claims": ["One claim needs a citation."],
        "source_reliability_concerns": ["Check metadata completeness."],
        "methodology_concerns": ["Clarify study designs."],
        "suggested_revisions": ["Add a limitations paragraph."],
        "final_recommendation": "revise_before_use",
        "confidence_score": 0.72,
        "action_items": ["Ground every claim in shortlisted papers."],
        "risk_level": "medium",
        "quality_scores": {
            "relevance": 0.8,
            "grounding": 0.7,
            "methodology": 0.6,
            "source_reliability": 0.7,
            "overall": 0.7,
        },
        "debate_summary": summary,
    }


def _role_payload(role_name, score):
    return {
        "role_name": role_name,
        "score": score,
        "findings": ["Finding"],
        "concerns": ["Concern"],
        "suggestions": ["Suggestion"],
    }


def _agent_outputs_payload(summary="Generated debate summary."):
    report = _debate_report_payload(summary)
    return {
        "relevance_reviewer": {
            **_role_payload("Relevance Reviewer", 0.8),
            "relevance_score": 0.8,
            "off_topic_sections": [],
            "query_alignment_comments": ["Aligned."],
        },
        "evidence_grounding_critic": {
            **_role_payload("Evidence Grounding Critic", 0.7),
            "grounding_score": 0.7,
            "unsupported_claims": [],
            "weakly_supported_claims": ["Weak claim."],
            "citation_issues": [],
        },
        "methodology_critic": {
            **_role_payload("Methodology Critic", 0.6),
            "methodology_score": 0.6,
            "methodology_gaps": ["Gap"],
            "overclaims": [],
            "missing_limitations": ["Limitation"],
            "suggested_methodology_revisions": ["Revision"],
        },
        "source_reliability_critic": {
            **_role_payload("Source Reliability Critic", 0.7),
            "source_quality_score": 0.7,
            "source_concerns": [],
            "low_trust_evidence": [],
            "metadata_warnings": [],
        },
        "final_arbiter": {
            **report,
            **_role_payload("Final Arbiter", 0.7),
        },
    }


def _grounding_payload():
    return {
        "valid": True,
        "unknown_citations": 0,
        "unknown_supporting_candidate_ids": 0,
        "sections_without_evidence": 0,
        "total_sections": 1,
        "total_supported_sections": 1,
    }


def _fake_debate_result(summary="Generated debate summary."):
    return SimpleNamespace(
        debate_report=_PayloadPart(_debate_report_payload(summary)),
        agent_outputs=_PayloadPart(_agent_outputs_payload(summary)),
        grounding_check=_PayloadPart(_grounding_payload()),
        markdown_report=f"# Debate Agent Report\n\n{summary}",
        debate_mode="structured_panel",
        used_llm=True,
        fallback_reason=None,
        prompt_version="test-v1",
        generated_at="2026-05-12T02:00:00Z",
    )


def teardown_function():
    app.dependency_overrides.clear()


def test_review_debate_requires_authentication():
    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 403


def test_review_debate_verifies_session_ownership():
    _install_db([_candidate()], session_obj=None)

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_review_debate_no_shortlisted_papers_returns_clean_error(monkeypatch):
    _install_db([
        _candidate(raw_metadata_json={"_review": {"status": "dismissed", "raw_prompt": "do-not-leak"}})
    ])

    def _should_not_call_debate(evidence_pack):
        raise AssertionError("Debate Agent should not run without shortlisted papers")

    monkeypatch.setattr("app.api.review.generate_debate_report", _should_not_call_debate)

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 400
    assert response.json() == {"detail": "No shortlisted papers for this session."}


def test_review_debate_mock_mode_returns_structured_panel(monkeypatch):
    _install_db([_candidate()])
    monkeypatch.setattr("app.services.debate_agent.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == 7
    assert data["research_query"] == "artificial intelligence for early cancer detection"
    assert data["paper_count"] == 1
    assert data["llm_used"] is True
    assert data["prompt_version"] == "v1"
    assert data["debate_mode"] == "structured_panel"
    assert data["markdown_report"].startswith("# Debate Agent Report")
    assert "fallback_reason" not in data
    assert set(data["agent_outputs"]) == {
        "relevance_reviewer",
        "evidence_grounding_critic",
        "methodology_critic",
        "source_reliability_critic",
        "final_arbiter",
    }
    assert data["debate_report"]["final_recommendation"] in FINAL_RECOMMENDATIONS
    assert 0 <= data["debate_report"]["confidence_score"] <= 1
    assert data["debate_report"]["action_items"]
    assert data["debate_report"]["risk_level"] in {"low", "medium", "high"}
    assert data["debate_report"]["debate_summary"]
    assert set(data["debate_report"]["quality_scores"]) == {
        "relevance",
        "grounding",
        "methodology",
        "source_reliability",
        "overall",
    }
    assert all(0 <= score <= 1 for score in data["debate_report"]["quality_scores"].values())
    for role_output in data["agent_outputs"].values():
        assert role_output["role_name"]
        assert 0 <= role_output["score"] <= 1
        assert isinstance(role_output["findings"], list)
        assert isinstance(role_output["concerns"], list)
        assert isinstance(role_output["suggestions"], list)
    assert data["grounding_check"]["valid"] is True


def test_review_debate_llm_disabled_fallback_does_not_500(monkeypatch):
    _install_db([_candidate()])
    monkeypatch.setattr("app.services.debate_agent.get_llm_client", lambda: LLMClient(_llm_settings(enabled=False)))

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_used"] is False
    assert data["fallback_reason"] == "llm_disabled"
    assert data["debate_report"]["overall_assessment"]
    assert len(data["debate_report"]["strengths"]) >= 2
    assert len(data["debate_report"]["weaknesses"]) >= 2
    assert data["debate_report"]["action_items"]
    assert data["debate_report"]["quality_scores"]["overall"] >= 0
    assert data["markdown_report"]
    assert data["agent_outputs"]["evidence_grounding_critic"]["citation_issues"]


def test_review_debate_response_does_not_leak_internal_metadata(monkeypatch):
    _install_db([_candidate()])
    monkeypatch.setattr("app.services.debate_agent.get_llm_client", lambda: LLMClient(_llm_settings(enabled=True)))

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 200
    response_text = json.dumps(response.json(), sort_keys=True)
    for forbidden in FORBIDDEN_RESPONSE_STRINGS:
        assert forbidden not in response_text
        assert forbidden not in response.json()["markdown_report"]


def test_review_debate_post_generates_and_persists_report(monkeypatch):
    db = _install_db([_candidate()])
    monkeypatch.setattr("app.api.review.generate_debate_report", lambda evidence_pack: _fake_debate_result())

    response = TestClient(app).post("/sessions/7/review/debate")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["generated_at"] == "2026-05-12T02:00:00Z"
    assert data["updated_at"]
    assert len(db.debate_reports) == 1
    assert db.debate_reports[0].payload_json["debate_report"]["debate_summary"] == "Generated debate summary."
    assert db.commit_count == 1


def test_review_debate_second_post_reuses_cached_report_without_generator(monkeypatch):
    db = _install_db([_candidate()])
    calls = {"count": 0}

    def _generate(evidence_pack):
        calls["count"] += 1
        return _fake_debate_result(summary="First result.")

    monkeypatch.setattr("app.api.review.generate_debate_report", _generate)
    first = TestClient(app).post("/sessions/7/review/debate")
    second = TestClient(app).post("/sessions/7/review/debate")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1
    assert second.json()["cached"] is True
    assert second.json()["debate_report"]["debate_summary"] == "First result."
    assert db.commit_count == 1


def test_review_debate_force_regenerates_and_updates_persisted_report(monkeypatch):
    db = _install_db([_candidate()])
    summaries = iter(["Old result.", "Fresh result."])

    def _generate(evidence_pack):
        return _fake_debate_result(summary=next(summaries))

    monkeypatch.setattr("app.api.review.generate_debate_report", _generate)
    first = TestClient(app).post("/sessions/7/review/debate")
    forced = TestClient(app).post("/sessions/7/review/debate?force=true")

    assert first.status_code == 200
    assert forced.status_code == 200
    assert forced.json()["cached"] is False
    assert forced.json()["debate_report"]["debate_summary"] == "Fresh result."
    assert len(db.debate_reports) == 1
    assert db.debate_reports[0].payload_json["debate_report"]["debate_summary"] == "Fresh result."
    assert db.commit_count == 2


def test_review_debate_limits_candidates_for_regeneration(monkeypatch):
    from app.api import review as review_api

    db = _install_db(
        [
            _candidate(id=21, title="Paper 1", final_score=0.99),
            _candidate(id=22, title="Paper 2", final_score=0.88),
            _candidate(id=23, title="Paper 3", final_score=0.77),
        ]
    )
    monkeypatch.setattr(review_api.settings, "llm_max_debate_candidates", 2)

    def _generate(evidence_pack):
        assert evidence_pack.paper_count == 2
        return _fake_debate_result()

    monkeypatch.setattr("app.api.review.generate_debate_report", _generate)

    response = TestClient(app).post("/sessions/7/review/debate?force=true")

    assert response.status_code == 200
    assert response.json()["paper_count"] == 2
    assert db.commit_count == 1


def test_review_debate_timeout_returns_fallback_instead_of_500(monkeypatch):
    _install_db([_candidate()])

    def _timeout(evidence_pack):
        raise TimeoutError("debate took too long")

    monkeypatch.setattr("app.api.review.generate_debate_report", _timeout)

    response = TestClient(app).post("/sessions/7/review/debate?force=true")

    assert response.status_code == 200
    data = response.json()
    assert data["fallback_reason"] == "timeout"
    assert data["llm_used"] is False
    assert data["cached"] is False
    assert data["debate_report"]["overall_assessment"]


def test_review_debate_get_returns_cached_report_and_does_not_generate(monkeypatch):
    db = _install_db([_candidate()])
    monkeypatch.setattr("app.api.review.generate_debate_report", lambda evidence_pack: _fake_debate_result())
    post_response = TestClient(app).post("/sessions/7/review/debate")

    def _should_not_generate(evidence_pack):
        raise AssertionError("GET must not regenerate Debate Agent output")

    monkeypatch.setattr("app.api.review.generate_debate_report", _should_not_generate)
    get_response = TestClient(app).get("/sessions/7/review/debate")

    assert post_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["cached"] is True
    assert get_response.json()["debate_report"]["debate_summary"] == "Generated debate summary."


def test_review_debate_get_missing_report_returns_clean_404(monkeypatch):
    _install_db([_candidate()])

    def _should_not_generate(evidence_pack):
        raise AssertionError("GET must not call Debate Agent")

    monkeypatch.setattr("app.api.review.generate_debate_report", _should_not_generate)

    response = TestClient(app).get("/sessions/7/review/debate")

    assert response.status_code == 404
    assert response.json() == {"detail": "Debate report not found"}


def test_review_debate_get_verifies_session_ownership():
    _install_db([_candidate()], session_obj=None)

    response = TestClient(app).get("/sessions/7/review/debate")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_review_debate_persisted_payload_does_not_leak_internal_metadata(monkeypatch):
    db = _install_db([_candidate()])
    monkeypatch.setattr("app.api.review.generate_debate_report", lambda evidence_pack: _fake_debate_result())

    response = TestClient(app).post("/sessions/7/review/debate")
    stored_text = json.dumps(db.debate_reports[0].payload_json, sort_keys=True)

    assert response.status_code == 200
    for forbidden in FORBIDDEN_RESPONSE_STRINGS:
        assert forbidden not in stored_text


def test_debate_parser_clamps_scores_recommendation_and_unknown_candidate_ids():
    pack = build_evidence_pack(
        _FakeDB(
            session_obj=_session(),
            candidates=[
                _candidate(id=21),
                _candidate(id=22, title="Clinical AI Detection", scopus_matched=False, citation_count=0),
                _candidate(
                    id=999,
                    title="Dismissed AI Paper",
                    raw_metadata_json={"_review": {"status": "dismissed"}},
                ),
            ],
        ),
        session_id=7,
        research_query="early cancer detection",
    )
    draft = fallback_review_draft(pack).draft
    client = _FakeDebateClient(
        {
            "agent_outputs": {
                "relevance_reviewer": {
                    "relevance_score": 1.8,
                    "off_topic_sections": ["Candidate 999 should not appear."],
                    "query_alignment_comments": ["Candidate 21 is aligned; [C999] is not."],
                },
                "evidence_grounding_critic": {
                    "grounding_score": -0.5,
                    "unsupported_claims": ["Unsupported claim cites [C999]."],
                    "weakly_supported_claims": ["Claim from C22 needs more detail."],
                    "citation_issues": ["raw_prompt do-not-leak"],
                },
                "methodology_critic": {
                    "methodology_score": 2,
                    "methodology_gaps": ["Candidate 999 method gap."],
                    "overclaims": ["No overclaim for candidate 21."],
                    "missing_limitations": ["Missing dataset limits."],
                    "suggested_methodology_revisions": ["Add methods for C22."],
                },
                "source_reliability_critic": {
                    "source_quality_score": -9,
                    "source_concerns": ["Paper 999 source concern."],
                    "low_trust_evidence": ["Candidate 22 has sparse reliability metadata."],
                    "metadata_warnings": ["api_key sk-test-secret"],
                },
                "final_arbiter": {
                    "overall_assessment": "Final assessment mentions [C999].",
                    "strengths": ["C21 is relevant."],
                    "weaknesses": ["Candidate 999 is weak."],
                    "missing_evidence": ["Need full text."],
                    "unsupported_or_weak_claims": ["[C999] unsupported."],
                    "source_reliability_concerns": ["Candidate 22 metadata is sparse."],
                    "methodology_concerns": ["Need methods."],
                    "suggested_revisions": ["Remove C999."],
                    "final_recommendation": "not_allowed",
                    "confidence_score": 5,
                    "risk_level": "extreme",
                    "quality_scores": {
                        "relevance": 9,
                        "grounding": -4,
                        "methodology": 0.5,
                        "source_reliability": 2,
                        "overall": 7,
                    },
                    "action_items": ["Remove paper 999."],
                    "debate_summary": "Raw summary references [C999].",
                },
            },
            "debate_report": {
                "overall_assessment": "Report mentions candidate 999.",
                "strengths": ["Candidate 21 is safely referenced."],
                "weaknesses": ["Candidate 999 should be removed."],
                "missing_evidence": ["Need full text for C22."],
                "unsupported_or_weak_claims": ["[C999] unsupported."],
                "source_reliability_concerns": ["Candidate 22 metadata warning."],
                "methodology_concerns": ["Method details are thin."],
                "suggested_revisions": ["Add limitations."],
                "final_recommendation": "revise_before_use",
                "confidence_score": -1,
                "risk_level": "critical",
                "quality_scores": {
                    "relevance": 1.8,
                    "grounding": -1,
                    "methodology": 0.4,
                    "source_reliability": 2,
                    "overall": 4,
                },
                "action_items": ["Check candidate 999."],
                "debate_summary": "Report summary mentions paper 999.",
            },
        }
    )

    result = generate_debate_report(pack, draft=draft, client=client)
    data = result.to_metadata()
    text = json.dumps(data, sort_keys=True)

    assert result.used_llm is True
    assert result.debate_report.final_recommendation == "revise_before_use"
    assert result.debate_report.confidence_score == 0.0
    assert result.debate_report.risk_level in {"low", "medium", "high"}
    assert result.debate_report.risk_level != "critical"
    assert result.debate_report.quality_scores == {
        "relevance": 1.0,
        "grounding": 0.0,
        "methodology": 0.4,
        "source_reliability": 1.0,
        "overall": 1.0,
    }
    assert result.markdown_report.startswith("# Debate Agent Report")
    assert result.agent_outputs.relevance_reviewer.relevance_score == 1.0
    assert result.agent_outputs.evidence_grounding_critic.grounding_score == 0.0
    assert result.agent_outputs.methodology_critic.methodology_score == 1.0
    assert result.agent_outputs.source_reliability_critic.source_quality_score == 0.0
    assert "999" not in text
    assert "C999" not in text
    assert "raw_prompt" not in text
    assert "sk-test-secret" not in text
    assert "critical" not in text
    assert set(client.payload["guardrails"]["allowed_candidate_ids"]) == {21, 22}
