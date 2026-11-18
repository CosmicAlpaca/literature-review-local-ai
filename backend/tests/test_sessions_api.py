from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user, get_db
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_debate_report import ReviewDebateReport
from app.db.models.review_draft import ReviewDraft
from app.db.models.review_session import ReviewSession
from app.db.models.search_query_context import SearchQueryContext
from app.db.models.selected_paper import SelectedPaper
from app.main import app


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
        for obj in self._db.collection(self._model):
            if self._matches(obj):
                return obj
        return None

    def all(self):
        return [obj for obj in self._db.collection(self._model) if self._matches(obj)]

    def delete(self, *args, **kwargs):
        collection = self._db.collection(self._model)
        kept = [obj for obj in collection if not self._matches(obj)]
        deleted = len(collection) - len(kept)
        self._db.replace_collection(self._model, kept)
        return deleted

    def _matches(self, obj):
        return all(getattr(obj, key, None) == value for key, value in self._filters.items())


class _FakeDB:
    def __init__(self, *, session_obj, candidates=None, selected=None, contexts=None, debates=None, drafts=None):
        self.sessions = [session_obj] if session_obj else []
        self.candidates = candidates or []
        self.selected = selected or []
        self.contexts = contexts or []
        self.debates = debates or []
        self.drafts = drafts or []
        self.commit_count = 0

    def query(self, model):
        return _FakeQuery(self, model)

    def delete(self, obj):
        if isinstance(obj, ReviewSession) and obj in self.sessions:
            self.sessions.remove(obj)

    def commit(self):
        self.commit_count += 1

    def collection(self, model):
        if model is ReviewSession:
            return self.sessions
        if model is PaperCandidate:
            return self.candidates
        if model is SelectedPaper:
            return self.selected
        if model is SearchQueryContext:
            return self.contexts
        if model is ReviewDebateReport:
            return self.debates
        if model is ReviewDraft:
            return self.drafts
        return []

    def replace_collection(self, model, values):
        if model is ReviewSession:
            self.sessions = values
        elif model is PaperCandidate:
            self.candidates = values
        elif model is SelectedPaper:
            self.selected = values
        elif model is SearchQueryContext:
            self.contexts = values
        elif model is ReviewDebateReport:
            self.debates = values
        elif model is ReviewDraft:
            self.drafts = values


def _override_current_user():
    return SimpleNamespace(id=1, email="demo@example.com", role="user")


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def _session(**overrides):
    base = {
        "id": 38,
        "user_id": 1,
        "topic": "Cancer AI review",
        "raw_query": "artificial intelligence for early cancer detection",
        "status": "created",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    base.update(overrides)
    return ReviewSession(**base)


def _candidate(**overrides):
    base = {
        "id": 21,
        "session_id": 38,
        "title": "Imaging AI for Early Cancer Detection",
        "abstract": "Deep learning models for diagnosis.",
        "year": 2024,
        "provider": "mock",
        "scopus_matched": False,
        "raw_metadata_json": {"_review": {"status": "shortlisted"}},
        "created_at": datetime.utcnow(),
    }
    base.update(overrides)
    return PaperCandidate(**base)


def _install_db(db):
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    return db


def teardown_function():
    app.dependency_overrides.clear()


def test_delete_empty_session_succeeds():
    db = _install_db(_FakeDB(session_obj=_session()))

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 204
    assert db.sessions == []
    assert db.commit_count == 1


def test_delete_session_with_candidates_succeeds_and_removes_children():
    db = _install_db(_FakeDB(session_obj=_session(), candidates=[_candidate()]))

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 204
    assert db.sessions == []
    assert db.candidates == []


def test_delete_session_with_selected_papers_succeeds_and_removes_children():
    candidate = _candidate()
    db = _install_db(
        _FakeDB(
            session_obj=_session(),
            candidates=[candidate],
            selected=[SelectedPaper(id=5, session_id=38, paper_candidate_id=21, rank_order=1)],
        )
    )

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 204
    assert db.selected == []
    assert db.candidates == []


def test_delete_session_with_query_context_succeeds_and_removes_children():
    db = _install_db(
        _FakeDB(
            session_obj=_session(),
            contexts=[SearchQueryContext(id=3, session_id=38, payload_json={"status": "ok"})],
        )
    )

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 204
    assert db.contexts == []
    assert db.sessions == []


def test_delete_session_with_debate_report_and_draft_succeeds_and_removes_children():
    db = _install_db(
        _FakeDB(
            session_obj=_session(),
            debates=[
                ReviewDebateReport(
                    id=7,
                    session_id=38,
                    payload_json={"agent_outputs": {"final_arbiter": {}}},
                )
            ],
            drafts=[
                ReviewDraft(
                    id=8,
                    session_id=38,
                    user_id=1,
                    title="Draft",
                    content="Draft content",
                    content_format="markdown",
                    status="draft",
                )
            ],
        )
    )

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 204
    assert db.debates == []
    assert db.drafts == []
    assert db.sessions == []


def test_delete_another_users_session_returns_404_and_keeps_rows():
    db = _install_db(_FakeDB(session_obj=_session(user_id=2), candidates=[_candidate()]))

    response = TestClient(app).delete("/sessions/38")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
    assert len(db.sessions) == 1
    assert len(db.candidates) == 1
    assert db.commit_count == 0
