from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_mock_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user, get_db
from app.db.base import Base
from app.db.models.review_draft import ReviewDraft
from app.db.models.review_session import ReviewSession
from app.main import app
from app.schemas.draft import DraftUpsertRequest
from app.services.review_drafts import upsert_draft_for_session


def assert_requires_auth(response):
    assert response.status_code in (401, 403)


class _FakeQuery:
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

    def first(self):
        if self._model is ReviewSession:
            rows = self._db.sessions
        elif self._model is ReviewDraft:
            rows = self._db.drafts
        else:
            rows = []
        for row in rows:
            if all(getattr(row, key, None) == value for key, value in self._filters.items()):
                return row
        return None

    def delete(self):
        if self._model is ReviewDraft:
            before = len(self._db.drafts)
            self._db.drafts = [
                row
                for row in self._db.drafts
                if not all(getattr(row, key, None) == value for key, value in self._filters.items())
            ]
            return before - len(self._db.drafts)
        return 0


class _DraftDB:
    def __init__(self, sessions=None, drafts=None):
        self.sessions = sessions or []
        self.drafts = drafts or []
        self.commit_count = 0

    def query(self, model):
        return _FakeQuery(self, model)

    def add(self, obj):
        if isinstance(obj, ReviewDraft) and obj not in self.drafts:
            if obj.id is None:
                obj.id = len(self.drafts) + 1
            self.drafts.append(obj)

    def commit(self):
        self.commit_count += 1

    def refresh(self, obj):
        return None


def _session(**overrides):
    base = {
        "id": 7,
        "user_id": 1,
        "topic": "Cancer AI review",
        "raw_query": "artificial intelligence for early cancer detection",
        "status": "created",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _draft(**overrides):
    base = {
        "id": 1,
        "session_id": 7,
        "user_id": 1,
        "title": "Initial draft",
        "content": "# Initial draft",
        "content_format": "markdown",
        "status": "draft",
        "version": 1,
        "metadata_json": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    base.update(overrides)
    return ReviewDraft(**base)


def _override_current_user():
    return SimpleNamespace(id=1, email="drafts@example.com", role="user")


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def teardown_function():
    app.dependency_overrides.clear()


def test_get_draft_before_save_returns_empty_envelope():
    db = _DraftDB(sessions=[_session()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).get("/sessions/7/draft")

    assert response.status_code == 200
    assert response.json() == {"session_id": 7, "exists": False, "draft": None}


def test_put_draft_creates_latest_draft():
    db = _DraftDB(sessions=[_session()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).put(
        "/sessions/7/draft",
        json={
            "title": "Cancer AI draft",
            "content": "# Cancer AI draft\n\nBody",
            "metadata": {"source": "manual"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["draft"]["session_id"] == 7
    assert data["draft"]["user_id"] == 1
    assert data["draft"]["title"] == "Cancer AI draft"
    assert data["draft"]["content"] == "# Cancer AI draft\n\nBody"
    assert data["draft"]["content_format"] == "markdown"
    assert data["draft"]["status"] == "draft"
    assert data["draft"]["version"] == 1
    assert data["draft"]["metadata"] == {"source": "manual"}
    assert len(db.drafts) == 1


def test_get_draft_after_put_returns_saved_content():
    db = _DraftDB(sessions=[_session()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)
    client = TestClient(app)

    put_response = client.put(
        "/sessions/7/draft",
        json={"title": "Saved", "content": "Saved markdown"},
    )
    get_response = client.get("/sessions/7/draft")

    assert put_response.status_code == 200
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["exists"] is True
    assert data["draft"]["title"] == "Saved"
    assert data["draft"]["content"] == "Saved markdown"


def test_put_draft_updates_same_row_and_increments_version():
    existing = _draft()
    db = _DraftDB(sessions=[_session()], drafts=[existing])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).put(
        "/sessions/7/draft",
        json={
            "title": "Updated draft",
            "content": "Updated markdown",
            "status": "generated",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(db.drafts) == 1
    assert db.drafts[0].id == existing.id
    assert data["draft"]["id"] == existing.id
    assert data["draft"]["content"] == "Updated markdown"
    assert data["draft"]["status"] == "generated"
    assert data["draft"]["version"] == 2


def test_user_cannot_access_another_users_session_draft():
    db = _DraftDB(
        sessions=[_session(id=7, user_id=2)],
        drafts=[_draft(session_id=7, user_id=2)],
    )
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    get_response = TestClient(app).get("/sessions/7/draft")
    put_response = TestClient(app).put("/sessions/7/draft", json={"content": "Nope"})

    assert get_response.status_code == 404
    assert put_response.status_code == 404


def test_draft_endpoints_require_authentication():
    app.dependency_overrides.clear()

    get_response = TestClient(app).get("/sessions/7/draft")
    put_response = TestClient(app).put("/sessions/7/draft", json={"content": "Draft"})

    assert_requires_auth(get_response)
    assert_requires_auth(put_response)


def test_empty_content_validation_returns_422():
    db = _DraftDB(sessions=[_session()])
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _override_db(db)

    response = TestClient(app).put("/sessions/7/draft", json={"content": "   "})

    assert response.status_code == 422
    assert db.drafts == []


def test_draft_metadata_is_json_safe_when_upserted():
    db = _DraftDB(sessions=[_session()])
    payload = DraftUpsertRequest(
        title="Metadata draft",
        content="Metadata content",
        metadata={
            "generated_at": datetime(2026, 5, 16, 12, 0, 0),
            "tags": {"ai", "review"},
            "nested": {"value": SimpleNamespace(name="object")},
        },
    )

    saved = upsert_draft_for_session(db, db.sessions[0], 1, payload)

    assert saved.metadata_json["generated_at"] == "2026-05-16T12:00:00"
    assert sorted(saved.metadata_json["tags"]) == ["ai", "review"]
    assert saved.metadata_json["nested"]["value"] == "namespace(name='object')"


def test_review_draft_model_is_registered_for_create_all():
    ddl_statements = []
    engine = create_mock_engine(
        "postgresql://",
        lambda sql, *args, **kwargs: ddl_statements.append(str(sql.compile(dialect=engine.dialect))),
    )

    table = Base.metadata.tables.get("review_drafts")
    assert table is not None
    assert {
        "id",
        "session_id",
        "user_id",
        "content",
        "version",
        "metadata_json",
        "created_at",
        "updated_at",
    }.issubset(table.columns.keys())

    Base.metadata.create_all(bind=engine)

    assert any("CREATE TABLE review_drafts" in statement for statement in ddl_statements)
    assert any("metadata_json JSONB" in statement for statement in ddl_statements)
