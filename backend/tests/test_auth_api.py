from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_db
from app.core.security import PASSWORD_TOO_LONG_MESSAGE, hash_password
from app.main import app


class _AuthQuery:
    def __init__(self, result=None):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _AuthDB:
    def __init__(self, existing_user=None):
        self.existing_user = existing_user
        self.added_user = None
        self.committed = False

    def query(self, model):
        return _AuthQuery(self.existing_user)

    def add(self, obj):
        self.added_user = obj
        if getattr(obj, "id", None) is None:
            obj.id = 41

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        return None


def _override_db(db):
    def _dependency():
        yield db

    return _dependency


def _client_with_db(db):
    app.dependency_overrides[get_db] = _override_db(db)
    return TestClient(app)


def _clear_overrides():
    app.dependency_overrides.clear()


def test_register_exact_production_short_password_case_succeeds():
    db = _AuthDB()
    client = _client_with_db(db)

    response = client.post(
        "/auth/register",
        json={
            "email": "fresh_test@example.com",
            "full_name": "Fresh User",
            "password": "shortpass123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"] == {
        "id": 41,
        "email": "fresh_test@example.com",
        "full_name": "Fresh User",
        "is_active": True,
        "is_admin": False,
    }
    assert db.committed is True
    assert db.added_user.email == "fresh_test@example.com"
    assert db.added_user.password_hash != "shortpass123"

    _clear_overrides()


def test_register_exactly_72_ascii_byte_password_succeeds():
    db = _AuthDB()
    client = _client_with_db(db)

    response = client.post(
        "/auth/register",
        json={
            "email": "exact72@example.com",
            "full_name": "Exact Password",
            "password": "a" * 72,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"] == {
        "id": 41,
        "email": "exact72@example.com",
        "full_name": "Exact Password",
        "is_active": True,
        "is_admin": False,
    }
    assert db.committed is True

    _clear_overrides()


def test_register_password_longer_than_72_utf8_bytes_returns_validation_error():
    db = _AuthDB()
    client = _client_with_db(db)

    response = client.post(
        "/auth/register",
        json={
            "email": "long@example.com",
            "full_name": "Long Password",
            "password": "a" * 73,
        },
    )

    assert response.status_code == 422
    assert PASSWORD_TOO_LONG_MESSAGE in response.text
    assert db.added_user is None
    assert db.committed is False

    _clear_overrides()


def test_register_multibyte_password_exceeding_72_bytes_returns_validation_error():
    db = _AuthDB()
    client = _client_with_db(db)

    response = client.post(
        "/auth/register",
        json={
            "email": "multi@example.com",
            "full_name": "Multibyte Password",
            "password": "\u00e9" * 37,
        },
    )

    assert response.status_code == 422
    assert PASSWORD_TOO_LONG_MESSAGE in response.text
    assert db.added_user is None

    _clear_overrides()


def test_login_password_longer_than_72_utf8_bytes_returns_clean_error():
    user = SimpleNamespace(
        id=7,
        email="login@example.com",
        full_name="Login User",
        role="user",
        password_hash=hash_password("shortpass123"),
    )
    db = _AuthDB(existing_user=user)
    client = _client_with_db(db)

    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "a" * 73},
    )

    assert response.status_code == 422
    assert PASSWORD_TOO_LONG_MESSAGE in response.text

    _clear_overrides()


def test_duplicate_email_returns_clean_client_error():
    existing = SimpleNamespace(id=9, email="exists@example.com", full_name="Existing User", role="user")
    db = _AuthDB(existing_user=existing)
    client = _client_with_db(db)

    response = client.post(
        "/auth/register",
        json={
            "email": "exists@example.com",
            "full_name": "Existing User",
            "password": "shortpass123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered."
    assert PASSWORD_TOO_LONG_MESSAGE not in response.text
    assert db.added_user is None

    _clear_overrides()


def test_login_with_short_password_succeeds():
    user = SimpleNamespace(
        id=7,
        email="login@example.com",
        full_name="Login User",
        role="user",
        password_hash=hash_password("shortpass123"),
    )
    db = _AuthDB(existing_user=user)
    client = _client_with_db(db)

    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "shortpass123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert "user" not in data

    _clear_overrides()
