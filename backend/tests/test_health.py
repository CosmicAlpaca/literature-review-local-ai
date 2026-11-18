from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def test_health_endpoint_returns_safe_status():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"]
    assert data["environment"] in {"development", "production"}
    assert isinstance(data["debug"], bool)
    assert data["search_provider_mode"]
    assert isinstance(data["llm_enabled"], bool)
    assert "secret_key" not in data
    assert "database_url" not in data
    assert "llm_api_key" not in data


def test_root_returns_same_safe_health_shape():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database_url" not in data


def test_cors_allows_local_frontend_origin():
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
