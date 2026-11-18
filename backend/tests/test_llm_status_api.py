from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def test_llm_status_when_disabled(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", False)
    monkeypatch.setattr("app.api.llm.settings.llm_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.llm_model", "mock-literature-review-v1")
    monkeypatch.setattr("app.api.llm.settings.llm_api_key", None)
    monkeypatch.setattr("app.api.llm.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.api.llm.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.api.llm.settings.llm_summary_enabled", False)

    response = TestClient(app).get("/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_enabled"] is False
    assert data["llm_provider"] == "mock"
    assert data["available"] is False
    assert data["ready_for_ai_search"] is False
    assert data["enabled_steps"] == []
    assert data["reason"] == "llm_disabled"
    assert data["api_key_configured"] is False
    assert "disabled" in data["disabled_reason"]
    assert "llm_api_key" not in data
    assert "database_url" not in data


def test_llm_status_when_mock_enabled(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.llm_model", "mock-test-model")
    monkeypatch.setattr("app.api.llm.settings.llm_api_key", None)
    monkeypatch.setattr("app.api.llm.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_summary_enabled", True)

    response = TestClient(app).get("/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_enabled"] is True
    assert data["llm_provider"] == "mock"
    assert data["llm_model"] == "mock-test-model"
    assert data["query_rewrite_enabled"] is True
    assert data["rerank_enabled"] is True
    assert data["summary_enabled"] is True
    assert data["enabled_steps"] == ["query_rewrite", "rerank", "summary"]
    assert data["api_key_configured"] is False
    assert data["available"] is True
    assert data["ready_for_ai_search"] is True
    assert data["reason"] is None
    assert data["disabled_reason"] is None


def test_llm_status_openai_compatible_missing_api_key(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_provider", "openai_compatible")
    monkeypatch.setattr("app.api.llm.settings.llm_model", "gpt-test")
    monkeypatch.setattr("app.api.llm.settings.llm_api_key", None)
    monkeypatch.setattr("app.api.llm.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_summary_enabled", True)

    response = TestClient(app).get("/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_enabled"] is True
    assert data["llm_provider"] == "openai_compatible"
    assert data["api_key_configured"] is False
    assert data["available"] is False
    assert data["ready_for_ai_search"] is False
    assert data["enabled_steps"] == ["query_rewrite", "rerank", "summary"]
    assert data["reason"] == "missing_api_key"
    assert "API key" in data["disabled_reason"]
    assert "sk-" not in str(data)


def test_llm_status_enabled_without_steps_not_ready(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.llm_model", "mock-test-model")
    monkeypatch.setattr("app.api.llm.settings.llm_api_key", None)
    monkeypatch.setattr("app.api.llm.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.api.llm.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.api.llm.settings.llm_summary_enabled", False)

    response = TestClient(app).get("/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["ready_for_ai_search"] is False
    assert data["enabled_steps"] == []
    assert data["reason"] == "step_disabled"
