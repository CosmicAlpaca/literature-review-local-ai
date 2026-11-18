from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.dependencies import get_current_user
from app.main import app


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _authenticated_client():
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1,
        email="llm-preview@example.com",
        role="user",
    )
    return TestClient(app)


def _enable_mock_llm(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.llm_model", "mock-preview-model")
    monkeypatch.setattr("app.api.llm.settings.llm_api_key", "sk-secret-test")
    monkeypatch.setattr("app.api.llm.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.llm_prompt_version", "preview-test-v1")
    monkeypatch.setattr("app.api.llm.settings.llm_max_candidates_for_rerank", 5)
    monkeypatch.setattr("app.api.llm.settings.llm_max_calls_per_search", 20)
    monkeypatch.setattr("app.api.llm.settings.llm_max_estimated_tokens_per_search", 20000)
    monkeypatch.setattr("app.api.llm.settings.llm_max_input_chars_per_call", 12000)
    monkeypatch.setattr("app.api.llm.settings.llm_usage_estimation_enabled", True)


def _query_payload():
    return {
        "topic": "artificial intelligence",
        "raw_query": "AI for early cancer detection",
    }


def _rerank_payload(candidate_count=1):
    return {
        "query": "artificial intelligence for early cancer detection",
        "candidates": [
            {
                "id": f"p{index + 1}",
                "title": "AI for Early Cancer Detection",
                "abstract": "Artificial intelligence models for cancer screening.",
                "source_title": "Journal of Medical AI",
                "year": 2023,
                "citation_count": 100,
            }
            for index in range(candidate_count)
        ],
    }


def _summary_payload():
    return {
        "query": "artificial intelligence for early cancer detection",
        "paper": {
            "title": "AI for Early Cancer Detection",
            "abstract": "Artificial intelligence models for cancer screening.",
            "source_title": "Journal of Medical AI",
            "year": 2023,
        },
    }


def _local_rerank_payload(candidate_count=1):
    return {
        "query": "artificial intelligence for early cancer detection",
        "candidates": [
            {
                "candidate_id": index + 1,
                "title": "AI for Early Cancer Detection",
                "abstract": "Artificial intelligence models for cancer screening.",
                "source_title": "Journal of Medical AI",
                "year": 2023,
                "citation_count": 100,
                "doi": "10.1000/ai",
                "provider": "openalex_strict",
            }
            for index in range(candidate_count)
        ],
    }


def _enable_mock_local_reranker(monkeypatch):
    monkeypatch.setattr("app.api.llm.settings.local_reranker_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_provider", "mock")
    monkeypatch.setattr("app.api.llm.settings.local_reranker_model_version", "local-preview-v1")
    monkeypatch.setattr("app.api.llm.settings.local_reranker_top_k", 20)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_model_path", "")


def test_llm_preview_endpoints_require_authentication():
    client = TestClient(app)

    cases = [
        ("/llm/query-preview", _query_payload()),
        ("/llm/rerank-preview", _rerank_payload()),
        ("/llm/summary-preview", _summary_payload()),
        ("/llm/local-rerank-preview", _local_rerank_payload()),
    ]
    for path, payload in cases:
        response = client.post(path, json=payload)
        assert response.status_code == 403


def test_authenticated_query_preview_returns_safe_mock_shape(monkeypatch):
    _enable_mock_llm(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/query-preview", json=_query_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == "AI for early cancer detection"
    assert data["optimized_query"]
    assert data["expanded_queries"]
    assert "cancer" in data["key_terms"]
    assert data["notes"]
    assert data["provider"] == "mock"
    assert data["model"] == "mock-preview-model"
    assert data["prompt_version"] == "preview-test-v1"
    assert data["estimated_usage"]["estimated_total_tokens"] > 0
    assert "fallback_reason" not in data


def test_authenticated_rerank_preview_returns_bounded_scores(monkeypatch):
    _enable_mock_llm(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/rerank-preview", json=_rerank_payload(candidate_count=2))

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "mock"
    assert data["prompt_version"] == "preview-test-v1"
    assert len(data["results"]) == 2
    assert data["results"][0]["candidate_index"] == 0
    assert data["results"][0]["candidate_id"] == "p1"
    assert 0.0 <= data["results"][0]["relevance_score"] <= 1.0
    assert data["results"][0]["reason"]
    assert data["estimated_usage"]["estimated_total_tokens"] > 0


def test_rerank_preview_rejects_too_many_candidates_cleanly(monkeypatch):
    _enable_mock_llm(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/rerank-preview", json=_rerank_payload(candidate_count=6))

    assert response.status_code == 422
    assert "at most 5 candidates" in response.text


def test_authenticated_summary_preview_returns_safe_mock_shape(monkeypatch):
    _enable_mock_llm(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/summary-preview", json=_summary_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["one_sentence_summary"]
    assert data["contribution"]
    assert data["method_or_data"]
    assert data["relevance_to_query"]
    assert data["limitations_or_caution"]
    assert data["provider"] == "mock"
    assert data["model"] == "mock-preview-model"
    assert data["estimated_usage"]["estimated_total_tokens"] > 0


def test_authenticated_local_rerank_preview_returns_safe_mock_shape(monkeypatch):
    _enable_mock_local_reranker(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/local-rerank-preview", json=_local_rerank_payload(candidate_count=2))

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["model_provider"] == "mock"
    assert data["model_version"] == "local-preview-v1"
    assert len(data["results"]) == 2
    assert data["results"][0]["candidate_id"] == 1
    assert 0.0 <= data["results"][0]["local_relevance_score"] <= 1.0
    assert data["results"][0]["reason"]
    assert "fallback_reason" not in data


def test_local_rerank_preview_empty_candidates_returns_clean_error(monkeypatch):
    _enable_mock_local_reranker(monkeypatch)
    client = _authenticated_client()
    payload = _local_rerank_payload(candidate_count=0)

    response = client.post("/llm/local-rerank-preview", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "At least one candidate is required."


def test_local_rerank_preview_candidate_cap_validation(monkeypatch):
    _enable_mock_local_reranker(monkeypatch)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_top_k", 1)
    client = _authenticated_client()

    response = client.post("/llm/local-rerank-preview", json=_local_rerank_payload(candidate_count=2))

    assert response.status_code == 422
    assert "at most 1 candidates" in response.text

    monkeypatch.setattr("app.api.llm.settings.local_reranker_top_k", 25)
    absolute_cap_response = client.post(
        "/llm/local-rerank-preview",
        json=_local_rerank_payload(candidate_count=21),
    )

    assert absolute_cap_response.status_code == 422
    assert "at most 20 candidates" in absolute_cap_response.text


def test_local_rerank_preview_missing_model_path_does_not_crash_or_leak(monkeypatch):
    _enable_mock_local_reranker(monkeypatch)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_enabled", True)
    monkeypatch.setattr("app.api.llm.settings.local_reranker_provider", "local")
    monkeypatch.setattr("app.api.llm.settings.local_reranker_model_path", "D:/private/models/secret-reranker")
    client = _authenticated_client()

    response = client.post("/llm/local-rerank-preview", json=_local_rerank_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["model_provider"] == "local"
    assert data["fallback_reason"] == "model_path_missing"
    assert data["results"][0]["fallback_reason"] == "model_path_missing"
    response_text = response.text
    assert "secret-reranker" not in response_text
    assert "D:/private" not in response_text
    assert "traceback" not in response_text.lower()


def test_llm_disabled_preview_returns_fallback_not_500(monkeypatch):
    _enable_mock_llm(monkeypatch)
    monkeypatch.setattr("app.api.llm.settings.llm_enabled", False)
    client = _authenticated_client()

    response = client.post("/llm/query-preview", json=_query_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["optimized_query"] == "AI for early cancer detection"
    assert data["provider"] == "fallback"
    assert data["model"] == "none"
    assert data["fallback_reason"] == "llm_disabled"


def test_preview_endpoints_do_not_call_retrieval_provider(monkeypatch):
    _enable_mock_llm(monkeypatch)

    def _fail_if_called():
        raise AssertionError("retrieval provider must not be called by LLM previews")

    monkeypatch.setattr("app.providers.provider_factory.get_search_provider", _fail_if_called)
    client = _authenticated_client()

    response = client.post("/llm/rerank-preview", json=_rerank_payload())

    assert response.status_code == 200
    assert response.json()["results"]


def test_preview_response_omits_secrets_raw_prompts_and_traces(monkeypatch):
    _enable_mock_llm(monkeypatch)
    client = _authenticated_client()

    response = client.post("/llm/query-preview", json=_query_payload())

    assert response.status_code == 200
    response_text = response.text.lower()
    assert "sk-secret-test" not in response_text
    assert "api_key" not in response_text
    assert "raw_model_response" not in response_text
    assert "messages" not in response_text
    assert "traceback" not in response_text
