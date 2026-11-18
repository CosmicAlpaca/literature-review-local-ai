from pathlib import Path
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.llm_client import LLMClient, LLMUnavailableError
from app.services.llm_client import LLMGeneration
from app.services.llm_client import (
    candidate_fingerprint,
    compact_payload_for_llm,
    estimate_llm_usage,
    estimate_text_tokens,
    query_fingerprint,
)
from app.services.llm_query import rewrite_query
from app.services.llm_reranker import rerank_candidates
from app.services.paper_summary import summarize_paper


def _settings(
    *,
    enabled: bool,
    provider: str = "mock",
    model: str = "mock-test-model",
    api_key: str | None = None,
    base_url: str = "https://llm.example.test/v1",
    timeout: float = 4.0,
    temperature: float = 0.0,
    prompt_version: str = "v1",
):
    return SimpleNamespace(
        llm_enabled=enabled,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_timeout_seconds=timeout,
        llm_temperature=temperature,
        llm_prompt_version=prompt_version,
    )


def test_query_rewrite_falls_back_when_llm_disabled():
    client = LLMClient(_settings(enabled=False))

    result = rewrite_query(
        raw_query="flood detection using deep learning",
        topic="remote sensing",
        client=client,
    )

    assert result.used_llm is False
    assert result.original_query == "flood detection using deep learning"
    assert result.optimized_query == "flood detection using deep learning"
    assert "flood" in result.key_terms


def test_usage_estimator_is_deterministic():
    usage = estimate_llm_usage(
        input_payload={"query": "abcd" * 10},
        output_payload={"answer": "ok"},
    )

    assert estimate_text_tokens("abcd") == 1
    assert usage.estimated_input_tokens == estimate_llm_usage(
        input_payload={"query": "abcd" * 10},
        output_payload={"answer": "ok"},
    ).estimated_input_tokens
    assert usage.estimated_total_tokens == usage.estimated_input_tokens + usage.estimated_output_tokens
    assert usage.estimated_call_count == 1


def test_payload_compaction_truncates_oversized_strings():
    payload, truncated = compact_payload_for_llm({"abstract": "x" * 5000}, max_chars=1200)

    assert truncated is True
    assert len(payload["abstract"]) < 5000


def test_query_rewrite_mock_is_deterministic():
    client = LLMClient(_settings(enabled=True, prompt_version="prompt-test-v2"))

    result = rewrite_query(
        raw_query="flood detection using deep learning",
        topic="remote sensing",
        client=client,
    )

    assert result.used_llm is True
    assert result.provider == "mock"
    assert result.optimized_query == "remote sensing flood detection deep learning"
    assert result.expanded_queries[0] == "flood detection using deep learning"
    assert result.prompt_version == "prompt-test-v2"
    assert result.generated_at
    assert result.query_fingerprint == query_fingerprint("flood detection using deep learning", "remote sensing")


def test_openai_compatible_client_builds_request_and_parses_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"optimized_query":"flood mapping","expanded_queries":["flood mapping"],'
                                '"key_terms":["flood","mapping"],"notes":"ok"}'
                            )
                        }
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(
        _settings(enabled=True, provider="openai_compatible", api_key="test-key"),
        http_client=http_client,
    )

    result = client.generate_structured_text(
        task="query_rewrite",
        payload={"raw_query": "flood", "topic": "mapping"},
    )

    assert result.provider == "openai_compatible"
    assert result.content["optimized_query"] == "flood mapping"
    assert captured["url"] == "https://llm.example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert '"response_format":{"type":"json_object"}' in captured["body"].replace(" ", "")

    http_client.close()


def test_openai_compatible_client_invalid_json_falls_back_via_unavailable():
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not json"}}]},
            )
        )
    )
    client = LLMClient(
        _settings(enabled=True, provider="openai_compatible", api_key="test-key"),
        http_client=http_client,
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        client.generate_structured_text(task="query_rewrite", payload={"raw_query": "flood"})
    assert exc_info.value.reason == "invalid_json"

    http_client.close()


def test_query_rewrite_real_client_invalid_json_returns_fallback():
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not json"}}]},
            )
        )
    )
    client = LLMClient(
        _settings(enabled=True, provider="openai_compatible", api_key="test-key"),
        http_client=http_client,
    )

    result = rewrite_query("flood detection", topic="urban", client=client)

    assert result.used_llm is False
    assert result.optimized_query == "flood detection"
    assert result.attempted is True
    assert result.fallback_reason == "invalid_json"

    http_client.close()


def test_openai_compatible_client_timeout_falls_back_via_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout", request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = LLMClient(
        _settings(enabled=True, provider="openai_compatible", api_key="test-key"),
        http_client=http_client,
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        client.generate_structured_text(task="query_rewrite", payload={"raw_query": "flood"})
    assert exc_info.value.reason == "timeout"

    http_client.close()


def test_openai_compatible_client_http_error_uses_safe_reason():
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, request=request))
    )
    client = LLMClient(
        _settings(enabled=True, provider="openai_compatible", api_key="test-key"),
        http_client=http_client,
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        client.generate_structured_text(task="query_rewrite", payload={"raw_query": "flood"})
    assert exc_info.value.reason == "http_error"

    http_client.close()


def test_reranker_falls_back_when_llm_disabled():
    client = LLMClient(_settings(enabled=False))
    candidates = [{"id": 1, "title": "Flood detection", "abstract": "CNN remote sensing"}]

    result = rerank_candidates("flood detection", candidates, client=client)

    assert result[0].used_llm is False
    assert result[0].llm_relevance_score == 0.0


def test_reranker_mock_scores_query_overlap_deterministically():
    client = LLMClient(_settings(enabled=True))
    candidates = [
        {"id": 1, "title": "Flood detection", "abstract": "CNN remote sensing"},
        {"id": 2, "title": "Crop yield", "abstract": "Agriculture"},
    ]

    result = rerank_candidates("flood detection", candidates, client=client)

    assert result[0].used_llm is True
    assert result[0].llm_relevance_score == 1.0
    assert result[0].query_fingerprint == query_fingerprint("flood detection")
    assert result[0].candidate_fingerprint == candidate_fingerprint(candidates[0])
    assert result[1].llm_relevance_score == 0.0


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
        )


class _CapturingStructuredClient(_FakeStructuredClient):
    def __init__(self, content):
        super().__init__(content)
        self.payload = None

    def generate_structured_text(self, task, payload):
        self.payload = payload
        return super().generate_structured_text(task, payload)


class _FailingClient:
    def generate_structured_text(self, task, payload):
        raise LLMUnavailableError("failed")


def test_query_rewrite_parses_fake_real_client_response():
    client = _FakeStructuredClient(
        {
            "optimized_query": "urban flood detection remote sensing",
            "expanded_queries": ["urban flood detection", "satellite flood mapping"],
            "key_terms": ["urban", "flood", "remote sensing"],
            "notes": "Expanded for scholarly retrieval.",
        }
    )

    result = rewrite_query("flood detection", topic="urban", client=client)

    assert result.used_llm is True
    assert result.provider == "openai_compatible"
    assert result.optimized_query == "urban flood detection remote sensing"
    assert result.key_terms == ["urban", "flood", "remote sensing"]


def test_query_rewrite_non_english_fake_result_uses_english_academic_terms():
    client = _FakeStructuredClient(
        {
            "optimized_query": "pharmacological treatment for insomnia sleep disorder medication",
            "expanded_queries": [
                "insomnia pharmacological therapy",
                "sleep disorder treatment medication",
                "hypnotics melatonin benzodiazepines insomnia treatment",
            ],
            "key_terms": ["insomnia", "sleep disorder", "medication", "hypnotics"],
            "notes": "Translated from Vietnamese for scholarly retrieval.",
        }
    )

    result = rewrite_query("thuốc trị mất ngủ", client=client)

    assert result.used_llm is True
    assert result.original_query == "thuốc trị mất ngủ"
    assert result.optimized_query == "pharmacological treatment for insomnia sleep disorder medication"
    assert "insomnia pharmacological therapy" in result.expanded_queries
    assert "insomnia" in result.key_terms
    assert all(ord(char) < 128 for char in result.optimized_query)


def test_query_rewrite_non_english_mock_avoids_raw_query_as_optimized():
    client = LLMClient(_settings(enabled=True))

    result = rewrite_query("thuốc trị mất ngủ", client=client)

    assert result.used_llm is True
    assert result.optimized_query != "thuốc trị mất ngủ"
    assert "insomnia" in result.optimized_query
    assert "medication" in result.optimized_query


def test_query_rewrite_missing_fields_uses_defaults():
    client = _FakeStructuredClient({"optimized_query": "flood mapping"})

    result = rewrite_query("flood detection", topic="urban", client=client)

    assert result.used_llm is True
    assert result.optimized_query == "flood mapping"
    assert result.expanded_queries
    assert "flood" in result.key_terms


def test_query_rewrite_invalid_contract_returns_safe_fallback():
    client = _FakeStructuredClient({"unexpected": "value"})

    result = rewrite_query("flood detection", topic="urban", client=client)

    assert result.used_llm is False
    assert result.fallback_reason == "invalid_json"
    assert result.optimized_query == "flood detection"


def test_query_rewrite_failing_client_returns_fallback():
    result = rewrite_query("flood detection", topic="urban", client=_FailingClient())

    assert result.used_llm is False
    assert result.optimized_query == "flood detection"
    assert result.attempted is True
    assert result.fallback_reason == "provider_unavailable"
    assert result.to_metadata()["fallback_reason"] == "provider_unavailable"


def test_reranker_parses_fake_real_client_response():
    client = _FakeStructuredClient(
        {
            "results": [
                {"candidate_index": 0, "candidate_id": 1, "relevance_score": 0.92, "reason": "Direct match"},
                {"candidate_index": 1, "candidate_id": 2, "relevance_score": 1.5},
            ]
        }
    )
    candidates = [
        {"id": 1, "title": "Flood detection", "abstract": "Satellite mapping"},
        {"id": 2, "title": "Flood risk", "abstract": "Hydrology"},
    ]

    result = rerank_candidates("flood detection", candidates, client=client)

    assert result[0].used_llm is True
    assert result[0].llm_relevance_score == 0.92
    assert result[0].reason == "Direct match"
    assert result[1].llm_relevance_score == 1.0
    assert result[1].reason == "LLM rerank returned no reason."


def test_reranker_clamps_negative_relevance_score_to_zero():
    client = _FakeStructuredClient(
        {
            "results": [
                {"candidate_index": 0, "candidate_id": 1, "relevance_score": -2, "reason": "Not relevant"}
            ]
        }
    )

    result = rerank_candidates(
        "flood detection",
        [{"id": 1, "title": "Flood detection", "abstract": "Satellite mapping"}],
        client=client,
    )

    assert result[0].used_llm is True
    assert result[0].llm_relevance_score == 0.0


def test_reranker_failing_client_returns_fallback():
    result = rerank_candidates(
        "flood detection",
        [{"id": 1, "title": "Flood detection"}],
        client=_FailingClient(),
    )

    assert result[0].used_llm is False
    assert result[0].llm_relevance_score == 0.0
    assert result[0].attempted is True
    assert result[0].fallback_reason == "provider_unavailable"


def test_reranker_invalid_json_shape_returns_safe_fallback():
    result = rerank_candidates(
        "flood detection",
        [{"id": 1, "title": "Flood detection"}],
        client=_FakeStructuredClient({"results": "not-a-list"}),
    )

    assert result[0].used_llm is False
    assert result[0].fallback_reason == "invalid_json"


def test_reranker_invalid_contract_returns_safe_fallback():
    result = rerank_candidates(
        "flood detection",
        [{"id": 1, "title": "Flood detection"}],
        client=_FakeStructuredClient({"results": [{"unexpected": "value"}]}),
    )

    assert result[0].used_llm is False
    assert result[0].fallback_reason == "invalid_json"


def test_paper_summary_falls_back_when_llm_disabled():
    client = LLMClient(_settings(enabled=False))
    candidate = SimpleNamespace(
        title="Flood Detection",
        abstract="A CNN approach for flood detection.",
        year=2024,
        source_title="Remote Sensing",
    )

    result = summarize_paper(candidate, query="flood detection", client=client)

    assert result.used_llm is False
    assert "Flood Detection" in result.one_sentence_summary
    assert "verify" in result.limitations_or_caution.lower()


def test_paper_summary_mock_is_deterministic():
    client = LLMClient(_settings(enabled=True))
    candidate = {
        "title": "Flood Detection",
        "abstract": "A CNN approach for flood detection.",
        "year": 2024,
        "source_title": "Remote Sensing",
    }

    result = summarize_paper(candidate, query="flood detection", client=client)

    assert result.used_llm is True
    assert result.provider == "mock"
    assert result.method_or_data == "Available abstract metadata"
    assert result.prompt_version == "v1"
    assert result.query_fingerprint == query_fingerprint("flood detection")
    assert result.candidate_fingerprint == candidate_fingerprint(candidate)


def test_summary_payload_omits_raw_internal_and_sensitive_metadata():
    client = _CapturingStructuredClient(
        {
            "one_sentence_summary": "A safe paper summary.",
            "contribution": "Uses safe candidate fields.",
            "method_or_data": "Metadata-only summary.",
            "relevance_to_query": "Relevant to flood detection.",
            "limitations_or_caution": "Verify before citation.",
        }
    )
    candidate = SimpleNamespace(
        title="Flood Detection",
        abstract="A CNN approach for flood detection.",
        year=2024,
        source_title="Remote Sensing",
        doi="10.1000/flood",
        url="https://example.com/flood",
        authors_json=["Author A", {"name": "Unsafe Object"}],
        citation_count=12,
        provider="semantic_scholar",
        raw_metadata_json={
            "api_key": "sk-secret-value",
            "secret": "top-secret",
            "token": "private-token",
            "_llm_diagnostics": {"raw_prompt": "do-not-send"},
            "_ranking": {"debug": "do-not-send"},
            "_trust": {"debug": "do-not-send"},
            "_source_match": {"debug": "do-not-send"},
            "raw_prompt": "do-not-send",
            "raw_model_response": "do-not-send",
            "nested": {"raw_response": "do-not-send"},
        },
    )

    result = summarize_paper(candidate, query="flood detection", client=client)

    assert result.used_llm is True
    assert client.payload == {
        "query": "flood detection",
        "title": "Flood Detection",
        "abstract": "A CNN approach for flood detection.",
        "source_title": "Remote Sensing",
        "doi": "10.1000/flood",
        "url": "https://example.com/flood",
        "provider": "semantic_scholar",
        "year": 2024,
        "citation_count": 12,
        "authors": ["Author A"],
    }

    payload_text = json.dumps(client.payload, sort_keys=True)
    for forbidden in (
        "sk-secret-value",
        "top-secret",
        "private-token",
        "_llm",
        "_ranking",
        "_trust",
        "_source_match",
        "raw_prompt",
        "raw_model_response",
        "raw_response",
        "raw_metadata_json",
        "internal",
        "debug",
        "diagnostics",
        "do-not-send",
    ):
        assert forbidden not in payload_text


def test_summary_parses_fake_real_client_response():
    client = _FakeStructuredClient(
        {
            "one_sentence_summary": "A flood mapping paper.",
            "contribution": "Shows a satellite method.",
            "method_or_data": "Remote sensing imagery.",
            "relevance_to_query": "Directly relevant.",
            "limitations_or_caution": "Metadata-only summary.",
        }
    )
    candidate = {"title": "Flood Detection", "abstract": "A CNN approach.", "source_title": "Remote Sensing"}

    result = summarize_paper(candidate, query="flood detection", client=client)

    assert result.used_llm is True
    assert result.provider == "openai_compatible"
    assert result.one_sentence_summary == "A flood mapping paper."
    assert result.method_or_data == "Remote sensing imagery."


def test_summary_missing_fields_uses_defaults():
    client = _FakeStructuredClient({"one_sentence_summary": "A flood mapping paper."})
    candidate = {"title": "Flood Detection", "abstract": "A CNN approach.", "source_title": "Remote Sensing"}

    result = summarize_paper(candidate, query="flood detection", client=client)

    assert result.used_llm is True
    assert result.one_sentence_summary == "A flood mapping paper."
    assert result.contribution == "A CNN approach."


def test_summary_invalid_contract_returns_safe_fallback():
    candidate = {"title": "Flood Detection", "abstract": "A CNN approach.", "source_title": "Remote Sensing"}

    result = summarize_paper(candidate, query="flood detection", client=_FakeStructuredClient({"unexpected": "value"}))

    assert result.used_llm is False
    assert result.fallback_reason == "invalid_json"


def test_summary_failing_client_returns_fallback():
    candidate = {"title": "Flood Detection", "abstract": "A CNN approach.", "source_title": "Remote Sensing"}

    result = summarize_paper(candidate, query="flood detection", client=_FailingClient())

    assert result.used_llm is False
    assert "Flood Detection" in result.one_sentence_summary
    assert result.attempted is True
    assert result.fallback_reason == "provider_unavailable"


def test_summary_invalid_json_shape_returns_safe_fallback():
    candidate = {"title": "Flood Detection", "abstract": "A CNN approach.", "source_title": "Remote Sensing"}

    result = summarize_paper(candidate, query="flood detection", client=_FakeStructuredClient("not-a-dict"))

    assert result.used_llm is False
    assert result.fallback_reason == "invalid_json"
