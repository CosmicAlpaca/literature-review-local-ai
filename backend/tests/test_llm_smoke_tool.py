from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.llm_client import LLMClient
from app.tools import llm_smoke


def _settings(
    *,
    enabled: bool = True,
    provider: str = "mock",
    model: str = "mock-test-model",
    api_key: str | None = None,
    query_rewrite: bool = True,
    rerank: bool = True,
    summary: bool = True,
):
    return SimpleNamespace(
        llm_enabled=enabled,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_query_rewrite_enabled=query_rewrite,
        llm_rerank_enabled=rerank,
        llm_summary_enabled=summary,
        llm_cache_enabled=True,
        llm_prompt_version="v1",
        llm_usage_estimation_enabled=True,
        llm_max_calls_per_search=20,
        llm_max_estimated_tokens_per_search=20000,
        llm_max_input_chars_per_call=12000,
    )


def test_smoke_tool_json_mock_outputs_expected_keys():
    settings = _settings()
    client = LLMClient(settings)
    stdout = StringIO()

    code = llm_smoke.main(
        ["--json", "--query", "flood detection"],
        client=client,
        settings_obj=settings,
        stdout=stdout,
    )

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert set(data) >= {"status", "query_rewrite", "rerank", "summary", "estimated_usage", "sample"}
    assert data["status"]["provider"] == "mock"
    assert data["status"]["ready_for_ai_search"] is True
    assert data["query_rewrite"]["used_llm"] is True
    assert data["rerank"][0]["used_llm"] is True
    assert data["summary"]["used_llm"] is True
    assert data["sample"]["retrieval_provider_called"] is False
    assert data["estimated_usage"]["estimated_call_count"] >= 3


def test_smoke_tool_text_output_does_not_include_secrets():
    settings = _settings(
        provider="openai_compatible",
        model="real-test-model",
        api_key="sk-secret-value",
    )
    stdout = StringIO()

    code = llm_smoke.main(["--mock-only"], settings_obj=settings, stdout=stdout)

    output = stdout.getvalue()
    assert code == 0
    assert "sk-secret-value" not in output
    assert "llm_api_key" not in output.lower()
    assert "raw prompt" not in output.lower()
    assert "provider: mock" in output


def test_smoke_tool_fallback_case_is_safe_when_llm_disabled():
    settings = _settings(enabled=False)
    client = LLMClient(settings)
    stdout = StringIO()

    code = llm_smoke.main(["--json"], client=client, settings_obj=settings, stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["status"]["available"] is False
    assert data["status"]["reason"] == "llm_disabled"
    assert data["query_rewrite"]["used_llm"] is False
    assert data["query_rewrite"]["fallback_reason"] == "llm_disabled"
    assert data["rerank"][0]["used_llm"] is False
    assert data["rerank"][0]["fallback_reason"] == "llm_disabled"
    assert data["summary"]["used_llm"] is False
    assert data["summary"]["fallback_reason"] == "llm_disabled"
    assert "sk-" not in stdout.getvalue()


def test_smoke_tool_mock_only_overrides_real_provider_without_network():
    settings = _settings(
        provider="openai_compatible",
        model="real-test-model",
        api_key="sk-secret-value",
    )
    stdout = StringIO()

    code = llm_smoke.main(["--json", "--mock-only", "--max-candidates", "2"], settings_obj=settings, stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["status"]["provider"] == "mock"
    assert data["status"]["model"] == "real-test-model"
    assert len(data["rerank"]) == 2
    assert data["query_rewrite"]["used_llm"] is True
    assert "sk-secret-value" not in stdout.getvalue()


def test_smoke_tool_reports_step_disabled_without_external_call():
    settings = _settings(enabled=True, query_rewrite=False, rerank=False, summary=False)
    client = LLMClient(settings)
    stdout = StringIO()

    code = llm_smoke.main(["--json"], client=client, settings_obj=settings, stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["status"]["ready_for_ai_search"] is False
    assert data["status"]["reason"] == "step_disabled"
    assert data["query_rewrite"]["fallback_reason"] == "step_disabled"
    assert data["rerank"][0]["fallback_reason"] == "step_disabled"
    assert data["summary"]["fallback_reason"] == "step_disabled"
