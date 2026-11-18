from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.llm_fixtures import load_llm_eval_cases
from app.services.llm_client import LLMClient
from app.tools import llm_eval


def _settings(
    *,
    enabled: bool = True,
    provider: str = "mock",
    model: str = "mock-test-model",
    api_key: str | None = None,
):
    return SimpleNamespace(
        llm_enabled=enabled,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_query_rewrite_enabled=True,
        llm_rerank_enabled=True,
        llm_summary_enabled=True,
        llm_cache_enabled=True,
        llm_prompt_version="v1",
        llm_usage_estimation_enabled=True,
        llm_max_calls_per_search=20,
        llm_max_estimated_tokens_per_search=20000,
        llm_max_input_chars_per_call=12000,
    )


def test_eval_fixtures_load_static_cases():
    cases = load_llm_eval_cases()

    assert len(cases) >= 5
    assert {case["case_id"] for case in cases} >= {
        "early_cancer_detection",
        "flood_detection_remote_sensing",
        "llm_hallucination_detection",
        "clinical_decision_support_ai",
        "systematic_review_automation",
    }
    for case in cases:
        assert case["topic"]
        assert case["raw_query"]
        assert case["expected_terms"]
        assert len(case["candidates"]) >= 3
        assert case["expected_top_candidate_title"]


def test_eval_mock_json_command_outputs_expected_report_keys():
    settings = _settings(provider="openai_compatible", model="real-test-model", api_key="sk-secret-value")
    stdout = StringIO()

    code = llm_eval.main(["--mock-only", "--json"], settings_obj=settings, stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert set(data) >= {
        "created_at",
        "provider",
        "model",
        "prompt_version",
        "cases",
        "summary",
        "retrieval_provider_called",
        "database_access_required",
        "cache",
    }
    assert data["provider"] == "mock"
    assert data["retrieval_provider_called"] is False
    assert data["database_access_required"] is False
    assert data["summary"]["total_cases"] >= 5
    assert data["summary"]["passed_cases"] == data["summary"]["total_cases"]
    assert "sk-secret-value" not in stdout.getvalue()
    assert "raw prompt" not in stdout.getvalue().lower()
    assert "traceback" not in stdout.getvalue().lower()


def test_eval_metrics_are_deterministic_for_mock_case():
    settings = _settings()
    client = LLMClient(settings)

    report = llm_eval.run_eval(
        cases=[case for case in load_llm_eval_cases() if case["case_id"] == "flood_detection_remote_sensing"],
        client=client,
        settings_obj=settings,
    )
    case = report["cases"][0]

    assert report["summary"] == {"total_cases": 1, "passed_cases": 1, "failed_cases": 0}
    assert case["query_rewrite"]["contains_expected_terms_count"] == 5
    assert case["query_rewrite"]["expanded_queries_count"] == 3
    assert case["rerank"]["expected_top_candidate_rank"] == 1
    assert case["rerank"]["relevance_scores_bounded"] is True
    assert case["rerank"]["reasons_non_empty"] is True
    assert case["summary"]["summary_fields_present"] is True
    assert case["summary"]["relevance_to_query_non_empty"] is True
    assert case["summary"]["limitations_or_caution_present"] is True
    assert case["operational"]["estimated_usage"]["estimated_call_count"] == 3


def test_eval_single_case_selection_works():
    stdout = StringIO()

    code = llm_eval.main(
        ["--mock-only", "--json", "--case", "early_cancer_detection"],
        settings_obj=_settings(),
        stdout=stdout,
    )

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["summary"]["total_cases"] == 1
    assert data["cases"][0]["case_id"] == "early_cancer_detection"


def test_eval_output_writes_json_report():
    report_path = Path(__file__).with_name("_llm_eval_report_test.json")
    if report_path.exists():
        report_path.unlink()
    stdout = StringIO()

    try:
        code = llm_eval.main(
            ["--mock-only", "--case", "clinical_decision_support_ai", "--output", str(report_path)],
            settings_obj=_settings(),
            stdout=stdout,
        )

        assert code == 0
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["summary"]["total_cases"] == 1
        assert data["cases"][0]["case_id"] == "clinical_decision_support_ai"
        assert "LLM eval report" in stdout.getvalue()
    finally:
        if report_path.exists():
            report_path.unlink()


def test_eval_fallback_case_is_represented_safely_when_llm_disabled():
    settings = _settings(enabled=False, provider="mock", api_key="sk-secret-value")
    client = LLMClient(settings)
    stdout = StringIO()

    code = llm_eval.main(["--json", "--case", "systematic_review_automation"], client=client, settings_obj=settings, stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    case = data["cases"][0]
    assert case["fallback_reasons"] == ["llm_disabled"]
    assert case["query_rewrite"]["used_llm"] is False
    assert case["rerank"]["used_llm"] is False
    assert case["summary"]["used_llm"] is False
    assert "sk-secret-value" not in stdout.getvalue()
    assert "traceback" not in stdout.getvalue().lower()


def test_eval_invalid_case_returns_safe_error_without_stack_trace():
    stdout = StringIO()

    code = llm_eval.main(["--json", "--case", "missing_case"], settings_obj=_settings(), stdout=stdout)

    assert code == 2
    data = json.loads(stdout.getvalue())
    assert data["error"] == "invalid_case"
    assert "missing_case" in data["detail"]
    assert "traceback" not in stdout.getvalue().lower()


def test_generated_eval_reports_are_ignored_by_default():
    gitignore = Path(__file__).resolve().parents[2] / ".gitignore"

    assert "backend/reports/" in gitignore.read_text(encoding="utf-8")
