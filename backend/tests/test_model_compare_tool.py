from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.model_comparison_fixtures import load_model_comparison_cases
from app.tools import model_compare


def _settings(
    *,
    llm_provider="mock",
    llm_model="mock-test-model",
    api_key="sk-secret-value",
    local_enabled=False,
    local_provider="mock",
    local_model_path="D:/private/models/secret-reranker",
):
    return SimpleNamespace(
        llm_enabled=True,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=api_key,
        llm_base_url="https://llm.example.test/v1",
        llm_timeout_seconds=4.0,
        llm_temperature=0.0,
        llm_prompt_version="v1",
        llm_max_input_chars_per_call=12000,
        local_reranker_enabled=local_enabled,
        local_reranker_provider=local_provider,
        local_reranker_model_path=local_model_path,
        local_reranker_model_version="local-test-v1",
        local_reranker_top_k=20,
    )


def test_model_comparison_fixtures_load_static_cases():
    cases = load_model_comparison_cases()

    assert len(cases) == 5
    assert {case["case_id"] for case in cases} == {
        "early_cancer_detection",
        "flood_detection_remote_sensing",
        "llm_hallucination_detection",
        "clinical_decision_support_ai",
        "systematic_review_automation",
    }
    for case in cases:
        assert case["query"]
        assert case["expected_best_candidate_id"]
        assert len(case["candidates"]) >= 3
        for candidate in case["candidates"]:
            assert "candidate_id" in candidate
            assert "baseline_score" in candidate
            assert "label" in candidate


def test_model_compare_mock_json_outputs_expected_top_level_keys():
    stdout = StringIO()

    code = model_compare.main(["--mock-only", "--json"], settings_obj=_settings(), stdout=stdout)

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert set(data) >= {
        "created_at",
        "local_reranker",
        "llm_rerank",
        "methods",
        "cases",
        "summary",
        "retrieval_provider_called",
        "database_access_required",
        "training_required",
    }
    assert data["methods"] == ["baseline", "local_reranker", "llm_rerank"]
    assert data["retrieval_provider_called"] is False
    assert data["database_access_required"] is False
    assert data["training_required"] is False
    assert data["local_reranker"]["model_provider"] == "mock"
    assert data["summary"]["total_cases"] == 5


def test_model_compare_metrics_are_deterministic():
    cases = [case for case in load_model_comparison_cases() if case["case_id"] == "flood_detection_remote_sensing"]

    first = model_compare.run_comparison(cases=cases, settings_obj=_settings(local_enabled=True))
    second = model_compare.run_comparison(cases=cases, settings_obj=_settings(local_enabled=True))

    assert first["cases"][0]["metrics"] == second["cases"][0]["metrics"]
    assert first["cases"][0]["rankings"] == second["cases"][0]["rankings"]
    assert first["cases"][0]["metrics"]["baseline"]["top1_candidate_id"] == 201
    assert first["cases"][0]["metrics"]["baseline"]["expected_best_rank"] == 1
    assert first["summary"]["average_precision_at_3"]["baseline"] == 0.6667


def test_precision_and_ndcg_metrics_on_simple_input():
    labels = {"a": 2, "b": 0, "c": 1}

    assert model_compare.precision_at_k(["a", "b", "c"], labels, 3) == 0.6667
    assert model_compare.precision_at_k(["b"], labels, 3) == 0.0
    assert model_compare.ndcg_at_k(["a", "c", "b"], labels, 3) == 1.0
    assert model_compare.ndcg_at_k(["b", "c", "a"], labels, 3) < 1.0


def test_model_compare_single_case_selection_works():
    stdout = StringIO()

    code = model_compare.main(
        ["--mock-only", "--json", "--case", "early_cancer_detection"],
        settings_obj=_settings(),
        stdout=stdout,
    )

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["summary"]["total_cases"] == 1
    assert data["cases"][0]["case_id"] == "early_cancer_detection"


def test_model_compare_output_omits_secrets_raw_prompts_paths_and_traces():
    stdout = StringIO()

    code = model_compare.main(["--mock-only", "--json"], settings_obj=_settings(), stdout=stdout)

    assert code == 0
    text = stdout.getvalue().lower()
    assert "sk-secret-value" not in text
    assert "raw_prompt" not in text
    assert "raw model" not in text
    assert "traceback" not in text
    assert "secret-reranker" not in text
    assert "d:/private" not in text


def test_model_compare_report_output_file_can_be_written():
    report_path = Path(__file__).with_name("_model_comparison_report_test.json")
    if report_path.exists():
        report_path.unlink()
    stdout = StringIO()
    try:
        code = model_compare.main(
            ["--mock-only", "--case", "clinical_decision_support_ai", "--output", str(report_path)],
            settings_obj=_settings(),
            stdout=stdout,
        )

        assert code == 0
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["summary"]["total_cases"] == 1
        assert data["cases"][0]["case_id"] == "clinical_decision_support_ai"
        assert "Model comparison report" in stdout.getvalue()
    finally:
        if report_path.exists():
            report_path.unlink()


def test_model_compare_does_not_call_provider_or_require_db(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval provider must not be called by model comparison")

    monkeypatch.setattr("app.providers.provider_factory.get_search_provider", _fail_if_called)
    stdout = StringIO()

    code = model_compare.main(
        ["--mock-only", "--json", "--case", "systematic_review_automation"],
        settings_obj=_settings(),
        stdout=stdout,
    )

    assert code == 0
    data = json.loads(stdout.getvalue())
    assert data["retrieval_provider_called"] is False
    assert data["database_access_required"] is False


def test_model_compare_invalid_case_returns_safe_error():
    stdout = StringIO()

    code = model_compare.main(["--json", "--case", "missing_case"], settings_obj=_settings(), stdout=stdout)

    assert code == 2
    data = json.loads(stdout.getvalue())
    assert data["error"] == "invalid_case"
    assert "missing_case" in data["detail"]
    assert "traceback" not in stdout.getvalue().lower()
