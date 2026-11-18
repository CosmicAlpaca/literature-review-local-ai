from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings


_LLM_ENV_VARS = (
    "LLM_ENABLED",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_TEMPERATURE",
    "LLM_QUERY_REWRITE_ENABLED",
    "LLM_RERANK_ENABLED",
    "LLM_SUMMARY_ENABLED",
    "LLM_CACHE_ENABLED",
    "LLM_PROMPT_VERSION",
    "LLM_MAX_CANDIDATES_FOR_RERANK",
    "LLM_MAX_CANDIDATES_FOR_SUMMARY",
    "LLM_MAX_SUMMARIES_PER_SEARCH",
    "LLM_SKIP_SUMMARY_IF_ABSTRACT_MISSING",
    "LLM_MAX_CALLS_PER_SEARCH",
    "LLM_MAX_ESTIMATED_TOKENS_PER_SEARCH",
    "LLM_MAX_INPUT_CHARS_PER_CALL",
    "LLM_USAGE_ESTIMATION_ENABLED",
    "LOCAL_RERANKER_ENABLED",
    "LOCAL_RERANKER_MODEL_PATH",
    "LOCAL_RERANKER_MODEL_VERSION",
    "LOCAL_RERANKER_TOP_K",
    "LOCAL_RERANKER_PROVIDER",
    "RERANKER_ENABLED",
    "RERANKER_MODEL_NAME",
    "RERANKER_MAX_CANDIDATES",
    "RERANKER_BATCH_SIZE",
    "RERANKER_TIMEOUT_SECONDS",
    "RERANKER_SCORE_WEIGHT",
)


def test_llm_settings_defaults_are_disabled(monkeypatch):
    for env_var in _LLM_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    parsed = Settings(_env_file=None)

    assert parsed.llm_enabled is False
    assert parsed.llm_provider == "mock"
    assert parsed.llm_base_url == "https://api.openai.com/v1"
    assert parsed.llm_timeout_seconds == 8.0
    assert parsed.llm_temperature == 0.0
    assert parsed.llm_query_rewrite_enabled is False
    assert parsed.llm_rerank_enabled is False
    assert parsed.llm_summary_enabled is False
    assert parsed.llm_cache_enabled is True
    assert parsed.llm_prompt_version == "v1"
    assert parsed.llm_max_candidates_for_rerank == 10
    assert parsed.llm_max_candidates_for_summary == 3
    assert parsed.llm_max_summaries_per_search == 3
    assert parsed.llm_skip_summary_if_abstract_missing is True
    assert parsed.llm_max_calls_per_search == 20
    assert parsed.llm_max_estimated_tokens_per_search == 20000
    assert parsed.llm_max_input_chars_per_call == 12000
    assert parsed.llm_usage_estimation_enabled is True
    assert parsed.local_reranker_enabled is False
    assert parsed.local_reranker_model_path == ""
    assert parsed.local_reranker_model_version == "v1"
    assert parsed.local_reranker_top_k == 20
    assert parsed.local_reranker_provider == "mock"
    assert parsed.reranker_enabled is False
    assert parsed.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert parsed.reranker_max_candidates == 50
    assert parsed.reranker_batch_size == 16
    assert parsed.reranker_timeout_seconds == 20.0
    assert parsed.reranker_score_weight == 0.35


def test_llm_provider_and_candidate_limits_are_normalized():
    parsed = Settings(
        _env_file=None,
        llm_provider=" MOCK ",
        llm_base_url=" https://llm.example.test/v1/ ",
        llm_timeout_seconds="-3",
        llm_temperature="9",
        llm_max_candidates_for_rerank="-3",
        llm_max_candidates_for_summary="2",
        llm_max_summaries_per_search="-8",
        llm_max_calls_per_search="-2",
        llm_max_estimated_tokens_per_search="-10",
        llm_max_input_chars_per_call="-100",
        llm_prompt_version=" phase2-cache ",
        local_reranker_provider=" MOCK ",
        local_reranker_model_path=" C:/models/reranker ",
        local_reranker_model_version=" local-v2 ",
        local_reranker_top_k="-5",
        reranker_model_name=" cross-encoder/test-model ",
        reranker_max_candidates="-10",
        reranker_batch_size="0",
        reranker_timeout_seconds="-2",
        reranker_score_weight="9",
    )

    assert parsed.llm_provider == "mock"
    assert parsed.llm_base_url == "https://llm.example.test/v1"
    assert parsed.llm_timeout_seconds == 0.1
    assert parsed.llm_temperature == 2.0
    assert parsed.llm_max_candidates_for_rerank == 0
    assert parsed.llm_max_candidates_for_summary == 2
    assert parsed.llm_max_summaries_per_search == 0
    assert parsed.llm_max_calls_per_search == 0
    assert parsed.llm_max_estimated_tokens_per_search == 0
    assert parsed.llm_max_input_chars_per_call == 0
    assert parsed.llm_prompt_version == "phase2-cache"
    assert parsed.local_reranker_provider == "mock"
    assert parsed.local_reranker_model_path == "C:/models/reranker"
    assert parsed.local_reranker_model_version == "local-v2"
    assert parsed.local_reranker_top_k == 0
    assert parsed.reranker_model_name == "cross-encoder/test-model"
    assert parsed.reranker_max_candidates == 0
    assert parsed.reranker_batch_size == 0
    assert parsed.reranker_timeout_seconds == 0.1
    assert parsed.reranker_score_weight == 1.0


def test_cors_origins_include_local_and_frontend_origin():
    parsed = Settings(
        _env_file=None,
        debug=False,
        frontend_origin=" https://frontend.example.com/ ",
        additional_cors_origins="https://preview.example.com, *",
    )

    origins = parsed.cors_origins()

    assert "http://localhost:3000" in origins
    assert "http://127.0.0.1:3000" in origins
    assert "https://frontend.example.com" in origins
    assert "https://preview.example.com" in origins
    assert "*" not in origins


def test_cors_wildcard_allowed_only_in_debug_mode():
    parsed = Settings(
        _env_file=None,
        debug=True,
        additional_cors_origins="*",
    )

    assert "*" in parsed.cors_origins()
