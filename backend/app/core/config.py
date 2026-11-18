from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LOCAL_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class Settings(BaseSettings):
    app_name: str = "Literature Review System"
    debug: bool = True
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    search_provider_mode: str = "mock"
    frontend_origin: str | None = None
    additional_cors_origins: str = ""
    llm_enabled: bool = False
    llm_provider: str = "mock"
    llm_model: str = "mock-literature-review-v1"
    llm_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    elsevier_api_key: str | None = None
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    llm_timeout_seconds: float = 8.0
    llm_temperature: float = 0.0
    llm_query_rewrite_enabled: bool = False
    llm_rerank_enabled: bool = False
    llm_summary_enabled: bool = False
    llm_cache_enabled: bool = True
    llm_prompt_version: str = "v1"
    llm_max_candidates_for_rerank: int = 10
    llm_max_candidates_for_summary: int = 3
    llm_max_summaries_per_search: int = 3
    llm_max_debate_candidates: int = 3
    llm_skip_summary_if_abstract_missing: bool = True
    llm_max_calls_per_search: int = 20
    llm_max_estimated_tokens_per_search: int = 20000
    llm_max_input_chars_per_call: int = 12000
    llm_usage_estimation_enabled: bool = True
    local_reranker_enabled: bool = False
    local_reranker_model_path: str = ""
    local_reranker_model_version: str = "v1"
    local_reranker_top_k: int = 20
    local_reranker_provider: str = "mock"
    local_analysis_enabled: bool = False
    reranker_enabled: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_max_candidates: int = 50
    reranker_batch_size: int = 16
    reranker_timeout_seconds: float = 20.0
    reranker_score_weight: float = 0.35

    database_url: str = "postgresql+psycopg2://postgres:1@localhost:5432/lit_review"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "dev", "development", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "prod", "production", "release"}:
                return False
        return value

    @field_validator("search_provider_mode", mode="before")
    @classmethod
    def parse_search_provider_mode(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("frontend_origin", mode="before")
    @classmethod
    def parse_frontend_origin(cls, value):
        if isinstance(value, str):
            normalized = value.strip().rstrip("/")
            return normalized or None
        return value

    @field_validator("additional_cors_origins", mode="before")
    @classmethod
    def parse_additional_cors_origins(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def parse_llm_provider(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("local_reranker_provider", mode="before")
    @classmethod
    def parse_local_reranker_provider(cls, value):
        if isinstance(value, str):
            return value.strip().lower() or "mock"
        return value

    @field_validator("local_reranker_model_path", mode="before")
    @classmethod
    def parse_local_reranker_model_path(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("local_reranker_model_version", mode="before")
    @classmethod
    def parse_local_reranker_model_version(cls, value):
        if isinstance(value, str):
            return value.strip() or "v1"
        return value

    @field_validator("reranker_model_name", mode="before")
    @classmethod
    def parse_reranker_model_name(cls, value):
        if isinstance(value, str):
            return value.strip() or "cross-encoder/ms-marco-MiniLM-L6-v2"
        return value

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def parse_llm_base_url(cls, value):
        if isinstance(value, str):
            return value.strip().rstrip("/")
        return value

    @field_validator("llm_prompt_version", mode="before")
    @classmethod
    def parse_llm_prompt_version(cls, value):
        if isinstance(value, str):
            return value.strip() or "v1"
        return value

    @field_validator("llm_timeout_seconds", mode="before")
    @classmethod
    def parse_llm_timeout_seconds(cls, value):
        parsed = float(value)
        return max(parsed, 0.1)

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def parse_llm_temperature(cls, value):
        parsed = float(value)
        return max(0.0, min(parsed, 2.0))

    @field_validator(
        "llm_max_candidates_for_rerank",
        "llm_max_candidates_for_summary",
        "llm_max_summaries_per_search",
        "llm_max_debate_candidates",
        "llm_max_calls_per_search",
        "llm_max_estimated_tokens_per_search",
        "llm_max_input_chars_per_call",
        "local_reranker_top_k",
        "reranker_max_candidates",
        "reranker_batch_size",
        mode="before",
    )
    @classmethod
    def parse_positive_candidate_limit(cls, value):
        parsed = int(value)
        return max(parsed, 0)

    @field_validator("reranker_timeout_seconds", mode="before")
    @classmethod
    def parse_reranker_timeout_seconds(cls, value):
        parsed = float(value)
        return max(parsed, 0.1)

    @field_validator("reranker_score_weight", mode="before")
    @classmethod
    def parse_reranker_score_weight(cls, value):
        parsed = float(value)
        return max(0.0, min(parsed, 1.0))

    def cors_origins(self) -> list[str]:
        origins = list(LOCAL_CORS_ORIGINS)

        if self.frontend_origin and self.frontend_origin not in origins:
            origins.append(self.frontend_origin)

        for raw_origin in self.additional_cors_origins.split(","):
            origin = raw_origin.strip().rstrip("/")
            if not origin:
                continue
            if origin == "*" and not self.debug:
                continue
            if origin not in origins:
                origins.append(origin)

        return origins

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
