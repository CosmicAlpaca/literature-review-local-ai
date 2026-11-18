from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    topic: str
    raw_query: str
    filters_json: dict | None = None


class SessionResponse(BaseModel):
    id: int
    topic: str
    raw_query: str
    status: str
    created_at: datetime


class LLMSummaryResponse(BaseModel):
    one_sentence_summary: str | None = None
    contribution: str | None = None
    method_or_data: str | None = None
    relevance_to_query: str | None = None
    limitations_or_caution: str | None = None


class CandidateResponse(BaseModel):
    id: int
    title: str
    abstract: str | None = None
    abstract_present: bool = False
    abstract_length: int = 0
    year: int | None = None
    source_title: str | None = None
    doi: str | None = None
    url: str | None = None
    citation_count: int | None = None
    provider: str
    scopus_matched: bool
    source_match_reason: str | None = None
    source_match_confidence: float | None = None
    matched_source_title: str | None = None
    matched_source_issn: str | None = None
    matched_source_eissn: str | None = None
    final_score: float | None = None
    reranker_status: Literal["applied", "unavailable", "disabled"] = "disabled"
    reranker_relative_score: float | None = None
    created_at: datetime
    review_status: Literal["shortlisted", "dismissed"] | None = None
    llm_relevance_score: float | None = None
    llm_relevance_reason: str | None = None
    llm_summary: LLMSummaryResponse | None = None


class CandidateSummaryResponse(BaseModel):
    session_id: int
    candidate_id: int
    summary: LLMSummaryResponse
    cached: bool
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None


class ReviewDecisionRequest(BaseModel):
    status: Literal["shortlisted", "dismissed"]


class SearchQueryContextResponse(BaseModel):
    status: Literal["ok", "partial", "empty", "running"] = "ok"
    original_query: str | None = None
    optimized_query: str | None = None
    effective_query: str | None = None
    steps: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    query_rewrite_notes: str | None = None
    llm_enabled: bool = False
    llm_requested: bool = False
    use_llm_requested: bool = False
    llm_available: bool = False
    llm_used: bool = False
    llm_steps: list[str] = Field(default_factory=list)
    llm_query_rewrite_used: bool = False
    llm_disabled_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    applied_filters: dict | None = None
    empty_result_retry_used: bool | None = None
    retry_queries: list[str] | None = None
    final_search_query: str | None = None


class SearchResponse(BaseModel):
    session_id: int | None = None
    candidates: list[CandidateResponse]
    top5: list[CandidateResponse]
    selected_papers: list[CandidateResponse] = Field(default_factory=list)
    query_context: SearchQueryContextResponse = Field(default_factory=SearchQueryContextResponse)


class SearchQueryContextStoredResponse(BaseModel):
    session_id: int
    exists: bool
    query_context: dict = Field(default_factory=dict)
    updated_at: datetime | None = None
