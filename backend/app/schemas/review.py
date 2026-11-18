from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReviewOutlineSectionResponse(BaseModel):
    section_id: str
    title: str
    purpose: str
    supporting_candidate_ids: list[int] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)


class ReviewOutlinePayloadResponse(BaseModel):
    title: str
    sections: list[ReviewOutlineSectionResponse] = Field(default_factory=list)


class ReviewOutlineResponse(BaseModel):
    session_id: int
    research_query: str
    paper_count: int
    outline: ReviewOutlinePayloadResponse
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None


class ReviewThemeResponse(BaseModel):
    theme_id: str
    name: str
    rationale: str
    keywords: list[str] = Field(default_factory=list)
    supporting_candidate_ids: list[int] = Field(default_factory=list)


class ReviewThemesResponse(BaseModel):
    session_id: int
    research_query: str
    paper_count: int
    themes: list[ReviewThemeResponse] = Field(default_factory=list)
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None


class ReviewDraftSectionResponse(BaseModel):
    section_id: str
    title: str
    content: str
    supporting_candidate_ids: list[int] = Field(default_factory=list)
    citation_placeholders: list[str] = Field(default_factory=list)
    key_claims: list[str] = Field(default_factory=list)
    limitations_or_cautions: list[str] = Field(default_factory=list)


class ReviewDraftPayloadResponse(BaseModel):
    title: str
    sections: list[ReviewDraftSectionResponse] = Field(default_factory=list)


class ReviewDraftResponse(BaseModel):
    session_id: int
    research_query: str
    paper_count: int
    draft: ReviewDraftPayloadResponse
    markdown_preview: str | None = None
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None


class DraftGroundingCheckResponse(BaseModel):
    valid: bool
    unknown_citations: int
    unknown_supporting_candidate_ids: int
    sections_without_evidence: int
    total_sections: int
    total_supported_sections: int


class ReviewMarkdownExportResponse(BaseModel):
    session_id: int
    filename: str
    markdown: str
    paper_count: int
    grounding_check: DraftGroundingCheckResponse
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None


class DebateRoleFieldsResponse(BaseModel):
    role_name: str
    score: float
    findings: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class RelevanceReviewerResponse(DebateRoleFieldsResponse):
    relevance_score: float
    off_topic_sections: list[str] = Field(default_factory=list)
    query_alignment_comments: list[str] = Field(default_factory=list)


class EvidenceGroundingCriticResponse(DebateRoleFieldsResponse):
    grounding_score: float
    unsupported_claims: list[str] = Field(default_factory=list)
    weakly_supported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)


class MethodologyCriticResponse(DebateRoleFieldsResponse):
    methodology_score: float
    methodology_gaps: list[str] = Field(default_factory=list)
    overclaims: list[str] = Field(default_factory=list)
    missing_limitations: list[str] = Field(default_factory=list)
    suggested_methodology_revisions: list[str] = Field(default_factory=list)


class SourceReliabilityCriticResponse(DebateRoleFieldsResponse):
    source_quality_score: float
    source_concerns: list[str] = Field(default_factory=list)
    low_trust_evidence: list[str] = Field(default_factory=list)
    metadata_warnings: list[str] = Field(default_factory=list)


class DebateQualityScoresResponse(BaseModel):
    relevance: float
    grounding: float
    methodology: float
    source_reliability: float
    overall: float


class DebateReportResponse(BaseModel):
    overall_assessment: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_or_weak_claims: list[str] = Field(default_factory=list)
    source_reliability_concerns: list[str] = Field(default_factory=list)
    methodology_concerns: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    final_recommendation: Literal[
        "strong_draft",
        "acceptable_with_minor_revision",
        "revise_before_use",
        "insufficient_evidence",
    ]
    confidence_score: float
    action_items: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]
    quality_scores: DebateQualityScoresResponse
    debate_summary: str


class FinalArbiterResponse(DebateRoleFieldsResponse, DebateReportResponse):
    pass


class DebateAgentOutputsResponse(BaseModel):
    relevance_reviewer: RelevanceReviewerResponse
    evidence_grounding_critic: EvidenceGroundingCriticResponse
    methodology_critic: MethodologyCriticResponse
    source_reliability_critic: SourceReliabilityCriticResponse
    final_arbiter: FinalArbiterResponse


class ReviewDebateResponse(BaseModel):
    session_id: int
    research_query: str
    paper_count: int
    debate_report: DebateReportResponse
    agent_outputs: DebateAgentOutputsResponse
    grounding_check: DraftGroundingCheckResponse
    markdown_report: str | None = None
    debate_mode: Literal["structured_panel"]
    llm_used: bool
    fallback_reason: str | None = None
    prompt_version: str | None = None
    cached: bool = False
    generated_at: str | None = None
    updated_at: str | None = None
