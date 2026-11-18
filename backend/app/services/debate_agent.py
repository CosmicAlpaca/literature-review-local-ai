from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from app.services.evidence_pack import EvidencePack, EvidencePaper
from app.services.input_guardrails import sanitize_public_text
from app.services.llm_client import (
    LLMClient,
    LLMUnavailableError,
    current_prompt_version,
    fallback_reason_from_exception,
    get_llm_client,
    safe_fallback_reason,
    utc_timestamp,
)
from app.services.llm_prompts import REVIEW_DEBATE_TASK
from app.services.review_writer import (
    DraftGroundingCheck,
    ReviewDraft,
    check_draft_grounding,
    fallback_review_draft,
    generate_review_draft,
)


FINAL_RECOMMENDATIONS = {
    "strong_draft",
    "acceptable_with_minor_revision",
    "revise_before_use",
    "insufficient_evidence",
}
RISK_LEVELS = {"low", "medium", "high"}
DEBATE_MODE = "structured_panel"

_FORBIDDEN_PUBLIC_MARKERS = (
    "raw_prompt",
    "raw prompt",
    "raw_model_response",
    "raw model response",
    "api_key",
    "api key",
    "secret",
    "system prompt",
    "developer message",
    "traceback",
    "stack trace",
    "_llm",
    "_ranking",
    "_trust",
    "sk-",
)


@dataclass(frozen=True)
class RelevanceReviewerOutput:
    relevance_score: float
    off_topic_sections: list[str]
    query_alignment_comments: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            _role_fields(
                role_name="Relevance Reviewer",
                score=self.relevance_score,
                findings=self.query_alignment_comments,
                concerns=self.off_topic_sections,
                suggestions=["Revise any section that does not directly answer the research query."],
            )
        )
        return data


@dataclass(frozen=True)
class EvidenceGroundingCriticOutput:
    grounding_score: float
    unsupported_claims: list[str]
    weakly_supported_claims: list[str]
    citation_issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            _role_fields(
                role_name="Evidence Grounding Critic",
                score=self.grounding_score,
                findings=self.citation_issues,
                concerns=self.unsupported_claims + self.weakly_supported_claims,
                suggestions=["Verify weak claims against the shortlisted papers before using the draft."],
            )
        )
        return data


@dataclass(frozen=True)
class MethodologyCriticOutput:
    methodology_score: float
    methodology_gaps: list[str]
    overclaims: list[str]
    missing_limitations: list[str]
    suggested_methodology_revisions: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            _role_fields(
                role_name="Methodology Critic",
                score=self.methodology_score,
                findings=self.missing_limitations,
                concerns=self.methodology_gaps + self.overclaims,
                suggestions=self.suggested_methodology_revisions,
            )
        )
        return data


@dataclass(frozen=True)
class SourceReliabilityCriticOutput:
    source_quality_score: float
    source_concerns: list[str]
    low_trust_evidence: list[str]
    metadata_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            _role_fields(
                role_name="Source Reliability Critic",
                score=self.source_quality_score,
                findings=self.metadata_warnings,
                concerns=self.source_concerns + self.low_trust_evidence,
                suggestions=["Verify source metadata and prioritize stronger evidence where reliability is unclear."],
            )
        )
        return data


@dataclass(frozen=True)
class DebateReport:
    overall_assessment: str
    strengths: list[str]
    weaknesses: list[str]
    missing_evidence: list[str]
    unsupported_or_weak_claims: list[str]
    source_reliability_concerns: list[str]
    methodology_concerns: list[str]
    suggested_revisions: list[str]
    final_recommendation: str
    confidence_score: float
    action_items: list[str]
    risk_level: str
    quality_scores: dict[str, float]
    debate_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebateAgentOutputs:
    relevance_reviewer: RelevanceReviewerOutput
    evidence_grounding_critic: EvidenceGroundingCriticOutput
    methodology_critic: MethodologyCriticOutput
    source_reliability_critic: SourceReliabilityCriticOutput
    final_arbiter: DebateReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance_reviewer": self.relevance_reviewer.to_dict(),
            "evidence_grounding_critic": self.evidence_grounding_critic.to_dict(),
            "methodology_critic": self.methodology_critic.to_dict(),
            "source_reliability_critic": self.source_reliability_critic.to_dict(),
            "final_arbiter": {
                **self.final_arbiter.to_dict(),
                **_role_fields(
                    role_name="Final Arbiter",
                    score=self.final_arbiter.quality_scores.get(
                        "overall",
                        self.final_arbiter.confidence_score,
                    ),
                    findings=self.final_arbiter.strengths,
                    concerns=self.final_arbiter.weaknesses
                    + self.final_arbiter.unsupported_or_weak_claims
                    + self.final_arbiter.source_reliability_concerns
                    + self.final_arbiter.methodology_concerns,
                    suggestions=self.final_arbiter.suggested_revisions
                    + self.final_arbiter.action_items,
                ),
            },
        }


@dataclass(frozen=True)
class DebateAgentResult:
    debate_report: DebateReport
    agent_outputs: DebateAgentOutputs
    grounding_check: DraftGroundingCheck
    markdown_report: str
    debate_mode: str = DEBATE_MODE
    provider: str = "fallback"
    model: str = "none"
    used_llm: bool = False
    attempted: bool = False
    fallback_reason: str | None = None
    prompt_version: str | None = None
    generated_at: str | None = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_call_count: int = 0
    input_truncated: bool = False

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["debate_report"] = self.debate_report.to_dict()
        data["agent_outputs"] = self.agent_outputs.to_dict()
        data["grounding_check"] = self.grounding_check.to_dict()
        return data


def _role_fields(
    *,
    role_name: str,
    score: float,
    findings: list[str],
    concerns: list[str],
    suggestions: list[str],
) -> dict[str, Any]:
    return {
        "role_name": role_name,
        "score": _clamp_score(score, 0.0),
        "findings": _dedupe_nonempty(findings, limit=8),
        "concerns": _dedupe_nonempty(concerns, limit=8),
        "suggestions": _dedupe_nonempty(suggestions, limit=8),
    }


def _clamp_score(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return round(max(0.0, min(float(value), 1.0)), 4)
    return default


def _dedupe_nonempty(values: list[str], *, limit: int = 8) -> list[str]:
    items: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            text = sanitize_public_text(value.strip())
            if text not in items:
                items.append(text)
    return items[:limit]


def _quality_scores(
    *,
    relevance: float,
    grounding: float,
    methodology: float,
    source_reliability: float,
) -> dict[str, float]:
    scores = {
        "relevance": _clamp_score(relevance, 0.0),
        "grounding": _clamp_score(grounding, 0.0),
        "methodology": _clamp_score(methodology, 0.0),
        "source_reliability": _clamp_score(source_reliability, 0.0),
    }
    scores["overall"] = _clamp_score(sum(scores.values()) / len(scores), 0.0)
    return scores


def _safe_quality_scores(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return fallback
    scores: dict[str, float] = {}
    for key in ("relevance", "grounding", "methodology", "source_reliability", "overall"):
        scores[key] = _clamp_score(value.get(key), fallback.get(key, 0.0))
    return scores


def _safe_risk_level(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        risk_level = value.strip().lower()
        if risk_level in RISK_LEVELS:
            return risk_level
    return fallback


def _derive_risk_level(
    quality_scores: dict[str, float],
    *,
    recommendation: str,
    grounding_check: DraftGroundingCheck,
) -> str:
    overall = quality_scores.get("overall", 0.0)
    if (
        recommendation == "insufficient_evidence"
        or not grounding_check.valid
        or overall < 0.6
    ):
        return "high"
    if recommendation == "strong_draft" and overall >= 0.8:
        return "low"
    return "medium"


def _candidate_label(paper: EvidencePaper) -> str:
    year = f", {paper.year}" if paper.year else ""
    source = f", {sanitize_public_text(paper.source_title)}" if paper.source_title else ""
    return f"C{paper.candidate_id}: {sanitize_public_text(paper.title)}{year}{source}"


def _contains_forbidden_public_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FORBIDDEN_PUBLIC_MARKERS)


def _sanitize_candidate_references(text: str, allowed_ids: set[int]) -> str:
    def _replace_bracket(match: re.Match[str]) -> str:
        candidate_id = int(match.group(1))
        return match.group(0) if candidate_id in allowed_ids else ""

    def _replace_named(match: re.Match[str]) -> str:
        candidate_id = int(match.group(2))
        return match.group(0) if candidate_id in allowed_ids else match.group(1).rstrip()

    cleaned = re.sub(r"\[C(\d+)\]", _replace_bracket, text)
    cleaned = re.sub(
        r"\b((?:candidate|paper)(?:[_\s-]?id)?\s*#?\s*)(\d+)\b",
        _replace_named,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bC(\d+)\b", _replace_bracket, cleaned)
    return " ".join(cleaned.split()).strip()


def _safe_public_string(value: Any, default: str, allowed_ids: set[int]) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    text = sanitize_public_text(value.strip())
    if _contains_forbidden_public_marker(text):
        return default
    text = _sanitize_candidate_references(text, allowed_ids)
    return text or default


def _safe_public_list(value: Any, allowed_ids: set[int], *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        text = sanitize_public_text(item.strip())
        if _contains_forbidden_public_marker(text):
            continue
        text = _sanitize_candidate_references(text, allowed_ids)
        if text and text not in items:
            items.append(text)
    return items[:limit]


def _safe_recommendation(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        recommendation = value.strip()
        if recommendation in FINAL_RECOMMENDATIONS:
            return recommendation
    return fallback


def _draft_to_payload(draft: ReviewDraft) -> dict[str, Any]:
    return draft.to_dict()


def _grounding_score(grounding_check: DraftGroundingCheck) -> float:
    if grounding_check.total_sections <= 0:
        return 0.0
    issue_count = (
        grounding_check.unknown_citations
        + grounding_check.unknown_supporting_candidate_ids
        + grounding_check.sections_without_evidence
    )
    return _clamp_score(1.0 - (issue_count / max(grounding_check.total_sections * 2, 1)), 0.5)


def _source_quality(evidence_pack: EvidencePack) -> tuple[float, list[str], list[str], list[str]]:
    if evidence_pack.paper_count == 0:
        return (
            0.0,
            ["No shortlisted evidence is available for source reliability review."],
            [],
            ["Source metadata could not be evaluated without evidence papers."],
        )

    concerns: list[str] = []
    low_reliability: list[str] = []
    warnings: list[str] = []
    scores: list[float] = []

    for paper in evidence_pack.papers:
        score = 0.45
        if paper.source_title:
            score += 0.15
        else:
            warnings.append(f"{_candidate_label(paper)} has no source title metadata.")
        if paper.year:
            score += 0.1
        else:
            warnings.append(f"{_candidate_label(paper)} has no publication year metadata.")
        if paper.citation_count is not None:
            if paper.citation_count >= 25:
                score += 0.15
            elif paper.citation_count > 0:
                score += 0.08
            else:
                concerns.append(f"{_candidate_label(paper)} has zero recorded citations.")
        else:
            warnings.append(f"{_candidate_label(paper)} has no citation count metadata.")
        if paper.scopus_matched:
            score += 0.15

        score = _clamp_score(score, 0.45)
        scores.append(score)
        if score < 0.55:
            low_reliability.append(f"{_candidate_label(paper)} has limited reliability metadata.")

    if not concerns:
        concerns.append("No major source reliability concerns were detected from safe metadata.")
    if not warnings:
        warnings.append("No source metadata warnings were detected from the shortlisted evidence.")

    average = round(sum(scores) / len(scores), 4) if scores else 0.0
    return average, concerns[:8], low_reliability[:8], warnings[:8]


def render_debate_markdown(
    *,
    evidence_pack: EvidencePack,
    report: DebateReport,
    grounding_check: DraftGroundingCheck,
) -> str:
    lines = [
        "# Debate Agent Report",
        "",
        f"Research query: {sanitize_public_text(evidence_pack.research_query)}",
        f"Paper count: {evidence_pack.paper_count}",
        "",
        "## Overall Assessment",
        "",
        report.overall_assessment,
        "",
        "## Debate Summary",
        "",
        report.debate_summary,
        "",
        "## Quality Scores",
        "",
    ]
    for key in ("relevance", "grounding", "methodology", "source_reliability", "overall"):
        lines.append(f"- {key.replace('_', ' ').title()}: {report.quality_scores.get(key, 0.0):.2f}")

    def _section(title: str, items: list[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- None reported.")

    _section("Strengths", report.strengths)
    _section("Weaknesses", report.weaknesses)
    _section("Missing Evidence", report.missing_evidence)
    _section("Unsupported Or Weak Claims", report.unsupported_or_weak_claims)
    _section("Source Reliability Concerns", report.source_reliability_concerns)
    _section("Methodology Concerns", report.methodology_concerns)
    _section("Suggested Revisions", report.suggested_revisions)
    _section("Action Items", report.action_items)

    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            f"- Recommendation: {report.final_recommendation}",
            f"- Risk level: {report.risk_level}",
            f"- Confidence score: {report.confidence_score:.2f}",
            "",
            "## Grounding Check",
            "",
            f"- Valid: {grounding_check.valid}",
            f"- Unknown citations: {grounding_check.unknown_citations}",
            f"- Unknown supporting candidate IDs: {grounding_check.unknown_supporting_candidate_ids}",
            f"- Sections without evidence: {grounding_check.sections_without_evidence}",
            f"- Total sections: {grounding_check.total_sections}",
            f"- Total supported sections: {grounding_check.total_supported_sections}",
        ]
    )
    return "\n".join(lines).strip()


def fallback_debate_report(
    evidence_pack: EvidencePack,
    draft: ReviewDraft,
    grounding_check: DraftGroundingCheck,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> DebateAgentResult:
    query = evidence_pack.research_query
    grounding = _grounding_score(grounding_check)
    source_score, source_concerns, low_reliability, metadata_warnings = _source_quality(evidence_pack)
    has_methods = any(
        any(term in f"{section.title} {section.content}".lower() for term in ("method", "data", "dataset"))
        for section in draft.sections
    )
    methodology_score = 0.72 if has_methods else 0.48
    relevance_score = 0.76 if evidence_pack.paper_count else 0.0

    relevance = RelevanceReviewerOutput(
        relevance_score=relevance_score,
        off_topic_sections=[],
        query_alignment_comments=[
            f"The draft is framed around the session research query: {query}.",
            f"The critique uses {evidence_pack.paper_count} shortlisted papers only.",
        ],
    )
    grounding_output = EvidenceGroundingCriticOutput(
        grounding_score=grounding,
        unsupported_claims=(
            ["Grounding check found citation or supporting-candidate issues."]
            if not grounding_check.valid
            else []
        ),
        weakly_supported_claims=[
            "Claims based on abstracts or metadata should be verified against full papers before final use."
        ],
        citation_issues=(
            [
                f"Unknown citations: {grounding_check.unknown_citations}.",
                f"Unknown supporting candidate IDs: {grounding_check.unknown_supporting_candidate_ids}.",
                f"Sections without evidence: {grounding_check.sections_without_evidence}.",
            ]
            if not grounding_check.valid
            else ["All draft citations and supporting candidate IDs map to shortlisted evidence."]
        ),
    )
    methodology = MethodologyCriticOutput(
        methodology_score=methodology_score,
        methodology_gaps=(
            []
            if has_methods
            else ["The draft does not yet compare study methods, datasets, or evaluation settings in detail."]
        ),
        overclaims=[
            "Avoid treating metadata-level summaries as verified full-text findings."
        ],
        missing_limitations=[
            "Add limitations tied to abstract-only evidence, source metadata quality, and study comparability."
        ],
        suggested_methodology_revisions=[
            "Separate claims about methods, datasets, and results by cited paper.",
            "Add a limitations paragraph before using the draft as a final manuscript section.",
        ],
    )
    source_reliability = SourceReliabilityCriticOutput(
        source_quality_score=source_score,
        source_concerns=source_concerns,
        low_trust_evidence=low_reliability,
        metadata_warnings=metadata_warnings,
    )

    quality_scores = _quality_scores(
        relevance=relevance_score,
        grounding=grounding,
        methodology=methodology_score,
        source_reliability=source_score,
    )
    average = quality_scores["overall"]
    if evidence_pack.paper_count == 0:
        recommendation = "insufficient_evidence"
    elif not grounding_check.valid or average < 0.6:
        recommendation = "revise_before_use"
    elif average >= 0.82:
        recommendation = "strong_draft"
    else:
        recommendation = "acceptable_with_minor_revision"
    risk_level = _derive_risk_level(
        quality_scores,
        recommendation=recommendation,
        grounding_check=grounding_check,
    )
    action_items = [
        "Verify the most important claims against full-text papers before final use.",
        "Revise weak or metadata-only claims so each one is tied to shortlisted evidence.",
        "Add methodology and limitation details where the available evidence supports them.",
    ]
    debate_summary = (
        f"Structured panel review of {evidence_pack.paper_count} shortlisted papers. "
        f"Overall quality is {average:.2f} with {risk_level} risk; recommendation: {recommendation}."
    )

    report = DebateReport(
        overall_assessment=(
            "This debate report is a conservative fallback critique generated from the "
            "shortlisted evidence pack, sanitized draft, and grounding check."
        ),
        strengths=[
            "The draft is constrained to shortlisted evidence.",
            "Citation placeholders and supporting candidate IDs were checked against the evidence pack.",
        ],
        weaknesses=[
            "The critique cannot verify full-text claims beyond available metadata, summaries, and abstracts.",
            "Methodology and source reliability judgments depend on the completeness of candidate metadata.",
        ],
        missing_evidence=[
            "Full-text verification is still needed for methods, datasets, results, and limitations."
        ],
        unsupported_or_weak_claims=grounding_output.unsupported_claims
        + grounding_output.weakly_supported_claims,
        source_reliability_concerns=source_concerns + low_reliability,
        methodology_concerns=methodology.methodology_gaps + methodology.overclaims,
        suggested_revisions=[
            "Add paper-by-paper methodology details where abstracts or summaries support them.",
            "Keep citation placeholders tied to shortlisted candidate IDs only.",
            "Verify source metadata and full-text claims before final submission.",
        ],
        final_recommendation=recommendation,
        confidence_score=_clamp_score(average, 0.65),
        action_items=action_items,
        risk_level=risk_level,
        quality_scores=quality_scores,
        debate_summary=debate_summary,
    )
    outputs = DebateAgentOutputs(
        relevance_reviewer=relevance,
        evidence_grounding_critic=grounding_output,
        methodology_critic=methodology,
        source_reliability_critic=source_reliability,
        final_arbiter=report,
    )
    safe_reason = safe_fallback_reason(fallback_reason) if fallback_reason else None
    return DebateAgentResult(
        debate_report=report,
        agent_outputs=outputs,
        grounding_check=grounding_check,
        markdown_report=render_debate_markdown(
            evidence_pack=evidence_pack,
            report=report,
            grounding_check=grounding_check,
        ),
        attempted=attempted,
        fallback_reason=safe_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
    )


def _parse_relevance(value: Any, fallback: RelevanceReviewerOutput, allowed_ids: set[int]) -> RelevanceReviewerOutput:
    data = value if isinstance(value, dict) else {}
    return RelevanceReviewerOutput(
        relevance_score=_clamp_score(data.get("relevance_score"), fallback.relevance_score),
        off_topic_sections=_safe_public_list(data.get("off_topic_sections"), allowed_ids)
        or fallback.off_topic_sections,
        query_alignment_comments=_safe_public_list(data.get("query_alignment_comments"), allowed_ids)
        or fallback.query_alignment_comments,
    )


def _parse_grounding(
    value: Any,
    fallback: EvidenceGroundingCriticOutput,
    allowed_ids: set[int],
) -> EvidenceGroundingCriticOutput:
    data = value if isinstance(value, dict) else {}
    return EvidenceGroundingCriticOutput(
        grounding_score=_clamp_score(data.get("grounding_score"), fallback.grounding_score),
        unsupported_claims=_safe_public_list(data.get("unsupported_claims"), allowed_ids)
        or fallback.unsupported_claims,
        weakly_supported_claims=_safe_public_list(data.get("weakly_supported_claims"), allowed_ids)
        or fallback.weakly_supported_claims,
        citation_issues=_safe_public_list(data.get("citation_issues"), allowed_ids)
        or fallback.citation_issues,
    )


def _parse_methodology(
    value: Any,
    fallback: MethodologyCriticOutput,
    allowed_ids: set[int],
) -> MethodologyCriticOutput:
    data = value if isinstance(value, dict) else {}
    return MethodologyCriticOutput(
        methodology_score=_clamp_score(data.get("methodology_score"), fallback.methodology_score),
        methodology_gaps=_safe_public_list(data.get("methodology_gaps"), allowed_ids)
        or fallback.methodology_gaps,
        overclaims=_safe_public_list(data.get("overclaims"), allowed_ids) or fallback.overclaims,
        missing_limitations=_safe_public_list(data.get("missing_limitations"), allowed_ids)
        or fallback.missing_limitations,
        suggested_methodology_revisions=_safe_public_list(
            data.get("suggested_methodology_revisions"),
            allowed_ids,
        )
        or fallback.suggested_methodology_revisions,
    )


def _parse_source_reliability(
    value: Any,
    fallback: SourceReliabilityCriticOutput,
    allowed_ids: set[int],
) -> SourceReliabilityCriticOutput:
    data = value if isinstance(value, dict) else {}
    return SourceReliabilityCriticOutput(
        source_quality_score=_clamp_score(data.get("source_quality_score"), fallback.source_quality_score),
        source_concerns=_safe_public_list(data.get("source_concerns"), allowed_ids)
        or fallback.source_concerns,
        low_trust_evidence=_safe_public_list(data.get("low_trust_evidence"), allowed_ids)
        or fallback.low_trust_evidence,
        metadata_warnings=_safe_public_list(data.get("metadata_warnings"), allowed_ids)
        or fallback.metadata_warnings,
    )


def _parse_report(value: Any, fallback: DebateReport, allowed_ids: set[int]) -> DebateReport:
    data = value if isinstance(value, dict) else {}
    recommendation = _safe_recommendation(
        data.get("final_recommendation"),
        fallback.final_recommendation,
    )
    quality_scores = _safe_quality_scores(data.get("quality_scores"), fallback.quality_scores)
    risk_level = _safe_risk_level(data.get("risk_level"), fallback.risk_level)
    return DebateReport(
        overall_assessment=_safe_public_string(
            data.get("overall_assessment"),
            fallback.overall_assessment,
            allowed_ids,
        ),
        strengths=_safe_public_list(data.get("strengths"), allowed_ids) or fallback.strengths,
        weaknesses=_safe_public_list(data.get("weaknesses"), allowed_ids) or fallback.weaknesses,
        missing_evidence=_safe_public_list(data.get("missing_evidence"), allowed_ids)
        or fallback.missing_evidence,
        unsupported_or_weak_claims=_safe_public_list(data.get("unsupported_or_weak_claims"), allowed_ids)
        or fallback.unsupported_or_weak_claims,
        source_reliability_concerns=_safe_public_list(
            data.get("source_reliability_concerns"),
            allowed_ids,
        )
        or fallback.source_reliability_concerns,
        methodology_concerns=_safe_public_list(data.get("methodology_concerns"), allowed_ids)
        or fallback.methodology_concerns,
        suggested_revisions=_safe_public_list(data.get("suggested_revisions"), allowed_ids)
        or fallback.suggested_revisions,
        final_recommendation=recommendation,
        confidence_score=_clamp_score(data.get("confidence_score"), fallback.confidence_score),
        action_items=_safe_public_list(data.get("action_items"), allowed_ids)
        or fallback.action_items,
        risk_level=risk_level,
        quality_scores=quality_scores,
        debate_summary=_safe_public_string(
            data.get("debate_summary"),
            fallback.debate_summary,
            allowed_ids,
        ),
    )


def _parse_debate_content(
    content: dict[str, Any],
    fallback: DebateAgentResult,
    allowed_ids: set[int],
) -> tuple[DebateReport, DebateAgentOutputs]:
    if not isinstance(content, dict):
        raise ValueError("debate output must be a JSON object")

    outputs_value = content.get("agent_outputs")
    if not isinstance(outputs_value, dict):
        raise ValueError("debate output must include agent_outputs")

    relevance = _parse_relevance(
        outputs_value.get("relevance_reviewer"),
        fallback.agent_outputs.relevance_reviewer,
        allowed_ids,
    )
    grounding = _parse_grounding(
        outputs_value.get("evidence_grounding_critic"),
        fallback.agent_outputs.evidence_grounding_critic,
        allowed_ids,
    )
    methodology = _parse_methodology(
        outputs_value.get("methodology_critic"),
        fallback.agent_outputs.methodology_critic,
        allowed_ids,
    )
    source_reliability = _parse_source_reliability(
        outputs_value.get("source_reliability_critic"),
        fallback.agent_outputs.source_reliability_critic,
        allowed_ids,
    )
    report_source = content.get("debate_report")
    if not isinstance(report_source, dict):
        report_source = outputs_value.get("final_arbiter")
    report = _parse_report(report_source, fallback.debate_report, allowed_ids)
    final_arbiter = _parse_report(outputs_value.get("final_arbiter"), report, allowed_ids)

    return report, DebateAgentOutputs(
        relevance_reviewer=relevance,
        evidence_grounding_critic=grounding,
        methodology_critic=methodology,
        source_reliability_critic=source_reliability,
        final_arbiter=final_arbiter,
    )


def _debate_payload(
    evidence_pack: EvidencePack,
    draft: ReviewDraft,
    grounding_check: DraftGroundingCheck,
) -> dict[str, Any]:
    return {
        "research_query": evidence_pack.research_query,
        "evidence_pack": evidence_pack.to_llm_payload(),
        "draft": _draft_to_payload(draft),
        "grounding_check": grounding_check.to_dict(),
        "guardrails": {
            "allowed_candidate_ids": [paper.candidate_id for paper in evidence_pack.papers],
            "use_shortlisted_evidence_only": True,
            "do_not_reference_unknown_candidate_ids": True,
        },
    }


def generate_debate_report(
    evidence_pack: EvidencePack,
    *,
    draft: ReviewDraft | None = None,
    grounding_check: DraftGroundingCheck | None = None,
    client: LLMClient | None = None,
) -> DebateAgentResult:
    client = client or get_llm_client()
    if draft is None:
        try:
            draft_result = generate_review_draft(evidence_pack, client=client)
        except TimeoutError as exc:
            draft_result = fallback_review_draft(
                evidence_pack,
                attempted=True,
                fallback_reason=fallback_reason_from_exception(exc),
            )
        draft = draft_result.draft
        if draft_result.fallback_reason:
            grounding_check = grounding_check or check_draft_grounding(evidence_pack, draft)
            return fallback_debate_report(
                evidence_pack,
                draft,
                grounding_check,
                attempted=True,
                fallback_reason=draft_result.fallback_reason,
            )
    if grounding_check is None:
        grounding_check = check_draft_grounding(evidence_pack, draft)

    fallback = fallback_debate_report(evidence_pack, draft, grounding_check)

    try:
        generation = client.generate_structured_text(
            task=REVIEW_DEBATE_TASK,
            payload=_debate_payload(evidence_pack, draft, grounding_check),
        )
    except (LLMUnavailableError, TimeoutError) as exc:
        return fallback_debate_report(
            evidence_pack,
            draft,
            grounding_check,
            attempted=True,
            fallback_reason=fallback_reason_from_exception(exc),
        )

    if generation.provider == "mock":
        mock_result = fallback_debate_report(evidence_pack, draft, grounding_check, attempted=True)
        return DebateAgentResult(
            debate_report=mock_result.debate_report,
            agent_outputs=mock_result.agent_outputs,
            grounding_check=grounding_check,
            markdown_report=render_debate_markdown(
                evidence_pack=evidence_pack,
                report=mock_result.debate_report,
                grounding_check=grounding_check,
            ),
            provider=generation.provider,
            model=generation.model,
            used_llm=True,
            attempted=True,
            prompt_version=generation.prompt_version or current_prompt_version(),
            generated_at=utc_timestamp(),
            estimated_input_tokens=generation.estimated_input_tokens,
            estimated_output_tokens=generation.estimated_output_tokens,
            estimated_total_tokens=generation.estimated_total_tokens,
            estimated_call_count=generation.estimated_call_count,
            input_truncated=generation.input_truncated,
        )

    try:
        report, outputs = _parse_debate_content(
            generation.content,
            fallback,
            {paper.candidate_id for paper in evidence_pack.papers},
        )
        return DebateAgentResult(
            debate_report=report,
            agent_outputs=outputs,
            grounding_check=grounding_check,
            markdown_report=render_debate_markdown(
                evidence_pack=evidence_pack,
                report=report,
                grounding_check=grounding_check,
            ),
            provider=generation.provider,
            model=generation.model,
            used_llm=True,
            attempted=True,
            prompt_version=generation.prompt_version or current_prompt_version(),
            generated_at=utc_timestamp(),
            estimated_input_tokens=generation.estimated_input_tokens,
            estimated_output_tokens=generation.estimated_output_tokens,
            estimated_total_tokens=generation.estimated_total_tokens,
            estimated_call_count=generation.estimated_call_count,
            input_truncated=generation.input_truncated,
        )
    except (AttributeError, TypeError, ValueError):
        return fallback_debate_report(
            evidence_pack,
            draft,
            grounding_check,
            attempted=True,
            fallback_reason="invalid_json",
        )
