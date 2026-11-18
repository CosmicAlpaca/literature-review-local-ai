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
    get_llm_client,
    safe_fallback_reason,
    utc_timestamp,
)
from app.services.llm_prompts import REVIEW_OUTLINE_FIELDS, REVIEW_OUTLINE_TASK
from app.services.llm_prompts import REVIEW_DRAFT_FIELDS, REVIEW_DRAFT_TASK


@dataclass(frozen=True)
class ReviewOutlineSection:
    section_id: str
    title: str
    purpose: str
    supporting_candidate_ids: list[int]
    key_points: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewOutline:
    title: str
    sections: list[ReviewOutlineSection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class ReviewOutlineResult:
    outline: ReviewOutline
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
        data["outline"] = self.outline.to_dict()
        return data


@dataclass(frozen=True)
class ReviewDraftSection:
    section_id: str
    title: str
    content: str
    supporting_candidate_ids: list[int]
    citation_placeholders: list[str]
    key_claims: list[str]
    limitations_or_cautions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewDraft:
    title: str
    sections: list[ReviewDraftSection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class ReviewDraftResult:
    draft: ReviewDraft
    markdown_preview: str
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
        data["draft"] = self.draft.to_dict()
        return data


@dataclass(frozen=True)
class DraftGroundingCheck:
    valid: bool
    unknown_citations: int
    unknown_supporting_candidate_ids: int
    sections_without_evidence: int
    total_sections: int
    total_supported_sections: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _short_text(value: str | None, *, fallback: str, limit: int = 160) -> str:
    text = sanitize_public_text(str(value or "").strip())
    if not text:
        return fallback
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _candidate_ids(papers: list[EvidencePaper], *, limit: int | None = None) -> list[int]:
    ids = [paper.candidate_id for paper in papers]
    return ids[:limit] if limit else ids


def _paper_key_point(paper: EvidencePaper) -> str:
    if paper.llm_summary:
        summary = (
            paper.llm_summary.get("one_sentence_summary")
            or paper.llm_summary.get("contribution")
            or paper.llm_summary.get("key_findings")
        )
        if summary:
            return _short_text(summary, fallback=paper.title)
    source = f" from {paper.source_title}" if paper.source_title else ""
    year = f" ({paper.year})" if paper.year else ""
    return _short_text(f"{paper.title}{year}{source}.", fallback=paper.title)


def fallback_review_outline(
    evidence_pack: EvidencePack,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> ReviewOutlineResult:
    papers = evidence_pack.papers
    all_ids = _candidate_ids(papers)
    top_ids = _candidate_ids(papers, limit=5)
    query = evidence_pack.research_query
    title_query = _short_text(query, fallback="shortlisted literature", limit=90)

    themes = [_paper_key_point(paper) for paper in papers[:3]]
    if not themes:
        themes = ["No shortlisted evidence was available for thematic synthesis."]

    outline = ReviewOutline(
        title=f"Literature Review Outline: {title_query}",
        sections=[
            ReviewOutlineSection(
                section_id="introduction",
                title="Introduction and Scope",
                purpose="Frame the research problem and define the review boundary.",
                supporting_candidate_ids=all_ids,
                key_points=[
                    f"Introduce the review topic: {title_query}.",
                    f"Explain how the {len(papers)} shortlisted papers support the review scope.",
                ],
            ),
            ReviewOutlineSection(
                section_id="themes",
                title="Major Themes in the Evidence",
                purpose="Group shortlisted papers into the main concepts and findings.",
                supporting_candidate_ids=top_ids,
                key_points=themes,
            ),
            ReviewOutlineSection(
                section_id="methods_and_evidence",
                title="Methods and Evidence Base",
                purpose="Compare study methods, data sources, venues, and publication context.",
                supporting_candidate_ids=all_ids,
                key_points=[
                    "Compare methods and datasets only where abstracts or summaries provide evidence.",
                    "Use DOI, source, year, and provider metadata to guide citation verification.",
                ],
            ),
            ReviewOutlineSection(
                section_id="gaps_and_synthesis",
                title="Gaps and Synthesis",
                purpose="Identify limitations, tensions, and opportunities for future work.",
                supporting_candidate_ids=all_ids,
                key_points=[
                    "Separate evidence-backed claims from assumptions requiring full-text verification.",
                    "Highlight gaps that recur across the shortlisted papers.",
                ],
            ),
        ],
    )

    safe_reason = safe_fallback_reason(fallback_reason) if fallback_reason else None
    return ReviewOutlineResult(
        outline=outline,
        attempted=attempted,
        fallback_reason=safe_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
    )


def _safe_string(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return sanitize_public_text(value.strip()) or default
    return default


def _safe_key_points(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    points: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            text = sanitize_public_text(item.strip())
            if text not in points:
                points.append(text)
    return points[:8]


def _safe_string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            text = sanitize_public_text(item.strip())
            if text not in items:
                items.append(text)
    return items[:limit]


def _safe_candidate_ids(value: Any, allowed_ids: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            candidate_id = int(item)
        except (TypeError, ValueError):
            continue
        if candidate_id in allowed_ids and candidate_id not in ids:
            ids.append(candidate_id)
    return ids


def _citation_placeholders(candidate_ids: list[int]) -> list[str]:
    return [f"[C{candidate_id}]" for candidate_id in candidate_ids]


def _sanitize_content_citations(content: str, candidate_ids: list[int]) -> str:
    allowed_placeholders = set(_citation_placeholders(candidate_ids))

    def _replace(match):
        placeholder = match.group(0)
        return placeholder if placeholder in allowed_placeholders else ""

    cleaned = re.sub(r"\[C\d+\]", _replace, sanitize_public_text(content))
    return " ".join(cleaned.split())


def render_draft_markdown(draft: ReviewDraft) -> str:
    lines = [f"# {draft.title}", ""]
    for section in draft.sections:
        lines.extend([f"## {section.title}", "", section.content, ""])
        if section.key_claims:
            lines.append("Key claims:")
            for claim in section.key_claims:
                lines.append(f"- {claim}")
            lines.append("")
        if section.limitations_or_cautions:
            lines.append("Limitations or cautions:")
            for caution in section.limitations_or_cautions:
                lines.append(f"- {caution}")
            lines.append("")
    return "\n".join(lines).strip()


def _format_reference_value(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text or "N/A"


def _reference_line(paper: EvidencePaper) -> str:
    doi_or_url = paper.doi or paper.url
    return (
        f"- [C{paper.candidate_id}] {sanitize_public_text(paper.title)}. "
        f"Year: {_format_reference_value(paper.year)}. "
        f"Source: {_format_reference_value(sanitize_public_text(paper.source_title))}. "
        f"DOI/URL: {_format_reference_value(doi_or_url)}."
    )


def render_review_markdown(evidence_pack: EvidencePack, draft: ReviewDraft) -> str:
    lines = [
        f"# {draft.title}",
        "",
        f"Research query: {sanitize_public_text(evidence_pack.research_query)}",
        "",
        f"Evidence summary: {evidence_pack.paper_count} shortlisted papers.",
        "",
    ]

    for section in draft.sections:
        lines.extend([f"## {section.title}", "", section.content, ""])
        lines.append(
            "Supporting candidate IDs: "
            + ", ".join(str(candidate_id) for candidate_id in section.supporting_candidate_ids)
        )
        lines.append("")
        if section.key_claims:
            lines.append("Key claims:")
            for claim in section.key_claims:
                lines.append(f"- {claim}")
            lines.append("")
        if section.limitations_or_cautions:
            lines.append("Limitations or cautions:")
            for caution in section.limitations_or_cautions:
                lines.append(f"- {caution}")
            lines.append("")

    lines.extend(["## References", ""])
    for paper in evidence_pack.papers:
        lines.append(_reference_line(paper))

    return "\n".join(lines).strip()


def check_draft_grounding(evidence_pack: EvidencePack, draft: ReviewDraft) -> DraftGroundingCheck:
    allowed_ids = {paper.candidate_id for paper in evidence_pack.papers}
    unknown_citations = 0
    unknown_supporting_candidate_ids = 0
    sections_without_evidence = 0
    total_supported_sections = 0

    for section in draft.sections:
        section_unknown_ids = [
            candidate_id
            for candidate_id in section.supporting_candidate_ids
            if candidate_id not in allowed_ids
        ]
        unknown_supporting_candidate_ids += len(section_unknown_ids)
        valid_supporting_ids = [
            candidate_id
            for candidate_id in section.supporting_candidate_ids
            if candidate_id in allowed_ids
        ]
        if valid_supporting_ids:
            total_supported_sections += 1
        else:
            sections_without_evidence += 1

        placeholders = list(section.citation_placeholders)
        placeholders.extend(re.findall(r"\[C\d+\]", section.content))
        for placeholder in placeholders:
            try:
                candidate_id = int(placeholder.strip()[2:-1])
            except (TypeError, ValueError):
                unknown_citations += 1
                continue
            if candidate_id not in allowed_ids:
                unknown_citations += 1

    return DraftGroundingCheck(
        valid=(
            unknown_citations == 0
            and unknown_supporting_candidate_ids == 0
            and sections_without_evidence == 0
        ),
        unknown_citations=unknown_citations,
        unknown_supporting_candidate_ids=unknown_supporting_candidate_ids,
        sections_without_evidence=sections_without_evidence,
        total_sections=len(draft.sections),
        total_supported_sections=total_supported_sections,
    )


def _paper_claim(paper: EvidencePaper) -> str:
    if paper.llm_summary:
        summary = (
            paper.llm_summary.get("one_sentence_summary")
            or paper.llm_summary.get("contribution")
            or paper.llm_summary.get("key_findings")
        )
        if summary:
            return _short_text(summary, fallback=paper.title)
    if paper.abstract:
        return _short_text(paper.abstract, fallback=paper.title)
    return f"{paper.title} is part of the shortlisted evidence base."


def fallback_review_draft(
    evidence_pack: EvidencePack,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> ReviewDraftResult:
    papers = evidence_pack.papers
    query = _short_text(evidence_pack.research_query, fallback="Shortlisted Literature", limit=90)
    title = f"{query}: A Grounded Literature Review Draft"

    sections: list[ReviewDraftSection] = []
    if papers:
        all_ids = _candidate_ids(papers)
        placeholders = _citation_placeholders(all_ids)
        intro_content = (
            f"This draft section synthesizes {len(papers)} shortlisted papers for the review topic "
            f"'{query}'. The current evidence base should be treated as metadata-grounded and verified "
            f"against full texts before final citation. {' '.join(placeholders)}"
        )
        sections.append(
            ReviewDraftSection(
                section_id="overview",
                title="Overview of Shortlisted Evidence",
                content=intro_content,
                supporting_candidate_ids=all_ids,
                citation_placeholders=placeholders,
                key_claims=[
                    f"The shortlisted evidence base contains {len(papers)} papers relevant to {query}.",
                    "The draft is grounded in candidate metadata and stored summaries only.",
                ],
                limitations_or_cautions=[
                    "Verify all claims against full papers before submission.",
                    "Metadata-only drafting may miss methods, results, or limitations not present in abstracts.",
                ],
            )
        )

        for index, paper in enumerate(papers[:3], start=1):
            citation = f"[C{paper.candidate_id}]"
            sections.append(
                ReviewDraftSection(
                    section_id=f"candidate_{paper.candidate_id}",
                    title=_short_text(paper.title, fallback=f"Evidence Theme {index}", limit=100),
                    content=(
                        f"{_paper_claim(paper)} This point is included as grounded draft material "
                        f"for the broader review of {query}. {citation}"
                    ),
                    supporting_candidate_ids=[paper.candidate_id],
                    citation_placeholders=[citation],
                    key_claims=[_paper_claim(paper)],
                    limitations_or_cautions=[
                        "This section is based on available metadata and should be checked against the full paper."
                    ],
                )
            )

    draft = ReviewDraft(title=title, sections=sections)
    safe_reason = safe_fallback_reason(fallback_reason) if fallback_reason else None
    return ReviewDraftResult(
        draft=draft,
        markdown_preview=render_draft_markdown(draft),
        attempted=attempted,
        fallback_reason=safe_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
    )


def _parse_outline_content(content: dict[str, Any], fallback: ReviewOutline, allowed_ids: set[int]) -> ReviewOutline:
    if not isinstance(content, dict) or not any(field in content for field in REVIEW_OUTLINE_FIELDS):
        raise ValueError("structured LLM content does not match review outline contract")

    sections_value = content.get("sections")
    if not isinstance(sections_value, list) or not sections_value:
        raise ValueError("review outline sections must be a non-empty list")

    sections: list[ReviewOutlineSection] = []
    for index, item in enumerate(sections_value, start=1):
        if not isinstance(item, dict):
            continue
        section_id = _safe_string(item.get("section_id"), f"section_{index}")
        title = _safe_string(item.get("title"), f"Section {index}")
        purpose = _safe_string(item.get("purpose"), "Synthesize shortlisted evidence.")
        candidate_ids = _safe_candidate_ids(item.get("supporting_candidate_ids"), allowed_ids)
        key_points = _safe_key_points(item.get("key_points"))
        if not key_points:
            key_points = ["Develop this section using the shortlisted evidence pack."]
        sections.append(
            ReviewOutlineSection(
                section_id=section_id,
                title=title,
                purpose=purpose,
                supporting_candidate_ids=candidate_ids,
                key_points=key_points,
            )
        )

    if not sections:
        raise ValueError("review outline sections did not contain valid objects")

    return ReviewOutline(
        title=_safe_string(content.get("title"), fallback.title),
        sections=sections,
    )


def _parse_draft_content(content: dict[str, Any], fallback: ReviewDraft, allowed_ids: set[int]) -> ReviewDraft:
    if not isinstance(content, dict) or not any(field in content for field in REVIEW_DRAFT_FIELDS):
        raise ValueError("structured LLM content does not match review draft contract")

    draft_value = content.get("draft")
    if not isinstance(draft_value, dict):
        raise ValueError("review draft output must include a draft object")

    sections_value = draft_value.get("sections")
    if not isinstance(sections_value, list) or not sections_value:
        raise ValueError("review draft sections must be a non-empty list")

    sections: list[ReviewDraftSection] = []
    fallback_ids = sorted(allowed_ids)
    for index, item in enumerate(sections_value, start=1):
        if not isinstance(item, dict):
            continue
        candidate_ids = _safe_candidate_ids(item.get("supporting_candidate_ids"), allowed_ids)
        if not candidate_ids and fallback_ids:
            candidate_ids = [fallback_ids[0]]
        if not candidate_ids:
            continue

        title = _safe_string(item.get("title"), f"Draft Section {index}")
        content_text = _safe_string(
            item.get("content"),
            "Draft section generated from shortlisted evidence.",
        )
        content_text = _sanitize_content_citations(content_text, candidate_ids)
        if not content_text:
            content_text = "Draft section generated from shortlisted evidence."

        sections.append(
            ReviewDraftSection(
                section_id=_safe_string(item.get("section_id"), f"section_{index}"),
                title=title,
                content=content_text,
                supporting_candidate_ids=candidate_ids,
                citation_placeholders=_citation_placeholders(candidate_ids),
                key_claims=_safe_string_list(item.get("key_claims")) or [
                    "Claim should be verified against the supporting evidence."
                ],
                limitations_or_cautions=_safe_string_list(item.get("limitations_or_cautions")) or [
                    "Verify this draft section against the full papers before use."
                ],
            )
        )

    if not sections:
        raise ValueError("review draft sections did not contain valid grounded references")

    return ReviewDraft(
        title=_safe_string(draft_value.get("title"), fallback.title),
        sections=sections,
    )


def generate_review_outline(
    evidence_pack: EvidencePack,
    client: LLMClient | None = None,
) -> ReviewOutlineResult:
    client = client or get_llm_client()
    fallback = fallback_review_outline(evidence_pack)

    try:
        generation = client.generate_structured_text(
            task=REVIEW_OUTLINE_TASK,
            payload=evidence_pack.to_llm_payload(),
        )
    except LLMUnavailableError as exc:
        return fallback_review_outline(
            evidence_pack,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        mock_outline = fallback_review_outline(evidence_pack, attempted=True).outline
        return ReviewOutlineResult(
            outline=mock_outline,
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
        outline = _parse_outline_content(
            generation.content,
            fallback.outline,
            {paper.candidate_id for paper in evidence_pack.papers},
        )
        return ReviewOutlineResult(
            outline=outline,
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
        return fallback_review_outline(
            evidence_pack,
            attempted=True,
            fallback_reason="invalid_json",
        )


def generate_review_draft(
    evidence_pack: EvidencePack,
    client: LLMClient | None = None,
) -> ReviewDraftResult:
    client = client or get_llm_client()
    fallback = fallback_review_draft(evidence_pack)

    try:
        generation = client.generate_structured_text(
            task=REVIEW_DRAFT_TASK,
            payload=evidence_pack.to_llm_payload(),
        )
    except LLMUnavailableError as exc:
        return fallback_review_draft(
            evidence_pack,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        mock_draft = fallback_review_draft(evidence_pack, attempted=True).draft
        return ReviewDraftResult(
            draft=mock_draft,
            markdown_preview=render_draft_markdown(mock_draft),
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
        draft = _parse_draft_content(
            generation.content,
            fallback.draft,
            {paper.candidate_id for paper in evidence_pack.papers},
        )
        return ReviewDraftResult(
            draft=draft,
            markdown_preview=render_draft_markdown(draft),
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
        return fallback_review_draft(
            evidence_pack,
            attempted=True,
            fallback_reason="invalid_json",
        )
