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
from app.services.llm_prompts import THEME_GROUPING_FIELDS, THEME_GROUPING_TASK


@dataclass(frozen=True)
class ThemeGroup:
    theme_id: str
    name: str
    rationale: str
    keywords: list[str]
    supporting_candidate_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThemeGroupingResult:
    themes: list[ThemeGroup]
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
        data["themes"] = [theme.to_dict() for theme in self.themes]
        return data


def _short_text(value: str | None, *, fallback: str, limit: int = 180) -> str:
    text = sanitize_public_text(str(value or "").strip())
    if not text:
        return fallback
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _theme_id(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (normalized or fallback)[:80]


def _paper_keywords(paper: EvidencePaper) -> list[str]:
    keywords: list[str] = []
    if paper.review_tags:
        keywords.extend(paper.review_tags)
    if paper.source_title:
        keywords.append(paper.source_title)
    if paper.year:
        keywords.append(str(paper.year))

    cleaned: list[str] = []
    for keyword in keywords:
        text = _short_text(keyword, fallback="", limit=50)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:6]


def _paper_summary_text(paper: EvidencePaper) -> str:
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
    return _short_text(paper.title, fallback="Shortlisted paper")


def fallback_theme_grouping(
    evidence_pack: EvidencePack,
    *,
    attempted: bool = False,
    fallback_reason: str | None = None,
) -> ThemeGroupingResult:
    papers = evidence_pack.papers
    if not papers:
        themes: list[ThemeGroup] = []
    else:
        all_keywords: list[str] = []
        for paper in papers:
            for keyword in _paper_keywords(paper):
                if keyword not in all_keywords:
                    all_keywords.append(keyword)

        first_points = [_paper_summary_text(paper) for paper in papers[:3]]
        rationale = (
            "These shortlisted papers form the initial evidence base for the review. "
            + " ".join(first_points)
        )
        themes = [
            ThemeGroup(
                theme_id="shortlisted_evidence",
                name="Shortlisted Evidence Base",
                rationale=_short_text(rationale, fallback="Theme generated from shortlisted evidence."),
                keywords=all_keywords[:8] or ["shortlisted evidence"],
                supporting_candidate_ids=[paper.candidate_id for paper in papers],
            )
        ]

        tagged: dict[str, list[EvidencePaper]] = {}
        for paper in papers:
            for tag in paper.review_tags or []:
                tagged.setdefault(tag, []).append(paper)
        for tag, tagged_papers in sorted(tagged.items())[:2]:
            theme_name = _short_text(tag.title(), fallback="Review Tag", limit=80)
            themes.append(
                ThemeGroup(
                    theme_id=_theme_id(tag, fallback="review_tag"),
                    name=theme_name,
                    rationale=f"These papers were shortlisted with the review tag '{tag}'.",
                    keywords=[tag],
                    supporting_candidate_ids=[paper.candidate_id for paper in tagged_papers],
                )
            )

    safe_reason = safe_fallback_reason(fallback_reason) if fallback_reason else None
    return ThemeGroupingResult(
        themes=themes,
        attempted=attempted,
        fallback_reason=safe_reason,
        prompt_version=current_prompt_version(),
        generated_at=utc_timestamp() if attempted else None,
    )


def _safe_string(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return sanitize_public_text(value.strip()) or default
    return default


def _safe_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            keyword = _short_text(item, fallback="", limit=60)
            if keyword and keyword not in keywords:
                keywords.append(keyword)
    return keywords[:12]


def _safe_candidate_ids(value: Any, allowed_ids: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    candidate_ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            candidate_id = int(item)
        except (TypeError, ValueError):
            continue
        if candidate_id in allowed_ids and candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    return candidate_ids


def _parse_theme_content(content: dict[str, Any], allowed_ids: set[int]) -> list[ThemeGroup]:
    if not isinstance(content, dict) or not any(field in content for field in THEME_GROUPING_FIELDS):
        raise ValueError("structured LLM content does not match theme grouping contract")

    themes_value = content.get("themes")
    if not isinstance(themes_value, list) or not themes_value:
        raise ValueError("theme grouping output must include a non-empty themes array")

    themes: list[ThemeGroup] = []
    for index, item in enumerate(themes_value, start=1):
        if not isinstance(item, dict):
            continue
        supporting_ids = _safe_candidate_ids(item.get("supporting_candidate_ids"), allowed_ids)
        if not supporting_ids:
            continue
        name = _safe_string(item.get("name"), f"Theme {index}")
        theme_id = _theme_id(_safe_string(item.get("theme_id"), name), fallback=f"theme_{index}")
        themes.append(
            ThemeGroup(
                theme_id=theme_id,
                name=name,
                rationale=_safe_string(item.get("rationale"), "Theme derived from shortlisted evidence."),
                keywords=_safe_keywords(item.get("keywords")) or [name],
                supporting_candidate_ids=supporting_ids,
            )
        )

    if not themes:
        raise ValueError("theme grouping output did not reference valid shortlisted candidate IDs")
    return themes


def generate_theme_groups(
    evidence_pack: EvidencePack,
    client: LLMClient | None = None,
) -> ThemeGroupingResult:
    client = client or get_llm_client()

    try:
        generation = client.generate_structured_text(
            task=THEME_GROUPING_TASK,
            payload=evidence_pack.to_llm_payload(),
        )
    except LLMUnavailableError as exc:
        return fallback_theme_grouping(
            evidence_pack,
            attempted=True,
            fallback_reason=exc.reason,
        )

    if generation.provider == "mock":
        mock_themes = fallback_theme_grouping(evidence_pack, attempted=True).themes
        return ThemeGroupingResult(
            themes=mock_themes,
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
        themes = _parse_theme_content(
            generation.content,
            {paper.candidate_id for paper in evidence_pack.papers},
        )
        return ThemeGroupingResult(
            themes=themes,
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
        return fallback_theme_grouping(
            evidence_pack,
            attempted=True,
            fallback_reason="invalid_json",
        )
