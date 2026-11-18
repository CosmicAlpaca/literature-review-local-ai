from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.paper_candidate import PaperCandidate


_SUMMARY_FIELDS = (
    "one_sentence_summary",
    "contribution",
    "method_or_data",
    "relevance_to_query",
    "limitations_or_caution",
)


@dataclass(frozen=True)
class EvidencePaper:
    candidate_id: int
    title: str
    abstract: str | None
    year: int | None
    source_title: str | None
    doi: str | None
    url: str | None
    citation_count: int | None
    provider: str
    scopus_matched: bool
    final_score: float | None
    llm_summary: dict[str, str] | None = None
    review_notes: str | None = None
    review_tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class EvidencePack:
    session_id: int
    research_query: str
    papers: list[EvidencePaper]

    @property
    def paper_count(self) -> int:
        return len(self.papers)

    def to_llm_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "research_query": self.research_query,
            "papers": [paper.to_dict() for paper in self.papers],
        }


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    metadata = getattr(candidate, "raw_metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        safe_item = _safe_string(item)
        if safe_item and safe_item not in items:
            items.append(safe_item)
    return items or None


def _review_metadata(candidate: Any) -> dict[str, Any]:
    review = _candidate_metadata(candidate).get("_review")
    return review if isinstance(review, dict) else {}


def is_shortlisted(candidate: Any) -> bool:
    return _review_metadata(candidate).get("status") == "shortlisted"


def _safe_llm_summary(candidate: Any) -> dict[str, str] | None:
    summary = _candidate_metadata(candidate).get("_llm_summary")
    if not isinstance(summary, dict):
        return None

    safe_summary: dict[str, str] = {}
    for field in _SUMMARY_FIELDS:
        value = _safe_string(summary.get(field))
        if not value and field == "contribution":
            value = _safe_string(summary.get("key_findings"))
        if not value and field == "method_or_data":
            value = _safe_string(summary.get("main_methodology"))
        if not value and field == "limitations_or_caution":
            value = (
                _safe_string(summary.get("research_gap"))
                or _safe_string(summary.get("potential_contributions"))
            )
        if value:
            safe_summary[field] = value
    return safe_summary or None


def _review_notes(candidate: Any) -> str | None:
    review = _review_metadata(candidate)
    return _safe_string(review.get("notes")) or _safe_string(review.get("note"))


def _review_tags(candidate: Any) -> list[str] | None:
    return _safe_string_list(_review_metadata(candidate).get("tags"))


def build_evidence_pack(
    db: Session,
    *,
    session_id: int,
    research_query: str,
    max_papers: int | None = None,
) -> EvidencePack:
    candidates = (
        db.query(PaperCandidate)
        .filter(PaperCandidate.session_id == session_id)
        .order_by(PaperCandidate.final_score.desc(), PaperCandidate.id.asc())
        .all()
    )

    papers = [
        EvidencePaper(
            candidate_id=int(candidate.id),
            title=str(candidate.title),
            abstract=_safe_string(getattr(candidate, "abstract", None)),
            year=_safe_int(getattr(candidate, "year", None)),
            source_title=_safe_string(getattr(candidate, "source_title", None)),
            doi=_safe_string(getattr(candidate, "doi", None)),
            url=_safe_string(getattr(candidate, "url", None)),
            citation_count=_safe_int(getattr(candidate, "citation_count", None)),
            provider=str(getattr(candidate, "provider", "")),
            scopus_matched=bool(getattr(candidate, "scopus_matched", False)),
            final_score=_safe_float(getattr(candidate, "final_score", None)),
            llm_summary=_safe_llm_summary(candidate),
            review_notes=_review_notes(candidate),
            review_tags=_review_tags(candidate),
        )
        for candidate in candidates
        if is_shortlisted(candidate)
    ]

    if max_papers is not None and max_papers > 0:
        papers = papers[:max_papers]

    return EvidencePack(
        session_id=session_id,
        research_query=research_query,
        papers=papers,
    )
