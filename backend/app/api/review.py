from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_current_user, get_db
from app.db.models.review_debate_report import ReviewDebateReport
from app.db.models.review_session import ReviewSession
from app.db.models.user import User
from app.schemas.draft import DraftUpsertRequest
from app.schemas.review import (
    ReviewDebateResponse,
    ReviewDraftResponse,
    ReviewMarkdownExportResponse,
    ReviewOutlineResponse,
    ReviewThemesResponse,
)
from app.services.evidence_pack import build_evidence_pack
from app.services.llm_client import safe_fallback_reason
from app.services.debate_agent import fallback_debate_report, generate_debate_report
from app.services.review_writer import (
    check_draft_grounding,
    fallback_review_draft,
    generate_review_draft,
    generate_review_outline,
    render_review_markdown,
)
from app.services.review_drafts import upsert_draft_for_session
from app.services.theme_grouping import generate_theme_groups


router = APIRouter(prefix="/sessions", tags=["review"])


def _get_user_session_or_404(db: Session, session_id: int, user_id: int) -> ReviewSession:
    session_obj = (
        db.query(ReviewSession)
        .filter(ReviewSession.id == session_id, ReviewSession.user_id == user_id)
        .first()
    )
    if not session_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session_obj


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _timestamp(value) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds") + "Z"
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _get_persisted_debate_report(db: Session, session_id: int) -> ReviewDebateReport | None:
    return (
        db.query(ReviewDebateReport)
        .filter(ReviewDebateReport.session_id == session_id)
        .first()
    )


def _safe_debate_payload(
    *,
    session_obj: ReviewSession,
    research_query: str,
    paper_count: int,
    result,
) -> dict:
    return {
        "session_id": session_obj.id,
        "research_query": research_query,
        "paper_count": paper_count,
        "debate_report": result.debate_report.to_dict(),
        "agent_outputs": result.agent_outputs.to_dict(),
        "grounding_check": result.grounding_check.to_dict(),
        "markdown_report": result.markdown_report,
        "debate_mode": result.debate_mode,
        "llm_used": result.used_llm,
        "fallback_reason": (
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        "prompt_version": result.prompt_version,
        "generated_at": result.generated_at or _utc_timestamp(),
    }


def _response_from_debate_payload(
    payload: dict,
    *,
    cached: bool,
    updated_at: str | None = None,
) -> ReviewDebateResponse:
    safe_payload = {
        "session_id": payload.get("session_id"),
        "research_query": payload.get("research_query"),
        "paper_count": payload.get("paper_count"),
        "debate_report": payload.get("debate_report"),
        "agent_outputs": payload.get("agent_outputs"),
        "grounding_check": payload.get("grounding_check"),
        "markdown_report": payload.get("markdown_report"),
        "debate_mode": payload.get("debate_mode"),
        "llm_used": payload.get("llm_used"),
        "fallback_reason": (
            safe_fallback_reason(payload.get("fallback_reason"))
            if payload.get("fallback_reason")
            else None
        ),
        "prompt_version": payload.get("prompt_version"),
        "generated_at": _timestamp(payload.get("generated_at")),
        "updated_at": updated_at,
        "cached": cached,
    }
    return ReviewDebateResponse(**safe_payload)


def _store_debate_payload(
    db: Session,
    *,
    session_id: int,
    payload: dict,
    existing: ReviewDebateReport | None,
) -> ReviewDebateReport | None:
    if not all(hasattr(db, method) for method in ("add", "commit")):
        return None

    now = datetime.utcnow()
    if existing:
        existing.payload_json = payload
        existing.updated_at = now
        report = existing
    else:
        report = ReviewDebateReport(
            session_id=session_id,
            payload_json=payload,
            created_at=now,
            updated_at=now,
        )
    db.add(report)
    db.commit()
    if hasattr(db, "refresh"):
        db.refresh(report)
    return report


@router.post(
    "/{session_id}/review/outline",
    response_model=ReviewOutlineResponse,
    response_model_exclude_none=True,
)
def create_review_outline(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    research_query = session_obj.raw_query or session_obj.topic
    evidence_pack = build_evidence_pack(
        db,
        session_id=session_obj.id,
        research_query=research_query,
    )

    if evidence_pack.paper_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No shortlisted papers for this session.",
        )

    result = generate_review_outline(evidence_pack)
    return ReviewOutlineResponse(
        session_id=session_obj.id,
        research_query=research_query,
        paper_count=evidence_pack.paper_count,
        outline=result.outline.to_dict(),
        llm_used=result.used_llm,
        fallback_reason=(
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        prompt_version=result.prompt_version,
    )


@router.post(
    "/{session_id}/review/themes",
    response_model=ReviewThemesResponse,
    response_model_exclude_none=True,
)
def create_review_themes(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    research_query = session_obj.raw_query or session_obj.topic
    evidence_pack = build_evidence_pack(
        db,
        session_id=session_obj.id,
        research_query=research_query,
    )

    if evidence_pack.paper_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No shortlisted papers for this session.",
        )

    result = generate_theme_groups(evidence_pack)
    return ReviewThemesResponse(
        session_id=session_obj.id,
        research_query=research_query,
        paper_count=evidence_pack.paper_count,
        themes=[theme.to_dict() for theme in result.themes],
        llm_used=result.used_llm,
        fallback_reason=(
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        prompt_version=result.prompt_version,
    )


@router.post(
    "/{session_id}/review/draft",
    response_model=ReviewDraftResponse,
    response_model_exclude_none=True,
)
def create_review_draft(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    research_query = session_obj.raw_query or session_obj.topic
    evidence_pack = build_evidence_pack(
        db,
        session_id=session_obj.id,
        research_query=research_query,
    )

    if evidence_pack.paper_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No shortlisted papers for this session.",
        )

    result = generate_review_draft(evidence_pack)
    if all(hasattr(db, method) for method in ("add", "commit")):
        upsert_draft_for_session(
            db,
            session_obj,
            current_user.id,
            DraftUpsertRequest(
                title=result.draft.title,
                content=result.markdown_preview,
                content_format="markdown",
                status="generated",
                metadata={
                    "research_query": research_query,
                    "paper_count": evidence_pack.paper_count,
                    "draft": result.draft.to_dict(),
                    "llm_used": result.used_llm,
                    "fallback_reason": (
                        safe_fallback_reason(result.fallback_reason)
                        if result.fallback_reason
                        else None
                    ),
                    "prompt_version": result.prompt_version,
                },
            ),
        )
    return ReviewDraftResponse(
        session_id=session_obj.id,
        research_query=research_query,
        paper_count=evidence_pack.paper_count,
        draft=result.draft.to_dict(),
        markdown_preview=result.markdown_preview,
        llm_used=result.used_llm,
        fallback_reason=(
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        prompt_version=result.prompt_version,
    )


@router.post(
    "/{session_id}/review/export/markdown",
    response_model=ReviewMarkdownExportResponse,
    response_model_exclude_none=True,
)
def export_review_markdown(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    research_query = session_obj.raw_query or session_obj.topic
    evidence_pack = build_evidence_pack(
        db,
        session_id=session_obj.id,
        research_query=research_query,
    )

    if evidence_pack.paper_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No shortlisted papers for this session.",
        )

    result = generate_review_draft(evidence_pack)
    grounding_check = check_draft_grounding(evidence_pack, result.draft)
    return ReviewMarkdownExportResponse(
        session_id=session_obj.id,
        filename=f"literature_review_session_{session_obj.id}.md",
        markdown=render_review_markdown(evidence_pack, result.draft),
        paper_count=evidence_pack.paper_count,
        grounding_check=grounding_check.to_dict(),
        llm_used=result.used_llm,
        fallback_reason=(
            safe_fallback_reason(result.fallback_reason)
            if result.fallback_reason
            else None
        ),
        prompt_version=result.prompt_version,
    )


@router.post(
    "/{session_id}/review/debate",
    response_model=ReviewDebateResponse,
    response_model_exclude_none=True,
)
def create_review_debate(
    session_id: int,
    force: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    persisted = _get_persisted_debate_report(db, session_obj.id)
    if persisted and not force:
        return _response_from_debate_payload(
            persisted.payload_json,
            cached=True,
            updated_at=_timestamp(getattr(persisted, "updated_at", None)),
        )

    research_query = session_obj.raw_query or session_obj.topic
    evidence_pack = build_evidence_pack(
        db,
        session_id=session_obj.id,
        research_query=research_query,
        max_papers=settings.llm_max_debate_candidates,
    )

    if evidence_pack.paper_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No shortlisted papers for this session.",
        )

    try:
        result = generate_debate_report(evidence_pack)
    except TimeoutError:
        draft = fallback_review_draft(
            evidence_pack,
            attempted=True,
            fallback_reason="timeout",
        ).draft
        grounding_check = check_draft_grounding(evidence_pack, draft)
        result = fallback_debate_report(
            evidence_pack,
            draft,
            grounding_check,
            attempted=True,
            fallback_reason="timeout",
        )
    payload = _safe_debate_payload(
        session_obj=session_obj,
        research_query=research_query,
        paper_count=evidence_pack.paper_count,
        result=result,
    )
    stored = _store_debate_payload(
        db,
        session_id=session_obj.id,
        payload=payload,
        existing=persisted,
    )
    return _response_from_debate_payload(
        payload,
        cached=False,
        updated_at=_timestamp(getattr(stored, "updated_at", None)),
    )


@router.get(
    "/{session_id}/review/debate",
    response_model=ReviewDebateResponse,
    response_model_exclude_none=True,
)
def get_review_debate(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    persisted = _get_persisted_debate_report(db, session_obj.id)
    if not persisted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Debate report not found",
        )
    return _response_from_debate_payload(
        persisted.payload_json,
        cached=True,
        updated_at=_timestamp(getattr(persisted, "updated_at", None)),
    )
