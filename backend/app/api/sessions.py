from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_debate_report import ReviewDebateReport
from app.db.models.review_draft import ReviewDraft
from app.db.models.review_session import ReviewSession
from app.db.models.search_query_context import SearchQueryContext
from app.db.models.selected_paper import SelectedPaper
from app.db.models.user import User
from app.schemas.draft import DraftEnvelopeResponse, DraftUpsertRequest
from app.schemas.session import CreateSessionRequest, SessionResponse
from app.services.orchestrator import SearchFilterError, normalize_search_filters
from app.services.review_drafts import (
    draft_to_response,
    get_draft_for_session,
    upsert_draft_for_session,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_user_session_or_404(db: Session, session_id: int, user_id: int) -> ReviewSession:
    session_obj = db.query(ReviewSession).filter(
        ReviewSession.id == session_id,
        ReviewSession.user_id == user_id,
    ).first()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_obj


@router.post("", response_model=SessionResponse)
def create_session(
    payload: CreateSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        normalize_search_filters(query=payload.raw_query, session_filters=payload.filters_json)
    except SearchFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_obj = ReviewSession(
        user_id=current_user.id,
        topic=payload.topic,
        raw_query=payload.raw_query,
        filters_json=payload.filters_json,
        status="created",
    )
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)

    return SessionResponse(
        id=session_obj.id,
        topic=session_obj.topic,
        raw_query=session_obj.raw_query,
        status=session_obj.status,
        created_at=session_obj.created_at,
    )


@router.get("", response_model=list[SessionResponse])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ReviewSession)
        .filter(ReviewSession.user_id == current_user.id)
        .order_by(ReviewSession.created_at.desc())
        .all()
    )
    return [
        SessionResponse(
            id=s.id,
            topic=s.topic,
            raw_query=s.raw_query,
            status=s.status,
            created_at=s.created_at,
        )
        for s in sessions
    ]


@router.get("/{session_id}/draft", response_model=DraftEnvelopeResponse)
def get_session_draft(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    draft = get_draft_for_session(db, session_obj.id, current_user.id)
    if not draft:
        return DraftEnvelopeResponse(session_id=session_obj.id, exists=False, draft=None)
    return DraftEnvelopeResponse(
        session_id=session_obj.id,
        exists=True,
        draft=draft_to_response(draft),
    )


@router.put("/{session_id}/draft", response_model=DraftEnvelopeResponse)
def upsert_session_draft(
    session_id: int,
    payload: DraftUpsertRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)
    draft = upsert_draft_for_session(db, session_obj, current_user.id, payload)
    return DraftEnvelopeResponse(
        session_id=session_obj.id,
        exists=True,
        draft=draft_to_response(draft),
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_obj = _get_user_session_or_404(db, session_id, current_user.id)

    # Keep this explicit because several routes use bulk inserts/updates and the
    # models do not configure database-level ON DELETE CASCADE.
    db.query(ReviewDebateReport).filter(ReviewDebateReport.session_id == session_id).delete(
        synchronize_session=False
    )
    db.query(SearchQueryContext).filter(SearchQueryContext.session_id == session_id).delete(
        synchronize_session=False
    )
    db.query(ReviewDraft).filter(ReviewDraft.session_id == session_id).delete(
        synchronize_session=False
    )
    db.query(SelectedPaper).filter(SelectedPaper.session_id == session_id).delete(
        synchronize_session=False
    )
    db.query(PaperCandidate).filter(PaperCandidate.session_id == session_id).delete(
        synchronize_session=False
    )
    db.delete(session_obj)
    db.commit()

    return None
