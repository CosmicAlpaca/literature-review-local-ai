from __future__ import annotations

from datetime import datetime
from typing import Any

from app.db.models.review_draft import ReviewDraft
from app.db.models.review_session import ReviewSession
from app.schemas.draft import DraftResponse, DraftUpsertRequest


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def json_safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return _json_safe_value(metadata)


def draft_to_response(draft: ReviewDraft) -> DraftResponse:
    return DraftResponse(
        id=draft.id,
        session_id=draft.session_id,
        user_id=draft.user_id,
        title=draft.title,
        content=draft.content,
        content_format=draft.content_format,
        status=draft.status,
        version=draft.version,
        metadata=draft.metadata_json,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
    )


def get_draft_for_session(db, session_id: int, user_id: int) -> ReviewDraft | None:
    return (
        db.query(ReviewDraft)
        .filter(ReviewDraft.session_id == session_id, ReviewDraft.user_id == user_id)
        .first()
    )


def upsert_draft_for_session(
    db,
    session: ReviewSession,
    user_id: int,
    payload: DraftUpsertRequest,
) -> ReviewDraft:
    existing = get_draft_for_session(db, session.id, user_id)
    now = datetime.utcnow()
    safe_metadata = json_safe_metadata(payload.metadata)

    if existing:
        existing.title = payload.title
        existing.content = payload.content
        existing.content_format = payload.content_format
        existing.status = payload.status
        existing.metadata_json = safe_metadata
        existing.version = int(existing.version or 1) + 1
        existing.updated_at = now
        draft = existing
    else:
        draft = ReviewDraft(
            session_id=session.id,
            user_id=user_id,
            title=payload.title,
            content=payload.content,
            content_format=payload.content_format,
            status=payload.status,
            version=1,
            metadata_json=safe_metadata,
            created_at=now,
            updated_at=now,
        )

    db.add(draft)
    db.commit()
    if hasattr(db, "refresh"):
        db.refresh(draft)
    return draft
