"""
api/pdf_chat.py – PDF Chat API (completely independent from search pipeline)

Endpoints:
  POST /pdf-chat/sessions               – Create a new chat session
  DELETE /pdf-chat/sessions/{key}       – Delete a session & its store
  POST /pdf-chat/sessions/{key}/upload  – Upload PDF(s) to a session
  GET  /pdf-chat/sessions/{key}/docs    – List uploaded documents
  POST /pdf-chat/sessions/{key}/ask     – Ask a question
  POST /pdf-chat/sessions/{key}/summary – Summarize a document
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from app.core.dependencies import get_current_user
from app.core.guardrails import (
    MAX_FILES_PER_SESSION,
    MAX_FILE_SIZE_BYTES,
    check_chunk_capacity,
    check_rate_limit,
    check_session_capacity,
    sanitize_doc_name,
    sanitize_question,
    scrub_error_message,
    validate_pdf_file,
    verify_session_key_format,
    verify_session_ownership,
)
from app.services.pdf_rag import (
    RagAnswer,
    answer_question,
    delete_store,
    get_document_summary,
    get_or_create_store,
    knowledge_graph,
)

router = APIRouter(prefix="/pdf-chat", tags=["pdf-chat"])

# Rate limits (per user)
_UPLOAD_MAX       = 20   # uploads per minute
_UPLOAD_WINDOW    = 60
_ASK_MAX          = 30   # questions per minute
_ASK_WINDOW       = 60
_SUMMARY_MAX      = 10   # summaries per minute
_SUMMARY_WINDOW   = 60
_SESSION_MAX      = 10   # session creates per hour
_SESSION_WINDOW   = 3600


# ── Pydantic models ────────────────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_key: str


class UploadResponse(BaseModel):
    session_key: str
    doc_name: str
    chunks_added: int
    total_chunks: int
    documents: list[str]


class DocumentListResponse(BaseModel):
    session_key: str
    documents: list[str]
    total_chunks: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=500)

class AskRequest(BaseModel):
    question: str
    history: list[ChatTurn] = Field(default_factory=list, max_length=4)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()


class CitedChunkResponse(BaseModel):
    doc_name: str
    page_num: int
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    cited_chunks: list[CitedChunkResponse]
    error: str | None = None


class SummaryRequest(BaseModel):
    doc_name: str | None = None


class SummaryResponse(BaseModel):
    summary: str
    doc_name: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _session_key(user_id: int, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def _require_owned_store(session_key: str, user_id: int):
    """
    Guards:
    1. session_key format (path traversal prevention)
    2. session_key ownership (IDOR prevention)
    3. store existence (404)
    """
    verify_session_key_format(session_key)
    verify_session_ownership(session_key, user_id)

    from app.services.pdf_rag import _stores
    if session_key not in _stores:
        raise HTTPException(status_code=404, detail="PDF chat session not found.")
    return _stores[session_key]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_pdf_session(
    current_user: Any = Depends(get_current_user),
):
    """Create a new isolated PDF chat session."""
    # Rate limit: prevent session flooding
    check_rate_limit(
        f"pdf_session_create:{current_user.id}",
        max_calls=_SESSION_MAX,
        window_seconds=_SESSION_WINDOW,
        action="session creation",
    )
    session_id = uuid.uuid4().hex
    key = _session_key(current_user.id, session_id)
    get_or_create_store(key)  # initialise empty store
    return CreateSessionResponse(session_key=key)


@router.delete("/sessions/{session_key:path}", status_code=204)
def delete_pdf_session(
    session_key: str,
    current_user: Any = Depends(get_current_user),
):
    """Delete a PDF chat session and free its memory."""
    verify_session_key_format(session_key)
    verify_session_ownership(session_key, current_user.id)
    delete_store(session_key)


@router.post(
    "/sessions/{session_key:path}/upload",
    response_model=UploadResponse,
)
async def upload_pdf(
    session_key: str,
    file: UploadFile = File(...),
    current_user: Any = Depends(get_current_user),
):
    """Upload a PDF and index it in the session's RAG store."""
    # Rate limit uploads
    check_rate_limit(
        f"pdf_upload:{current_user.id}",
        max_calls=_UPLOAD_MAX,
        window_seconds=_UPLOAD_WINDOW,
        action="upload",
    )

    # Ownership + format guards
    verify_session_key_format(session_key)
    verify_session_ownership(session_key, current_user.id)
    store = _require_owned_store(session_key, current_user.id)

    # Capacity guards (before reading file content)
    check_session_capacity(store, max_files=MAX_FILES_PER_SESSION)
    check_chunk_capacity(store)

    # Read file
    content = await file.read(MAX_FILE_SIZE_BYTES + 1)

    # File validation (extension, magic bytes, size)
    doc_name = validate_pdf_file(file.filename, content)

    # Duplicate document guard
    if doc_name in store.document_names:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A document named '{doc_name}' is already uploaded in this session.",
        )

    try:
        added = store.add_document(content, doc_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=scrub_error_message(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=scrub_error_message(exc),
        )

    return UploadResponse(
        session_key=session_key,
        doc_name=doc_name,
        chunks_added=added,
        total_chunks=store.total_chunks,
        documents=store.document_names,
    )


@router.get(
    "/sessions/{session_key:path}/docs",
    response_model=DocumentListResponse,
)
def list_documents(
    session_key: str,
    current_user: Any = Depends(get_current_user),
):
    """List all documents indexed in this session."""
    store = _require_owned_store(session_key, current_user.id)
    return DocumentListResponse(
        session_key=session_key,
        documents=store.document_names,
        total_chunks=store.total_chunks,
    )


@router.post(
    "/sessions/{session_key:path}/ask",
    response_model=AskResponse,
)
def ask_question(
    session_key: str,
    payload: AskRequest,
    current_user: Any = Depends(get_current_user),
):
    """Ask a question over the uploaded PDFs."""
    # Rate limit Q&A
    check_rate_limit(
        f"pdf_ask:{current_user.id}",
        max_calls=_ASK_MAX,
        window_seconds=_ASK_WINDOW,
        action="question",
    )

    store = _require_owned_store(session_key, current_user.id)

    # Full question validation (length + injection)
    safe_question = sanitize_question(payload.question)

    try:
        result: RagAnswer = answer_question(store, safe_question, [turn.model_dump() for turn in payload.history])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=scrub_error_message(exc),
        )

    return AskResponse(
        answer=result.answer,
        cited_chunks=[
            CitedChunkResponse(
                doc_name=c.doc_name,
                page_num=c.page_num,
                text_preview=c.text[:200] + ("..." if len(c.text) > 200 else ""),
            )
            for c in result.cited_chunks
        ],
        error=result.error,
    )


@router.post(
    "/sessions/{session_key:path}/summary",
    response_model=SummaryResponse,
)
def summarize_document(
    session_key: str,
    payload: SummaryRequest,
    current_user: Any = Depends(get_current_user),
):
    """Generate a summary of one document (or all documents)."""
    # Rate limit summaries
    check_rate_limit(
        f"pdf_summary:{current_user.id}",
        max_calls=_SUMMARY_MAX,
        window_seconds=_SUMMARY_WINDOW,
        action="summary",
    )

    store = _require_owned_store(session_key, current_user.id)

    # Sanitize doc_name (injection + length)
    safe_doc_name = sanitize_doc_name(payload.doc_name)

    # Verify the doc actually exists in this session
    if safe_doc_name and safe_doc_name not in store.document_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{safe_doc_name}' not found in this session.",
        )

    try:
        summary = get_document_summary(store, doc_name=safe_doc_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=scrub_error_message(exc),
        )

    return SummaryResponse(summary=summary, doc_name=safe_doc_name)


@router.get("/sessions/{session_key:path}/graph")
def graph_for_session(session_key: str, current_user: Any = Depends(get_current_user)):
    """Return a co-occurrence map with page-level evidence."""
    store = _require_owned_store(session_key, current_user.id)
    return knowledge_graph(store)
