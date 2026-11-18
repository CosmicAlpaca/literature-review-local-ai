"""
guardrails.py – Centralized security and safety guardrails.

Covers:
  - Input sanitization (query, question, doc_name)
  - Prompt-injection detection
  - File validation (magic bytes, size, name)
  - Session ownership verification (PDF chat)
  - Rate-limit helpers (in-process sliding window)
  - Canary / info-leak scrubbing for error messages
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# PDF guardrails
PDF_MAGIC = b"%PDF"
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024        # 20 MB
MAX_FILES_PER_SESSION = 10
MAX_TOTAL_CHUNKS_PER_SESSION = 2_000          # ~4 MB of text
MAX_DOC_NAME_LENGTH = 120
MAX_QUESTION_LENGTH = 1_000
MAX_SUMMARY_DOC_NAME_LENGTH = 120

# Search guardrails
MAX_QUERY_LENGTH = 400
MAX_SOURCE_FILTER_LENGTH = 120
MAX_MIN_CITATIONS = 100_000

# Prompt-injection detection — patterns that try to override system instructions
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"you\s+are\s+now\s+(a\s+)?(?!a\s+research)",   # "you are now a [different role]"
        r"act\s+as\s+(?!a\s+(research|academic|paper))",
        r"jailbreak",
        r"DAN\s+mode",
        r"<\s*system\s*>",          # XML-style system tag injection
        r"\[INST\]",                 # Llama-style instruction injection
        r"###\s*(Instruction|System|Human|Assistant)\s*:",
        r"SYSTEM\s*:\s*you",
    ]
]

# Characters/sequences that are safe to strip from names/queries
_DANGEROUS_CHARS_RE = re.compile(r"[<>&\"'`\\]")

# In-process sliding-window rate limiter state
_rate_state: dict[str, deque] = defaultdict(deque)
_rate_lock = Lock()


# ── Rate Limiter ──────────────────────────────────────────────────────────────

def check_rate_limit(
    key: str,
    *,
    max_calls: int,
    window_seconds: int,
    action: str = "request",
) -> None:
    """
    Sliding-window in-process rate limiter.
    Raises HTTP 429 if the key has exceeded max_calls in window_seconds.
    Thread-safe via a single global lock (adequate for a single-process server).
    """
    now = time.monotonic()
    cutoff = now - window_seconds
    with _rate_lock:
        dq = _rate_state[key]
        # evict expired timestamps
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= max_calls:
            wait = int(window_seconds - (now - dq[0])) + 1
            logger.warning("Rate limit hit: key=%s action=%s", key, action)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Too many {action}s. "
                    f"Please wait {wait}s before retrying."
                ),
                headers={"Retry-After": str(wait)},
            )
        dq.append(now)


# ── Input Sanitizers ──────────────────────────────────────────────────────────

def sanitize_query(query: str | None) -> str:
    """
    Validate and sanitize a search query.
    Raises 422 for empty, too-long, or injection-detected inputs.
    Returns a stripped, safe query string.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query cannot be empty.",
        )
    q = query.strip()
    if len(q) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Search query is too long (max {MAX_QUERY_LENGTH} characters).",
        )
    _check_injection(q, field="query")
    return q


def sanitize_question(question: str | None) -> str:
    """
    Validate and sanitize a PDF chat question.
    Raises 422 for empty, too-long, or injection-detected inputs.
    """
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )
    q = question.strip()
    if len(q) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question is too long (max {MAX_QUESTION_LENGTH} characters).",
        )
    _check_injection(q, field="question")
    return q


def sanitize_doc_name(name: str | None) -> str | None:
    """
    Sanitize a doc_name used for summaries. Returns None if name is None/empty.
    Strips dangerous characters; rejects names that are too long.
    """
    if not name:
        return None
    name = name.strip()
    if len(name) > MAX_SUMMARY_DOC_NAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document name is too long (max {MAX_SUMMARY_DOC_NAME_LENGTH} characters).",
        )
    # Strip characters that could cause template/path injection
    cleaned = _DANGEROUS_CHARS_RE.sub("", name).strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document name contains only invalid characters.",
        )
    return cleaned


def sanitize_source_filter(source: str | None) -> str | None:
    """Sanitize the `source` search filter string."""
    if not source:
        return None
    s = source.strip()
    if len(s) > MAX_SOURCE_FILTER_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Source filter is too long (max {MAX_SOURCE_FILTER_LENGTH} characters).",
        )
    return _DANGEROUS_CHARS_RE.sub("", s).strip() or None


# ── PDF File Validators ───────────────────────────────────────────────────────

def validate_pdf_file(
    filename: str | None,
    content: bytes,
) -> str:
    """
    Validate an uploaded PDF file.
    Checks: filename extension, magic bytes, file size.
    Returns the sanitized document name (filename without .pdf).
    """
    # 1. Filename presence and extension
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No filename provided.",
        )
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted (.pdf extension required).",
        )

    # 2. Magic bytes — verify actual PDF content regardless of extension
    if not content[:4] == PDF_MAGIC:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file does not appear to be a valid PDF (magic bytes check failed).",
        )

    # 3. Size check
    size_bytes = len(content)
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Max allowed: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )

    # 4. Sanitize doc name derived from filename
    base = filename[:-4]  # strip .pdf
    if len(base) > MAX_DOC_NAME_LENGTH:
        base = base[:MAX_DOC_NAME_LENGTH]
    # Strip dangerous chars from doc name
    safe_name = _DANGEROUS_CHARS_RE.sub("", base).strip()
    if not safe_name:
        safe_name = "document"
    return safe_name


# ── Session Ownership ─────────────────────────────────────────────────────────

def verify_session_ownership(session_key: str, user_id: int) -> None:
    """
    Verify that the session_key belongs to the requesting user.
    PDF chat session keys are formatted as '{user_id}:{uuid}'.
    Raises 403 if the key does not start with the user's ID.
    """
    expected_prefix = f"{user_id}:"
    if not session_key.startswith(expected_prefix):
        logger.warning(
            "Session ownership violation: user_id=%s key=%s",
            user_id,
            session_key[:30],
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: this session does not belong to you.",
        )


def verify_session_key_format(session_key: str) -> None:
    """
    Verify that the session_key looks well-formed (prevents path traversal).
    Format: '{int}:{hex32}'.
    """
    # Allow only digits, colon, and hex characters (a-f, 0-9)
    if not re.fullmatch(r"\d+:[0-9a-f]{32}", session_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session key format.",
        )


# ── Session Capacity ──────────────────────────────────────────────────────────

def check_session_capacity(store: Any, *, max_files: int = MAX_FILES_PER_SESSION) -> None:
    """Raise 422 if the session already has too many documents."""
    if len(store.document_names) >= max_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum {max_files} files per session reached.",
        )


def check_chunk_capacity(store: Any) -> None:
    """Raise 422 if the session's total chunks would exceed the hard limit."""
    if store.total_chunks >= MAX_TOTAL_CHUNKS_PER_SESSION:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Session chunk limit reached ({MAX_TOTAL_CHUNKS_PER_SESSION} chunks max). "
                "Remove some documents before uploading more."
            ),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_injection(text: str, *, field: str = "input") -> None:
    """Detect prompt-injection patterns. Raises 422 with a safe message."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Prompt injection attempt detected in field=%s", field)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {field}: contains disallowed content.",
            )


def scrub_error_message(exc: Exception) -> str:
    """
    Return a safe error message that does not leak internal details.
    Strips file paths, API keys, stack traces, and SQL fragments.
    """
    raw = str(exc)
    # Patterns that should never reach the client
    sensitive_patterns = [
        r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[=:]\s*\S+",
        r"File \"[^\"]+\"",          # Python traceback file paths
        r"/[a-z0-9_/.-]{10,}",       # absolute Unix paths
        r"[A-Z]:\\[^\"<>|?*\n]+",   # Windows paths
        r"(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\s+\w+",  # SQL
        r"Traceback \(most recent",
    ]
    for p in sensitive_patterns:
        if re.search(p, raw, re.IGNORECASE):
            return "An internal error occurred. Please try again later."
    # Truncate long messages
    return raw[:300] if len(raw) > 300 else raw
