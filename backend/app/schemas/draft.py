from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DraftResponse(BaseModel):
    id: int
    session_id: int
    user_id: int
    title: str | None = None
    content: str
    content_format: str = "markdown"
    status: str = "draft"
    version: int = 1
    metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class DraftEnvelopeResponse(BaseModel):
    session_id: int
    exists: bool
    draft: DraftResponse | None = None


class DraftUpsertRequest(BaseModel):
    title: str | None = None
    content: str
    content_format: str = "markdown"
    status: str = "draft"
    metadata: dict[str, Any] | None = Field(default=None)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("content must not be empty")
        return value

    @field_validator("content_format", "status")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must not be empty")
        return value.strip()

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
