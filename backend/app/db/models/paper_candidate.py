from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Float, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PaperCandidate(Base):
    __tablename__ = "paper_candidates"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("review_sessions.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    authors_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    normalized_source_title: Mapped[str | None] = mapped_column(String(1000), nullable=True, index=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    scopus_matched: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    scopus_source_id: Mapped[int | None] = mapped_column(ForeignKey("scopus_sources.id"), nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)