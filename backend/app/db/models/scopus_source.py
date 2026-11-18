from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ScopusSource(Base):
    __tablename__ = "scopus_sources"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    source_record_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_title: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    normalized_source_title: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    issn: Mapped[str | None] = mapped_column(String(50), nullable=True)
    eissn: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(500), nullable=True)
    active_status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    asjc_codes: Mapped[str | None] = mapped_column(Text, nullable=True)
    open_access_status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    discontinued_due_to_quality: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    coverage: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_discontinued: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)