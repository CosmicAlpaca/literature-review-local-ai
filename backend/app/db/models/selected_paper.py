from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SelectedPaper(Base):
    __tablename__ = "selected_papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("review_sessions.id"))
    paper_candidate_id: Mapped[int] = mapped_column(ForeignKey("paper_candidates.id"))
    rank_order: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)