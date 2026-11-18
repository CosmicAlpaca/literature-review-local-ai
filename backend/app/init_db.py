from app.db.base import Base
from app.db.session import engine

from app.db.models.user import User
from app.db.models.review_session import ReviewSession
from app.db.models.scopus_source import ScopusSource
from app.db.models.scopus_import_job import ScopusImportJob
from app.db.models.paper_candidate import PaperCandidate
from app.db.models.selected_paper import SelectedPaper
from app.db.models.review_debate_report import ReviewDebateReport
from app.db.models.search_query_context import SearchQueryContext
from app.db.models.review_draft import ReviewDraft


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()
