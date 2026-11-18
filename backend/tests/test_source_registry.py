from pathlib import Path
import uuid
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import admin as admin_api
from app.core.dependencies import get_db, require_admin
from app.db.models.scopus_import_job import ScopusImportJob
from app.db.models.scopus_source import ScopusSource
from app.main import app
from app.services.source_registry import (
    ingest_scopus_excel,
    ingest_scopus_excel_with_progress,
    load_registry_workbook,
)


class _FakeQuery:
    def __init__(self, results):
        self._results = results

    def all(self):
        return self._results

    def filter(self, *args, **kwargs):
        return self


class _FakeDB:
    def __init__(self):
        self.sources = []
        self.commit_count = 0

    def query(self, model):
        return _FakeQuery(self.sources)

    def add(self, obj):
        self.sources.append(obj)

    def commit(self):
        self.commit_count += 1


def _build_registry_workbook(path: Path):
    wb = Workbook()

    ws = wb.active
    ws.title = "Scopus Sources Feb. 2026"
    ws.append(
        [
            "Sourcerecord ID",
            "Source Title",
            "ISSN",
            "EISSN",
            "Active or Inactive",
            "Coverage",
            "Titles Discontinued by Scopus",
            "Medline-sourced Title? (See additional details under separate tab.)",
            "Open Access Status",
            "Source Type",
            "Related Title 1",
            "Other Related Title 2",
            "Other Related Title 3",
            "Other Related Title 4",
            "Publisher",
            "Publisher Imprints Grouped to Main Publisher",
            "All Science Journal Classification Codes (ASJC)",
            "Top level:\n\nLife Sciences",
            "Top level:\n\nSocial Sciences",
            "Top level:\n\nPhysical Sciences",
            "Top level:\n\nHealth Sciences",
        ]
    )
    ws.append(
        [
            "1001",
            "Remote Sensing",
            "20724292",
            "",
            "Active",
            "2010-2026",
            "",
            "Yes",
            "Gold",
            "Journal",
            "Remote Sensing Journal",
            "",
            "",
            "",
            "MDPI",
            "MDPI",
            "1900; 2300",
            "Life Sciences",
            "",
            "Physical Sciences",
            "",
        ]
    )

    ws2 = wb.create_sheet("Accepted Titles Feb. 2026")
    ws2.append(["Status", "", "", "", ""])
    ws2.append(["Notice", "", "", "", ""])
    ws2.append(["Source Title", "ISSN", "EISSN", "Date of Acceptance", "Publisher"])
    ws2.append(["Future Energy", "", "12345678", "2026-02-01", "Springer"])

    ws3 = wb.create_sheet("Discontinued Titles Feb. 2026")
    ws3.append(["Status", "", "", "", "", "", "", "", "", ""])
    ws3.append(
        [
            "Sourcerecord ID",
            "Source Title (newly added titles are highlighted in red)",
            "ISSN",
            "EISSN",
            "Publisher",
            "Indexation Change",
            "Year",
            "Volume",
            "Issue",
            "Page Range",
        ]
    )
    ws3.append(["2002", "Old Journal", "87654321", "", "Legacy Pub", "Discontinuation", 2024, 13, 1, "1-10"])

    ws4 = wb.create_sheet("Serial Conf. Proc. with Profile")
    ws4.append(
        [
            "Sourcerecord ID",
            "Source Title",
            "ISSN",
            "EISSN",
            "Titles Discontinued by Scopus",
            "Coverage",
            "All Science Journal Classification Codes (ASJC)",
        ]
    )
    ws4.append(["3003", "Conference Series A", "22223333", "", "", "2024-2026", "1700; 2200"])

    wb.save(path)


def _build_registry_workbook_with_main_rows(path: Path, row_count: int):
    wb = Workbook()

    ws = wb.active
    ws.title = "Scopus Sources Feb. 2026"
    ws.append(
        [
            "Sourcerecord ID",
            "Source Title",
            "ISSN",
            "EISSN",
            "Active or Inactive",
            "Coverage",
            "Titles Discontinued by Scopus",
            "Medline-sourced Title? (See additional details under separate tab.)",
            "Open Access Status",
            "Source Type",
            "Related Title 1",
            "Other Related Title 2",
            "Other Related Title 3",
            "Other Related Title 4",
            "Publisher",
            "Publisher Imprints Grouped to Main Publisher",
            "All Science Journal Classification Codes (ASJC)",
        ]
    )
    for index in range(row_count):
        ws.append(
            [
                f"row-{index}",
                f"Journal {index}",
                f"1000{index:04d}",
                "",
                "Active",
                "2020-2026",
                "",
                "",
                "Gold",
                "Journal",
                "",
                "",
                "",
                "",
                "Publisher",
                "Publisher",
                "1900",
            ]
        )

    wb.save(path)


def _build_eissn_only_registry_workbook(path: Path):
    wb = Workbook()

    ws = wb.active
    ws.title = "Scopus Sources Feb. 2026"
    ws.append(
        [
            "Sourcerecord ID",
            "Source Title",
            "ISSN",
            "EISSN",
            "Active or Inactive",
            "Coverage",
            "Titles Discontinued by Scopus",
            "Medline-sourced Title? (See additional details under separate tab.)",
            "Open Access Status",
            "Source Type",
            "Related Title 1",
            "Other Related Title 2",
            "Other Related Title 3",
            "Other Related Title 4",
            "Publisher",
            "Publisher Imprints Grouped to Main Publisher",
            "All Science Journal Classification Codes (ASJC)",
        ]
    )
    ws.append(
        [
            "21100374601",
            "IEEE Access",
            "",
            "21693536",
            "Active",
            "2013-2026",
            "",
            "",
            "Gold",
            "Journal",
            "",
            "",
            "",
            "",
            "IEEE",
            "IEEE",
            "1700; 2200",
        ]
    )
    ws.append(
        [
            "10600153309",
            "PLOS ONE",
            "",
            "19326203",
            "Active",
            "2006-2026",
            "",
            "",
            "Gold",
            "Journal",
            "",
            "",
            "",
            "",
            "Public Library of Science",
            "Public Library of Science",
            "1100; 1300",
        ]
    )
    ws.append(
        [
            "21100275443",
            "Frontiers in Oncology",
            "",
            "2234943X",
            "Active",
            "2011-2026",
            "",
            "",
            "Gold",
            "Journal",
            "",
            "",
            "",
            "",
            "Frontiers Media",
            "Frontiers Media",
            "2700",
        ]
    )
    ws.append(
        [
            "",
            "No Identifier Journal",
            "",
            "",
            "Active",
            "2020-2026",
            "",
            "",
            "",
            "Journal",
            "",
            "",
            "",
            "",
            "Unknown",
            "Unknown",
            "",
        ]
    )

    wb.save(path)


def _local_workbook_path() -> Path:
    return Path(__file__).resolve().parent / f"_registry_test_{uuid.uuid4().hex}.xlsx"


def test_load_registry_workbook_uses_feb_2026_sheets():
    workbook_path = _local_workbook_path()
    try:
        _build_registry_workbook(workbook_path)
        load_registry_workbook.cache_clear()

        registry = load_registry_workbook(str(workbook_path))

        assert any(entry.registry_category == "active_indexed_journal" for entry in registry.entries)
        assert any(entry.registry_category == "accepted_title" for entry in registry.entries)
        assert any(entry.registry_category == "discontinued_title" for entry in registry.entries)
        assert any(entry.registry_category == "conference_proceeding" for entry in registry.entries)
        assert "remote sensing journal" in registry.by_related_title
        assert "20724292" in registry.by_issn
    finally:
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)


def test_ingest_scopus_excel_preserves_primary_registry_metadata():
    workbook_path = _local_workbook_path()
    try:
        _build_registry_workbook(workbook_path)
        load_registry_workbook.cache_clear()
        db = _FakeDB()

        inserted = ingest_scopus_excel(db, str(workbook_path))

        assert inserted == 4
        assert db.commit_count == 1

        sources_by_title = {source.source_title: source for source in db.sources}

        active = sources_by_title["Remote Sensing"]
        assert active.source_record_id == "1001"
        assert active.source_type == "Journal"
        assert active.active_status == "Active"
        assert active.open_access_status == "Gold"
        assert active.asjc_codes == "1900; 2300"

        accepted = sources_by_title["Future Energy"]
        assert accepted.is_accepted is True
        assert accepted.active_status == "Accepted"

        discontinued = sources_by_title["Old Journal"]
        assert discontinued.is_discontinued is True
        assert discontinued.discontinued_due_to_quality is True

        conference = sources_by_title["Conference Series A"]
        assert conference.source_type == "Conference Proceeding"
        assert conference.active_status == "Conference"
    finally:
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)


def test_ingest_scopus_excel_imports_eissn_only_main_sheet_rows():
    workbook_path = _local_workbook_path()
    try:
        _build_eissn_only_registry_workbook(workbook_path)
        load_registry_workbook.cache_clear()
        db = _FakeDB()

        inserted = ingest_scopus_excel(db, str(workbook_path))

        assert inserted == 3
        assert db.commit_count == 1

        sources_by_title = {source.normalized_source_title: source for source in db.sources}
        expected = {
            "ieee access": "21693536",
            "plos one": "19326203",
            "frontiers in oncology": "2234943X",
        }
        assert set(sources_by_title) == set(expected)
        for normalized_title, eissn in expected.items():
            source = sources_by_title[normalized_title]
            assert source.issn is None
            assert source.eissn == eissn
    finally:
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)


def _sqlite_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ScopusSource.__table__.create(bind=engine)
    ScopusImportJob.__table__.create(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def test_ingest_scopus_excel_with_progress_commits_each_batch():
    workbook_path = _local_workbook_path()
    SessionLocal = _sqlite_session_factory()
    progress_events = []
    counts_after_batch_commits = []

    def _progress(**updates):
        progress_events.append(updates)
        processed_rows = updates.get("processed_rows") or 0
        total_rows = updates.get("total_rows") or 0
        if processed_rows and processed_rows < total_rows:
            db = SessionLocal()
            try:
                counts_after_batch_commits.append(
                    db.execute(text("SELECT COUNT(*) FROM scopus_sources")).scalar()
                )
            finally:
                db.close()

    try:
        _build_registry_workbook_with_main_rows(workbook_path, row_count=5)
        load_registry_workbook.cache_clear()

        inserted = ingest_scopus_excel_with_progress(
            SessionLocal,
            str(workbook_path),
            batch_size=2,
            progress_callback=_progress,
        )

        db = SessionLocal()
        try:
            count = db.execute(text("SELECT COUNT(*) FROM scopus_sources")).scalar()
        finally:
            db.close()

        assert inserted == 5
        assert count == 5
        assert counts_after_batch_commits[0] == 2
        assert progress_events[-1]["status"] == "completed"
        assert progress_events[-1]["processed_rows"] == 5
        assert progress_events[-1]["inserted"] == 5
    finally:
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)


def test_ingest_scopus_excel_with_progress_persists_eissn_only_rows():
    workbook_path = _local_workbook_path()
    SessionLocal = _sqlite_session_factory()

    try:
        _build_eissn_only_registry_workbook(workbook_path)
        load_registry_workbook.cache_clear()

        inserted = ingest_scopus_excel_with_progress(
            SessionLocal,
            str(workbook_path),
            batch_size=2,
        )

        db = SessionLocal()
        try:
            rows = db.query(ScopusSource).all()
        finally:
            db.close()

        sources_by_title = {source.normalized_source_title: source for source in rows}
        assert inserted == 3
        assert set(sources_by_title) == {
            "ieee access",
            "plos one",
            "frontiers in oncology",
        }
        assert sources_by_title["ieee access"].eissn == "21693536"
        assert sources_by_title["plos one"].eissn == "19326203"
        assert sources_by_title["frontiers in oncology"].eissn == "2234943X"
        assert {source.issn for source in rows} == {None}
    finally:
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)


def test_upload_scopus_status_reports_completed_progress(monkeypatch):
    workbook_path = _local_workbook_path()
    SessionLocal = _sqlite_session_factory()

    def _override_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[require_admin] = lambda: SimpleNamespace(id=1, role="admin")
    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(admin_api, "IMPORT_DIR", workbook_path.parent)
    monkeypatch.setattr(admin_api, "SessionLocal", SessionLocal)

    try:
        _build_registry_workbook_with_main_rows(workbook_path, row_count=3)
        load_registry_workbook.cache_clear()
        client = TestClient(app)
        with workbook_path.open("rb") as upload:
            response = client.post(
                "/admin/upload-scopus",
                files={
                    "file": (
                        "small_scopus.xlsx",
                        upload,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status_response = client.get(f"/admin/scopus-imports/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["id"] == job_id
        assert status["job_id"] == job_id
        assert status["status"] == "completed"
        assert status["total_rows"] == 3
        assert status["processed_rows"] == 3
        assert status["inserted"] == 3

        db = SessionLocal()
        try:
            count = db.execute(text("SELECT COUNT(*) FROM scopus_sources")).scalar()
            persisted_job = db.get(ScopusImportJob, job_id)
            persisted_job_status = persisted_job.status if persisted_job else None
        finally:
            db.close()
        assert count == 3
        assert persisted_job is not None
        assert persisted_job_status == "completed"

        list_response = client.get("/admin/scopus-imports")
        assert list_response.status_code == 200
        assert list_response.json()["jobs"][0]["job_id"] == job_id
    finally:
        app.dependency_overrides.clear()
        load_registry_workbook.cache_clear()
        workbook_path.unlink(missing_ok=True)
        for saved_upload in workbook_path.parent.glob("*_small_scopus.xlsx"):
            saved_upload.unlink(missing_ok=True)
