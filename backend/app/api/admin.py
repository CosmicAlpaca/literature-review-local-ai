from __future__ import annotations

import shutil
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, require_admin
from app.db.models.scopus_import_job import ScopusImportJob
from app.db.session import SessionLocal
from app.services.source_registry import ingest_scopus_excel_with_progress

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)

IMPORT_DIR = Path("data/imports")
IMPORT_DIR.mkdir(parents=True, exist_ok=True)

def _utc_now() -> datetime:
    return datetime.utcnow()


def _ensure_import_jobs_table(db: Session) -> None:
    ScopusImportJob.__table__.create(bind=db.get_bind(), checkfirst=True)


def _job_to_dict(job: ScopusImportJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "job_id": job.id,
        "status": job.status,
        "message": job.message,
        "filename": job.filename,
        "saved_path": job.saved_path,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
        "inserted": job.inserted,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _set_job_status(job_id: str, **updates: Any) -> None:
    db = SessionLocal()
    try:
        _ensure_import_jobs_table(db)
        job = db.get(ScopusImportJob, job_id)
        if not job:
            logger.warning("Scopus import job not found while updating status: %s", job_id)
            return

        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = _utc_now()
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to update Scopus import job status: job_id=%s", job_id)
        raise
    finally:
        db.close()


def _run_scopus_import_job(job_id: str, file_path: str) -> None:
    """
    Runs outside the request lifecycle.

    Important:
    - Do not reuse the request-scoped db session here.
    - Do not create a DB session until Excel has been read and a DB batch is ready.
    """
    try:
        _set_job_status(
            job_id,
            status="running",
            message="Scopus import is running.",
            started_at=_utc_now(),
            total_rows=None,
            processed_rows=0,
            inserted=0,
        )

        def _progress(**updates: Any) -> None:
            _set_job_status(job_id, **updates)

        inserted = ingest_scopus_excel_with_progress(
            SessionLocal,
            file_path,
            batch_size=1000,
            progress_callback=_progress,
        )

        _set_job_status(
            job_id,
            status="completed",
            message="Scopus import completed.",
            inserted=inserted,
            finished_at=_utc_now(),
        )
        logger.info("Scopus import completed: job_id=%s inserted=%s", job_id, inserted)

    except Exception as exc:
        logger.exception("Scopus import failed: job_id=%s file_path=%s", job_id, file_path)
        _set_job_status(
            job_id,
            status="failed",
            message=str(exc),
            finished_at=_utc_now(),
        )


@router.post("/upload-scopus")
def upload_scopus(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    """
    Upload a Scopus/Source Registry Excel file and start ingestion in background.

    Returns immediately with job_id so the client does not hit Railway first-byte timeout.
    """
    filename = file.filename or "scopus_sources.xlsx"

    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Only Excel files are supported (.xlsx, .xls).",
        )

    job_id = str(uuid.uuid4())
    safe_filename = filename.replace("\\", "_").replace("/", "_")
    saved_path = IMPORT_DIR / f"{job_id}_{safe_filename}"

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info("Scopus import file saved: job_id=%s path=%s", job_id, saved_path)

    _ensure_import_jobs_table(db)
    created_at = _utc_now()
    db.add(
        ScopusImportJob(
            id=job_id,
            status="queued",
            message="Scopus import queued.",
            filename=filename,
            saved_path=str(saved_path),
            total_rows=None,
            processed_rows=0,
            inserted=0,
            created_at=created_at,
            updated_at=created_at,
            started_at=None,
            finished_at=None,
        )
    )
    db.commit()

    background_tasks.add_task(_run_scopus_import_job, job_id, str(saved_path))

    return {
        "message": "Scopus import started.",
        "job_id": job_id,
        "status_url": f"/admin/scopus-imports/{job_id}",
    }


@router.get("/scopus-imports/{job_id}")
def get_scopus_import_status(
    job_id: str,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    _ensure_import_jobs_table(db)
    job = db.get(ScopusImportJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Import job not found.")

    return _job_to_dict(job)


@router.get("/scopus-imports")
def list_scopus_imports(
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    _ensure_import_jobs_table(db)
    jobs = (
        db.query(ScopusImportJob)
        .order_by(ScopusImportJob.created_at.desc())
        .all()
    )
    return {
        "jobs": [_job_to_dict(job) for job in jobs],
    }


@router.get("/scopus-sources/status")
def get_scopus_sources_status(
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    count = db.execute(text("SELECT COUNT(*) FROM scopus_sources")).scalar()

    samples = db.execute(
        text(
            """
            SELECT *
            FROM scopus_sources
            LIMIT 5
            """
        )
    ).mappings().all()

    return {
        "source_count": count,
        "samples": [dict(row) for row in samples],
    }
