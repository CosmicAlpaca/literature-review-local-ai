from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Callable, Iterable

from openpyxl import load_workbook
from sqlalchemy.orm import Session

from app.db.models.scopus_source import ScopusSource
from app.utils.normalize import normalize_text

logger = logging.getLogger(__name__)

PRIMARY_WORKBOOK = Path("data/ext_list_Feb_2026.xlsx")
LEGACY_WORKBOOK = Path("data/ext_list_January_2025.xlsx")

MAIN_SHEET = "Scopus Sources Feb. 2026"
LEGACY_MAIN_SHEET = "Scopus Sources Jan. 2025"
ACCEPTED_SHEET = "Accepted Titles Feb. 2026"
DISCONTINUED_SHEET = "Discontinued Titles Feb. 2026"
CONFERENCE_SHEET = "Serial Conf. Proc. with Profile"

ProgressCallback = Callable[..., None]


@dataclass(frozen=True)
class RegistryEntry:
    source_record_id: str | None
    source_title: str
    normalized_source_title: str
    issn: str | None
    eissn: str | None
    active_status: str | None
    coverage: str | None
    source_type: str | None
    open_access_status: str | None
    publisher: str | None
    main_publisher: str | None
    asjc_codes: str | None
    top_level_fields: tuple[str, ...]
    related_titles: tuple[str, ...]
    medline_sourced: bool
    is_accepted: bool
    is_discontinued: bool
    discontinued_due_to_quality: bool
    registry_category: str


@dataclass(frozen=True)
class RegistryWorkbookData:
    entries: tuple[RegistryEntry, ...]
    by_record_id: dict[str, RegistryEntry]
    by_normalized_title: dict[str, tuple[RegistryEntry, ...]]
    by_issn: dict[str, tuple[RegistryEntry, ...]]
    by_eissn: dict[str, tuple[RegistryEntry, ...]]
    by_related_title: dict[str, tuple[RegistryEntry, ...]]
    by_all_identifiers: dict[str, tuple[RegistryEntry, ...]]  # ISSN + EISSN combined
    by_publisher: dict[str, tuple[RegistryEntry, ...]]  # normalized publisher name


def get_primary_registry_workbook_path() -> Path:
    if PRIMARY_WORKBOOK.exists():
        return PRIMARY_WORKBOOK
    return LEGACY_WORKBOOK


def _clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_identifier(value) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    normalized = "".join(ch for ch in text.upper() if ch.isalnum())
    return normalized or None


def _boolish(value) -> bool:
    text = (_clean_text(value) or "").lower()
    if not text:
        return False
    return text not in {"no", "n", "false", "0"}


def _coverage_from_discontinued_row(row: dict[str, object]) -> str | None:
    year = _clean_text(row.get("Year"))
    volume = _clean_text(row.get("Volume"))
    issue = _clean_text(row.get("Issue"))
    page_range = _clean_text(row.get("Page Range"))

    parts = []
    if year:
        parts.append(str(year))
    if volume:
        parts.append(f"vol {volume}")
    if issue:
        parts.append(f"issue {issue}")
    if page_range:
        parts.append(f"pp {page_range}")
    return ", ".join(parts) or None


def _sheet_rows(
    workbook_path: str,
    sheet_name: str,
    header_row: int,
    required_columns: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    wb = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            return []
        ws = wb[sheet_name]
        rows_iter = ws.iter_rows(values_only=True)
        headers: list[str] | None = None
        records: list[dict[str, object]] = []

        for row_number, row in enumerate(rows_iter, start=1):
            if row_number < header_row:
                continue
            if row_number == header_row:
                headers = [str(cell).strip() if cell is not None else "" for cell in row]
                missing = [column for column in required_columns if column not in headers]
                if missing:
                    raise ValueError(
                        f"Sheet '{sheet_name}' is missing required columns: "
                        f"{', '.join(missing)}"
                    )
                continue

            if headers is None:
                break

            record = {
                headers[idx]: row[idx] if idx < len(row) else None
                for idx in range(len(headers))
                if headers[idx]
            }
            if any(value not in (None, "") for value in record.values()):
                records.append(record)
    finally:
        wb.close()

    return records


def _top_level_fields(row: dict[str, object]) -> tuple[str, ...]:
    fields = []
    for key, value in row.items():
        if key.startswith("Top level:") and _clean_text(value):
            fields.append(str(value).strip())
    return tuple(fields)


def _merge_entries(existing: RegistryEntry, incoming: RegistryEntry) -> RegistryEntry:
    related_titles = tuple(
        dict.fromkeys([*existing.related_titles, *incoming.related_titles]).keys()
    )
    top_level_fields = tuple(
        dict.fromkeys([*existing.top_level_fields, *incoming.top_level_fields]).keys()
    )

    return RegistryEntry(
        source_record_id=existing.source_record_id or incoming.source_record_id,
        source_title=existing.source_title or incoming.source_title,
        normalized_source_title=existing.normalized_source_title or incoming.normalized_source_title,
        issn=existing.issn or incoming.issn,
        eissn=existing.eissn or incoming.eissn,
        active_status=incoming.active_status or existing.active_status,
        coverage=incoming.coverage or existing.coverage,
        source_type=incoming.source_type or existing.source_type,
        open_access_status=incoming.open_access_status or existing.open_access_status,
        publisher=incoming.publisher or existing.publisher,
        main_publisher=incoming.main_publisher or existing.main_publisher,
        asjc_codes=incoming.asjc_codes or existing.asjc_codes,
        top_level_fields=top_level_fields,
        related_titles=related_titles,
        medline_sourced=existing.medline_sourced or incoming.medline_sourced,
        is_accepted=existing.is_accepted or incoming.is_accepted,
        is_discontinued=existing.is_discontinued or incoming.is_discontinued,
        discontinued_due_to_quality=(
            existing.discontinued_due_to_quality or incoming.discontinued_due_to_quality
        ),
        registry_category=incoming.registry_category or existing.registry_category,
    )


def _entry_key(entry: RegistryEntry) -> str:
    if entry.source_record_id:
        return f"id:{entry.source_record_id}"
    return (
        f"title:{entry.normalized_source_title}|"
        f"{entry.issn or ''}|{entry.eissn or ''}|{entry.source_type or ''}"
    )


def _has_required_source_identity(entry: RegistryEntry) -> bool:
    return bool(
        entry.source_title
        and entry.normalized_source_title
        and (entry.source_record_id or entry.issn or entry.eissn)
    )


_PUBLISHER_SUFFIXES = {
    "publishing", "publishers", "publications", "press",
    "inc", "inc.", "ltd", "ltd.", "llc", "group", "co",
    "gmbh", "b.v.", "bv", "ag", "sa", "s.a.",
}


def _normalize_publisher(name: str | None) -> str:
    """Normalize publisher name for fuzzy matching (strip suffixes, lowercase)."""
    if not name:
        return ""
    tokens = name.strip().lower().split()
    cleaned = [t for t in tokens if t.rstrip(".,") not in _PUBLISHER_SUFFIXES
               and t.replace(".", "").rstrip(",") not in _PUBLISHER_SUFFIXES]
    return " ".join(cleaned).strip()


def _build_entry_maps(entries: Iterable[RegistryEntry]) -> RegistryWorkbookData:
    merged: dict[str, RegistryEntry] = {}
    for entry in entries:
        if not _has_required_source_identity(entry):
            continue
        key = _entry_key(entry)
        merged[key] = _merge_entries(merged[key], entry) if key in merged else entry

    final_entries = tuple(merged.values())

    by_record_id: dict[str, RegistryEntry] = {}
    by_normalized_title: dict[str, list[RegistryEntry]] = {}
    by_issn: dict[str, list[RegistryEntry]] = {}
    by_eissn: dict[str, list[RegistryEntry]] = {}
    by_related_title: dict[str, list[RegistryEntry]] = {}
    by_all_identifiers: dict[str, list[RegistryEntry]] = {}
    by_publisher: dict[str, list[RegistryEntry]] = {}

    for entry in final_entries:
        if entry.source_record_id:
            by_record_id[entry.source_record_id] = entry

        by_normalized_title.setdefault(entry.normalized_source_title, []).append(entry)

        if entry.issn:
            by_issn.setdefault(entry.issn, []).append(entry)
            by_all_identifiers.setdefault(entry.issn, []).append(entry)
        if entry.eissn:
            by_eissn.setdefault(entry.eissn, []).append(entry)
            by_all_identifiers.setdefault(entry.eissn, []).append(entry)

        # Normalized publisher index
        pub_name = _normalize_publisher(entry.publisher or entry.main_publisher)
        if pub_name:
            by_publisher.setdefault(pub_name, []).append(entry)

        for related_title in entry.related_titles:
            normalized = normalize_text(related_title)
            if normalized:
                by_related_title.setdefault(normalized, []).append(entry)

    return RegistryWorkbookData(
        entries=final_entries,
        by_record_id=by_record_id,
        by_normalized_title={key: tuple(value) for key, value in by_normalized_title.items()},
        by_issn={key: tuple(value) for key, value in by_issn.items()},
        by_eissn={key: tuple(value) for key, value in by_eissn.items()},
        by_related_title={key: tuple(value) for key, value in by_related_title.items()},
        by_all_identifiers={key: tuple(value) for key, value in by_all_identifiers.items()},
        by_publisher={key: tuple(value) for key, value in by_publisher.items()},
    )


def _main_entries(workbook_path: str) -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    rows = _sheet_rows(
        workbook_path,
        MAIN_SHEET,
        header_row=1,
        required_columns=("Source Title",),
    )
    if not rows:
        rows = _sheet_rows(
            workbook_path,
            LEGACY_MAIN_SHEET,
            header_row=1,
            required_columns=("Source Title",),
        )

    for row in rows:
        title = _clean_text(row.get("Source Title"))
        if not title:
            continue

        source_type = _clean_text(row.get("Source Type"))
        active_status = _clean_text(row.get("Active or Inactive"))
        is_discontinued = _boolish(row.get("Titles Discontinued by Scopus"))
        registry_category = "active_indexed_journal"
        if source_type and "conference" in source_type.lower():
            registry_category = "conference_proceeding"
        elif source_type and source_type.lower() != "journal":
            registry_category = "non_journal_source"
        elif active_status and active_status.lower() != "active":
            registry_category = "inactive_indexed_source"

        related_titles = tuple(
            value
            for value in (
                _clean_text(row.get("Related Title 1")),
                _clean_text(row.get("Other Related Title 2")),
                _clean_text(row.get("Other Related Title 3")),
                _clean_text(row.get("Other Related Title 4")),
            )
            if value
        )

        entries.append(
            RegistryEntry(
                source_record_id=_clean_text(row.get("Sourcerecord ID")),
                source_title=title,
                normalized_source_title=normalize_text(title),
                issn=_normalize_identifier(row.get("ISSN")),
                eissn=_normalize_identifier(row.get("EISSN")),
                active_status=active_status,
                coverage=_clean_text(row.get("Coverage")),
                source_type=source_type,
                open_access_status=_clean_text(row.get("Open Access Status")),
                publisher=_clean_text(row.get("Publisher")),
                main_publisher=_clean_text(
                    row.get("Publisher Imprints Grouped to Main Publisher")
                ),
                asjc_codes=_clean_text(row.get("All Science Journal Classification Codes (ASJC)")),
                top_level_fields=_top_level_fields(row),
                related_titles=related_titles,
                medline_sourced=_boolish(
                    row.get("Medline-sourced Title? (See additional details under separate tab.)")
                ),
                is_accepted=False,
                is_discontinued=is_discontinued,
                discontinued_due_to_quality=is_discontinued,
                registry_category=registry_category,
            )
        )
    return entries


def _accepted_entries(workbook_path: str) -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    for row in _sheet_rows(
        workbook_path,
        ACCEPTED_SHEET,
        header_row=3,
        required_columns=("Source Title",),
    ):
        title = _clean_text(row.get("Source Title"))
        if not title:
            continue

        acceptance_date = _clean_text(row.get("Date of Acceptance"))
        entries.append(
            RegistryEntry(
                source_record_id=None,
                source_title=title,
                normalized_source_title=normalize_text(title),
                issn=_normalize_identifier(row.get("ISSN")),
                eissn=_normalize_identifier(row.get("EISSN")),
                active_status="Accepted",
                coverage=acceptance_date,
                source_type="Journal",
                open_access_status=None,
                publisher=_clean_text(row.get("Publisher")),
                main_publisher=_clean_text(row.get("Publisher")),
                asjc_codes=None,
                top_level_fields=(),
                related_titles=(),
                medline_sourced=False,
                is_accepted=True,
                is_discontinued=False,
                discontinued_due_to_quality=False,
                registry_category="accepted_title",
            )
        )
    return entries


def _discontinued_entries(workbook_path: str) -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    for row in _sheet_rows(
        workbook_path,
        DISCONTINUED_SHEET,
        header_row=2,
        required_columns=("Source Title (newly added titles are highlighted in red)",),
    ):
        title = _clean_text(row.get("Source Title (newly added titles are highlighted in red)"))
        if not title:
            continue

        entries.append(
            RegistryEntry(
                source_record_id=_clean_text(row.get("Sourcerecord ID")),
                source_title=title,
                normalized_source_title=normalize_text(title),
                issn=_normalize_identifier(row.get("ISSN")),
                eissn=_normalize_identifier(row.get("EISSN")),
                active_status="Discontinued",
                coverage=_coverage_from_discontinued_row(row),
                source_type="Journal",
                open_access_status=None,
                publisher=_clean_text(row.get("Publisher")),
                main_publisher=_clean_text(row.get("Publisher")),
                asjc_codes=None,
                top_level_fields=(),
                related_titles=(),
                medline_sourced=False,
                is_accepted=False,
                is_discontinued=True,
                discontinued_due_to_quality=True,
                registry_category="discontinued_title",
            )
        )
    return entries


def _conference_entries(workbook_path: str) -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    for row in _sheet_rows(
        workbook_path,
        CONFERENCE_SHEET,
        header_row=1,
        required_columns=("Source Title",),
    ):
        title = _clean_text(row.get("Source Title"))
        if not title:
            continue

        entries.append(
            RegistryEntry(
                source_record_id=_clean_text(row.get("Sourcerecord ID")),
                source_title=title,
                normalized_source_title=normalize_text(title),
                issn=_normalize_identifier(row.get("ISSN")),
                eissn=_normalize_identifier(row.get("EISSN")),
                active_status="Conference",
                coverage=_clean_text(row.get("Coverage")),
                source_type="Conference Proceeding",
                open_access_status=None,
                publisher=None,
                main_publisher=None,
                asjc_codes=_clean_text(row.get("All Science Journal Classification Codes (ASJC)")),
                top_level_fields=(),
                related_titles=(),
                medline_sourced=False,
                is_accepted=False,
                is_discontinued=_boolish(row.get("Titles Discontinued by Scopus")),
                discontinued_due_to_quality=_boolish(row.get("Titles Discontinued by Scopus")),
                registry_category="conference_proceeding",
            )
        )
    return entries


@lru_cache(maxsize=4)
def load_registry_workbook(workbook_path: str | None = None) -> RegistryWorkbookData:
    path = Path(workbook_path) if workbook_path else get_primary_registry_workbook_path()
    workbook_path_str = str(path)

    entries = [
        *_main_entries(workbook_path_str),
        *_accepted_entries(workbook_path_str),
        *_discontinued_entries(workbook_path_str),
        *_conference_entries(workbook_path_str),
    ]
    return _build_entry_maps(entries)


def _find_existing_source(existing_sources: list[ScopusSource], entry: RegistryEntry) -> ScopusSource | None:
    if entry.source_record_id:
        for source in existing_sources:
            if source.source_record_id == entry.source_record_id:
                return source

    entry_identifiers = {value for value in (entry.issn, entry.eissn) if value}
    for source in existing_sources:
        source_identifiers = {
            value
            for value in (
                _normalize_identifier(source.issn),
                _normalize_identifier(source.eissn),
            )
            if value
        }
        if entry_identifiers & source_identifiers:
            return source

    for source in existing_sources:
        if source.normalized_source_title == entry.normalized_source_title:
            return source

    return None


def _entry_to_source(entry: RegistryEntry) -> ScopusSource:
    return ScopusSource(
        source_record_id=entry.source_record_id,
        source_title=entry.source_title,
        normalized_source_title=entry.normalized_source_title,
        issn=entry.issn,
        eissn=entry.eissn,
        source_type=entry.source_type,
        publisher=entry.main_publisher or entry.publisher,
        active_status=entry.active_status,
        asjc_codes=entry.asjc_codes,
        open_access_status=entry.open_access_status,
        discontinued_due_to_quality=entry.discontinued_due_to_quality,
        coverage=entry.coverage,
        is_accepted=entry.is_accepted,
        is_discontinued=entry.is_discontinued,
    )


def _apply_entry_to_source(existing: ScopusSource, entry: RegistryEntry) -> None:
    existing.source_record_id = existing.source_record_id or entry.source_record_id
    existing.source_title = existing.source_title or entry.source_title
    existing.normalized_source_title = (
        existing.normalized_source_title or entry.normalized_source_title
    )
    existing.issn = existing.issn or entry.issn
    existing.eissn = existing.eissn or entry.eissn
    existing.source_type = entry.source_type or existing.source_type
    existing.publisher = entry.main_publisher or entry.publisher or existing.publisher
    existing.active_status = entry.active_status or existing.active_status
    existing.asjc_codes = entry.asjc_codes or existing.asjc_codes
    existing.open_access_status = entry.open_access_status or existing.open_access_status
    existing.coverage = entry.coverage or existing.coverage
    existing.is_accepted = existing.is_accepted or entry.is_accepted
    existing.is_discontinued = existing.is_discontinued or entry.is_discontinued
    existing.discontinued_due_to_quality = (
        existing.discontinued_due_to_quality or entry.discontinued_due_to_quality
    )


def _source_lookup_keys(source: ScopusSource) -> tuple[tuple[str, str], ...]:
    keys = []
    if source.source_record_id:
        keys.append(("source_record_id", source.source_record_id))
    if source.issn:
        normalized = _normalize_identifier(source.issn) or source.issn
        keys.append(("issn", normalized))
        keys.append(("identifier", normalized))
    if source.eissn:
        normalized = _normalize_identifier(source.eissn) or source.eissn
        keys.append(("eissn", normalized))
        keys.append(("identifier", normalized))
    if source.normalized_source_title:
        keys.append(("normalized_source_title", source.normalized_source_title))
    return tuple(keys)


def _entry_lookup_keys(entry: RegistryEntry) -> tuple[tuple[str, str], ...]:
    keys = []
    if entry.source_record_id:
        keys.append(("source_record_id", entry.source_record_id))
    if entry.issn:
        keys.append(("issn", entry.issn))
        keys.append(("identifier", entry.issn))
    if entry.eissn:
        keys.append(("eissn", entry.eissn))
        keys.append(("identifier", entry.eissn))
    if entry.normalized_source_title:
        keys.append(("normalized_source_title", entry.normalized_source_title))
    return tuple(keys)


def _find_existing_source_in_lookup(
    lookup: dict[tuple[str, str], ScopusSource],
    entry: RegistryEntry,
) -> ScopusSource | None:
    for key in _entry_lookup_keys(entry):
        existing = lookup.get(key)
        if existing is not None:
            return existing
    return None


def _index_source(lookup: dict[tuple[str, str], ScopusSource], source: ScopusSource) -> None:
    for key in _source_lookup_keys(source):
        lookup[key] = source


def _existing_sources_for_batch(db: Session, entries: tuple[RegistryEntry, ...]) -> list[ScopusSource]:
    source_record_ids = {entry.source_record_id for entry in entries if entry.source_record_id}
    issns = {entry.issn for entry in entries if entry.issn}
    eissns = {entry.eissn for entry in entries if entry.eissn}
    identifiers = issns | eissns
    normalized_titles = {
        entry.normalized_source_title for entry in entries if entry.normalized_source_title
    }

    candidates: dict[int, ScopusSource] = {}
    filters = (
        (ScopusSource.source_record_id, source_record_ids),
        (ScopusSource.issn, identifiers),
        (ScopusSource.eissn, identifiers),
        (ScopusSource.normalized_source_title, normalized_titles),
    )
    for column, values in filters:
        if not values:
            continue
        for source in db.query(ScopusSource).filter(column.in_(values)).all():
            candidates[source.id or id(source)] = source

    return list(candidates.values())


def _iter_batches(entries: tuple[RegistryEntry, ...], batch_size: int):
    for offset in range(0, len(entries), batch_size):
        yield offset, entries[offset : offset + batch_size]


def _report_progress(progress_callback: ProgressCallback | None, **updates: object) -> None:
    if progress_callback is not None:
        progress_callback(**updates)


def ingest_scopus_excel(db: Session, file_path: str | None = None, batch_size: int = 1000):
    workbook = load_registry_workbook(file_path)
    inserted = 0
    entries = workbook.entries

    for _, batch in _iter_batches(entries, batch_size):
        existing_sources = _existing_sources_for_batch(db, batch)
        lookup: dict[tuple[str, str], ScopusSource] = {}
        for source in existing_sources:
            _index_source(lookup, source)

        for entry in batch:
            existing = _find_existing_source_in_lookup(lookup, entry)
            if existing is None:
                existing = _entry_to_source(entry)
                db.add(existing)
                inserted += 1
            else:
                _apply_entry_to_source(existing, entry)

            for key in _entry_lookup_keys(entry):
                lookup[key] = existing

    db.commit()
    return inserted


def ingest_scopus_excel_with_progress(
    session_factory: Callable[[], Session],
    file_path: str | None = None,
    batch_size: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """
    Import Scopus registry rows without holding a database transaction during
    Excel parsing, and commit each database batch independently.
    """
    logger.info("Excel read started for Scopus import: %s", file_path)
    _report_progress(
        progress_callback,
        status="running",
        message="Reading Scopus Excel file.",
        processed_rows=0,
        inserted=0,
    )
    workbook = load_registry_workbook(file_path)
    entries = workbook.entries
    total_rows = len(entries)
    logger.info("Excel read completed for Scopus import: %s rows=%s", file_path, total_rows)
    _report_progress(
        progress_callback,
        status="running",
        message="Scopus Excel file parsed.",
        total_rows=total_rows,
        processed_rows=0,
        inserted=0,
    )

    inserted = 0
    processed_rows = 0

    for batch_number, (offset, batch) in enumerate(_iter_batches(entries, batch_size), start=1):
        db = session_factory()
        try:
            logger.info(
                "DB batch insert started for Scopus import: batch=%s offset=%s size=%s",
                batch_number,
                offset,
                len(batch),
            )
            _report_progress(
                progress_callback,
                status="running",
                message=f"Importing rows {offset + 1}-{offset + len(batch)} of {total_rows}.",
                total_rows=total_rows,
                processed_rows=processed_rows,
                inserted=inserted,
            )

            existing_sources = _existing_sources_for_batch(db, batch)
            lookup: dict[tuple[str, str], ScopusSource] = {}
            for source in existing_sources:
                _index_source(lookup, source)

            batch_inserted = 0
            for entry in batch:
                existing = _find_existing_source_in_lookup(lookup, entry)
                if existing is None:
                    existing = _entry_to_source(entry)
                    db.add(existing)
                    batch_inserted += 1
                else:
                    _apply_entry_to_source(existing, entry)

                for key in _entry_lookup_keys(entry):
                    lookup[key] = existing

            db.commit()
            inserted += batch_inserted
            processed_rows += len(batch)
            logger.info(
                "DB batch committed for Scopus import: batch=%s processed=%s/%s inserted=%s",
                batch_number,
                processed_rows,
                total_rows,
                inserted,
            )
            _report_progress(
                progress_callback,
                status="running",
                message=f"Imported {processed_rows} of {total_rows} rows.",
                total_rows=total_rows,
                processed_rows=processed_rows,
                inserted=inserted,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "DB batch failed for Scopus import: batch=%s offset=%s size=%s",
                batch_number,
                offset,
                len(batch),
            )
            raise
        finally:
            db.close()

    logger.info("Import completed for Scopus import: rows=%s inserted=%s", total_rows, inserted)
    _report_progress(
        progress_callback,
        status="completed",
        message="Scopus import completed.",
        total_rows=total_rows,
        processed_rows=processed_rows,
        inserted=inserted,
    )
    return inserted
