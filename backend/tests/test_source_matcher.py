from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.source_registry import RegistryEntry, RegistryWorkbookData, _normalize_publisher
from app.services.source_matcher import _extract_identifiers, resolve_scopus_match
from app.utils.normalize import normalize_text


class _FakeQuery:
    def __init__(self, results):
        self._results = results

    def all(self):
        return self._results


class _FakeDB:
    def __init__(self, sources):
        self._sources = sources

    def query(self, model):
        return _FakeQuery(self._sources)


def _empty_registry():
    return _registry_data(())


def _registry_data(entries: tuple[RegistryEntry, ...]) -> RegistryWorkbookData:
    by_record_id: dict[str, RegistryEntry] = {}
    by_normalized_title: dict[str, list[RegistryEntry]] = {}
    by_issn: dict[str, list[RegistryEntry]] = {}
    by_eissn: dict[str, list[RegistryEntry]] = {}
    by_related_title: dict[str, list[RegistryEntry]] = {}
    by_all_identifiers: dict[str, list[RegistryEntry]] = {}
    by_publisher: dict[str, list[RegistryEntry]] = {}

    for entry in entries:
        if entry.source_record_id:
            by_record_id[entry.source_record_id] = entry
        if entry.normalized_source_title:
            by_normalized_title.setdefault(entry.normalized_source_title, []).append(entry)
        if entry.issn:
            by_issn.setdefault(entry.issn, []).append(entry)
            by_all_identifiers.setdefault(entry.issn, []).append(entry)
        if entry.eissn:
            by_eissn.setdefault(entry.eissn, []).append(entry)
            by_all_identifiers.setdefault(entry.eissn, []).append(entry)
        publisher = _normalize_publisher(entry.publisher or entry.main_publisher)
        if publisher:
            by_publisher.setdefault(publisher, []).append(entry)
        for related_title in entry.related_titles:
            normalized = normalize_text(related_title)
            if normalized:
                by_related_title.setdefault(normalized, []).append(entry)

    return RegistryWorkbookData(
        entries=entries,
        by_record_id=by_record_id,
        by_normalized_title={
            key: tuple(value) for key, value in by_normalized_title.items()
        },
        by_issn={key: tuple(value) for key, value in by_issn.items()},
        by_eissn={key: tuple(value) for key, value in by_eissn.items()},
        by_related_title={
            key: tuple(value) for key, value in by_related_title.items()
        },
        by_all_identifiers={
            key: tuple(value) for key, value in by_all_identifiers.items()
        },
        by_publisher={key: tuple(value) for key, value in by_publisher.items()},
    )


def test_source_matcher_exact_normalized_title_match(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=1,
                source_title="IEEE Access",
                normalized_source_title="ieee access",
                issn=None,
                eissn=None,
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(db, "IEEE Access")

    assert result.source is not None
    assert result.source.id == 1
    assert result.reason == "exact_normalized_title"
    assert result.confidence >= 0.99


def test_source_matcher_identifier_match_has_priority(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=2,
                source_title="Remote Sensing",
                normalized_source_title="remote sensing",
                issn="2072-4292",
                eissn=None,
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(
        db,
        "Remote Sensing Journal",
        raw_metadata={"issn": "2072-4292"},
    )

    assert result.source is not None
    assert result.source.id == 2
    assert result.reason.startswith("identifier_match:")
    assert result.confidence == 1.0


def test_source_matcher_identifier_match_handles_list_valued_issn(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=7,
                source_title="Remote Sensing",
                normalized_source_title="remote sensing",
                issn=None,
                eissn="2072-4293",
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(
        db,
        "Remote Sensing Journal",
        raw_metadata={
            "source_openalex_id": "https://openalex.org/S123",
            "issn_l": "2072-4292",
            "issn": ["2072-4292", "2072-4293"],
            "eissn": None,
            "source_type": "journal",
        },
    )

    assert result.source is not None
    assert result.source.id == 7
    assert result.reason == "identifier_match:issn_2"
    assert result.input_identifiers == {
        "issn": "20724292",
        "issn_2": "20724293",
    }


def test_source_matcher_matches_candidate_eissn_against_stored_issn(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=9,
                source_title="Cross Field Journal",
                normalized_source_title="cross field journal",
                issn="21693536",
                eissn=None,
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(
        db,
        "Different Title",
        raw_metadata={"eissn": "2169-3536"},
    )

    assert result.source is not None
    assert result.source.id == 9
    assert result.reason == "identifier_match:eissn"


def test_source_matcher_matches_candidate_issn_against_stored_eissn(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=10,
                source_title="IEEE Access",
                normalized_source_title="ieee access",
                issn=None,
                eissn="21693536",
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(
        db,
        "Different Title",
        raw_metadata={"issn": "2169-3536"},
    )

    assert result.source is not None
    assert result.source.id == 10
    assert result.reason == "identifier_match:issn"


def test_source_matcher_matches_eissn_only_scopus_sources_by_title_and_eissn(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=21,
                source_title="IEEE Access",
                normalized_source_title="ieee access",
                issn=None,
                eissn="21693536",
                source_record_id="21100374601",
            ),
            SimpleNamespace(
                id=22,
                source_title="PLOS ONE",
                normalized_source_title="plos one",
                issn=None,
                eissn="19326203",
                source_record_id="10600153309",
            ),
            SimpleNamespace(
                id=23,
                source_title="Frontiers in Oncology",
                normalized_source_title="frontiers in oncology",
                issn=None,
                eissn="2234943X",
                source_record_id="21100275443",
            ),
        ]
    )

    cases = [
        ("IEEE Access", "2169-3536", 21),
        ("PLOS ONE", "1932-6203", 22),
        ("Frontiers in Oncology", "2234-943X", 23),
    ]
    for title, eissn, source_id in cases:
        title_result = resolve_scopus_match(db, title)
        eissn_result = resolve_scopus_match(
            db,
            "Different upstream title",
            raw_metadata={"eissn": eissn},
        )

        assert title_result.source is not None
        assert title_result.source.id == source_id
        assert title_result.reason == "exact_normalized_title"
        assert eissn_result.source is not None
        assert eissn_result.source.id == source_id
        assert eissn_result.reason == "identifier_match:eissn"


def test_source_matcher_missing_issn_metadata_does_not_crash(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=8,
                source_title="IEEE Access",
                normalized_source_title="ieee access",
                issn=None,
                eissn=None,
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(db, "IEEE Access", raw_metadata={"publisher": "IEEE"})

    assert result.source is not None
    assert result.source.id == 8
    assert result.reason == "exact_normalized_title"
    assert result.input_identifiers is None


def test_extract_identifiers_ignores_malformed_non_string_issn_metadata():
    identifiers = _extract_identifiers(
        {
            "issn": {"unexpected": "shape"},
            "issn_l": object(),
            "eissn": {"nested", "set"},
            "external_ids": {"issn": None, "eissn": ["", {"bad": "value"}]},
        }
    )

    assert identifiers == {}


def test_source_matcher_high_confidence_fuzzy_fallback(monkeypatch):
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: _empty_registry())
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))
    db = _FakeDB(
        [
            SimpleNamespace(
                id=3,
                source_title="Nature Communications",
                normalized_source_title="nature communications",
                issn=None,
                eissn=None,
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(db, "Nature Communication")

    assert result.source is not None
    assert result.source.id == 3
    assert result.reason in {
        "exact_title_after_light_singularization",
        "high_confidence_fuzzy_title",
    }
    assert result.confidence >= 0.82


def test_source_matcher_related_title_lookup_uses_registry(monkeypatch):
    registry_entry = RegistryEntry(
        source_record_id="1001",
        source_title="Remote Sensing",
        normalized_source_title="remote sensing",
        issn="20724292",
        eissn=None,
        active_status="Active",
        coverage="2010-2026",
        source_type="Journal",
        open_access_status="Gold",
        publisher="MDPI",
        main_publisher="MDPI",
        asjc_codes="1900;2300",
        top_level_fields=("Physical Sciences",),
        related_titles=("Remote Sensing Journal",),
        medline_sourced=False,
        is_accepted=False,
        is_discontinued=False,
        discontinued_due_to_quality=False,
        registry_category="active_indexed_journal",
    )
    registry = _registry_data((registry_entry,))
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: registry)
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))

    db = _FakeDB(
        [
            SimpleNamespace(
                id=4,
                source_title="Remote Sensing",
                normalized_source_title="remote sensing",
                issn="20724292",
                eissn=None,
                source_record_id="1001",
            )
        ]
    )

    result = resolve_scopus_match(db, "Remote Sensing Journal")

    assert result.source is not None
    assert result.source.id == 4
    assert result.reason == "related_title_match"
    assert result.registry_metadata["registry_category"] == "active_indexed_journal"


def test_source_matcher_related_title_lookup_can_resolve_by_eissn(monkeypatch):
    registry_entry = RegistryEntry(
        source_record_id=None,
        source_title="IEEE Access",
        normalized_source_title="ieee access",
        issn=None,
        eissn="21693536",
        active_status="Active",
        coverage="2013-2026",
        source_type="Journal",
        open_access_status="Gold",
        publisher="IEEE",
        main_publisher="IEEE",
        asjc_codes="1700;2200",
        top_level_fields=("Physical Sciences",),
        related_titles=("IEEE Open Access",),
        medline_sourced=False,
        is_accepted=False,
        is_discontinued=False,
        discontinued_due_to_quality=False,
        registry_category="active_indexed_journal",
    )
    registry = _registry_data((registry_entry,))
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: registry)
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))

    db = _FakeDB(
        [
            SimpleNamespace(
                id=11,
                source_title="IEEE Access",
                normalized_source_title="ieee access",
                issn=None,
                eissn="21693536",
                source_record_id=None,
            )
        ]
    )

    result = resolve_scopus_match(db, "IEEE Open Access")

    assert result.source is not None
    assert result.source.id == 11
    assert result.reason == "related_title_match"


def test_source_matcher_accepted_and_discontinued_categories_adjust_confidence(monkeypatch):
    accepted_entry = RegistryEntry(
        source_record_id="2001",
        source_title="Future Energy",
        normalized_source_title="future energy",
        issn=None,
        eissn="12345678",
        active_status="Accepted",
        coverage="2026-02-01",
        source_type="Journal",
        open_access_status=None,
        publisher="Springer",
        main_publisher="Springer",
        asjc_codes=None,
        top_level_fields=(),
        related_titles=(),
        medline_sourced=False,
        is_accepted=True,
        is_discontinued=False,
        discontinued_due_to_quality=False,
        registry_category="accepted_title",
    )
    discontinued_entry = RegistryEntry(
        source_record_id="2002",
        source_title="Old Journal",
        normalized_source_title="old journal",
        issn="87654321",
        eissn=None,
        active_status="Discontinued",
        coverage="2024, vol 13, issue 1",
        source_type="Journal",
        open_access_status=None,
        publisher="Legacy Pub",
        main_publisher="Legacy Pub",
        asjc_codes=None,
        top_level_fields=(),
        related_titles=(),
        medline_sourced=False,
        is_accepted=False,
        is_discontinued=True,
        discontinued_due_to_quality=True,
        registry_category="discontinued_title",
    )
    registry = _registry_data((accepted_entry, discontinued_entry))
    monkeypatch.setattr("app.services.source_matcher.load_registry_workbook", lambda path=None: registry)
    monkeypatch.setattr("app.services.source_matcher.get_primary_registry_workbook_path", lambda: Path("unused.xlsx"))

    db = _FakeDB(
        [
            SimpleNamespace(
                id=5,
                source_title="Future Energy",
                normalized_source_title="future energy",
                issn=None,
                eissn="12345678",
                source_record_id="2001",
            ),
            SimpleNamespace(
                id=6,
                source_title="Old Journal",
                normalized_source_title="old journal",
                issn="87654321",
                eissn=None,
                source_record_id="2002",
            ),
        ]
    )

    accepted = resolve_scopus_match(db, "Future Energy")
    discontinued = resolve_scopus_match(db, "Old Journal")

    assert accepted.registry_metadata["registry_category"] == "accepted_title"
    assert discontinued.registry_metadata["registry_category"] == "discontinued_title"
    assert accepted.confidence > discontinued.confidence
