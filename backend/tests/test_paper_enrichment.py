from pathlib import Path
import sys
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.paper_enrichment import _fetch_crossref, enrich_candidate_abstract


def test_enrichment_recovers_openalex_abstract_from_metadata():
    candidate = SimpleNamespace(
        id=21,
        abstract=None,
        raw_metadata_json={
            "abstract_inverted_index": {
                "OpenAlex": [0],
                "metadata": [1],
                "contains": [2],
                "a": [3],
                "recoverable": [4],
                "abstract": [5],
                "for": [6],
                "summaries.": [7],
            }
        },
    )

    result = enrich_candidate_abstract(candidate)

    assert result.found is True
    assert result.source == "openalex_metadata"
    assert candidate.abstract == "OpenAlex metadata contains a recoverable abstract for summaries."
    assert candidate.raw_metadata_json["_abstract_enrichment"]["source"] == "openalex_metadata"


def test_enrichment_returns_missing_without_identifiers(monkeypatch):
    candidate = SimpleNamespace(id=22, abstract=None, raw_metadata_json={}, doi=None, url=None, title=None)

    def _fail_external(*args, **kwargs):
        raise AssertionError("external enrichment should not run without identifiers or title")

    monkeypatch.setattr("app.services.paper_enrichment._fetch_openalex_detail", _fail_external)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_semantic_scholar", _fail_external)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_crossref", _fail_external)

    result = enrich_candidate_abstract(candidate)

    assert result.found is False
    assert result.reason == "no_abstract"


def test_enrichment_existing_abstract_skips_providers(monkeypatch):
    candidate = SimpleNamespace(id=23, abstract="Existing abstract.", raw_metadata_json={}, doi="10.1000/x")

    def _fail_external(*args, **kwargs):
        raise AssertionError("external enrichment should not run when abstract exists")

    monkeypatch.setattr("app.services.paper_enrichment._fetch_openalex_detail", _fail_external)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_semantic_scholar", _fail_external)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_crossref", _fail_external)

    result = enrich_candidate_abstract(candidate)

    assert result.found is True
    assert result.source == "existing"
    assert candidate.abstract == "Existing abstract."


def test_enrichment_provider_errors_do_not_fail_endpoint_path(monkeypatch):
    candidate = SimpleNamespace(
        id=24,
        abstract=None,
        raw_metadata_json={"openalex_id": "https://openalex.org/W123"},
        doi="10.1000/example",
        url="https://openalex.org/W123",
        title="Recoverable paper",
    )

    def _timeout(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    calls = {"semantic": 0, "crossref": 0}

    def _semantic(*args, **kwargs):
        calls["semantic"] += 1
        return None, None

    def _crossref(*args, **kwargs):
        calls["crossref"] += 1
        return "Crossref fallback abstract with enough detail for summary generation."

    monkeypatch.setattr("app.services.paper_enrichment._fetch_openalex_detail", _timeout)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_semantic_scholar", _semantic)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_crossref", _crossref)

    result = enrich_candidate_abstract(candidate, timeout=0.1)

    assert result.found is True
    assert result.source == "crossref"
    assert calls == {"semantic": 1, "crossref": 1}
    assert candidate.abstract.startswith("Crossref fallback abstract")


def test_enrichment_stops_after_first_found_provider(monkeypatch):
    candidate = SimpleNamespace(
        id=25,
        abstract=None,
        raw_metadata_json={"openalex_id": "https://openalex.org/W123"},
        doi="10.1000/example",
        url="https://openalex.org/W123",
        title="OpenAlex paper",
    )

    def _openalex(*args, **kwargs):
        return "OpenAlex detail abstract with enough detail for summary generation."

    def _fail_downstream(*args, **kwargs):
        raise AssertionError("downstream provider should not be called after success")

    monkeypatch.setattr("app.services.paper_enrichment._fetch_openalex_detail", _openalex)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_semantic_scholar", _fail_downstream)
    monkeypatch.setattr("app.services.paper_enrichment._fetch_crossref", _fail_downstream)

    result = enrich_candidate_abstract(candidate)

    assert result.found is True
    assert result.source == "openalex_detail"


def test_crossref_xml_abstract_is_stripped_before_return(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "abstract": (
                        "<jats:p>This <jats:bold>Crossref</jats:bold> abstract "
                        "contains XML tags and should be cleaned.</jats:p>"
                    )
                }
            }

    class _FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("app.services.paper_enrichment.httpx.Client", _FakeClient)

    candidate = SimpleNamespace(doi="10.1000/xml")
    abstract = _fetch_crossref(candidate, timeout=0.1)

    assert abstract == "This Crossref abstract contains XML tags and should be cleaned."
