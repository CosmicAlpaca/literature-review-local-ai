from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.openalex_strict_provider import OpenAlexStrictProvider, reconstruct_openalex_abstract


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"bad status: {self.status_code}")

    def json(self):
        return self._payload


def test_openalex_provider_builds_strict_filter_request(monkeypatch):
    calls = {}

    class _FakeClient:
        def __init__(self, timeout):
            calls["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            calls["url"] = url
            calls["params"] = params
            calls["headers"] = headers
            return _FakeResponse({"results": []})

    monkeypatch.setattr("app.providers.openalex_strict_provider.httpx.Client", _FakeClient)

    provider = OpenAlexStrictProvider()
    provider.fetch_papers("flood detection", limit=5)

    assert calls["url"] == "https://api.openalex.org/works"
    assert calls["params"]["search"] == "flood detection"
    assert calls["params"]["per-page"] == 5
    assert "abstract_inverted_index" in calls["params"]["select"]
    assert "primary_location" in calls["params"]["select"]
    assert "type:article" in calls["params"]["filter"]
    assert "primary_location.source.type:journal" in calls["params"]["filter"]
    assert "primary_location.source.is_core:true" in calls["params"]["filter"]
    assert "primary_location.source.is_in_doaj:true" in calls["params"]["filter"]
    assert "has_doi:true" in calls["params"]["filter"]
    assert "is_retracted:false" in calls["params"]["filter"]


def test_openalex_provider_maps_strict_results_to_paper_result():
    provider = OpenAlexStrictProvider()
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "display_name": "Flood Detection with Remote Sensing",
                "doi": "https://doi.org/10.1000/flood",
                "publication_year": 2024,
                "cited_by_count": 17,
                "is_retracted": False,
                "abstract_inverted_index": {
                    "Flood": [0],
                    "detection": [1],
                    "abstract": [2],
                    "from": [3],
                    "OpenAlex": [4],
                    "inverted": [5],
                    "index": [6],
                },
                "authorships": [
                    {"author": {"display_name": "Author One"}},
                    {"author": {"display_name": "Author Two"}},
                ],
                "primary_location": {
                    "source": {
                        "id": "https://openalex.org/S123",
                        "display_name": "Remote Sensing",
                        "type": "journal",
                        "is_core": True,
                        "is_in_doaj": True,
                        "issn_l": "2072-4292",
                        "issn": ["2072-4292", "2072-4293"],
                        "host_organization": "https://openalex.org/P123",
                        "host_organization_name": "MDPI",
                    }
                },
            },
            {
                "id": "https://openalex.org/W999",
                "display_name": "Should Be Excluded",
                "doi": "https://doi.org/10.1000/excluded",
                "publication_year": 2024,
                "cited_by_count": 1,
                "is_retracted": False,
                "authorships": [],
                "primary_location": {
                    "source": {
                        "id": "https://openalex.org/S999",
                        "display_name": "Conference Source",
                        "type": "conference",
                        "is_core": True,
                        "is_in_doaj": True,
                    }
                },
            },
        ]
    }

    works = provider._parse_response(payload)

    assert len(works) == 1
    work = works[0]
    assert work["title"] == "Flood Detection with Remote Sensing"
    assert work["abstract"] == "Flood detection abstract from OpenAlex inverted index"
    assert work["doi"] == "10.1000/flood"
    assert work["source_title"] == "Remote Sensing"
    assert work["raw_metadata"]["issn_l"] == "2072-4292"
    assert work["raw_metadata"]["source_type"] == "journal"
    assert work["raw_metadata"]["is_core"] is True
    assert work["raw_metadata"]["is_in_doaj"] is True
    assert work["raw_metadata"]["abstract_source"] == "openalex_inverted_index"
    assert "abstract_inverted_index" in work["raw_metadata"]
    assert work["raw_metadata"]["filter_provenance"]["provider_mode"] == "openalex_strict"


def test_openalex_reconstructs_abstract_from_inverted_index():
    abstract = reconstruct_openalex_abstract(
        {
            "Deep": [0],
            "learning": [1],
            "supports": [2],
            "flood": [3],
            "detection": [4],
            "from": [5],
            "satellite": [6],
            "images.": [7],
        }
    )

    assert abstract == "Deep learning supports flood detection from satellite images."


def test_openalex_reconstruct_malformed_abstract_returns_none():
    assert reconstruct_openalex_abstract(None) is None
    assert reconstruct_openalex_abstract({}) is None
    assert reconstruct_openalex_abstract({"Too": ["bad"], "short": [1]}) is None
    assert reconstruct_openalex_abstract({"word": "not-a-list"}) is None
    assert reconstruct_openalex_abstract({"word": [999999999]}) is None
