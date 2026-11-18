from __future__ import annotations

from dataclasses import dataclass
import html
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.providers.openalex_strict_provider import reconstruct_openalex_abstract


logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
MIN_ABSTRACT_CHARS = 30


@dataclass(frozen=True)
class EnrichmentResult:
    found: bool
    abstract: str | None = None
    source: str | None = None
    reason: str | None = None


def has_summarizable_abstract(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _metadata(candidate: Any) -> dict[str, Any]:
    metadata = getattr(candidate, "raw_metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def _set_metadata(candidate: Any, metadata: dict[str, Any]) -> None:
    setattr(candidate, "raw_metadata_json", metadata)


def _candidate_value(candidate: Any, name: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name)
    return getattr(candidate, name, None)


def _clean_abstract(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= MIN_ABSTRACT_CHARS else None


def _normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    doi = value.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower() or None


def _openalex_work_id(candidate: Any, metadata: dict[str, Any]) -> str | None:
    raw_id = metadata.get("openalex_id") or _candidate_value(candidate, "url")
    if not isinstance(raw_id, str):
        return None
    text = raw_id.strip()
    if not text:
        return None
    if "openalex.org/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(r"W\d+", text, flags=re.IGNORECASE):
        return text
    return None


def _headers(provider: str) -> dict[str, str]:
    headers = {
        "User-Agent": f"A20-App-050/1.0 (literature review backend; {provider})",
    }
    semantic_key = getattr(settings, "semantic_scholar_api_key", None)
    if provider == "semantic_scholar" and semantic_key:
        headers["x-api-key"] = str(semantic_key)
    return headers


def _fetch_openalex_detail(candidate: Any, metadata: dict[str, Any], timeout: float) -> str | None:
    work_id = _openalex_work_id(candidate, metadata)
    if not work_id:
        return None

    params = {
        "select": "id,abstract_inverted_index",
    }
    api_key = getattr(settings, "openalex_api_key", None)
    if api_key:
        params["api_key"] = str(api_key)

    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{OPENALEX_WORKS_URL}/{quote(work_id, safe='')}",
            params=params,
            headers=_headers("openalex_detail"),
        )
        response.raise_for_status()
        data = response.json()
    return reconstruct_openalex_abstract(data.get("abstract_inverted_index"))


def _fetch_semantic_scholar(candidate: Any, timeout: float) -> tuple[str | None, str | None]:
    doi = _normalize_doi(_candidate_value(candidate, "doi"))
    title = _candidate_value(candidate, "title")
    fields = "paperId,title,abstract,externalIds"

    with httpx.Client(timeout=timeout) as client:
        if doi:
            response = client.get(
                f"{SEMANTIC_SCHOLAR_GRAPH_URL}/DOI:{quote(doi, safe='')}",
                params={"fields": fields},
                headers=_headers("semantic_scholar"),
            )
            if response.status_code != 404:
                response.raise_for_status()
                abstract = _clean_abstract(response.json().get("abstract"))
                if abstract:
                    return abstract, "semantic_scholar"

        if isinstance(title, str) and title.strip():
            response = client.get(
                f"{SEMANTIC_SCHOLAR_GRAPH_URL}/search",
                params={
                    "query": title.strip(),
                    "limit": 1,
                    "fields": fields,
                },
                headers=_headers("semantic_scholar"),
            )
            response.raise_for_status()
            data = response.json()
            first = (data.get("data") or [None])[0]
            if isinstance(first, dict):
                abstract = _clean_abstract(first.get("abstract"))
                if abstract:
                    return abstract, "semantic_scholar"

    return None, None


def _fetch_crossref(candidate: Any, timeout: float) -> str | None:
    doi = _normalize_doi(_candidate_value(candidate, "doi"))
    if not doi:
        return None

    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{CROSSREF_WORKS_URL}/{quote(doi, safe='')}",
            headers=_headers("crossref"),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict):
        return None
    return _clean_abstract(message.get("abstract"))


def _persist_abstract(candidate: Any, abstract: str, source: str) -> None:
    setattr(candidate, "abstract", abstract)
    metadata = dict(_metadata(candidate))
    metadata["_abstract_enrichment"] = {
        "source": source,
        "abstract_length": len(abstract),
    }
    _set_metadata(candidate, metadata)


def enrich_candidate_abstract(candidate: Any, *, timeout: float = 6.0) -> EnrichmentResult:
    if has_summarizable_abstract(_candidate_value(candidate, "abstract")):
        return EnrichmentResult(found=True, abstract=_candidate_value(candidate, "abstract"), source="existing")

    metadata = dict(_metadata(candidate))
    from_metadata = reconstruct_openalex_abstract(metadata.get("abstract_inverted_index"))
    if from_metadata:
        _persist_abstract(candidate, from_metadata, "openalex_metadata")
        return EnrichmentResult(found=True, abstract=from_metadata, source="openalex_metadata")

    attempts = []
    if _openalex_work_id(candidate, metadata):
        attempts.append(("openalex_detail", lambda: _fetch_openalex_detail(candidate, metadata, timeout)))
    title = _candidate_value(candidate, "title")
    if _normalize_doi(_candidate_value(candidate, "doi")) or (isinstance(title, str) and title.strip()):
        attempts.append(("semantic_scholar", lambda: _fetch_semantic_scholar(candidate, timeout)[0]))
    if _normalize_doi(_candidate_value(candidate, "doi")):
        attempts.append(("crossref", lambda: _fetch_crossref(candidate, timeout)))
    for source, fetch in attempts:
        try:
            abstract = fetch()
        except httpx.HTTPError as exc:
            logger.info("Abstract enrichment failed via %s for candidate %s: %s", source, _candidate_value(candidate, "id"), exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            logger.info("Abstract enrichment failed via %s for candidate %s: %s", source, _candidate_value(candidate, "id"), exc)
            continue
        if abstract:
            _persist_abstract(candidate, abstract, source)
            logger.info("Abstract enrichment succeeded via %s for candidate %s", source, _candidate_value(candidate, "id"))
            return EnrichmentResult(found=True, abstract=abstract, source=source)

    return EnrichmentResult(found=False, reason="no_abstract")
