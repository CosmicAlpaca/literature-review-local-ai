import time
from typing import Any

import httpx

from app.providers.base import PaperResult

_OPENALEX_BASE_URL = "https://api.openalex.org/works"
_DEFAULT_TIMEOUT = 20.0
_MAX_ABSTRACT_POSITION = 5000
_OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "authorships",
        "primary_location",
        "cited_by_count",
        "abstract_inverted_index",
        "ids",
        "best_oa_location",
        "open_access",
        "type",
        "is_retracted",
        "language",
    ]
)
_STRICT_FILTER = ",".join(
    [
        "type:article",
        "primary_location.source.type:journal",
        "primary_location.source.is_core:true",
        "primary_location.source.is_in_doaj:true",
        "has_doi:true",
        "is_retracted:false",
    ]
)


def reconstruct_openalex_abstract(abstract_inverted_index: Any) -> str | None:
    if not isinstance(abstract_inverted_index, dict) or not abstract_inverted_index:
        return None

    positioned_words: dict[int, str] = {}
    max_position = -1
    for word, positions in abstract_inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        cleaned_word = word.strip()
        if not cleaned_word:
            continue
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                continue
            if position > _MAX_ABSTRACT_POSITION:
                continue
            positioned_words[position] = cleaned_word
            max_position = max(max_position, position)

    if max_position < 0:
        return None

    words = [positioned_words.get(position, "") for position in range(max_position + 1)]
    abstract = " ".join(word for word in words if word).strip()
    return abstract if len(abstract) >= 30 else None


class OpenAlexStrictProvider:
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT, retry: int = 2):
        self.timeout = timeout
        self.retry = retry

    def fetch_papers(
        self,
        query: str,
        limit: int = 20,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access_only: bool = False,
        min_citations: int | None = None,
    ) -> list[PaperResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not (1 <= limit <= 100):
            raise ValueError(f"limit must be between 1 and 100, got: {limit}")

        data = self._call_api(
            query=query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            min_citations=min_citations,
        )
        works = self._parse_response(data)
        return [
            PaperResult(
                title=work["title"],
                abstract=work["abstract"],
                year=work["year"],
                authors=work["authors"],
                doi=work["doi"],
                url=work["url"],
                source_title=work["source_title"],
                citation_count=work["citation_count"],
                provider="openalex_strict",
                raw_metadata=work["raw_metadata"],
            )
            for work in works
        ]

    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        return self.fetch_papers(query=query, limit=20)

    def _call_api(
        self,
        query: str,
        limit: int,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        open_access_only: bool = False,
        min_citations: int | None = None,
    ) -> dict[str, Any]:
        # Build dynamic filter by appending to the strict base
        filters = [_STRICT_FILTER]
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:>{year_from - 1}")
        elif year_to:
            filters.append(f"publication_year:<{year_to + 1}")
        if open_access_only:
            filters.append("open_access.is_oa:true")
        if min_citations and min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations - 1}")

        params = {
            "search": query,
            "filter": ",".join(filters),
            "per-page": limit,
            "select": _OPENALEX_SELECT_FIELDS,
            "sort": "cited_by_count:desc",
        }
        
        from app.core.config import settings
        api_key = settings.openalex_api_key or ""
        if api_key:
            params["api_key"] = api_key
        headers = {
            "User-Agent": "A20-App-050/1.0 (literature review backend; openalex_strict)",
        }

        last_exc: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retry + 1):
                try:
                    response = client.get(_OPENALEX_BASE_URL, params=params, headers=headers)
                    if response.status_code == 429:
                        time.sleep(2**attempt)
                        last_exc = RuntimeError("OpenAlex rate-limit (429)")
                        continue

                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as exc:
                    last_exc = exc
                    time.sleep(1)

        raise RuntimeError(
            f"OpenAlex API failed after {self.retry + 1} attempts. Last error: {last_exc}"
        )

    def _extract_eissn(self, source: dict[str, Any]) -> str | None:
        """Extract electronic ISSN from source.issn array by excluding issn_l."""
        issn_l = (source.get("issn_l") or "").strip()
        issn_list = source.get("issn") or []
        if not isinstance(issn_list, list):
            return None
        for issn in issn_list:
            if isinstance(issn, str) and issn.strip() and issn.strip() != issn_l:
                return issn.strip()
        return None

    def _extract_all_issns(self, source: dict[str, Any]) -> list[str]:
        """Return all unique ISSNs from the source for downstream matching."""
        issns: list[str] = []
        issn_l = (source.get("issn_l") or "").strip()
        if issn_l:
            issns.append(issn_l)
        for issn in (source.get("issn") or []):
            if isinstance(issn, str) and issn.strip() and issn.strip() not in issns:
                issns.append(issn.strip())
        return issns

    def _parse_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        works: list[dict[str, Any]] = []

        for item in data.get("results", []):
            title = (item.get("display_name") or "").strip()
            doi = (item.get("doi") or "").strip()
            if not title or not doi:
                continue

            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            source_title = source.get("display_name")
            source_type = source.get("type")
            is_core = bool(source.get("is_core"))
            is_in_doaj = bool(source.get("is_in_doaj"))
            is_retracted = bool(item.get("is_retracted"))

            if source_type != "journal" or not is_core or not is_in_doaj or is_retracted:
                continue

            # Extract enriched metadata
            eissn = self._extract_eissn(source)
            all_issns = self._extract_all_issns(source)
            open_access = item.get("open_access") or {}
            abstract_inverted_index = item.get("abstract_inverted_index")
            abstract = reconstruct_openalex_abstract(abstract_inverted_index)
            language = item.get("language")
            work_type = item.get("type")

            # Publisher hierarchy
            host_org = source.get("host_organization_name")
            host_org_lineage = source.get("host_organization_lineage_names") or []

            works.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "year": item.get("publication_year"),
                    "authors": [
                        (authorship.get("author") or {}).get("display_name", "").strip()
                        for authorship in (item.get("authorships") or [])
                        if (authorship.get("author") or {}).get("display_name")
                    ],
                    "doi": doi.replace("https://doi.org/", "").strip(),
                    "url": item.get("id") or doi,
                    "source_title": source_title,
                    "citation_count": item.get("cited_by_count") or 0,
                    "raw_metadata": {
                        "openalex_id": item.get("id"),
                        "ids": item.get("ids"),
                        "abstract_inverted_index": abstract_inverted_index,
                        "abstract_source": "openalex_inverted_index" if abstract else None,
                        "best_oa_location": item.get("best_oa_location"),
                        "source_openalex_id": source.get("id"),
                        "issn_l": source.get("issn_l"),
                        "issn": all_issns,
                        "eissn": eissn,
                        "publisher": host_org,
                        "host_organization": source.get("host_organization"),
                        "host_organization_lineage": host_org_lineage,
                        "source_type": source_type,
                        "work_type": work_type,
                        "is_core": is_core,
                        "is_in_doaj": is_in_doaj,
                        "is_retracted": is_retracted,
                        "language": language,
                        "open_access": {
                            "is_oa": open_access.get("is_oa"),
                            "oa_status": open_access.get("oa_status"),
                            "oa_url": open_access.get("oa_url"),
                        },
                        "filter_provenance": {
                            "provider_mode": "openalex_strict",
                            "endpoint": _OPENALEX_BASE_URL,
                            "filter": _STRICT_FILTER,
                            "intent": [
                                "article_only",
                                "journal_only",
                                "core_indexed_only",
                                "doaj_only",
                                "doi_required",
                                "exclude_retracted",
                            ],
                        },
                    },
                }
            )

        return works
