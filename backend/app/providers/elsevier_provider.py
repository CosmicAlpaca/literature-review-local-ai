import logging
import os
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.providers.base import PaperResult

_SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
_DEFAULT_TIMEOUT = 20.0

logger = logging.getLogger(__name__)


@dataclass
class ScopusConfirmResult:
    """Result of attempting to confirm a paper in Scopus via Elsevier API."""
    confirmed: bool
    scopus_url: str | None = None
    scopus_doi: str | None = None
    scopus_eid: str | None = None
    source_title: str | None = None
    title_similarity: float = 0.0
    reason: str = "not_checked"


class ElsevierProvider:
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT, retry: int = 2):
        from app.core.config import settings
        self.timeout = timeout
        self.retry = retry
        self.api_key = settings.elsevier_api_key or ""

    # ------------------------------------------------------------------ #
    # Public: verify a specific paper exists in Scopus by title / DOI     #
    # ------------------------------------------------------------------ #

    def verify_paper_in_scopus(
        self,
        title: str,
        doi: str | None = None,
    ) -> ScopusConfirmResult:
        """
        Use the Scopus Search API to confirm whether a paper (identified by
        its title and optionally its DOI) is indexed in Scopus.

        Returns a ScopusConfirmResult with:
          - confirmed=True if Scopus returns a matching entry
          - scopus_url set to the Scopus record link if confirmed
          - title_similarity score used for the match decision
        """
        if not self.api_key:
            return ScopusConfirmResult(confirmed=False, reason="no_api_key")

        # --- Try DOI first: most reliable ---
        if doi:
            result = self._search_by_doi(doi)
            if result.confirmed:
                return result

        # --- Fall back to title search ---
        if title:
            result = self._search_by_title(title)
            return result

        return ScopusConfirmResult(confirmed=False, reason="no_query_material")

    def _search_by_doi(self, doi: str) -> ScopusConfirmResult:
        """Query Scopus with DOI({doi}) — returns unambiguous match."""
        clean_doi = doi.strip().lstrip("https://doi.org/").strip()
        query = f'DOI("{clean_doi}")'
        try:
            entries = self._raw_search(query, count=1)
        except Exception as exc:
            logger.warning("Elsevier DOI search failed: %s", exc)
            return ScopusConfirmResult(confirmed=False, reason=f"api_error:{exc}")

        if not entries:
            return ScopusConfirmResult(confirmed=False, reason="doi_not_in_scopus")

        entry = entries[0]
        scopus_url = self._extract_scopus_url(entry)
        return ScopusConfirmResult(
            confirmed=True,
            scopus_url=scopus_url,
            scopus_doi=clean_doi,
            scopus_eid=entry.get("eid"),
            source_title=entry.get("prism:publicationName"),
            title_similarity=1.0,
            reason="doi_match",
        )

    def _search_by_title(self, title: str) -> ScopusConfirmResult:
        """
        Search Scopus by TITLE(exact) and compare returned title with input
        using sequence similarity. Threshold: 0.82.
        """
        # Scopus TITLE() query requires exact phrase in quotes for best recall
        query = f'TITLE("{title}")'
        try:
            entries = self._raw_search(query, count=3)
        except Exception as exc:
            logger.warning("Elsevier title search failed: %s", exc)
            return ScopusConfirmResult(confirmed=False, reason=f"api_error:{exc}")

        if not entries:
            # Try looser keyword search
            try:
                query_loose = f'TITLE-ABS-KEY("{title}")'
                entries = self._raw_search(query_loose, count=3)
            except Exception:
                pass

        if not entries:
            return ScopusConfirmResult(confirmed=False, reason="title_not_found_in_scopus")

        # Pick the entry whose title best matches our input
        input_norm = title.strip().lower()
        best_entry: dict | None = None
        best_sim = 0.0

        for entry in entries:
            scopus_title = (entry.get("dc:title") or "").strip().lower()
            sim = SequenceMatcher(None, input_norm, scopus_title).ratio()
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is None or best_sim < 0.92:
            return ScopusConfirmResult(
                confirmed=False,
                title_similarity=round(best_sim, 4),
                reason=f"title_similarity_below_threshold:{round(best_sim, 4)}",
            )

        scopus_url = self._extract_scopus_url(best_entry)
        doi_raw = best_entry.get("prism:doi", "")
        clean_doi = doi_raw.replace("https://doi.org/", "").strip() if doi_raw else None

        return ScopusConfirmResult(
            confirmed=True,
            scopus_url=scopus_url,
            scopus_doi=clean_doi,
            scopus_eid=best_entry.get("eid"),
            source_title=best_entry.get("prism:publicationName"),
            title_similarity=round(best_sim, 4),
            reason="title_match",
        )

    # ------------------------------------------------------------------ #
    # Public: bulk search (used as search provider in cascade)             #
    # ------------------------------------------------------------------ #

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
        if not self.api_key:
            return []

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        # Build Scopus boolean query with filter pushdown
        scopus_parts = [f'TITLE-ABS-KEY("{query}")']
        if year_from and year_to:
            scopus_parts.append(f"PUBYEAR > {year_from - 1} AND PUBYEAR < {year_to + 1}")
        elif year_from:
            scopus_parts.append(f"PUBYEAR > {year_from - 1}")
        elif year_to:
            scopus_parts.append(f"PUBYEAR < {year_to + 1}")
        if open_access_only:
            scopus_parts.append("OPENACCESS(1)")

        scopus_query = " AND ".join(scopus_parts)

        # Sort by citation count for better results
        sort_param = "-citedby-count" if not min_citations else "-citedby-count"

        try:
            entries = self._raw_search(
                scopus_query,
                count=min(limit, 25),
                sort=sort_param,
            )
        except Exception as exc:
            raise RuntimeError(f"Elsevier API failed: {exc}") from exc

        results = [self._entry_to_paper_result(e) for e in entries if e.get("dc:title")]

        # Post-filter min_citations (Scopus doesn't support this in query syntax)
        if min_citations and min_citations > 0:
            results = [r for r in results if (r.citation_count or 0) >= min_citations]

        return results

    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        return self.fetch_papers(query=query, limit=20)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _raw_search(self, query: str, count: int = 5, sort: str | None = None) -> list[dict[str, Any]]:
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
        }
        params: dict[str, Any] = {
            "query": query,
            "count": count,
            "field": (
                "dc:title,dc:description,dc:creator,"
                "prism:publicationDate,prism:doi,prism:url,"
                "citedby-count,prism:publicationName,eid,link"
            ),
        }
        if sort:
            params["sort"] = sort

        last_exc: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retry + 1):
                try:
                    response = client.get(_SCOPUS_SEARCH_URL, headers=headers, params=params)
                    if response.status_code == 429:
                        time.sleep(2 ** attempt)
                        last_exc = RuntimeError("Elsevier rate-limit (429)")
                        continue
                    response.raise_for_status()
                    data = response.json()
                    return data.get("search-results", {}).get("entry", [])
                except httpx.HTTPError as exc:
                    last_exc = exc
                    time.sleep(1)

        raise RuntimeError(
            f"Elsevier API failed after {self.retry + 1} attempts. Last error: {last_exc}"
        )

    def _extract_scopus_url(self, entry: dict[str, Any]) -> str | None:
        """Extract the Scopus record URL from the entry links array."""
        links = entry.get("link") or []
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("@ref") == "scopus":
                    return link.get("@href")
        # Fallback: build from EID
        eid = entry.get("eid")
        if eid:
            return f"https://www.scopus.com/record/display.uri?eid={eid}&origin=inward"
        doi = entry.get("prism:doi", "")
        if doi:
            return f"https://doi.org/{doi.replace('https://doi.org/', '').strip()}"
        return None

    def _entry_to_paper_result(self, e: dict[str, Any]) -> PaperResult:
        title = e.get("dc:title", "").strip()
        creator = e.get("dc:creator", "")
        authors = [creator] if creator else []
        pub_date = e.get("prism:publicationDate", "")
        year = (
            int(pub_date[:4])
            if pub_date and len(pub_date) >= 4 and pub_date[:4].isdigit()
            else None
        )
        doi = e.get("prism:doi", "")
        clean_doi = doi.replace("https://doi.org/", "").strip() if doi else None
        scopus_url = self._extract_scopus_url(e)
        url = scopus_url or (f"https://doi.org/{clean_doi}" if clean_doi else e.get("prism:url", ""))

        return PaperResult(
            title=title,
            abstract=e.get("dc:description"),
            year=year,
            authors=authors,
            doi=clean_doi,
            url=url,
            source_title=e.get("prism:publicationName"),
            citation_count=int(e.get("citedby-count") or 0),
            provider="elsevier",
            raw_metadata={
                "doi": clean_doi,
                "eid": e.get("eid"),
                "scopus_url": scopus_url,
                "source_title": e.get("prism:publicationName"),
                "publisher": "Elsevier/Scopus",
            },
        )
