import time
from typing import Any

import httpx

from app.providers.base import PaperResult

_SS_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = "paperId,title,abstract,citationCount,url,authors,year,venue"
_DEFAULT_TIMEOUT = 20.0


class SemanticScholarProviderAdapter:
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT, retry: int = 2):
        self.timeout = timeout
        self.retry = retry

    def fetch_papers(self, query: str, limit: int = 20) -> list[PaperResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not (1 <= limit <= 100):
            raise ValueError(f"limit must be between 1 and 100, got: {limit}")

        data = self._call_api(query=query, limit=limit)
        papers = self._parse_response(data)
        return [
            PaperResult(
                title=paper["title"],
                abstract=paper["abstract"],
                year=paper["year"],
                authors=paper["authors"],
                doi=None,
                url=paper["url"],
                source_title=paper["source_title"],
                citation_count=paper["citation_count"],
                provider="semantic_scholar",
                raw_metadata={"paper_id": paper["paper_id"]},
            )
            for paper in papers
        ]

    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        return self.fetch_papers(query=query, limit=20)

    def _call_api(self, query: str, limit: int) -> dict[str, Any]:
        params = {
            "query": query,
            "limit": limit,
            "fields": _SS_FIELDS,
        }
        headers = {
            "User-Agent": "A20-App-050/1.0 (literature review backend)",
        }

        last_exc: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retry + 1):
                try:
                    response = client.get(_SS_BASE_URL, params=params, headers=headers)
                    if response.status_code == 429:
                        time.sleep(2**attempt)
                        last_exc = RuntimeError("Semantic Scholar rate-limit (429)")
                        continue

                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as exc:
                    last_exc = exc
                    time.sleep(1)

        raise RuntimeError(
            f"Semantic Scholar API failed after {self.retry + 1} attempts. Last error: {last_exc}"
        )

    def _parse_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []

        for item in data.get("data", []):
            title = (item.get("title") or "").strip()
            paper_id = (item.get("paperId") or "").strip()
            if not title or not paper_id:
                continue

            url = (
                item.get("url")
                or f"https://www.semanticscholar.org/paper/{paper_id}"
            )

            papers.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": item.get("abstract"),
                    "authors": [
                        author.get("name", "").strip()
                        for author in (item.get("authors") or [])
                        if author.get("name")
                    ],
                    "year": item.get("year"),
                    "url": url,
                    "source_title": item.get("venue"),
                    "citation_count": item.get("citationCount") or 0,
                }
            )

        return papers
