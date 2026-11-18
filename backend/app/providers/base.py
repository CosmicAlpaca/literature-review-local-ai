from dataclasses import dataclass
from typing import Protocol


@dataclass
class PaperResult:
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] | None = None
    doi: str | None = None
    url: str | None = None
    source_title: str | None = None
    citation_count: int | None = None
    provider: str = "unknown"
    raw_metadata: dict | None = None


class PaperSearchProvider(Protocol):
    def fetch_papers(self, query: str, limit: int = 20) -> list[PaperResult]:
        ...

    # Backward-compatible alias for older provider implementations.
    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        ...
