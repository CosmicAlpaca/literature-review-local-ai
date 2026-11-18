from app.core.config import settings
from app.providers.base import PaperResult, PaperSearchProvider
from app.providers.mock_provider import mock_search


class MockSearchProvider:
    def fetch_papers(self, query: str, limit: int = 20) -> list[PaperResult]:
        return mock_search(query)[:limit]

    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        return self.fetch_papers(query=query, limit=20)


class CascadeProvider:
    def __init__(self, providers: list[PaperSearchProvider]):
        self.providers = providers

    def fetch_papers(self, query: str, limit: int = 20) -> list[PaperResult]:
        import logging
        logger = logging.getLogger(__name__)
        all_papers = []
        for provider in self.providers:
            try:
                papers = provider.fetch_papers(query, limit)
                if papers:
                    logger.info("CascadeProvider: %s returned %d papers.", type(provider).__name__, len(papers))
                    all_papers.extend(papers)
            except Exception as exc:
                logger.warning("CascadeProvider: %s failed with %s. Continuing with others...", type(provider).__name__, exc)
        return all_papers

    def search(self, query: str, page: int = 1) -> list[PaperResult]:
        return self.fetch_papers(query=query, limit=20)


def get_search_provider() -> PaperSearchProvider:
    mode = settings.search_provider_mode.strip().lower()

    if mode == "mock":
        return MockSearchProvider()

    if mode == "crossref":
        from app.providers.crossref_provider import CrossrefProvider
        return CrossrefProvider()

    if mode == "semantic_scholar":
        from app.providers.semantic_scholar_adapter import SemanticScholarProviderAdapter

        return SemanticScholarProviderAdapter()

    if mode == "openalex_strict":
        from app.providers.openalex_strict_provider import OpenAlexStrictProvider

        return OpenAlexStrictProvider()

    if mode == "cascade":
        from app.providers.elsevier_provider import ElsevierProvider
        from app.providers.openalex_strict_provider import OpenAlexStrictProvider
        from app.providers.semantic_scholar_adapter import SemanticScholarProviderAdapter
        
        # Priority order: Elsevier (Scopus native) > OpenAlex (rich metadata)
        # Semantic Scholar is low-priority due to frequent 429 rate limits
        # and lack of DOI / source metadata.
        return CascadeProvider([
            ElsevierProvider(),
            OpenAlexStrictProvider(),
            SemanticScholarProviderAdapter(),
        ])

    raise ValueError(
        "Invalid SEARCH_PROVIDER_MODE. Supported values: 'mock', 'crossref', 'semantic_scholar', 'openalex_strict', 'cascade'."
    )
