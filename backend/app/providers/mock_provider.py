from app.providers.base import PaperResult


def mock_search(query: str):
    return [
        PaperResult(
            title="Deep Learning for Flood Detection",
            abstract="Study on flood detection using CNN models",
            year=2022,
            authors=["A", "B"],
            source_title="IEEE Access",
            citation_count=120,
            provider="mock",
        ),
        PaperResult(
            title="Multimodal Flood Analysis",
            abstract="Combining image and text data",
            year=2023,
            authors=["C"],
            source_title="Remote Sensing",
            citation_count=80,
            provider="mock",
        ),
        PaperResult(
            title="Satellite-based Flood Monitoring",
            year=2021,
            source_title="Nature Communications",
            citation_count=200,
            provider="mock",
        ),
    ]