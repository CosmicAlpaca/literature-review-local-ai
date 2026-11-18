from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.providers.provider_factory import MockSearchProvider, get_search_provider


def test_mock_mode_returns_mock_provider(monkeypatch):
    monkeypatch.setattr(settings, "search_provider_mode", "mock")

    provider = get_search_provider()

    assert isinstance(provider, MockSearchProvider)


def test_semantic_scholar_mode_resolves_provider_without_network(monkeypatch):
    monkeypatch.setattr(settings, "search_provider_mode", "semantic_scholar")

    provider = get_search_provider()

    assert type(provider).__name__ == "SemanticScholarProviderAdapter"


def test_openalex_strict_mode_resolves_provider_without_network(monkeypatch):
    monkeypatch.setattr(settings, "search_provider_mode", "openalex_strict")

    provider = get_search_provider()

    assert type(provider).__name__ == "OpenAlexStrictProvider"


def test_invalid_provider_mode_raises_value_error(monkeypatch):
    monkeypatch.setattr(settings, "search_provider_mode", "invalid")

    with pytest.raises(ValueError, match="Invalid SEARCH_PROVIDER_MODE"):
        get_search_provider()
