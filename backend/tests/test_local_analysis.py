from types import SimpleNamespace
from app.services import paper_summary, reranker


def test_local_extraction_returns_verbatim_query_relevant_sentence(monkeypatch):
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda name: object())
    monkeypatch.setattr(reranker, "_predict_with_timeout", lambda *args, **kwargs: [0.1, 0.9])
    paper = SimpleNamespace(title="Study", abstract="First sentence. Relevant result.", raw_metadata_json={})
    result = paper_summary.local_extractive_summary(paper, "result")
    assert result.relevance_to_query == "Relevant result."
    assert result.provider == "local_extractive"
    assert not result.used_llm


def test_local_extraction_reports_missing_abstract():
    result = paper_summary.local_extractive_summary(SimpleNamespace(title="Study", abstract=""), "query")
    assert result.fallback_reason == "no_abstract"
