from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models.paper_candidate import PaperCandidate
from app.db.models.review_session import ReviewSession
from app.db.models.selected_paper import SelectedPaper
from app.providers.base import PaperResult
from app.services.llm_client import candidate_fingerprint, query_fingerprint
from app.services.llm_query import QueryRewriteResult
from app.services.orchestrator import (
    SearchFilters,
    SearchLLMOptions,
    _apply_llm_summaries,
    _select_top_candidates,
    run_search_pipeline,
)
from app.services.paper_summary import PaperSummaryResult
from app.services.llm_reranker import RerankResult


class _DeleteQuery:
    def filter(self, *args, **kwargs):
        return self

    def delete(self):
        return 0


class _PaperCandidateQuery:
    def __init__(self, db):
        self._db = db

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._db.existing_candidates)

    def delete(self):
        return 0


class _ReviewSessionQuery:
    def __init__(self, session_obj):
        self._session_obj = session_obj

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._session_obj


class _FakeDB:
    def __init__(self, review_session, existing_candidates=None):
        self.review_session = review_session
        self.existing_candidates = existing_candidates or []
        self.added = []
        self._next_candidate_id = 1
        self.commit_count = 0

    def query(self, model):
        if model is SelectedPaper:
            return _DeleteQuery()
        if model is PaperCandidate:
            return _PaperCandidateQuery(self)
        if model is ReviewSession:
            return _ReviewSessionQuery(self.review_session)
        raise AssertionError(f"Unexpected model queried: {model}")

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        if not self.added:
            return
        obj = self.added[-1]
        if isinstance(obj, PaperCandidate) and obj.id is None:
            obj.id = self._next_candidate_id
            self._next_candidate_id += 1

    def commit(self):
        self.commit_count += 1


class _StubProvider:
    def fetch_papers(self, query: str, limit: int = 20):
        return [
            PaperResult(
                title="Paper A",
                abstract="Abstract A",
                year=2024,
                authors=["Author A"],
                doi="10.1000/a",
                url="https://example.com/a",
                source_title="Journal A",
                citation_count=12,
                provider="semantic_scholar",
                raw_metadata={"paper_id": "a"},
            ),
            PaperResult(
                title="Paper B",
                abstract="Abstract B",
                year=2023,
                authors=["Author B"],
                doi="10.1000/b",
                url="https://example.com/b",
                source_title="Journal B",
                citation_count=8,
                provider="semantic_scholar",
                raw_metadata={"paper_id": "b"},
            ),
        ][:limit]


class _RecordingProvider:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def fetch_papers(self, query: str, limit: int = 20):
        self.calls.append(query)
        return list(self.results_by_query.get(query, []))[:limit]


def _paper(title="Paper A", paper_id="a"):
    return PaperResult(
        title=title,
        abstract="Abstract A",
        year=2024,
        authors=["Author A"],
        doi=f"10.1000/{paper_id}",
        url=f"https://example.com/{paper_id}",
        source_title="Journal A",
        citation_count=12,
        provider="semantic_scholar",
        raw_metadata={"paper_id": paper_id},
    )


def _patch_phase1_dependencies(monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestrator.get_search_provider",
        lambda: _StubProvider(),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.resolve_scopus_match",
        lambda db_session, source_title, raw_metadata=None: SimpleNamespace(
            source=SimpleNamespace(id=99, source_title=source_title),
            reason="exact_normalized_title",
            confidence=0.99,
            matched_value=source_title,
            normalized_input_title=source_title.lower(),
            input_identifiers=None,
            registry_metadata={"registry_category": "active_indexed_journal"},
        ),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.compute_score",
        lambda query, paper, scopus_valid, source_match=None: 0.77,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_paper",
        lambda query, paper, scopus_valid, source_match=None: {
            "weights": {"title_relevance": 0.35},
            "components": {"title_relevance": 1.0},
            "weighted": {"title_relevance": 0.35},
            "total": 0.77,
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_trust",
        lambda paper, source_match=None: {
            "weights": {"source_match_confidence": 0.35},
            "components": {"source_match_confidence": 0.99},
            "weighted": {"source_match_confidence": 0.3465},
            "total": 0.81,
            "bucket": "high",
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.build_selection_debug",
        lambda ranking_score, trust_debug: {
            "weights": {"ranking_score": 0.75, "trust_score": 0.25},
            "bucket_adjustment": 0.03,
            "ranking_score": ranking_score,
            "trust_score": trust_debug["total"],
            "trust_bucket": trust_debug["bucket"],
            "combined_score": 0.83,
            "policy": "ranking_weighted_plus_trust_bucket_adjustment_v1",
        },
    )


def test_inline_llm_summaries_are_hard_capped(monkeypatch):
    candidates = [_summary_candidate(index) for index in range(1, 6)]
    calls = []

    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 10)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_summaries_per_search", 10)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_calls_per_search", 20)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_estimated_tokens_per_search", 20000)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_input_chars_per_call", 8000)

    def _summarize(candidate, query):
        calls.append(candidate.id)
        return PaperSummaryResult(
            one_sentence_summary="Short summary.",
            contribution="Contribution.",
            method_or_data="Method.",
            relevance_to_query="Relevant.",
            limitations_or_caution="Caution.",
            used_llm=True,
            attempted=True,
            estimated_call_count=1,
        )

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summarize)

    _apply_llm_summaries(
        candidates,
        query="sleep neurons",
        llm_options=SearchLLMOptions(use_llm=True, use_llm_summary=True),
        cache_snapshot={"summary": {}},
        budget=SimpleNamespace(
            reserve=lambda payload, default_output_tokens=512: ({}, None),
            call_count=0,
            remaining=lambda: {"calls": 20, "estimated_tokens": 20000},
        ),
    )

    assert calls == [1, 2, 3]
    assert all("_llm_summary" in candidates[index].raw_metadata_json for index in range(3))
    assert candidates[3].raw_metadata_json["_llm_diagnostics"]["summary"]["fallback_reason"] == "summary_cap_exceeded"
    assert candidates[4].raw_metadata_json["_llm_diagnostics"]["summary"]["fallback_reason"] == "summary_cap_exceeded"


def test_inline_summary_skip_does_not_crash_when_abstract_missing(monkeypatch):
    candidate = SimpleNamespace(
        id=1,
        title="No abstract paper",
        abstract=None,
        raw_metadata_json={},
    )

    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_skip_summary_if_abstract_missing", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 3)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_summaries_per_search", 3)
    monkeypatch.setattr(
        "app.services.orchestrator.enrich_candidate_abstract",
        lambda candidate, timeout=3.0: None,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.summarize_paper",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("summary should be skipped")),
    )

    _apply_llm_summaries(
        [candidate],
        query="sleep neurons",
        llm_options=SearchLLMOptions(use_llm=True, use_llm_summary=True),
        cache_snapshot={"summary": {}},
        budget=SimpleNamespace(
            reserve=lambda payload, default_output_tokens=512: ({}, None),
            call_count=0,
            remaining=lambda: {"calls": 20, "estimated_tokens": 20000},
        ),
    )

    assert candidate.raw_metadata_json["_llm_diagnostics"]["summary"]["fallback_reason"] == "no_abstract"
    assert "_llm_summary" not in candidate.raw_metadata_json


def test_reranker_module_does_not_import_sentence_transformers_at_startup():
    assert "sentence_transformers" not in sys.modules


def _enable_llm_cache_settings(monkeypatch, *, rerank=True, summary=True):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_model", "mock-test-model")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", rerank)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", summary)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_cache_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_prompt_version", "v1")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_rerank", 1)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 1)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_summaries_per_search", 1)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_skip_summary_if_abstract_missing", True)


def _existing_candidate(title="Paper A", query="flood detection", metadata=None, abstract="Abstract A"):
    candidate = SimpleNamespace(
        id=101,
        title=title,
        abstract=abstract,
        year=2024,
        doi="10.1000/a",
        source_title="Journal A",
        provider="semantic_scholar",
        raw_metadata_json={"paper_id": "a"},
        final_score=0.77,
    )
    candidate_fp = candidate_fingerprint(candidate)
    query_fp = query_fingerprint(query)
    raw_metadata = {"paper_id": "a"}
    if metadata:
        raw_metadata.update(metadata(candidate_fp, query_fp))
    candidate.raw_metadata_json = raw_metadata
    return candidate


def _summary_candidate(index: int):
    return SimpleNamespace(
        id=index,
        title=f"Summary Paper {index}",
        abstract="This abstract has enough detail for a short search-time summary.",
        raw_metadata_json={},
    )


def test_orchestrator_persists_candidates_and_marks_session(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)

    review_session = SimpleNamespace(id=7, status="created")
    db = _FakeDB(review_session)

    monkeypatch.setattr(
        "app.services.orchestrator.get_search_provider",
        lambda: _StubProvider(),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.resolve_scopus_match",
        lambda db_session, source_title, raw_metadata=None: SimpleNamespace(
            source=SimpleNamespace(id=99, source_title=source_title),
            reason="exact_normalized_title",
            confidence=0.99,
            matched_value=source_title,
            normalized_input_title=source_title.lower(),
            input_identifiers=None,
            registry_metadata={"registry_category": "active_indexed_journal"},
        ),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.compute_score",
        lambda query, paper, scopus_valid, source_match=None: 0.77 if scopus_valid else 0.42,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_paper",
        lambda query, paper, scopus_valid, source_match=None: {
            "weights": {"title_relevance": 0.35},
            "components": {
                "title_relevance": 1.0,
                "source_match_strength": 0.99,
                "source_match_reason_bonus": 0.95,
            },
            "weighted": {"title_relevance": 0.35},
            "total": 0.77 if scopus_valid else 0.42,
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_trust",
        lambda paper, source_match=None: {
            "weights": {"source_match_confidence": 0.35},
            "components": {"source_match_confidence": 0.99},
            "weighted": {"source_match_confidence": 0.3465},
            "total": 0.81,
            "bucket": "high",
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.build_selection_debug",
        lambda ranking_score, trust_debug: {
            "weights": {"ranking_score": 0.75, "trust_score": 0.25},
            "bucket_adjustment": 0.03,
            "ranking_score": ranking_score,
            "trust_score": trust_debug["total"],
            "trust_bucket": trust_debug["bucket"],
            "combined_score": 0.83,
            "policy": "ranking_weighted_plus_trust_bucket_adjustment_v1",
        },
    )

    candidates, top5 = run_search_pipeline(db, session_id=7, query="flood detection")

    assert len(candidates) == 2
    assert len(top5) == 2
    assert review_session.status == "searched"
    assert db.commit_count == 2

    saved_candidates = [item for item in db.added if isinstance(item, PaperCandidate)]
    selected_papers = [item for item in db.added if isinstance(item, SelectedPaper)]

    assert len(saved_candidates) == 2
    assert len(selected_papers) == 2
    assert saved_candidates[0].provider == "semantic_scholar"
    assert saved_candidates[0].abstract == "Abstract A"
    assert saved_candidates[0].final_score == 0.77
    assert "_source_match" in saved_candidates[0].raw_metadata_json
    assert "_ranking" in saved_candidates[0].raw_metadata_json
    assert "_trust" in saved_candidates[0].raw_metadata_json
    assert "_selection" in saved_candidates[0].raw_metadata_json
    assert saved_candidates[0].raw_metadata_json["_trust"]["bucket"] == "high"
    assert saved_candidates[0].raw_metadata_json["_selection"]["combined_score"] == 0.83
    assert saved_candidates[0].raw_metadata_json["_ranking"]["components"]["source_match_strength"] == 0.99
    assert selected_papers[0].rank_order == 1
    assert selected_papers[0].paper_candidate_id == candidates[0].id


def test_production_reranker_disabled_keeps_phase1_order(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    def _fail_load(*args, **kwargs):
        raise AssertionError("production reranker should not load when disabled")

    monkeypatch.setattr("app.services.reranker._load_cross_encoder", _fail_load)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, top5 = run_search_pipeline(db, session_id=7, query="flood detection")

    assert [candidate.title for candidate in candidates] == ["Paper A", "Paper B"]
    assert [candidate.title for candidate in top5] == ["Paper A", "Paper B"]
    assert all("_reranker" not in (candidate.raw_metadata_json or {}) for candidate in candidates)


def test_production_reranker_model_load_error_falls_back(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_model_name", "cross-encoder/test")
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_max_candidates", 50)
    _patch_phase1_dependencies(monkeypatch)
    monkeypatch.setattr(
        "app.services.reranker._load_cross_encoder",
        lambda model_name: (_ for _ in ()).throw(RuntimeError("model_load_failed")),
    )

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, top5 = run_search_pipeline(db, session_id=7, query="flood detection")

    assert [candidate.title for candidate in candidates] == ["Paper A", "Paper B"]
    assert len(top5) == 2
    assert candidates[0].raw_metadata_json["_reranker"]["reranker_applied"] is False
    assert candidates[0].raw_metadata_json["_reranker"]["skipped_reason"] == "model_load_failed"


def test_production_reranker_fake_scores_affect_pipeline_order(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_model_name", "cross-encoder/test")
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_max_candidates", 50)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_score_weight", 1.0)
    _patch_phase1_dependencies(monkeypatch)
    monkeypatch.setattr("app.services.reranker._load_cross_encoder", lambda model_name: object())
    monkeypatch.setattr("app.services.reranker._predict_with_timeout", lambda *args, **kwargs: [0.1, 0.9])

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, top5 = run_search_pipeline(db, session_id=7, query="flood detection")

    assert [candidate.title for candidate in candidates] == ["Paper B", "Paper A"]
    assert [candidate.title for candidate in top5] == ["Paper B", "Paper A"]
    assert candidates[0].raw_metadata_json["_reranker"]["reranker_applied"] is True
    assert candidates[0].raw_metadata_json["_reranker"]["reranker_score"] == 1.0
    assert candidates[0].raw_metadata_json["_reranker_selection"]["final_score_after_rerank"] == 1.0


def test_top_selection_keeps_abstract_quota_when_available():
    candidates = [
        PaperCandidate(
            id=1,
            session_id=7,
            title="High score without abstract 1",
            abstract=None,
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.99}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.99,
        ),
        PaperCandidate(
            id=2,
            session_id=7,
            title="High score without abstract 2",
            abstract=None,
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.98}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.98,
        ),
        PaperCandidate(
            id=3,
            session_id=7,
            title="High score without abstract 3",
            abstract=None,
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.97}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.97,
        ),
        PaperCandidate(
            id=4,
            session_id=7,
            title="Abstract paper 1",
            abstract="A sufficiently long abstract for summary generation and evidence review.",
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.72}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.72,
        ),
        PaperCandidate(
            id=5,
            session_id=7,
            title="Abstract paper 2",
            abstract="Another sufficiently long abstract for summary generation and review.",
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.71}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.71,
        ),
        PaperCandidate(
            id=6,
            session_id=7,
            title="Abstract paper 3",
            abstract="Third sufficiently long abstract for summary generation and review.",
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.70}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.70,
        ),
    ]

    selected = _select_top_candidates(candidates, limit=5)

    assert len(selected) == 5
    assert sum(1 for candidate in selected if candidate.abstract) >= 3
    assert selected[0].title == "Abstract paper 1"


def test_top_selection_quota_does_not_promote_irrelevant_abstract_paper():
    candidates = [
        PaperCandidate(
            id=1,
            session_id=7,
            title="Strong no abstract",
            abstract=None,
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.9}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.9,
        ),
        PaperCandidate(
            id=2,
            session_id=7,
            title="Relevant abstract",
            abstract="Relevant abstract with enough content for summary generation.",
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.58}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.58,
        ),
        PaperCandidate(
            id=3,
            session_id=7,
            title="Irrelevant abstract",
            abstract="Irrelevant abstract with enough content but very weak ranking signals.",
            provider="mock",
            raw_metadata_json={"_selection": {"combined_score": 0.05}, "_trust": {"total": 0.8}},
            scopus_matched=True,
            final_score=0.05,
        ),
    ]

    selected = _select_top_candidates(candidates, limit=2)

    assert [candidate.id for candidate in selected] == [2, 1]
    assert all(candidate.id != 3 for candidate in selected)


class _MutableQuery:
    def __init__(self, db, model):
        self._db = db
        self._model = model
        self._filters = {}

    def filter(self, *expressions, **kwargs):
        for expression in expressions:
            key = getattr(getattr(expression, "left", None), "key", None)
            value = getattr(getattr(expression, "right", None), "value", None)
            if key is not None:
                self._filters[key] = value
        return self

    def first(self):
        if self._model is ReviewSession:
            return self._db.review_session if self._matches(self._db.review_session) else None
        return None

    def all(self):
        if self._model is PaperCandidate:
            return [candidate for candidate in self._db.candidates if self._matches(candidate)]
        return []

    def delete(self):
        if self._model is SelectedPaper:
            before = len(self._db.selected_papers)
            self._db.selected_papers = [
                selected for selected in self._db.selected_papers if not self._matches(selected)
            ]
            self._db.operations.append(("delete", "selected_papers", before - len(self._db.selected_papers)))
            return before - len(self._db.selected_papers)
        if self._model is PaperCandidate:
            before = len(self._db.candidates)
            self._db.candidates = [
                candidate for candidate in self._db.candidates if not self._matches(candidate)
            ]
            self._db.operations.append(("delete", "paper_candidates", before - len(self._db.candidates)))
            return before - len(self._db.candidates)
        return 0

    def _matches(self, obj):
        if obj is None:
            return False
        return all(getattr(obj, key, None) == value for key, value in self._filters.items())


class _MutableSearchDB:
    def __init__(self):
        self.review_session = SimpleNamespace(id=7, status="created", topic="Flood review")
        self.candidates = [
            PaperCandidate(
                id=90,
                session_id=8,
                title="Other Session Candidate",
                abstract="Other abstract",
                year=2024,
                authors_json=[],
                doi="10.1000/other",
                url="https://example.com/other",
                source_title="Other Journal",
                normalized_source_title="other journal",
                citation_count=1,
                provider="existing",
                raw_metadata_json={},
                scopus_matched=False,
                scopus_source_id=None,
                final_score=0.1,
            )
        ]
        self.selected_papers = [
            SelectedPaper(id=90, session_id=8, paper_candidate_id=90, rank_order=1)
        ]
        self._pending = []
        self._next_candidate_id = 1
        self._next_selected_id = 1
        self.operations = []

    def query(self, model):
        return _MutableQuery(self, model)

    def add(self, obj):
        self._pending.append(obj)

    def flush(self):
        for obj in self._pending:
            if isinstance(obj, PaperCandidate) and obj.id is None:
                obj.id = self._next_candidate_id
                self._next_candidate_id += 1
            if isinstance(obj, SelectedPaper) and obj.id is None:
                obj.id = self._next_selected_id
                self._next_selected_id += 1

    def commit(self):
        self.flush()
        for obj in self._pending:
            if isinstance(obj, PaperCandidate) and obj not in self.candidates:
                self.candidates.append(obj)
            if isinstance(obj, SelectedPaper) and obj not in self.selected_papers:
                self.selected_papers.append(obj)
        self._pending.clear()


def test_orchestrator_rerun_replaces_candidates_and_selected_papers(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    _patch_phase1_dependencies(monkeypatch)
    db = _MutableSearchDB()

    run_search_pipeline(db, session_id=7, query="flood detection")
    run_search_pipeline(db, session_id=7, query="flood detection")

    session_candidates = [candidate for candidate in db.candidates if candidate.session_id == 7]
    session_selected = [selected for selected in db.selected_papers if selected.session_id == 7]

    assert len(session_candidates) == 2
    assert len(session_selected) == 2
    assert {candidate.title for candidate in session_candidates} == {"Paper A", "Paper B"}
    assert {selected.paper_candidate_id for selected in session_selected} == {
        candidate.id for candidate in session_candidates
    }


def test_orchestrator_cleanup_is_session_scoped(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    _patch_phase1_dependencies(monkeypatch)
    db = _MutableSearchDB()

    run_search_pipeline(db, session_id=7, query="flood detection")

    other_candidates = [candidate for candidate in db.candidates if candidate.session_id == 8]
    other_selected = [selected for selected in db.selected_papers if selected.session_id == 8]

    assert len(other_candidates) == 1
    assert other_candidates[0].title == "Other Session Candidate"
    assert len(other_selected) == 1
    assert other_selected[0].paper_candidate_id == 90
    assert ("delete", "selected_papers", 0) in db.operations
    assert ("delete", "paper_candidates", 0) in db.operations


def test_orchestrator_mock_llm_adds_internal_metadata_without_schema_changes(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_model", "mock-test-model")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_rerank", 2)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 2)

    review_session = SimpleNamespace(id=7, status="created", topic="flood detection")
    db = _FakeDB(review_session)
    provider = _StubProvider()

    monkeypatch.setattr(
        "app.services.orchestrator.get_search_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.resolve_scopus_match",
        lambda db_session, source_title, raw_metadata=None: SimpleNamespace(
            source=SimpleNamespace(id=99, source_title=source_title),
            reason="exact_normalized_title",
            confidence=0.99,
            matched_value=source_title,
            normalized_input_title=source_title.lower(),
            input_identifiers=None,
            registry_metadata={"registry_category": "active_indexed_journal"},
        ),
    )
    monkeypatch.setattr(
        "app.services.orchestrator.compute_score",
        lambda query, paper, scopus_valid, source_match=None: 0.77,
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_paper",
        lambda query, paper, scopus_valid, source_match=None: {
            "weights": {"title_relevance": 0.35},
            "components": {"title_relevance": 1.0},
            "weighted": {"title_relevance": 0.35},
            "total": 0.77,
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.score_trust",
        lambda paper, source_match=None: {
            "weights": {"source_match_confidence": 0.35},
            "components": {"source_match_confidence": 0.99},
            "weighted": {"source_match_confidence": 0.3465},
            "total": 0.81,
            "bucket": "high",
        },
    )
    monkeypatch.setattr(
        "app.services.orchestrator.build_selection_debug",
        lambda ranking_score, trust_debug: {
            "weights": {"ranking_score": 0.75, "trust_score": 0.25},
            "bucket_adjustment": 0.03,
            "ranking_score": ranking_score,
            "trust_score": trust_debug["total"],
            "trust_bucket": trust_debug["bucket"],
            "combined_score": 0.83,
            "policy": "ranking_weighted_plus_trust_bucket_adjustment_v1",
        },
    )

    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection using deep learning",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert review_session.status == "searched"

    saved_candidates = [item for item in db.added if isinstance(item, PaperCandidate)]
    selected_papers = [item for item in db.added if isinstance(item, SelectedPaper)]

    assert len(saved_candidates) == 2
    assert len(selected_papers) == 2
    assert saved_candidates[0].final_score == 0.77
    assert saved_candidates[0].raw_metadata_json["_llm_query"]["used_llm"] is True
    assert saved_candidates[0].raw_metadata_json["_llm_rerank"]["used_llm"] is True
    assert "_llm_selection" in saved_candidates[0].raw_metadata_json
    assert saved_candidates[0].raw_metadata_json["_llm_summary"]["used_llm"] is True
    diagnostics = saved_candidates[0].raw_metadata_json["_llm_diagnostics"]
    assert diagnostics["query_rewrite"]["attempted"] is True
    assert diagnostics["query_rewrite"]["used"] is True
    assert diagnostics["query_rewrite"]["cache_hit"] is False
    assert diagnostics["query_rewrite"]["cache_miss_reason"] == "no_cached_artifact"
    assert diagnostics["query_rewrite"]["estimated_usage"]["estimated_total_tokens"] > 0
    assert diagnostics["query_rewrite"]["budget_guard"]["allowed"] is True
    assert diagnostics["rerank"]["attempted"] is True
    assert diagnostics["rerank"]["used"] is True
    assert diagnostics["rerank"]["cache_hit"] is False
    assert diagnostics["rerank"]["cache_miss_reason"] == "no_cached_artifact"
    assert diagnostics["rerank"]["estimated_usage"]["estimated_total_tokens"] > 0
    assert diagnostics["summary"]["attempted"] is True
    assert diagnostics["summary"]["used"] is True
    assert diagnostics["summary"]["cache_hit"] is False
    assert diagnostics["summary"]["cache_miss_reason"] == "no_cached_artifact"
    assert diagnostics["summary"]["estimated_usage"]["estimated_total_tokens"] > 0
    assert selected_papers[0].paper_candidate_id == top5[0].id


def test_orchestrator_llm_failures_fall_back_to_phase1(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "openai_compatible")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_rerank", 2)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 2)
    _patch_phase1_dependencies(monkeypatch)

    def _raise(*args, **kwargs):
        raise RuntimeError("fake llm failure")

    monkeypatch.setattr("app.services.orchestrator.rewrite_query", _raise)
    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _raise)
    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _raise)

    review_session = SimpleNamespace(id=7, status="created", topic="flood detection")
    db = _FakeDB(review_session)

    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert review_session.status == "searched"
    assert all((candidate.raw_metadata_json or {})["_llm_query"]["used_llm"] is False for candidate in candidates)
    assert all("_llm_rerank" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_summary" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    for candidate in candidates:
        diagnostics = candidate.raw_metadata_json["_llm_diagnostics"]
        assert diagnostics["query_rewrite"]["attempted"] is True
        assert diagnostics["query_rewrite"]["used"] is False
        assert diagnostics["query_rewrite"]["fallback_reason"] == "provider_unavailable"
        assert diagnostics["query_rewrite"]["cache_miss_reason"] == "no_cached_artifact"
        assert diagnostics["query_rewrite"]["budget_guard"]["allowed"] is True
        assert diagnostics["rerank"]["attempted"] is True
        assert diagnostics["rerank"]["used"] is False
        assert diagnostics["rerank"]["fallback_reason"] == "provider_unavailable"
        assert diagnostics["rerank"]["cache_miss_reason"] == "no_cached_artifact"
        assert diagnostics["rerank"]["budget_guard"]["allowed"] is True
        assert diagnostics["summary"]["attempted"] is True
        assert diagnostics["summary"]["used"] is False
        assert diagnostics["summary"]["fallback_reason"] == "provider_unavailable"
        assert diagnostics["summary"]["cache_miss_reason"] == "no_cached_artifact"
        assert diagnostics["summary"]["budget_guard"]["allowed"] is True


def test_orchestrator_uses_rewritten_query_for_provider_search(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    rewritten = "pharmacological treatment insomnia sleep disorder medication"
    provider = _RecordingProvider({rewritten: [_paper()]})
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)
    monkeypatch.setattr(
        "app.services.orchestrator.rewrite_query",
        lambda raw_query, topic=None: QueryRewriteResult(
            original_query=raw_query,
            optimized_query=rewritten,
            expanded_queries=["insomnia pharmacological therapy"],
            key_terms=["insomnia", "medication"],
            notes="Translated for scholarly retrieval.",
            provider="fake",
            model="fake",
            used_llm=True,
            attempted=True,
        ),
    )

    search_filters = SearchFilters(query="thuốc trị mất ngủ")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="sleep medicine"))

    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="thuốc trị mất ngủ",
        llm_options=SearchLLMOptions(use_llm=True),
        search_filters=search_filters,
    )

    assert provider.calls == [rewritten]
    assert len(candidates) == 1
    assert search_filters.query_context_metadata["llm_query_rewrite_used"] is True
    assert search_filters.query_context_metadata["final_search_query"] == rewritten


def test_orchestrator_empty_result_retry_uses_expanded_query(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    optimized = "pharmacological treatment insomnia sleep disorder medication"
    retry = "insomnia pharmacological therapy"
    provider = _RecordingProvider({retry: [_paper(title="Retry Paper", paper_id="retry")]})
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)
    monkeypatch.setattr(
        "app.services.orchestrator.rewrite_query",
        lambda raw_query, topic=None: QueryRewriteResult(
            original_query=raw_query,
            optimized_query=optimized,
            expanded_queries=[retry, "sleep disorder treatment medication"],
            key_terms=["insomnia", "medication"],
            notes="Translated for scholarly retrieval.",
            provider="fake",
            model="fake",
            used_llm=True,
            attempted=True,
        ),
    )

    search_filters = SearchFilters(query="thuốc trị mất ngủ")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="sleep medicine"))

    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="thuốc trị mất ngủ",
        llm_options=SearchLLMOptions(use_llm=True),
        search_filters=search_filters,
    )

    assert provider.calls == [optimized, retry]
    assert len(candidates) == 1
    assert len(top5) == 1
    assert search_filters.query_context_metadata["empty_result_retry_used"] is True
    assert search_filters.query_context_metadata["retry_queries"] == [retry]
    assert search_filters.query_context_metadata["final_search_query"] == retry
    assert candidates[0].raw_metadata_json["_llm_query"]["empty_result_retry_used"] is True


def test_orchestrator_empty_result_retry_is_capped_and_returns_empty_json_shape(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    optimized = "pharmacological treatment insomnia sleep disorder medication"
    expanded = [
        "insomnia pharmacological therapy",
        "sleep disorder treatment medication",
        "hypnotics melatonin benzodiazepines insomnia treatment",
    ]
    provider = _RecordingProvider({})
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)
    monkeypatch.setattr(
        "app.services.orchestrator.rewrite_query",
        lambda raw_query, topic=None: QueryRewriteResult(
            original_query=raw_query,
            optimized_query=optimized,
            expanded_queries=expanded,
            key_terms=["insomnia", "medication"],
            notes="Translated for scholarly retrieval.",
            provider="fake",
            model="fake",
            used_llm=True,
            attempted=True,
        ),
    )

    search_filters = SearchFilters(query="thuốc trị mất ngủ")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="sleep medicine"))

    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="thuốc trị mất ngủ",
        llm_options=SearchLLMOptions(use_llm=True),
        search_filters=search_filters,
    )

    assert provider.calls == [optimized, expanded[0], expanded[1]]
    assert candidates == []
    assert top5 == []
    assert search_filters.query_context_metadata["empty_result_retry_used"] is True
    assert search_filters.query_context_metadata["retry_queries"] == expanded[:2]
    assert search_filters.query_context_metadata["final_search_query"] == expanded[1]


def test_orchestrator_use_llm_false_does_not_rewrite_or_retry(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    provider = _RecordingProvider({})
    monkeypatch.setattr("app.services.orchestrator.get_search_provider", lambda: provider)

    def _fail_rewrite(*args, **kwargs):
        raise AssertionError("rewrite should not run when use_llm=false")

    monkeypatch.setattr("app.services.orchestrator.rewrite_query", _fail_rewrite)

    search_filters = SearchFilters(query="thuốc trị mất ngủ")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="sleep medicine"))
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="thuốc trị mất ngủ",
        llm_options=SearchLLMOptions(use_llm=False),
        search_filters=search_filters,
    )

    assert provider.calls == ["thuốc trị mất ngủ"]
    assert candidates == []
    assert top5 == []
    assert search_filters.query_context_metadata["llm_query_rewrite_used"] is False


def test_orchestrator_request_can_force_standard_search_when_llm_env_enabled(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_rerank", 2)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 2)
    _patch_phase1_dependencies(monkeypatch)

    review_session = SimpleNamespace(id=7, status="created", topic="flood detection")
    db = _FakeDB(review_session)

    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=False),
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert all("_llm_query" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_rerank" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_summary" not in (candidate.raw_metadata_json or {}) for candidate in candidates)


def test_orchestrator_query_rewrite_timeout_returns_results_with_original_query(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    def _timeout_rewrite(*args, **kwargs):
        raise TimeoutError("query rewrite timed out")

    monkeypatch.setattr("app.services.orchestrator.rewrite_query", _timeout_rewrite)

    search_filters = SearchFilters(query="weight loss pills")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="elderly weight loss"))
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="weight loss pills",
        llm_options=SearchLLMOptions(use_llm=True, use_query_rewrite=True),
        search_filters=search_filters,
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert search_filters.query_context_metadata["effective_query"] == "weight loss pills"
    assert search_filters.query_context_metadata["status"] == "partial"
    assert "query_rewrite_timeout" in search_filters.query_context_metadata["warnings"]
    assert "query_rewrite:timeout" in search_filters.query_context_metadata["errors"]


def test_orchestrator_llm_rerank_timeout_keeps_phase1_ranking(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    def _timeout_rerank(*args, **kwargs):
        raise TimeoutError("rerank timed out")

    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _timeout_rerank)

    search_filters = SearchFilters(query="weight loss pills")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="elderly weight loss"))
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="weight loss pills",
        llm_options=SearchLLMOptions(use_llm=True, use_llm_rerank=True),
        search_filters=search_filters,
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert all("_llm_rerank" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert search_filters.query_context_metadata["status"] == "partial"
    assert "llm_rerank_timeout" in search_filters.query_context_metadata["warnings"]
    assert "llm_rerank:timeout" in search_filters.query_context_metadata["errors"]


def test_orchestrator_llm_summary_timeout_returns_candidates(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_summary", 3)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_summaries_per_search", 3)
    monkeypatch.setattr("app.services.orchestrator.settings.reranker_enabled", False)
    _patch_phase1_dependencies(monkeypatch)

    def _timeout_summary(*args, **kwargs):
        raise TimeoutError("summary timed out")

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _timeout_summary)

    search_filters = SearchFilters(query="weight loss pills")
    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="elderly weight loss"))
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="weight loss pills",
        llm_options=SearchLLMOptions(use_llm=True, use_llm_summary=True),
        search_filters=search_filters,
    )

    assert len(candidates) == 2
    assert len(top5) == 2
    assert all("_llm_summary" not in (candidate.raw_metadata_json or {}) for candidate in top5)
    assert search_filters.query_context_metadata["status"] == "partial"
    assert "llm_summary_timeout" in search_filters.query_context_metadata["warnings"]
    assert "llm_summary:timeout" in search_filters.query_context_metadata["errors"]


def test_orchestrator_default_request_stays_phase1_when_llm_env_enabled(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    _patch_phase1_dependencies(monkeypatch)

    def _fail_llm_call(*args, **kwargs):
        raise AssertionError("LLM steps should require explicit request opt-in")

    monkeypatch.setattr("app.services.orchestrator.rewrite_query", _fail_llm_call)
    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _fail_llm_call)
    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _fail_llm_call)

    review_session = SimpleNamespace(id=7, status="created", topic="flood detection")
    db = _FakeDB(review_session)

    candidates, top5 = run_search_pipeline(db, session_id=7, query="flood detection")

    assert len(candidates) == 2
    assert len(top5) == 2
    assert all("_llm_query" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_rerank" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_summary" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_diagnostics" not in (candidate.raw_metadata_json or {}) for candidate in candidates)


def test_orchestrator_specific_step_flag_can_opt_in_one_llm_step(monkeypatch):
    monkeypatch.setattr("app.services.orchestrator.settings.llm_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_provider", "mock")
    monkeypatch.setattr("app.services.orchestrator.settings.llm_query_rewrite_enabled", False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_rerank_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_summary_enabled", True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_candidates_for_rerank", 2)
    _patch_phase1_dependencies(monkeypatch)

    def _fail_summary(*args, **kwargs):
        raise AssertionError("summary should not run when only rerank is requested")

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _fail_summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm_rerank=True),
    )

    assert any("_llm_rerank" in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_query" not in (candidate.raw_metadata_json or {}) for candidate in candidates)
    assert all("_llm_summary" not in (candidate.raw_metadata_json or {}) for candidate in candidates)


def test_summary_cache_hit_avoids_llm_regeneration(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    _patch_phase1_dependencies(monkeypatch)

    existing = _existing_candidate(
        metadata=lambda candidate_fp, query_fp: {
            "_llm_summary": {
                "one_sentence_summary": "Cached summary.",
                "contribution": "Cached contribution.",
                "method_or_data": "Cached method.",
                "relevance_to_query": "Cached relevance.",
                "limitations_or_caution": "Cached caution.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": True,
                "attempted": True,
                "prompt_version": "v1",
                "generated_at": "2026-04-28T00:00:00Z",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _fail_summary(*args, **kwargs):
        raise AssertionError("summary should have been served from cache")

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _fail_summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert candidates[0].raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Cached summary."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]["cache_hit"] is True
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]["budget_guard"]["call_count"] == 0


def test_summary_recomputes_when_prompt_version_changes(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_prompt_version", "v2")
    _patch_phase1_dependencies(monkeypatch)
    calls = {"count": 0}
    existing = _existing_candidate(
        metadata=lambda candidate_fp, query_fp: {
            "_llm_summary": {
                "one_sentence_summary": "Old cached summary.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": True,
                "prompt_version": "v1",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _summary(candidate, query=None):
        calls["count"] += 1
        return PaperSummaryResult(
            one_sentence_summary="Fresh summary.",
            contribution="Fresh contribution.",
            method_or_data="Fresh method.",
            relevance_to_query="Fresh relevance.",
            limitations_or_caution="Fresh caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v2",
            generated_at="2026-04-28T00:00:01Z",
            query_fingerprint=query_fingerprint(query),
            candidate_fingerprint=candidate_fingerprint(candidate),
        )

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert calls["count"] == 1
    assert candidates[0].raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Fresh summary."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]["cache_miss_reason"] == "prompt_version_changed"


def test_rerank_cache_hit_uses_existing_metadata(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=True, summary=False)
    _patch_phase1_dependencies(monkeypatch)
    existing = _existing_candidate(
        metadata=lambda candidate_fp, query_fp: {
            "_llm_rerank": {
                "candidate_id": 101,
                "candidate_index": 0,
                "relevance_score": 0.91,
                "reason": "Cached rerank.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": True,
                "attempted": True,
                "prompt_version": "v1",
                "generated_at": "2026-04-28T00:00:00Z",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _fail_rerank(*args, **kwargs):
        raise AssertionError("rerank should have been served from cache")

    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _fail_rerank)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert candidates[0].raw_metadata_json["_llm_rerank"]["reason"] == "Cached rerank."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["rerank"]["cache_hit"] is True


def test_rerank_recomputes_when_query_changes(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=True, summary=False)
    _patch_phase1_dependencies(monkeypatch)
    calls = {"count": 0}
    existing = _existing_candidate(
        query="old query",
        metadata=lambda candidate_fp, query_fp: {
            "_llm_rerank": {
                "candidate_id": 101,
                "candidate_index": 0,
                "relevance_score": 0.2,
                "reason": "Old query rerank.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": True,
                "prompt_version": "v1",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _rerank(query, candidates):
        calls["count"] += 1
        return [
            RerankResult(
                candidate_id=candidates[0].id,
                candidate_index=0,
                llm_relevance_score=0.88,
                reason="Fresh rerank.",
                provider="mock",
                model="mock-test-model",
                used_llm=True,
                attempted=True,
                prompt_version="v1",
                generated_at="2026-04-28T00:00:01Z",
                query_fingerprint=query_fingerprint(query),
                candidate_fingerprint=candidate_fingerprint(candidates[0]),
            )
        ]

    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _rerank)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert calls["count"] == 1
    assert candidates[0].raw_metadata_json["_llm_rerank"]["reason"] == "Fresh rerank."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["rerank"]["cache_hit"] is False


def test_malformed_summary_cache_does_not_crash(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    _patch_phase1_dependencies(monkeypatch)
    calls = {"count": 0}
    existing = _existing_candidate(
        metadata=lambda candidate_fp, query_fp: {
            "_llm_summary": {
                "one_sentence_summary": "Malformed cached summary.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": False,
                "prompt_version": "v1",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _summary(candidate, query=None):
        calls["count"] += 1
        return PaperSummaryResult(
            one_sentence_summary="Recovered summary.",
            contribution="Recovered contribution.",
            method_or_data="Recovered method.",
            relevance_to_query="Recovered relevance.",
            limitations_or_caution="Recovered caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
            prompt_version="v1",
            generated_at="2026-04-28T00:00:01Z",
            query_fingerprint=query_fingerprint(query),
            candidate_fingerprint=candidate_fingerprint(candidate),
        )

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert calls["count"] == 1
    assert candidates[0].raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Recovered summary."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]["cache_miss_reason"] == "malformed_cached_artifact"


def test_llm_cache_disabled_bypasses_valid_summary_cache(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_cache_enabled", False)
    _patch_phase1_dependencies(monkeypatch)
    calls = {"count": 0}
    existing = _existing_candidate(
        metadata=lambda candidate_fp, query_fp: {
            "_llm_summary": {
                "one_sentence_summary": "Cached summary.",
                "provider": "mock",
                "model": "mock-test-model",
                "used_llm": True,
                "prompt_version": "v1",
                "query_fingerprint": query_fp,
                "candidate_fingerprint": candidate_fp,
            }
        }
    )

    def _summary(candidate, query=None):
        calls["count"] += 1
        return PaperSummaryResult(
            one_sentence_summary="Cache disabled summary.",
            contribution="Fresh contribution.",
            method_or_data="Fresh method.",
            relevance_to_query="Fresh relevance.",
            limitations_or_caution="Fresh caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
        )

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"), existing_candidates=[existing])
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert calls["count"] == 1
    assert candidates[0].raw_metadata_json["_llm_summary"]["one_sentence_summary"] == "Cache disabled summary."
    assert candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]["cache_miss_reason"] == "cache_disabled"


def test_llm_candidate_limits_are_respected(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=True, summary=True)
    _patch_phase1_dependencies(monkeypatch)
    rerank_sizes = []
    summary_calls = {"count": 0}

    def _rerank(query, candidates):
        rerank_sizes.append(len(candidates))
        return [
            RerankResult(
                candidate_id=candidate.id,
                candidate_index=index,
                llm_relevance_score=0.5,
                reason="Limited rerank.",
                provider="mock",
                model="mock-test-model",
                used_llm=True,
                attempted=True,
            )
            for index, candidate in enumerate(candidates)
        ]

    def _summary(candidate, query=None):
        summary_calls["count"] += 1
        return PaperSummaryResult(
            one_sentence_summary="Limited summary.",
            contribution="Limited contribution.",
            method_or_data="Limited method.",
            relevance_to_query="Limited relevance.",
            limitations_or_caution="Limited caution.",
            provider="mock",
            model="mock-test-model",
            used_llm=True,
            attempted=True,
        )

    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _rerank)
    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert rerank_sizes == [1]
    assert summary_calls["count"] == 1


def test_max_call_limit_prevents_additional_summary_calls(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_calls_per_search", 0)
    _patch_phase1_dependencies(monkeypatch)

    def _summary(*args, **kwargs):
        raise AssertionError("summary should be skipped by max call guard")

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    diagnostics = candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]
    assert diagnostics["used"] is False
    assert diagnostics["fallback_reason"] == "max_calls_exceeded"
    assert diagnostics["budget_guard"]["allowed"] is False


def test_estimated_token_limit_skips_llm_work(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=True, summary=False)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_estimated_tokens_per_search", 1)
    _patch_phase1_dependencies(monkeypatch)

    def _rerank(*args, **kwargs):
        raise AssertionError("rerank should be skipped by token guard")

    monkeypatch.setattr("app.services.orchestrator.rerank_candidates", _rerank)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    diagnostics = candidates[0].raw_metadata_json["_llm_diagnostics"]["rerank"]
    assert diagnostics["fallback_reason"] == "estimated_tokens_exceeded"
    assert diagnostics["estimated_usage"]["estimated_total_tokens"] > 1
    assert diagnostics["budget_guard"]["allowed"] is False


class _LongAbstractProvider:
    def fetch_papers(self, query: str, limit: int = 20):
        return [
            PaperResult(
                title="Long Abstract Paper",
                abstract="x" * 5000,
                year=2024,
                authors=["Author A"],
                doi="10.1000/long",
                url="https://example.com/long",
                source_title="Journal A",
                citation_count=12,
                provider="semantic_scholar",
                raw_metadata={"paper_id": "long"},
            )
        ]


def test_oversized_input_is_truncated_safely(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_input_chars_per_call", 1800)
    _patch_phase1_dependencies(monkeypatch)
    monkeypatch.setattr(
        "app.services.orchestrator.get_search_provider",
        lambda: _LongAbstractProvider(),
    )

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, _ = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    summary = candidates[0].raw_metadata_json["_llm_summary"]
    diagnostics = candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]
    assert summary["input_truncated"] is True
    assert diagnostics["input_truncated"] is True
    assert diagnostics["used"] is True


def test_input_too_large_budget_skip_does_not_crash(monkeypatch):
    _enable_llm_cache_settings(monkeypatch, rerank=False, summary=True)
    monkeypatch.setattr("app.services.orchestrator.settings.llm_max_input_chars_per_call", 10)
    _patch_phase1_dependencies(monkeypatch)
    monkeypatch.setattr(
        "app.services.orchestrator.get_search_provider",
        lambda: _LongAbstractProvider(),
    )

    def _summary(*args, **kwargs):
        raise AssertionError("summary should be skipped before LLM call")

    monkeypatch.setattr("app.services.orchestrator.summarize_paper", _summary)

    db = _FakeDB(SimpleNamespace(id=7, status="created", topic="flood detection"))
    candidates, top5 = run_search_pipeline(
        db,
        session_id=7,
        query="flood detection",
        llm_options=SearchLLMOptions(use_llm=True),
    )

    assert len(candidates) == 1
    assert len(top5) == 1
    diagnostics = candidates[0].raw_metadata_json["_llm_diagnostics"]["summary"]
    assert diagnostics["fallback_reason"] == "input_too_large"
    assert diagnostics["budget_guard"]["allowed"] is False
