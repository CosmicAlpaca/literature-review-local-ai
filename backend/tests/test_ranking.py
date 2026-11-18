from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.base import PaperResult
from app.services.ranking import build_selection_debug, compute_score, score_paper, score_trust


def test_ranking_rewards_query_overlap_and_metadata():
    strong = PaperResult(
        title="Flood Detection with Deep Learning for Satellite Images",
        abstract="This paper studies flood detection with deep learning and remote sensing data.",
        year=2024,
        source_title="Remote Sensing",
        citation_count=120,
        provider="semantic_scholar",
    )
    weak = PaperResult(
        title="General Image Compression Survey",
        abstract="Compression techniques for images.",
        year=2015,
        source_title=None,
        citation_count=3,
        provider="mock",
    )

    strong_score = compute_score(
        "flood detection deep learning",
        strong,
        scopus_valid=True,
        source_match={"confidence": 0.99, "reason": "exact_normalized_title"},
    )
    weak_score = compute_score(
        "flood detection deep learning",
        weak,
        scopus_valid=False,
        source_match={"confidence": 0.0, "reason": "no_match"},
    )

    assert strong_score > weak_score


def test_ranking_breakdown_is_transparent_and_bounded():
    paper = PaperResult(
        title="Flood Detection for Rivers",
        abstract="Flood detection and segmentation for rivers.",
        year=2023,
        source_title="Water Research",
        citation_count=10,
        provider="semantic_scholar",
    )

    breakdown = score_paper(
        "flood detection",
        paper,
        scopus_valid=True,
        source_match={"confidence": 0.96, "reason": "exact_normalized_title"},
    )

    assert set(breakdown.keys()) == {"weights", "components", "weighted", "total"}
    assert 0.0 <= breakdown["total"] <= 1.0
    assert breakdown["components"]["title_relevance"] > 0.0
    assert breakdown["components"]["abstract_relevance"] > 0.0
    assert breakdown["components"]["source_match_strength"] == 0.96
    assert breakdown["components"]["source_match_reason_bonus"] == 0.95


def test_ranking_prefers_stronger_source_match_when_other_signals_are_equal():
    paper = PaperResult(
        title="Flood Detection for Rivers",
        abstract="Flood detection and segmentation for rivers.",
        year=2023,
        source_title="Water Research",
        citation_count=10,
        provider="semantic_scholar",
    )

    stronger = compute_score(
        "flood detection",
        paper,
        scopus_valid=True,
        source_match={"confidence": 1.0, "reason": "identifier_match:issn"},
    )
    weaker = compute_score(
        "flood detection",
        paper,
        scopus_valid=True,
        source_match={"confidence": 0.82, "reason": "high_confidence_fuzzy_title"},
    )

    assert stronger > weaker


def test_trust_scoring_assigns_high_medium_low_buckets():
    high = PaperResult(
        title="Flood Detection with Deep Learning",
        abstract="Flood detection study with satellite imagery.",
        year=2024,
        doi="10.1000/high",
        source_title="Remote Sensing",
        citation_count=120,
        provider="semantic_scholar",
    )
    medium = PaperResult(
        title="Flood Monitoring Overview",
        abstract=None,
        year=2012,
        doi=None,
        source_title="Water",
        citation_count=0,
        provider="semantic_scholar",
    )
    low = PaperResult(
        title="Short Note",
        abstract=None,
        year=None,
        doi=None,
        source_title=None,
        citation_count=None,
        provider="mock",
    )

    high_trust = score_trust(
        high,
        source_match={
            "confidence": 1.0,
            "reason": "identifier_match:issn",
            "registry_metadata": {
                "registry_category": "active_indexed_journal",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
                "medline_sourced": True,
            },
        },
    )
    medium_trust = score_trust(
        medium,
        source_match={
            "confidence": 0.55,
            "reason": "high_confidence_fuzzy_title",
            "registry_metadata": {
                "registry_category": "accepted_title",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
            },
        },
    )
    low_trust = score_trust(
        low,
        source_match={"confidence": 0.0, "reason": "no_match"},
    )

    assert high_trust["bucket"] == "high"
    assert medium_trust["bucket"] == "medium"
    assert low_trust["bucket"] == "low"
    assert high_trust["total"] > medium_trust["total"] > low_trust["total"]


def test_selection_policy_penalizes_low_trust_when_relevance_is_close():
    stronger_ranking = build_selection_debug(
        ranking_score=0.82,
        trust_debug={"total": 0.28, "bucket": "low"},
    )
    stronger_trust = build_selection_debug(
        ranking_score=0.79,
        trust_debug={"total": 0.82, "bucket": "high"},
    )

    assert stronger_trust["combined_score"] > stronger_ranking["combined_score"]


def test_trust_distinguishes_active_accepted_and_discontinued_registry_categories():
    paper = PaperResult(
        title="Remote Sensing for Flood Detection",
        abstract="Flood detection from satellite images.",
        year=2024,
        doi="10.1000/example",
        source_title="Remote Sensing",
        citation_count=42,
        provider="semantic_scholar",
    )

    active = score_trust(
        paper,
        source_match={
            "confidence": 0.97,
            "reason": "exact_normalized_title",
            "registry_metadata": {
                "registry_category": "active_indexed_journal",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
                "medline_sourced": True,
            },
        },
    )
    accepted = score_trust(
        paper,
        source_match={
            "confidence": 0.72,
            "reason": "related_title_match",
            "registry_metadata": {
                "registry_category": "accepted_title",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
                "medline_sourced": False,
            },
        },
    )
    discontinued = score_trust(
        paper,
        source_match={
            "confidence": 0.44,
            "reason": "exact_normalized_title",
            "registry_metadata": {
                "registry_category": "discontinued_title",
                "source_type": "Journal",
                "is_discontinued": True,
                "discontinued_due_to_quality": True,
                "medline_sourced": False,
            },
        },
    )

    assert active["total"] > accepted["total"] > discontinued["total"]
    assert active["components"]["registry_category_strength"] == 1.0
    assert accepted["components"]["registry_category_strength"] == 0.58
    assert discontinued["components"]["discontinued_safety"] == 0.0


def test_trust_penalizes_conference_sources_relative_to_journals():
    journal_paper = PaperResult(
        title="Flood Detection Methods",
        abstract="Flood detection methods in journal literature.",
        year=2023,
        doi="10.1000/journal",
        source_title="Hydrology Journal",
        citation_count=25,
        provider="semantic_scholar",
    )
    conference_paper = PaperResult(
        title="Flood Detection Methods",
        abstract="Flood detection methods in conference literature.",
        year=2023,
        doi="10.1000/conf",
        source_title="Proceedings of Hydrology AI",
        citation_count=25,
        provider="semantic_scholar",
        raw_metadata={"source_type": "conference"},
    )

    journal_trust = score_trust(
        journal_paper,
        source_match={
            "confidence": 0.95,
            "reason": "exact_normalized_title",
            "registry_metadata": {
                "registry_category": "active_indexed_journal",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
            },
        },
    )
    conference_trust = score_trust(
        conference_paper,
        source_match={
            "confidence": 0.78,
            "reason": "exact_normalized_title",
            "registry_metadata": {
                "registry_category": "conference_proceeding",
                "source_type": "Conference Proceeding",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
            },
        },
    )

    assert journal_trust["total"] > conference_trust["total"]
    assert journal_trust["components"]["journal_appropriateness"] == 1.0
    assert conference_trust["components"]["journal_appropriateness"] == 0.15


def test_openalex_strict_evidence_is_small_and_cannot_override_poor_registry_signals():
    active_semantic = PaperResult(
        title="Flood Detection Study",
        abstract="Trustworthy journal article.",
        year=2024,
        doi="10.1000/active",
        source_title="Remote Sensing",
        citation_count=30,
        provider="semantic_scholar",
    )
    openalex_poor_registry = PaperResult(
        title="Flood Detection Study",
        abstract="Strictly filtered retrieval candidate.",
        year=2024,
        doi="10.1000/openalex",
        source_title="Questionable Proceedings",
        citation_count=30,
        provider="openalex_strict",
        raw_metadata={
            "source_type": "journal",
            "is_core": True,
            "is_in_doaj": True,
            "is_retracted": False,
            "filter_provenance": {
                "provider_mode": "openalex_strict",
                "filter": "type:article,...",
            },
        },
    )

    active_trust = score_trust(
        active_semantic,
        source_match={
            "confidence": 0.96,
            "reason": "identifier_match:issn",
            "registry_metadata": {
                "registry_category": "active_indexed_journal",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
            },
        },
    )
    openalex_trust = score_trust(
        openalex_poor_registry,
        source_match={
            "confidence": 0.55,
            "reason": "high_confidence_fuzzy_title",
            "registry_metadata": {
                "registry_category": "accepted_title",
                "source_type": "Journal",
                "is_discontinued": False,
                "discontinued_due_to_quality": False,
            },
        },
    )

    assert openalex_trust["components"]["strict_provider_evidence"] == 1.0
    assert active_trust["total"] > openalex_trust["total"]
    assert openalex_trust["bucket"] in {"medium", "high"}
