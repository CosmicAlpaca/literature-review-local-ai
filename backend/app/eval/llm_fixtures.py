from __future__ import annotations

from copy import deepcopy
from typing import Any


LLM_EVAL_CASES: list[dict[str, Any]] = [
    {
        "case_id": "early_cancer_detection",
        "topic": "early cancer detection with AI",
        "raw_query": "early cancer detection using artificial intelligence",
        "expected_terms": ["early", "cancer", "detection", "artificial", "intelligence"],
        "expected_top_candidate_title": "Artificial intelligence for early detection of cancer from imaging biomarkers",
        "candidates": [
            {
                "id": 101,
                "title": "Artificial intelligence for early detection of cancer from imaging biomarkers",
                "abstract": (
                    "This review examines machine learning and deep learning models for early cancer "
                    "detection using radiology, pathology, and multimodal imaging biomarkers."
                ),
                "source_title": "Cancer Imaging",
                "year": 2024,
                "citation_count": 42,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-cancer-101"},
            },
            {
                "id": 102,
                "title": "Hospital staffing patterns in oncology departments",
                "abstract": (
                    "The study evaluates staffing levels, shift schedules, and administrative workload "
                    "in oncology departments across regional hospitals."
                ),
                "source_title": "Health Services Management",
                "year": 2021,
                "citation_count": 11,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-cancer-102"},
            },
            {
                "id": 103,
                "title": "Blood pressure prediction using wearable sensors",
                "abstract": (
                    "Wearable sensor signals are used to predict blood pressure trends with supervised "
                    "learning models in ambulatory care."
                ),
                "source_title": "Biomedical Signal Processing",
                "year": 2022,
                "citation_count": 28,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-cancer-103"},
            },
        ],
    },
    {
        "case_id": "flood_detection_remote_sensing",
        "topic": "flood detection with remote sensing",
        "raw_query": "flood detection from satellite images",
        "expected_terms": ["flood", "detection", "satellite", "remote", "sensing"],
        "expected_top_candidate_title": "Satellite remote sensing and deep learning for rapid flood detection",
        "candidates": [
            {
                "id": 201,
                "title": "Satellite remote sensing and deep learning for rapid flood detection",
                "abstract": (
                    "A convolutional neural network maps flood extent from multispectral satellite "
                    "imagery after severe rainfall and river overflow events."
                ),
                "source_title": "Remote Sensing of Environment",
                "year": 2023,
                "citation_count": 67,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-flood-201"},
            },
            {
                "id": 202,
                "title": "Urban traffic congestion prediction using graph neural networks",
                "abstract": (
                    "Road network sensors and graph neural networks are used to forecast traffic "
                    "congestion during peak commuting hours."
                ),
                "source_title": "Transportation Research",
                "year": 2022,
                "citation_count": 54,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-flood-202"},
            },
            {
                "id": 203,
                "title": "Hydrologic simulation of drainage capacity under storm scenarios",
                "abstract": (
                    "A physically based hydrologic model estimates drainage capacity and pluvial flood "
                    "risk under synthetic storm events."
                ),
                "source_title": "Journal of Hydrology",
                "year": 2020,
                "citation_count": 33,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-flood-203"},
            },
        ],
    },
    {
        "case_id": "llm_hallucination_detection",
        "topic": "LLM hallucination detection",
        "raw_query": "detect hallucinations in large language model answers",
        "expected_terms": ["hallucinations", "large", "language", "model", "detection"],
        "expected_top_candidate_title": "Detecting hallucinations in large language model generated answers",
        "candidates": [
            {
                "id": 301,
                "title": "Detecting hallucinations in large language model generated answers",
                "abstract": (
                    "The paper benchmarks uncertainty estimation, retrieval grounding, and factuality "
                    "classifiers for detecting hallucinations in generated LLM answers."
                ),
                "source_title": "Transactions on Machine Learning Research",
                "year": 2024,
                "citation_count": 91,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-hallucination-301"},
            },
            {
                "id": 302,
                "title": "Tokenization efficiency for multilingual language models",
                "abstract": (
                    "This work compares tokenization strategies for multilingual transformer models "
                    "across low-resource languages."
                ),
                "source_title": "Computational Linguistics",
                "year": 2023,
                "citation_count": 40,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-hallucination-302"},
            },
            {
                "id": 303,
                "title": "Knowledge graph completion with link prediction",
                "abstract": (
                    "Embedding models are evaluated for knowledge graph completion and missing link "
                    "prediction tasks."
                ),
                "source_title": "Semantic Web Journal",
                "year": 2021,
                "citation_count": 39,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-hallucination-303"},
            },
        ],
    },
    {
        "case_id": "clinical_decision_support_ai",
        "topic": "clinical decision support AI",
        "raw_query": "artificial intelligence clinical decision support systems",
        "expected_terms": ["artificial", "intelligence", "clinical", "decision", "support"],
        "expected_top_candidate_title": "Artificial intelligence clinical decision support for diagnosis and triage",
        "candidates": [
            {
                "id": 401,
                "title": "Artificial intelligence clinical decision support for diagnosis and triage",
                "abstract": (
                    "Machine learning decision support systems assist diagnosis, triage, and risk "
                    "stratification from electronic health record and imaging data."
                ),
                "source_title": "Journal of Medical Internet Research",
                "year": 2024,
                "citation_count": 76,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-clinical-401"},
            },
            {
                "id": 402,
                "title": "Patient satisfaction with outpatient appointment scheduling",
                "abstract": (
                    "Survey data are analyzed to understand patient satisfaction with outpatient "
                    "appointment scheduling systems."
                ),
                "source_title": "BMC Health Services Research",
                "year": 2020,
                "citation_count": 18,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-clinical-402"},
            },
            {
                "id": 403,
                "title": "Federated learning for hospital data privacy",
                "abstract": (
                    "Federated learning allows multiple hospitals to train predictive models while "
                    "limiting direct exchange of patient-level data."
                ),
                "source_title": "npj Digital Medicine",
                "year": 2022,
                "citation_count": 82,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-clinical-403"},
            },
        ],
    },
    {
        "case_id": "systematic_review_automation",
        "topic": "systematic review automation",
        "raw_query": "automation tools for systematic literature review screening",
        "expected_terms": ["automation", "systematic", "literature", "review", "screening"],
        "expected_top_candidate_title": "Automation tools for systematic review screening and study selection",
        "candidates": [
            {
                "id": 501,
                "title": "Automation tools for systematic review screening and study selection",
                "abstract": (
                    "The study evaluates active learning, prioritization, and text classification tools "
                    "for title and abstract screening in systematic literature reviews."
                ),
                "source_title": "Systematic Reviews",
                "year": 2023,
                "citation_count": 58,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-review-501"},
            },
            {
                "id": 502,
                "title": "Reference manager adoption among graduate students",
                "abstract": (
                    "A survey examines reference manager preferences, citation workflows, and training "
                    "needs among graduate students."
                ),
                "source_title": "Journal of Academic Librarianship",
                "year": 2021,
                "citation_count": 16,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-review-502"},
            },
            {
                "id": 503,
                "title": "Bibliometric mapping of renewable energy research",
                "abstract": (
                    "Bibliometric methods identify publication clusters and collaboration networks in "
                    "renewable energy research."
                ),
                "source_title": "Scientometrics",
                "year": 2020,
                "citation_count": 45,
                "provider": "static_eval",
                "raw_metadata_json": {"paper_id": "eval-review-503"},
            },
        ],
    },
]


def load_llm_eval_cases() -> list[dict[str, Any]]:
    return deepcopy(LLM_EVAL_CASES)


def get_llm_eval_case(case_id: str) -> dict[str, Any]:
    for case in LLM_EVAL_CASES:
        if case["case_id"] == case_id:
            return deepcopy(case)
    available = ", ".join(case["case_id"] for case in LLM_EVAL_CASES)
    raise ValueError(f"unknown eval case '{case_id}'. Available cases: {available}")
