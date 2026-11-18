from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_COMPARISON_CASES: list[dict[str, Any]] = [
    {
        "case_id": "early_cancer_detection",
        "query": "early cancer detection using artificial intelligence",
        "expected_best_candidate_id": 101,
        "candidates": [
            {
                "candidate_id": 101,
                "title": "Artificial intelligence for early detection of cancer from imaging biomarkers",
                "abstract": (
                    "Machine learning and deep learning models support early cancer detection "
                    "using radiology, pathology, and multimodal imaging biomarkers."
                ),
                "source_title": "Cancer Imaging",
                "year": 2024,
                "citation_count": 42,
                "baseline_score": 0.91,
                "label": 2,
                "provider": "static_eval",
            },
            {
                "candidate_id": 102,
                "title": "Hospital staffing patterns in oncology departments",
                "abstract": "Staffing levels and administrative workload are evaluated in oncology departments.",
                "source_title": "Health Services Management",
                "year": 2021,
                "citation_count": 11,
                "baseline_score": 0.35,
                "label": 0,
                "provider": "static_eval",
            },
            {
                "candidate_id": 103,
                "title": "Wearable sensor models for blood pressure prediction",
                "abstract": "Wearable sensor signals are used to predict blood pressure trends.",
                "source_title": "Biomedical Signal Processing",
                "year": 2022,
                "citation_count": 28,
                "baseline_score": 0.42,
                "label": 1,
                "provider": "static_eval",
            },
        ],
    },
    {
        "case_id": "flood_detection_remote_sensing",
        "query": "flood detection from satellite remote sensing images",
        "expected_best_candidate_id": 201,
        "candidates": [
            {
                "candidate_id": 201,
                "title": "Satellite remote sensing and deep learning for rapid flood detection",
                "abstract": (
                    "A convolutional neural network maps flood extent from multispectral satellite "
                    "imagery after severe rainfall and river overflow events."
                ),
                "source_title": "Remote Sensing of Environment",
                "year": 2023,
                "citation_count": 67,
                "baseline_score": 0.94,
                "label": 2,
                "provider": "static_eval",
            },
            {
                "candidate_id": 202,
                "title": "Urban traffic congestion prediction using graph neural networks",
                "abstract": "Road sensors and graph neural networks forecast traffic congestion.",
                "source_title": "Transportation Research",
                "year": 2022,
                "citation_count": 54,
                "baseline_score": 0.31,
                "label": 0,
                "provider": "static_eval",
            },
            {
                "candidate_id": 203,
                "title": "Hydrologic simulation of drainage capacity under storm scenarios",
                "abstract": "Hydrologic simulation estimates drainage capacity and pluvial flood risk.",
                "source_title": "Journal of Hydrology",
                "year": 2020,
                "citation_count": 33,
                "baseline_score": 0.63,
                "label": 1,
                "provider": "static_eval",
            },
        ],
    },
    {
        "case_id": "llm_hallucination_detection",
        "query": "detect hallucinations in large language model answers",
        "expected_best_candidate_id": 301,
        "candidates": [
            {
                "candidate_id": 301,
                "title": "Detecting hallucinations in large language model generated answers",
                "abstract": (
                    "The paper benchmarks retrieval grounding, uncertainty estimation, and factuality "
                    "classifiers for detecting hallucinations in generated LLM answers."
                ),
                "source_title": "Transactions on Machine Learning Research",
                "year": 2024,
                "citation_count": 91,
                "baseline_score": 0.96,
                "label": 2,
                "provider": "static_eval",
            },
            {
                "candidate_id": 302,
                "title": "Tokenization efficiency for multilingual language models",
                "abstract": "Tokenization strategies are compared across multilingual transformer models.",
                "source_title": "Computational Linguistics",
                "year": 2023,
                "citation_count": 40,
                "baseline_score": 0.52,
                "label": 1,
                "provider": "static_eval",
            },
            {
                "candidate_id": 303,
                "title": "Knowledge graph completion with link prediction",
                "abstract": "Embedding models are evaluated for knowledge graph completion tasks.",
                "source_title": "Semantic Web Journal",
                "year": 2021,
                "citation_count": 39,
                "baseline_score": 0.27,
                "label": 0,
                "provider": "static_eval",
            },
        ],
    },
    {
        "case_id": "clinical_decision_support_ai",
        "query": "artificial intelligence clinical decision support systems",
        "expected_best_candidate_id": 401,
        "candidates": [
            {
                "candidate_id": 401,
                "title": "Artificial intelligence clinical decision support for diagnosis and triage",
                "abstract": (
                    "Machine learning decision support systems assist diagnosis, triage, and risk "
                    "stratification from electronic health record and imaging data."
                ),
                "source_title": "Journal of Medical Internet Research",
                "year": 2024,
                "citation_count": 76,
                "baseline_score": 0.93,
                "label": 2,
                "provider": "static_eval",
            },
            {
                "candidate_id": 402,
                "title": "Patient satisfaction with outpatient appointment scheduling",
                "abstract": "Survey data describe patient satisfaction with appointment scheduling systems.",
                "source_title": "BMC Health Services Research",
                "year": 2020,
                "citation_count": 18,
                "baseline_score": 0.36,
                "label": 0,
                "provider": "static_eval",
            },
            {
                "candidate_id": 403,
                "title": "Federated learning for hospital data privacy",
                "abstract": "Federated learning trains predictive models while limiting patient data exchange.",
                "source_title": "npj Digital Medicine",
                "year": 2022,
                "citation_count": 82,
                "baseline_score": 0.58,
                "label": 1,
                "provider": "static_eval",
            },
        ],
    },
    {
        "case_id": "systematic_review_automation",
        "query": "automation tools for systematic literature review screening",
        "expected_best_candidate_id": 501,
        "candidates": [
            {
                "candidate_id": 501,
                "title": "Automation tools for systematic review screening and study selection",
                "abstract": (
                    "Active learning, prioritization, and text classification tools are evaluated for "
                    "title and abstract screening in systematic literature reviews."
                ),
                "source_title": "Systematic Reviews",
                "year": 2023,
                "citation_count": 58,
                "baseline_score": 0.95,
                "label": 2,
                "provider": "static_eval",
            },
            {
                "candidate_id": 502,
                "title": "Reference manager adoption among graduate students",
                "abstract": "A survey examines citation workflows and reference manager preferences.",
                "source_title": "Journal of Academic Librarianship",
                "year": 2021,
                "citation_count": 16,
                "baseline_score": 0.41,
                "label": 0,
                "provider": "static_eval",
            },
            {
                "candidate_id": 503,
                "title": "Bibliometric mapping of renewable energy research",
                "abstract": "Bibliometric methods identify publication clusters in renewable energy research.",
                "source_title": "Scientometrics",
                "year": 2020,
                "citation_count": 45,
                "baseline_score": 0.44,
                "label": 0,
                "provider": "static_eval",
            },
        ],
    },
]


def load_model_comparison_cases() -> list[dict[str, Any]]:
    return deepcopy(MODEL_COMPARISON_CASES)


def get_model_comparison_case(case_id: str) -> dict[str, Any]:
    for case in MODEL_COMPARISON_CASES:
        if case["case_id"] == case_id:
            return deepcopy(case)
    available = ", ".join(case["case_id"] for case in MODEL_COMPARISON_CASES)
    raise ValueError(f"unknown model comparison case '{case_id}'. Available cases: {available}")
