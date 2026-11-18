from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, TextIO

from app.core.config import settings
from app.eval.model_comparison_fixtures import (
    get_model_comparison_case,
    load_model_comparison_cases,
)
from app.services.llm_client import LLMClient, utc_timestamp
from app.services.llm_reranker import rerank_candidates
from app.services.local_reranker import score_candidates


METHODS = ["baseline", "local_reranker", "llm_rerank"]


def main(
    argv: list[str] | None = None,
    *,
    settings_obj: Any = settings,
    client: LLMClient | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline fixture scores, LLM rerank scores, and local reranker scores."
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force deterministic mock LLM/local reranker behavior and avoid network calls.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--case", help="Run one fixture case by case_id.")
    parser.add_argument("--output", help="Write a JSON report to this path.")
    args = parser.parse_args(argv)

    out = stdout or sys.stdout
    try:
        cases = [get_model_comparison_case(args.case)] if args.case else load_model_comparison_cases()
        run_settings = _mock_settings(settings_obj) if args.mock_only else settings_obj
        run_client = client or LLMClient(run_settings)
        report = run_comparison(cases=cases, settings_obj=run_settings, client=run_client)
        if args.output:
            _write_json_report(report, args.output)
    except ValueError as exc:
        report = _safe_error_report(reason="invalid_case", detail=str(exc))
        _write_output(report, json_output=args.json, stdout=out)
        return 2
    except Exception:
        report = _safe_error_report(reason="comparison_failed")
        _write_output(report, json_output=args.json, stdout=out)
        return 1

    _write_output(report, json_output=args.json, stdout=out)
    return 0


def run_comparison(
    *,
    cases: list[dict[str, Any]] | None = None,
    settings_obj: Any = settings,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    selected_cases = cases or load_model_comparison_cases()
    client = client or LLMClient(settings_obj)
    case_reports = [
        _evaluate_case(case=case, settings_obj=settings_obj, client=client)
        for case in selected_cases
    ]
    passed_cases = sum(1 for case in case_reports if case["passed"])

    return {
        "created_at": utc_timestamp(),
        "local_reranker": {
            "enabled": bool(getattr(settings_obj, "local_reranker_enabled", False)),
            "model_provider": str(getattr(settings_obj, "local_reranker_provider", "mock") or "mock"),
            "model_version": str(getattr(settings_obj, "local_reranker_model_version", "v1") or "v1"),
        },
        "llm_rerank": {
            "provider": client.provider,
            "model": client.model,
            "prompt_version": str(getattr(settings_obj, "llm_prompt_version", "v1") or "v1"),
        },
        "methods": METHODS,
        "retrieval_provider_called": False,
        "database_access_required": False,
        "training_required": False,
        "cases": case_reports,
        "summary": _summary(case_reports, passed_cases=passed_cases),
    }


def precision_at_k(ranked_candidate_ids: list[Any], labels_by_candidate_id: dict[Any, int], k: int = 3) -> float:
    if k <= 0:
        return 0.0
    top_ids = ranked_candidate_ids[:k]
    if not top_ids:
        return 0.0
    relevant = sum(1 for candidate_id in top_ids if int(labels_by_candidate_id.get(candidate_id, 0) or 0) > 0)
    return round(relevant / len(top_ids), 4)


def ndcg_at_k(ranked_candidate_ids: list[Any], labels_by_candidate_id: dict[Any, int], k: int = 3) -> float:
    if k <= 0:
        return 0.0
    gains = [int(labels_by_candidate_id.get(candidate_id, 0) or 0) for candidate_id in ranked_candidate_ids[:k]]
    ideal = sorted((int(value or 0) for value in labels_by_candidate_id.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return round(_dcg(gains) / ideal_dcg, 4)


def topk_overlap(first_ids: list[Any], second_ids: list[Any], k: int = 3) -> float:
    if k <= 0:
        return 0.0
    first = set(first_ids[:k])
    second = set(second_ids[:k])
    if not first and not second:
        return 0.0
    return round(len(first & second) / max(len(first | second), 1), 4)


def _evaluate_case(*, case: dict[str, Any], settings_obj: Any, client: LLMClient) -> dict[str, Any]:
    query = str(case["query"])
    candidates = list(case["candidates"])
    labels = {candidate["candidate_id"]: int(candidate.get("label", 0) or 0) for candidate in candidates}
    expected_best = case.get("expected_best_candidate_id")

    baseline = _baseline_ranking(candidates)
    local = _local_ranking(query=query, candidates=candidates, settings_obj=settings_obj)
    llm = _llm_ranking(query=query, candidates=candidates, client=client)
    rankings = {
        "baseline": baseline,
        "local_reranker": local,
        "llm_rerank": llm,
    }
    metrics = {
        method: _method_metrics(
            ranking=ranking,
            labels=labels,
            expected_best_candidate_id=expected_best,
        )
        for method, ranking in rankings.items()
    }
    method_topk_overlap = {
        "baseline_local_reranker": topk_overlap(
            _ranking_ids(baseline),
            _ranking_ids(local),
            3,
        ),
        "baseline_llm_rerank": topk_overlap(
            _ranking_ids(baseline),
            _ranking_ids(llm),
            3,
        ),
        "local_reranker_llm_rerank": topk_overlap(
            _ranking_ids(local),
            _ranking_ids(llm),
            3,
        ),
    }

    return {
        "case_id": case["case_id"],
        "query": query,
        "expected_best_candidate_id": expected_best,
        "metrics": metrics,
        "method_topk_overlap": method_topk_overlap,
        "rankings": rankings,
        "passed": all(metric["passed"] for metric in metrics.values()),
    }


def _baseline_ranking(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -_float_or_default(candidate.get("baseline_score"), 0.0),
            str(candidate.get("candidate_id")),
        ),
    )
    return [
        _ranking_item(
            candidate_id=candidate.get("candidate_id"),
            score=_float_or_default(candidate.get("baseline_score"), 0.0),
            reason="Fixture baseline_score.",
            fallback_reason=None,
        )
        for candidate in ranked
    ]


def _local_ranking(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    settings_obj: Any,
) -> list[dict[str, Any]]:
    result = score_candidates(query, candidates, settings_obj=settings_obj)
    return sorted(
        [
            _ranking_item(
                candidate_id=item.candidate_id,
                score=item.local_relevance_score,
                reason=item.reason,
                fallback_reason=item.fallback_reason,
            )
            for item in result.results
        ],
        key=lambda item: (-item["score"], str(item["candidate_id"])),
    )


def _llm_ranking(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    client: LLMClient,
) -> list[dict[str, Any]]:
    llm_candidates = [
        {
            "id": candidate.get("candidate_id"),
            "title": candidate.get("title"),
            "abstract": candidate.get("abstract"),
            "source_title": candidate.get("source_title"),
            "year": candidate.get("year"),
            "citation_count": candidate.get("citation_count"),
            "provider": candidate.get("provider"),
        }
        for candidate in candidates
    ]
    try:
        results = rerank_candidates(query=query, candidates=llm_candidates, client=client)
    except Exception:
        results = []

    return sorted(
        [
            _ranking_item(
                candidate_id=result.candidate_id,
                score=result.llm_relevance_score,
                reason=result.reason,
                fallback_reason=result.fallback_reason,
            )
            for result in results
        ],
        key=lambda item: (-item["score"], str(item["candidate_id"])),
    )


def _method_metrics(
    *,
    ranking: list[dict[str, Any]],
    labels: dict[Any, int],
    expected_best_candidate_id: Any,
) -> dict[str, Any]:
    ranked_ids = _ranking_ids(ranking)
    expected_rank = _rank_of(ranked_ids, expected_best_candidate_id)
    return {
        "top1_candidate_id": ranked_ids[0] if ranked_ids else None,
        "expected_best_rank": expected_rank,
        "precision_at_3": precision_at_k(ranked_ids, labels, 3),
        "ndcg_at_3": ndcg_at_k(ranked_ids, labels, 3),
        "expected_best_in_top3": expected_rank is not None and expected_rank <= 3,
        "passed": expected_rank is not None and expected_rank <= 3,
    }


def _summary(case_reports: list[dict[str, Any]], *, passed_cases: int) -> dict[str, Any]:
    average_precision = {}
    average_ndcg = {}
    for method in METHODS:
        method_metrics = [case["metrics"][method] for case in case_reports]
        average_precision[method] = _average(item["precision_at_3"] for item in method_metrics)
        average_ndcg[method] = _average(item["ndcg_at_3"] for item in method_metrics)
    return {
        "total_cases": len(case_reports),
        "passed_cases": passed_cases,
        "failed_cases": len(case_reports) - passed_cases,
        "average_precision_at_3": average_precision,
        "average_ndcg_at_3": average_ndcg,
    }


def _mock_settings(settings_obj: Any) -> SimpleNamespace:
    return SimpleNamespace(
        llm_enabled=True,
        llm_provider="mock",
        llm_model=getattr(settings_obj, "llm_model", "mock-literature-review-v1")
        or "mock-literature-review-v1",
        llm_api_key=None,
        llm_base_url=getattr(settings_obj, "llm_base_url", "https://api.openai.com/v1"),
        llm_timeout_seconds=getattr(settings_obj, "llm_timeout_seconds", 8.0),
        llm_temperature=getattr(settings_obj, "llm_temperature", 0.0),
        llm_prompt_version=getattr(settings_obj, "llm_prompt_version", "v1"),
        local_reranker_enabled=True,
        local_reranker_provider="mock",
        local_reranker_model_path="",
        local_reranker_model_version=getattr(settings_obj, "local_reranker_model_version", "v1") or "v1",
        local_reranker_top_k=getattr(settings_obj, "local_reranker_top_k", 20),
        llm_max_input_chars_per_call=getattr(settings_obj, "llm_max_input_chars_per_call", 12000),
    )


def _safe_error_report(*, reason: str, detail: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "local_reranker": {"enabled": False, "model_provider": "unknown", "model_version": "unknown"},
        "methods": METHODS,
        "cases": [],
        "summary": {
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "average_precision_at_3": {},
            "average_ndcg_at_3": {},
        },
        "error": reason,
    }
    if detail:
        report["detail"] = detail
    return report


def _write_json_report(report: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_output(report: dict[str, Any], *, json_output: bool, stdout: TextIO) -> None:
    if json_output:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), file=stdout)
        return

    print("Model comparison report", file=stdout)
    print(f"- methods: {', '.join(report.get('methods', []))}", file=stdout)
    local = report.get("local_reranker", {})
    print(
        f"- local_reranker: provider={local.get('model_provider')} "
        f"version={local.get('model_version')} enabled={local.get('enabled')}",
        file=stdout,
    )
    summary = report.get("summary", {})
    print(f"- cases: {summary.get('passed_cases')}/{summary.get('total_cases')} passed", file=stdout)
    if report.get("error"):
        print(f"- error: {report.get('error')}", file=stdout)
        return

    for case in report.get("cases", []):
        print(f"\n{case.get('case_id')}", file=stdout)
        for method in METHODS:
            metrics = case.get("metrics", {}).get(method, {})
            print(
                f"- {method}: top1={metrics.get('top1_candidate_id')} "
                f"expected_rank={metrics.get('expected_best_rank')} "
                f"p@3={metrics.get('precision_at_3')} ndcg@3={metrics.get('ndcg_at_3')}",
                file=stdout,
            )


def _ranking_item(
    *,
    candidate_id: Any,
    score: Any,
    reason: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "score": round(max(0.0, min(_float_or_default(score, 0.0), 1.0)), 4),
        "reason": str(reason or "No explanation available."),
        "fallback_reason": fallback_reason,
    }


def _ranking_ids(ranking: list[dict[str, Any]]) -> list[Any]:
    return [item.get("candidate_id") for item in ranking]


def _rank_of(ranked_ids: list[Any], candidate_id: Any) -> int | None:
    for index, ranked_id in enumerate(ranked_ids, start=1):
        if ranked_id == candidate_id:
            return index
    return None


def _dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
