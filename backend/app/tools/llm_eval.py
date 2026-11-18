from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, TextIO

from app.core.config import settings
from app.eval.llm_fixtures import get_llm_eval_case, load_llm_eval_cases
from app.services.llm_client import LLMClient, safe_fallback_reason, utc_timestamp
from app.services.llm_query import rewrite_query
from app.services.llm_reranker import rerank_candidates
from app.services.llm_prompts import PAPER_SUMMARY_FIELDS
from app.services.paper_summary import summarize_paper


def main(
    argv: list[str] | None = None,
    *,
    client: LLMClient | None = None,
    settings_obj: Any = settings,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run backend-only LLM eval cases for query rewrite, rerank, and paper summary."
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force deterministic mock LLM behavior and avoid network calls.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--case", help="Run one fixture case by case_id.")
    parser.add_argument("--output", help="Write a JSON report to this path.")
    args = parser.parse_args(argv)

    out = stdout or sys.stdout
    try:
        cases = [get_llm_eval_case(args.case)] if args.case else load_llm_eval_cases()
        run_settings = _mock_settings(settings_obj) if args.mock_only else settings_obj
        run_client = client or LLMClient(run_settings)
        report = run_eval(cases=cases, client=run_client, settings_obj=run_settings)
        if args.output:
            _write_json_report(report, args.output)
    except ValueError as exc:
        report = _safe_error_report(settings_obj=settings_obj, reason="invalid_case", detail=str(exc))
        _write_output(report, json_output=args.json, stdout=out)
        return 2
    except Exception:
        report = _safe_error_report(settings_obj=settings_obj, reason="provider_unavailable")
        _write_output(report, json_output=args.json, stdout=out)
        return 1

    _write_output(report, json_output=args.json, stdout=out)
    return 0


def run_eval(
    *,
    cases: list[dict[str, Any]] | None = None,
    client: LLMClient | None = None,
    settings_obj: Any = settings,
) -> dict[str, Any]:
    client = client or LLMClient(settings_obj)
    selected_cases = cases or load_llm_eval_cases()
    case_reports = [_evaluate_case(case=case, client=client) for case in selected_cases]
    passed_cases = sum(1 for case in case_reports if case["passed"])

    return {
        "created_at": utc_timestamp(),
        "provider": client.provider,
        "model": client.model,
        "prompt_version": str(getattr(settings_obj, "llm_prompt_version", "v1") or "v1"),
        "retrieval_provider_called": False,
        "database_access_required": False,
        "cache": {
            "enabled": bool(getattr(settings_obj, "llm_cache_enabled", True)),
            "status": "not_available_for_service_layer_eval",
        },
        "cases": case_reports,
        "summary": {
            "total_cases": len(case_reports),
            "passed_cases": passed_cases,
            "failed_cases": len(case_reports) - passed_cases,
        },
    }


def _evaluate_case(*, case: dict[str, Any], client: LLMClient) -> dict[str, Any]:
    query_result = _safe_dataclass(
        rewrite_query(raw_query=case["raw_query"], topic=case.get("topic"), client=client)
    )
    rerank_results = [
        _safe_dataclass(result) for result in rerank_candidates(case["raw_query"], case["candidates"], client=client)
    ]
    top_candidate = _top_candidate(case["candidates"], rerank_results)
    summary_result = _safe_dataclass(summarize_paper(candidate=top_candidate, query=case["raw_query"], client=client))

    query_metrics = _query_rewrite_metrics(query_result, case.get("expected_terms", []))
    rerank_metrics = _rerank_metrics(rerank_results, case)
    summary_metrics = _summary_metrics(summary_result)
    fallback_reasons = _fallback_reasons(query_result, rerank_results, summary_result)
    usage = _aggregate_eval_usage(query_result, rerank_results, summary_result)
    passed = bool(query_metrics["passed"] and rerank_metrics["passed"] and summary_metrics["passed"])

    return {
        "case_id": case["case_id"],
        "topic": case["topic"],
        "query_rewrite": {
            "optimized_query": query_result.get("optimized_query"),
            "expanded_queries_count": query_metrics["expanded_queries_count"],
            "expected_terms_matched": query_metrics["matched_terms"],
            "contains_expected_terms_count": query_metrics["contains_expected_terms_count"],
            "optimized_query_non_empty": query_metrics["optimized_query_non_empty"],
            "used_llm": bool(query_result.get("used_llm")),
            "fallback_reason": query_result.get("fallback_reason"),
            "metrics": query_metrics,
        },
        "rerank": {
            "order": rerank_metrics["order"],
            "expected_top_candidate_rank": rerank_metrics["expected_top_candidate_rank"],
            "relevance_scores_bounded": rerank_metrics["relevance_scores_bounded"],
            "reasons_non_empty": rerank_metrics["reasons_non_empty"],
            "used_llm": any(bool(item.get("used_llm")) for item in rerank_results),
            "fallback_reasons": [
                item.get("fallback_reason") for item in rerank_results if item.get("fallback_reason")
            ],
            "metrics": rerank_metrics,
        },
        "summary": {
            "top_candidate_title": top_candidate.get("title"),
            "fields_present": summary_metrics["fields_present"],
            "summary_fields_present": summary_metrics["summary_fields_present"],
            "relevance_to_query_non_empty": summary_metrics["relevance_to_query_non_empty"],
            "limitations_or_caution_present": summary_metrics["limitations_or_caution_present"],
            "used_llm": bool(summary_result.get("used_llm")),
            "fallback_reason": summary_result.get("fallback_reason"),
            "metrics": summary_metrics,
        },
        "operational": {
            "fallback_reasons": fallback_reasons,
            "estimated_usage": usage,
            "cache_status": "not_available_for_service_layer_eval",
        },
        "fallback_reasons": fallback_reasons,
        "passed": passed,
    }


def _query_rewrite_metrics(result: dict[str, Any], expected_terms: list[str]) -> dict[str, Any]:
    expected = [_normalize_term(term) for term in expected_terms if _normalize_term(term)]
    searchable_text = _normalize_text(
        " ".join(
            [
                str(result.get("optimized_query") or ""),
                " ".join(result.get("expanded_queries") or []),
                " ".join(result.get("key_terms") or []),
            ]
        )
    )
    matched_terms = [term for term in expected if term in searchable_text]
    expanded_queries_count = len(result.get("expanded_queries") or [])
    optimized_query_non_empty = bool(str(result.get("optimized_query") or "").strip())

    return {
        "contains_expected_terms_count": len(matched_terms),
        "expected_terms_total": len(expected),
        "matched_terms": matched_terms,
        "optimized_query_non_empty": optimized_query_non_empty,
        "expanded_queries_count": expanded_queries_count,
        "passed": optimized_query_non_empty and expanded_queries_count > 0 and len(matched_terms) > 0,
    }


def _rerank_metrics(results: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for item in results:
        index = int(item.get("candidate_index") or 0)
        score = _float_or_default(item.get("llm_relevance_score"), 0.0)
        candidate = case["candidates"][index]
        scored.append(
            {
                "candidate_index": index,
                "title": candidate.get("title"),
                "score": score,
                "reason": item.get("reason"),
            }
        )

    order = sorted(scored, key=lambda item: (-item["score"], item["candidate_index"]))
    expected_title = case.get("expected_top_candidate_title")
    expected_rank = None
    if expected_title:
        for rank, item in enumerate(order, start=1):
            if item["title"] == expected_title:
                expected_rank = rank
                break

    scores_bounded = all(0.0 <= item["score"] <= 1.0 for item in scored)
    reasons_non_empty = all(bool(str(item.get("reason") or "").strip()) for item in scored)
    expected_top_ok = expected_rank in (None, 1)

    return {
        "order": [{"rank": rank, **item} for rank, item in enumerate(order, start=1)],
        "expected_top_candidate_rank": expected_rank,
        "relevance_scores_bounded": scores_bounded,
        "reasons_non_empty": reasons_non_empty,
        "passed": scores_bounded and reasons_non_empty and expected_top_ok,
    }


def _summary_metrics(result: dict[str, Any]) -> dict[str, Any]:
    fields_present = {
        field: bool(str(result.get(field) or "").strip())
        for field in PAPER_SUMMARY_FIELDS
    }
    return {
        "fields_present": fields_present,
        "summary_fields_present": all(fields_present.values()),
        "relevance_to_query_non_empty": fields_present["relevance_to_query"],
        "limitations_or_caution_present": fields_present["limitations_or_caution"],
        "passed": all(fields_present.values()),
    }


def _top_candidate(candidates: list[dict[str, Any]], rerank_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    if not rerank_results:
        return candidates[0]
    top = max(
        rerank_results,
        key=lambda item: (_float_or_default(item.get("llm_relevance_score"), 0.0), -int(item.get("candidate_index") or 0)),
    )
    index = int(top.get("candidate_index") or 0)
    return candidates[max(0, min(index, len(candidates) - 1))]


def _fallback_reasons(
    query_result: dict[str, Any],
    rerank_results: list[dict[str, Any]],
    summary_result: dict[str, Any],
) -> list[str]:
    reasons = []
    for reason in [
        query_result.get("fallback_reason"),
        *(item.get("fallback_reason") for item in rerank_results),
        summary_result.get("fallback_reason"),
    ]:
        if reason:
            safe_reason = safe_fallback_reason(str(reason))
            if safe_reason not in reasons:
                reasons.append(safe_reason)
    return reasons


def _safe_dataclass(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _usage_metadata(data: dict[str, Any]) -> dict[str, int | bool]:
    return {
        "estimated_input_tokens": int(data.get("estimated_input_tokens") or 0),
        "estimated_output_tokens": int(data.get("estimated_output_tokens") or 0),
        "estimated_total_tokens": int(data.get("estimated_total_tokens") or 0),
        "estimated_call_count": int(data.get("estimated_call_count") or 0),
        "input_truncated": bool(data.get("input_truncated") or False),
    }


def _aggregate_eval_usage(
    query_result: dict[str, Any],
    rerank_results: list[dict[str, Any]],
    summary_result: dict[str, Any],
) -> dict[str, int | bool]:
    usage_items = [_usage_metadata(query_result)]
    if rerank_results:
        usage_items.append(_usage_metadata(rerank_results[0]))
    usage_items.append(_usage_metadata(summary_result))
    return {
        "estimated_input_tokens": sum(int(item["estimated_input_tokens"]) for item in usage_items),
        "estimated_output_tokens": sum(int(item["estimated_output_tokens"]) for item in usage_items),
        "estimated_total_tokens": sum(int(item["estimated_total_tokens"]) for item in usage_items),
        "estimated_call_count": sum(int(item["estimated_call_count"]) for item in usage_items),
        "input_truncated": any(bool(item["input_truncated"]) for item in usage_items),
    }


def _mock_settings(settings_obj: Any) -> SimpleNamespace:
    return SimpleNamespace(
        llm_enabled=True,
        llm_provider="mock",
        llm_model=getattr(settings_obj, "llm_model", "mock-literature-review-v1") or "mock-literature-review-v1",
        llm_api_key=None,
        llm_base_url=getattr(settings_obj, "llm_base_url", "https://api.openai.com/v1"),
        llm_timeout_seconds=getattr(settings_obj, "llm_timeout_seconds", 8.0),
        llm_temperature=getattr(settings_obj, "llm_temperature", 0.0),
        llm_query_rewrite_enabled=True,
        llm_rerank_enabled=True,
        llm_summary_enabled=True,
        llm_cache_enabled=getattr(settings_obj, "llm_cache_enabled", True),
        llm_prompt_version=getattr(settings_obj, "llm_prompt_version", "v1"),
        llm_usage_estimation_enabled=getattr(settings_obj, "llm_usage_estimation_enabled", True),
        llm_max_calls_per_search=getattr(settings_obj, "llm_max_calls_per_search", 20),
        llm_max_estimated_tokens_per_search=getattr(settings_obj, "llm_max_estimated_tokens_per_search", 20000),
        llm_max_input_chars_per_call=getattr(settings_obj, "llm_max_input_chars_per_call", 12000),
    )


def _safe_error_report(*, settings_obj: Any, reason: str, detail: str | None = None) -> dict[str, Any]:
    safe_reason = reason if reason == "invalid_case" else safe_fallback_reason(reason)
    report: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "provider": str(getattr(settings_obj, "llm_provider", "unknown") or "unknown"),
        "model": str(getattr(settings_obj, "llm_model", "unknown") or "unknown"),
        "prompt_version": str(getattr(settings_obj, "llm_prompt_version", "v1") or "v1"),
        "cases": [],
        "summary": {"total_cases": 0, "passed_cases": 0, "failed_cases": 0},
        "error": safe_reason,
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

    print("LLM eval report", file=stdout)
    print(f"- provider: {report.get('provider')}", file=stdout)
    print(f"- model: {report.get('model')}", file=stdout)
    print(f"- prompt_version: {report.get('prompt_version')}", file=stdout)
    summary = report.get("summary", {})
    print(f"- cases: {summary.get('passed_cases')}/{summary.get('total_cases')} passed", file=stdout)
    if report.get("error"):
        print(f"- error: {report.get('error')}", file=stdout)
        return

    for case in report.get("cases", []):
        rewrite = case.get("query_rewrite", {})
        rerank = case.get("rerank", {})
        summary_metrics = case.get("summary", {})
        print(f"\n{case.get('case_id')}", file=stdout)
        print(f"- optimized_query: {rewrite.get('optimized_query')}", file=stdout)
        print(
            "- expected_terms_matched: "
            f"{rewrite.get('contains_expected_terms_count')} ({', '.join(rewrite.get('expected_terms_matched', []))})",
            file=stdout,
        )
        print(
            f"- expected_top_candidate_rank: {rerank.get('expected_top_candidate_rank')}",
            file=stdout,
        )
        print(f"- summary_fields_present: {summary_metrics.get('summary_fields_present')}", file=stdout)
        print(f"- fallback_reasons: {', '.join(case.get('fallback_reasons') or []) or 'none'}", file=stdout)


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _normalize_term(value: str) -> str:
    return _normalize_text(value)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
