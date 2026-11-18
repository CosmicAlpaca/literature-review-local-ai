from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, TextIO

from app.core.config import settings
from app.services.llm_client import LLMClient, safe_fallback_reason
from app.services.llm_query import rewrite_query
from app.services.llm_reranker import rerank_candidates
from app.services.paper_summary import summarize_paper


DEFAULT_QUERY = "flood detection remote sensing"

SAMPLE_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": 1,
        "title": "Deep learning for flood detection from satellite imagery",
        "abstract": (
            "This paper evaluates convolutional neural networks for detecting flood extent "
            "from multispectral satellite imagery after extreme rainfall events."
        ),
        "year": 2023,
        "source_title": "Remote Sensing of Environment",
        "doi": "10.0000/smoke.flood.2023",
        "provider": "static_smoke",
        "raw_metadata_json": {"paper_id": "smoke-flood-1", "doi": "10.0000/smoke.flood.2023"},
    },
    {
        "id": 2,
        "title": "Urban drainage risk assessment using hydrologic simulation",
        "abstract": (
            "A hydrologic simulation workflow estimates pluvial flood risk in dense urban "
            "catchments using rainfall scenarios and drainage network constraints."
        ),
        "year": 2021,
        "source_title": "Journal of Hydrology",
        "doi": "10.0000/smoke.hydro.2021",
        "provider": "static_smoke",
        "raw_metadata_json": {"paper_id": "smoke-hydro-2", "doi": "10.0000/smoke.hydro.2021"},
    },
    {
        "id": 3,
        "title": "Crop yield forecasting with climate indicators",
        "abstract": (
            "The study forecasts regional crop yield from seasonal climate indicators and "
            "historical production records."
        ),
        "year": 2020,
        "source_title": "Agricultural Systems",
        "doi": "10.0000/smoke.crop.2020",
        "provider": "static_smoke",
        "raw_metadata_json": {"paper_id": "smoke-crop-3", "doi": "10.0000/smoke.crop.2020"},
    },
]


def main(
    argv: list[str] | None = None,
    *,
    client: LLMClient | None = None,
    settings_obj: Any = settings,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run a safe backend-only smoke check for query rewrite, rerank, and paper summary."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Sample research query to evaluate.")
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Force deterministic mock LLM behavior and avoid network calls.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="Number of static sample candidates to include in rerank smoke.",
    )
    args = parser.parse_args(argv)

    out = stdout or sys.stdout
    try:
        run_settings = _mock_settings(settings_obj) if args.mock_only else settings_obj
        run_client = client or LLMClient(run_settings)
        result = run_smoke(
            query=args.query,
            max_candidates=args.max_candidates,
            client=run_client,
            settings_obj=run_settings,
        )
    except Exception:
        result = {
            "status": {"available": False, "reason": "provider_unavailable"},
            "error": "smoke_failed",
            "fallback_reason": "provider_unavailable",
        }
        _write_output(result, json_output=args.json, stdout=out)
        return 1

    _write_output(result, json_output=args.json, stdout=out)
    return 0


def run_smoke(
    *,
    query: str = DEFAULT_QUERY,
    max_candidates: int = 3,
    client: LLMClient | None = None,
    settings_obj: Any = settings,
) -> dict[str, Any]:
    client = client or LLMClient(settings_obj)
    candidates = SAMPLE_CANDIDATES[: max(1, min(int(max_candidates or 1), len(SAMPLE_CANDIDATES)))]
    status = _status_snapshot(client=client, settings_obj=settings_obj)

    query_result = _run_query_rewrite(query=query, client=client, settings_obj=settings_obj)
    rerank_results = _run_rerank(query=query, candidates=candidates, client=client, settings_obj=settings_obj)
    summary_result = _run_summary(query=query, candidate=candidates[0], client=client, settings_obj=settings_obj)

    return {
        "status": status,
        "query_rewrite": query_result,
        "rerank": rerank_results,
        "summary": summary_result,
        "estimated_usage": _aggregate_smoke_usage(query_result, rerank_results, summary_result),
        "sample": {
            "query": query,
            "candidate_count": len(candidates),
            "retrieval_provider_called": False,
        },
    }


def _run_query_rewrite(*, query: str, client: LLMClient, settings_obj: Any) -> dict[str, Any]:
    if _step_blocked(settings_obj, "llm_query_rewrite_enabled"):
        return _step_disabled_result("query_rewrite")
    return _safe_dataclass(rewrite_query(raw_query=query, topic="LLM smoke test", client=client))


def _run_rerank(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    client: LLMClient,
    settings_obj: Any,
) -> list[dict[str, Any]]:
    if _step_blocked(settings_obj, "llm_rerank_enabled"):
        return [_step_disabled_result("rerank", title=candidate.get("title")) for candidate in candidates]
    results = []
    for candidate, result in zip(candidates, rerank_candidates(query=query, candidates=candidates, client=client)):
        data = _safe_dataclass(result)
        data["title"] = candidate.get("title")
        results.append(data)
    return results


def _run_summary(
    *,
    query: str,
    candidate: dict[str, Any],
    client: LLMClient,
    settings_obj: Any,
) -> dict[str, Any]:
    if _step_blocked(settings_obj, "llm_summary_enabled"):
        return _step_disabled_result("summary", title=candidate.get("title"))
    return _safe_dataclass(summarize_paper(candidate=candidate, query=query, client=client))


def _step_blocked(settings_obj: Any, flag_name: str) -> bool:
    if not bool(getattr(settings_obj, "llm_enabled", False)):
        return False
    return not bool(getattr(settings_obj, flag_name, False))


def _step_disabled_result(step: str, *, title: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "step": step,
        "title": title,
        "used_llm": False,
        "attempted": False,
        "fallback_reason": "step_disabled",
        "estimated_usage": _usage_metadata({}),
    }
    return {key: value for key, value in result.items() if value is not None}


def _status_snapshot(*, client: LLMClient, settings_obj: Any) -> dict[str, Any]:
    enabled_steps = []
    if bool(getattr(settings_obj, "llm_query_rewrite_enabled", False)):
        enabled_steps.append("query_rewrite")
    if bool(getattr(settings_obj, "llm_rerank_enabled", False)):
        enabled_steps.append("rerank")
    if bool(getattr(settings_obj, "llm_summary_enabled", False)):
        enabled_steps.append("summary")

    reason = client.availability_reason()
    ready = reason is None and bool(enabled_steps)
    if reason is None and not enabled_steps:
        reason = "step_disabled"

    return {
        "llm_enabled": client.is_enabled(),
        "provider": client.provider,
        "model": client.model,
        "available": client.is_available(),
        "ready_for_ai_search": ready,
        "reason": safe_fallback_reason(reason) if reason else None,
        "enabled_steps": enabled_steps,
        "query_rewrite_enabled": bool(getattr(settings_obj, "llm_query_rewrite_enabled", False)),
        "rerank_enabled": bool(getattr(settings_obj, "llm_rerank_enabled", False)),
        "summary_enabled": bool(getattr(settings_obj, "llm_summary_enabled", False)),
        "cache_enabled": bool(getattr(settings_obj, "llm_cache_enabled", True)),
        "prompt_version": str(getattr(settings_obj, "llm_prompt_version", "v1") or "v1"),
        "usage_estimation_enabled": bool(getattr(settings_obj, "llm_usage_estimation_enabled", True)),
        "max_calls_per_search": int(getattr(settings_obj, "llm_max_calls_per_search", 20) or 0),
        "max_estimated_tokens_per_search": int(
            getattr(settings_obj, "llm_max_estimated_tokens_per_search", 20000) or 0
        ),
        "max_input_chars_per_call": int(getattr(settings_obj, "llm_max_input_chars_per_call", 12000) or 0),
    }


def _safe_dataclass(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        data = asdict(value)
    elif isinstance(value, dict):
        data = dict(value)
    else:
        data = {}

    data["estimated_usage"] = _usage_metadata(data)
    return data


def _usage_metadata(data: dict[str, Any]) -> dict[str, int | bool]:
    return {
        "estimated_input_tokens": int(data.get("estimated_input_tokens") or 0),
        "estimated_output_tokens": int(data.get("estimated_output_tokens") or 0),
        "estimated_total_tokens": int(data.get("estimated_total_tokens") or 0),
        "estimated_call_count": int(data.get("estimated_call_count") or 0),
        "input_truncated": bool(data.get("input_truncated") or False),
    }


def _aggregate_usage(items: list[dict[str, Any]]) -> dict[str, int | bool]:
    usage_items = [_usage_metadata(item) for item in items]
    return {
        "estimated_input_tokens": sum(int(item["estimated_input_tokens"]) for item in usage_items),
        "estimated_output_tokens": sum(int(item["estimated_output_tokens"]) for item in usage_items),
        "estimated_total_tokens": sum(int(item["estimated_total_tokens"]) for item in usage_items),
        "estimated_call_count": sum(int(item["estimated_call_count"]) for item in usage_items),
        "input_truncated": any(bool(item["input_truncated"]) for item in usage_items),
    }


def _aggregate_smoke_usage(
    query_result: dict[str, Any],
    rerank_results: list[dict[str, Any]],
    summary_result: dict[str, Any],
) -> dict[str, int | bool]:
    # Rerank is one batched call; service metadata is copied onto each returned candidate.
    rerank_usage = [rerank_results[0]] if rerank_results else []
    return _aggregate_usage([query_result, *rerank_usage, summary_result])


def _mock_settings(settings_obj: Any) -> SimpleNamespace:
    fields = {
        "llm_enabled": True,
        "llm_provider": "mock",
        "llm_model": getattr(settings_obj, "llm_model", "mock-literature-review-v1")
        or "mock-literature-review-v1",
        "llm_api_key": None,
        "llm_base_url": getattr(settings_obj, "llm_base_url", "https://api.openai.com/v1"),
        "llm_timeout_seconds": getattr(settings_obj, "llm_timeout_seconds", 8.0),
        "llm_temperature": getattr(settings_obj, "llm_temperature", 0.0),
        "llm_query_rewrite_enabled": True,
        "llm_rerank_enabled": True,
        "llm_summary_enabled": True,
        "llm_cache_enabled": getattr(settings_obj, "llm_cache_enabled", True),
        "llm_prompt_version": getattr(settings_obj, "llm_prompt_version", "v1"),
        "llm_usage_estimation_enabled": getattr(settings_obj, "llm_usage_estimation_enabled", True),
        "llm_max_calls_per_search": getattr(settings_obj, "llm_max_calls_per_search", 20),
        "llm_max_estimated_tokens_per_search": getattr(
            settings_obj,
            "llm_max_estimated_tokens_per_search",
            20000,
        ),
        "llm_max_input_chars_per_call": getattr(settings_obj, "llm_max_input_chars_per_call", 12000),
    }
    return SimpleNamespace(**fields)


def _write_output(result: dict[str, Any], *, json_output: bool, stdout: TextIO) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), file=stdout)
        return

    status = result.get("status", {})
    print("LLM smoke status", file=stdout)
    print(f"- provider: {status.get('provider', 'unknown')}", file=stdout)
    print(f"- model: {status.get('model', 'unknown')}", file=stdout)
    print(f"- available: {status.get('available')}", file=stdout)
    print(f"- ready_for_ai_search: {status.get('ready_for_ai_search')}", file=stdout)
    print(f"- reason: {status.get('reason')}", file=stdout)
    print(f"- enabled_steps: {', '.join(status.get('enabled_steps', [])) or 'none'}", file=stdout)

    query = result.get("query_rewrite", {})
    print("\nQuery rewrite", file=stdout)
    print(f"- used_llm: {query.get('used_llm')}", file=stdout)
    print(f"- fallback_reason: {query.get('fallback_reason')}", file=stdout)
    print(f"- optimized_query: {query.get('optimized_query')}", file=stdout)
    print(f"- expanded_queries: {', '.join(query.get('expanded_queries', []) or [])}", file=stdout)

    print("\nRerank", file=stdout)
    for item in result.get("rerank", []):
        title = item.get("title") or item.get("candidate_id") or item.get("candidate_index")
        score = item.get("llm_relevance_score")
        print(
            f"- {title}: used_llm={item.get('used_llm')} score={score} "
            f"fallback_reason={item.get('fallback_reason')} reason={item.get('reason')}",
            file=stdout,
        )

    summary = result.get("summary", {})
    print("\nSummary", file=stdout)
    print(f"- used_llm: {summary.get('used_llm')}", file=stdout)
    print(f"- fallback_reason: {summary.get('fallback_reason')}", file=stdout)
    print(f"- one_sentence_summary: {summary.get('one_sentence_summary')}", file=stdout)
    print(f"- relevance_to_query: {summary.get('relevance_to_query')}", file=stdout)

    usage = result.get("estimated_usage", {})
    print("\nEstimated usage", file=stdout)
    print(f"- estimated_call_count: {usage.get('estimated_call_count')}", file=stdout)
    print(f"- estimated_total_tokens: {usage.get('estimated_total_tokens')}", file=stdout)
    print(f"- input_truncated: {usage.get('input_truncated')}", file=stdout)


if __name__ == "__main__":
    raise SystemExit(main())
