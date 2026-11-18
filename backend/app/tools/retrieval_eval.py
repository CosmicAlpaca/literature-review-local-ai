"""
Retrieval Quality Evaluation Tool
==================================

Runs evaluation queries through the search pipeline and produces a
comprehensive quality report.

Usage:
    cd backend
    python -m app.tools.retrieval_eval
    python -m app.tools.retrieval_eval --output docs/retrieval_quality_report.md
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_QUERIES_PATH = _PROJECT_ROOT / "data" / "eval_queries_phase3.json"
_DEFAULT_REPORT_PATH = _PROJECT_ROOT / "docs" / "retrieval_quality_report.md"


def _load_eval_queries(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _EVAL_QUERIES_PATH
    if not p.exists():
        raise FileNotFoundError(f"Evaluation query file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _doi_present_rate(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    return round(sum(1 for c in candidates if c.get("doi")) / len(candidates), 4)


def _abstract_present_rate(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    return round(sum(1 for c in candidates if c.get("abstract")) / len(candidates), 4)


def _source_matched_rate(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    return round(sum(1 for c in candidates if c.get("scopus_matched")) / len(candidates), 4)


def _source_not_matched_rate(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    return round(sum(1 for c in candidates if not c.get("scopus_matched")) / len(candidates), 4)


def _provider_distribution(candidates: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in candidates:
        p = c.get("provider", "unknown")
        dist[p] = dist.get(p, 0) + 1
    return dist


def _trust_bucket_distribution(candidates: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for c in candidates:
        meta = c.get("raw_metadata_json") or {}
        trust = meta.get("_authoring_trust") or {}
        bucket = str(trust.get("trust_bucket", "low")).lower()
        dist[bucket] = dist.get(bucket, 0) + 1
    return dist


def _source_status_counts(candidates: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"active": 0, "discontinued": 0, "unknown": 0}
    for c in candidates:
        meta = c.get("raw_metadata_json") or {}
        trust = meta.get("_authoring_trust") or {}
        if trust.get("source_discontinued"):
            counts["discontinued"] += 1
        elif trust.get("source_active"):
            counts["active"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _match_method_distribution(candidates: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in candidates:
        meta = c.get("raw_metadata_json") or {}
        sm = meta.get("_source_match") or {}
        method = sm.get("match_method", "none")
        dist[method] = dist.get(method, 0) + 1
    return dist


def _dedup_stats_from_candidates(candidates: list[dict]) -> dict[str, Any]:
    """Extract dedup stats from first candidate's metadata (same for all in a search)."""
    for c in candidates:
        meta = c.get("raw_metadata_json") or {}
        stats = meta.get("_dedup_stats")
        if stats:
            return stats
    return {}


def _not_matched_papers(candidates: list[dict]) -> list[dict[str, Any]]:
    """Return details of papers that were NOT matched to Scopus."""
    unmatched = []
    for c in candidates:
        if c.get("scopus_matched"):
            continue
        meta = c.get("raw_metadata_json") or {}
        sm = meta.get("_source_match") or {}
        unmatched.append({
            "title": c.get("title", "N/A"),
            "source_title": c.get("source_title", "N/A"),
            "provider": c.get("provider", "unknown"),
            "doi": c.get("doi"),
            "match_reason": sm.get("reason", "no_match"),
            "confidence": sm.get("confidence", 0.0),
        })
    return unmatched


# ---------------------------------------------------------------------------
# Run a single query through the live pipeline
# ---------------------------------------------------------------------------

def _run_single_query_live(query_data: dict, db_session) -> dict[str, Any]:
    """Execute a single evaluation query through the full pipeline."""
    from app.services.orchestrator import run_search_pipeline

    query = query_data["query"]
    query_id = query_data["id"]
    expected_topic = query_data.get("expected_topic", "")

    logger.info("Running eval query [%s]: %s", query_id, query)
    start = time.monotonic()

    # Create a temporary session for this eval
    from app.db.models.review_session import ReviewSession
    from app.db.models.user import User

    # Use the first existing user, or create a system eval user
    eval_user = db_session.query(User).first()
    if eval_user is None:
        logger.error("No users found in database. Cannot create eval session.")
        return {"query_id": query_id, "query": query, "error": "no_users_in_db", "num_candidates": 0}

    session = ReviewSession(
        user_id=eval_user.id,
        topic=expected_topic,
        raw_query=query,
        status="created",
    )
    db_session.add(session)
    db_session.flush()

    try:
        candidates, top5 = run_search_pipeline(
            db=db_session,
            session_id=session.id,
            query=query,
        )
        elapsed = round(time.monotonic() - start, 2)

        # Serialize candidates for metrics
        candidate_dicts = []
        for c in candidates:
            candidate_dicts.append({
                "id": c.id,
                "title": c.title,
                "abstract": c.abstract,
                "year": c.year,
                "doi": c.doi,
                "url": c.url,
                "source_title": c.source_title,
                "citation_count": c.citation_count,
                "provider": c.provider,
                "scopus_matched": c.scopus_matched,
                "final_score": c.final_score,
                "raw_metadata_json": c.raw_metadata_json,
            })

        return {
            "query_id": query_id,
            "query": query,
            "expected_topic": expected_topic,
            "elapsed_seconds": elapsed,
            "num_candidates": len(candidate_dicts),
            "num_top5": len(top5),
            "doi_present_rate": _doi_present_rate(candidate_dicts),
            "abstract_present_rate": _abstract_present_rate(candidate_dicts),
            "source_matched_rate": _source_matched_rate(candidate_dicts),
            "source_not_matched_rate": _source_not_matched_rate(candidate_dicts),
            "provider_distribution": _provider_distribution(candidate_dicts),
            "trust_bucket_distribution": _trust_bucket_distribution(candidate_dicts),
            "source_status": _source_status_counts(candidate_dicts),
            "match_method_distribution": _match_method_distribution(candidate_dicts),
            "dedup_stats": _dedup_stats_from_candidates(candidate_dicts),
            "not_matched_papers": _not_matched_papers(candidate_dicts),
        }
    except Exception as exc:
        logger.exception("Eval query [%s] failed: %s", query_id, exc)
        return {
            "query_id": query_id,
            "query": query,
            "expected_topic": expected_topic,
            "error": str(exc),
            "num_candidates": 0,
        }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_candidates = sum(r.get("num_candidates", 0) for r in results)
    successful = [r for r in results if "error" not in r and r.get("num_candidates", 0) > 0]

    if not successful:
        return {"total_queries": len(results), "successful": 0, "total_candidates": 0}

    avg_doi = round(sum(r["doi_present_rate"] for r in successful) / len(successful), 4)
    avg_abstract = round(sum(r["abstract_present_rate"] for r in successful) / len(successful), 4)
    avg_matched = round(sum(r["source_matched_rate"] for r in successful) / len(successful), 4)
    avg_not_matched = round(sum(r["source_not_matched_rate"] for r in successful) / len(successful), 4)

    # Aggregate provider distribution
    agg_providers: dict[str, int] = {}
    for r in successful:
        for p, count in r.get("provider_distribution", {}).items():
            agg_providers[p] = agg_providers.get(p, 0) + count

    # Aggregate trust
    agg_trust: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in successful:
        for bucket, count in r.get("trust_bucket_distribution", {}).items():
            agg_trust[bucket] = agg_trust.get(bucket, 0) + count

    # Aggregate dedup
    total_raw = sum(r.get("dedup_stats", {}).get("total_raw", 0) for r in successful)
    total_removed = sum(r.get("dedup_stats", {}).get("duplicates_removed", 0) for r in successful)

    # Count all not matched
    total_not_matched_papers = sum(len(r.get("not_matched_papers", [])) for r in successful)

    return {
        "total_queries": len(results),
        "successful_queries": len(successful),
        "failed_queries": len(results) - len(successful),
        "total_candidates": total_candidates,
        "avg_candidates_per_query": round(total_candidates / len(successful), 1) if successful else 0,
        "avg_doi_present_rate": avg_doi,
        "avg_abstract_present_rate": avg_abstract,
        "avg_source_matched_rate": avg_matched,
        "avg_source_not_matched_rate": avg_not_matched,
        "provider_distribution": agg_providers,
        "trust_bucket_distribution": agg_trust,
        "total_raw_before_dedup": total_raw,
        "total_duplicates_removed": total_removed,
        "total_not_matched_papers": total_not_matched_papers,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Retrieval Quality Report",
        "",
        f"> Auto-generated by `retrieval_eval.py` on {ts}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total queries | {aggregate['total_queries']} |",
        f"| Successful queries | {aggregate.get('successful_queries', 0)} |",
        f"| Failed queries | {aggregate.get('failed_queries', 0)} |",
        f"| Total candidates | {aggregate['total_candidates']} |",
        f"| Avg candidates/query | {aggregate.get('avg_candidates_per_query', 0)} |",
        f"| Avg DOI present rate | {aggregate.get('avg_doi_present_rate', 0):.1%} |",
        f"| Avg abstract present rate | {aggregate.get('avg_abstract_present_rate', 0):.1%} |",
        f"| Avg source matched rate | {aggregate.get('avg_source_matched_rate', 0):.1%} |",
        f"| Avg source NOT matched rate | {aggregate.get('avg_source_not_matched_rate', 0):.1%} |",
        f"| Total raw (pre-dedup) | {aggregate.get('total_raw_before_dedup', 0)} |",
        f"| Total duplicates removed | {aggregate.get('total_duplicates_removed', 0)} |",
        f"| Total unmatched papers | {aggregate.get('total_not_matched_papers', 0)} |",
        "",
        "## Provider Distribution",
        "",
        "| Provider | Count |",
        "|----------|-------|",
    ]
    for provider, count in sorted(aggregate.get("provider_distribution", {}).items()):
        lines.append(f"| {provider} | {count} |")

    lines += [
        "",
        "## Trust Bucket Distribution",
        "",
        "| Bucket | Count |",
        "|--------|-------|",
    ]
    for bucket, count in aggregate.get("trust_bucket_distribution", {}).items():
        lines.append(f"| {bucket} | {count} |")

    lines += [
        "",
        "---",
        "",
        "## Per-Query Results",
        "",
    ]

    for r in results:
        qid = r.get("query_id", "?")
        query = r.get("query", "")
        lines.append(f"### {qid}: `{query}`")
        lines.append("")

        if "error" in r:
            lines.append(f"> **ERROR**: {r['error']}")
            lines.append("")
            continue

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Expected topic | {r.get('expected_topic', 'N/A')} |")
        lines.append(f"| Candidates | {r.get('num_candidates', 0)} |")
        lines.append(f"| Top 5 | {r.get('num_top5', 0)} |")
        lines.append(f"| Elapsed | {r.get('elapsed_seconds', 0)}s |")
        lines.append(f"| DOI present rate | {r.get('doi_present_rate', 0):.1%} |")
        lines.append(f"| Abstract present rate | {r.get('abstract_present_rate', 0):.1%} |")
        lines.append(f"| Source matched rate | {r.get('source_matched_rate', 0):.1%} |")
        lines.append(f"| Source NOT matched rate | {r.get('source_not_matched_rate', 0):.1%} |")
        lines.append("")

        # Provider breakdown
        pd = r.get("provider_distribution", {})
        if pd:
            lines.append("**Providers**: " + ", ".join(f"{p} ({c})" for p, c in pd.items()))
            lines.append("")

        # Match method breakdown
        mm = r.get("match_method_distribution", {})
        if mm:
            lines.append("**Match methods**: " + ", ".join(f"{m} ({c})" for m, c in mm.items()))
            lines.append("")

        # Dedup stats
        ds = r.get("dedup_stats", {})
        if ds:
            lines.append(
                f"**Dedup**: {ds.get('total_raw', 0)} raw → "
                f"{ds.get('after_dedup', 0)} after dedup "
                f"(removed {ds.get('duplicates_removed', 0)}: "
                f"DOI={ds.get('by_doi', 0)}, "
                f"title={ds.get('by_title', 0)}, "
                f"provider_id={ds.get('by_provider_id', 0)}, "
                f"fuzzy={ds.get('by_fuzzy_title', 0)})"
            )
            lines.append("")

        # NOT matched papers
        unmatched = r.get("not_matched_papers", [])
        if unmatched:
            lines.append(f"**Unmatched papers** ({len(unmatched)}):")
            lines.append("")
            lines.append("| Title | Source | Provider | Reason |")
            lines.append("|-------|--------|----------|--------|")
            for u in unmatched:
                title_short = (u["title"][:60] + "...") if len(u["title"]) > 60 else u["title"]
                lines.append(
                    f"| {title_short} | {u['source_title']} | {u['provider']} | {u['match_reason']} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval Quality Evaluation")
    parser.add_argument("--queries", type=str, default=None, help="Path to eval queries JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to output report")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    queries_path = Path(args.queries) if args.queries else _EVAL_QUERIES_PATH
    output_path = Path(args.output) if args.output else _DEFAULT_REPORT_PATH

    queries = _load_eval_queries(queries_path)
    if args.max_queries:
        queries = queries[:args.max_queries]

    logger.info("Loaded %d evaluation queries from %s", len(queries), queries_path)

    # Set up database — import all models so SQLAlchemy resolves FK references
    import app.db.models.user  # noqa: F401
    import app.db.models.review_session  # noqa: F401
    import app.db.models.paper_candidate  # noqa: F401
    import app.db.models.selected_paper  # noqa: F401
    import app.db.models.scopus_source  # noqa: F401
    import app.db.models.scopus_import_job  # noqa: F401
    from app.db.session import SessionLocal
    db = SessionLocal()

    results = []
    try:
        for query_data in queries:
            result = _run_single_query_live(query_data, db)
            results.append(result)
            logger.info(
                "  → [%s] %d candidates, matched=%.0f%%",
                result.get("query_id", "?"),
                result.get("num_candidates", 0),
                result.get("source_matched_rate", 0) * 100,
            )
    finally:
        db.close()

    aggregate = _aggregate_metrics(results)
    report = _generate_markdown_report(results, aggregate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Report written to %s", output_path)
    print(f"\n[OK] Report saved to: {output_path}")
    print(f"   Queries: {aggregate['total_queries']}, "
          f"Candidates: {aggregate['total_candidates']}, "
          f"Avg matched: {aggregate.get('avg_source_matched_rate', 0):.1%}")


if __name__ == "__main__":
    main()
