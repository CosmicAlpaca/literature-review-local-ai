import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "@/lib/axios";
import type { CandidateItem, LLMStatus } from "@/lib/session";

/**
 * Public contracts for the Phase 2 LLM preview endpoints documented in
 * `docs/phase2_llm_backend_handoff.md`. These mirror the Pydantic models in
 * `backend/app/api/llm.py` but only expose the frontend-safe fields.
 */

export type LlmEstimatedUsage = {
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_total_tokens: number;
  estimated_call_count: number;
  input_truncated: boolean;
};

/* ── /llm/query-preview ─────────────────────────────────────────────── */

export type QueryPreviewRequest = {
  topic?: string | null;
  raw_query: string;
};

export type QueryPreviewResponse = {
  original_query: string;
  optimized_query: string;
  expanded_queries: string[];
  key_terms: string[];
  notes: string;
  fallback_reason?: string | null;
  provider: string;
  model: string;
  prompt_version?: string | null;
  estimated_usage?: LlmEstimatedUsage | null;
};

/* ── /llm/rerank-preview ────────────────────────────────────────────── */

/**
 * The rerank-preview endpoint caps candidates at
 * `MAX_PREVIEW_RERANK_CANDIDATES = 10` server-side; we enforce the same cap
 * client-side to surface friendly UI feedback before the request fires.
 */
export const RERANK_PREVIEW_MAX_CANDIDATES = 10;

export type RerankPreviewCandidate = {
  id: string | number | null;
  title: string;
  abstract?: string | null;
  source_title?: string | null;
  year?: number | null;
  citation_count?: number | null;
};

export type RerankPreviewRequest = {
  query: string;
  candidates: RerankPreviewCandidate[];
};

export type RerankPreviewResult = {
  candidate_index: number;
  candidate_id: string | number | null;
  relevance_score: number;
  reason: string;
};

export type RerankPreviewResponse = {
  results: RerankPreviewResult[];
  fallback_reason?: string | null;
  provider: string;
  model: string;
  prompt_version?: string | null;
  estimated_usage?: LlmEstimatedUsage | null;
};

/* ── /llm/summary-preview ───────────────────────────────────────────── */

export type SummaryPreviewPaper = {
  title: string;
  abstract?: string | null;
  source_title?: string | null;
  year?: number | null;
};

export type SummaryPreviewRequest = {
  query?: string | null;
  paper: SummaryPreviewPaper;
};

export type SummaryPreviewResponse = {
  one_sentence_summary: string;
  contribution: string;
  method_or_data: string;
  relevance_to_query: string;
  limitations_or_caution: string;
  fallback_reason?: string | null;
  provider: string;
  model: string;
  prompt_version?: string | null;
  estimated_usage?: LlmEstimatedUsage | null;
};

/* ── Shared helpers ─────────────────────────────────────────────────── */

/**
 * Safe fallback codes documented in the LLM service layer.
 * Anything not in this list is rendered verbatim (no crash).
 */
export type LlmFallbackReason =
  | "llm_disabled"
  | "step_disabled"
  | "missing_api_key"
  | "provider_unavailable"
  | "invalid_response"
  | "invalid_json"
  | "http_error"
  | "rate_limited"
  | "timeout"
  | "input_too_large"
  | "max_calls_exceeded"
  | "estimated_tokens_exceeded"
  | "no_abstract"
  | "insufficient_content"
  | "budget_exceeded"
  | "summary_cap_exceeded";

export function llmFallbackLabel(reason: string | null | undefined): string {
  if (!reason) return "";
  switch (reason) {
    case "llm_disabled":
      return "AI assistance is disabled on this deployment.";
    case "step_disabled":
      return "This AI step is disabled; a simpler template was used instead.";
    case "missing_api_key":
      return "AI provider is not configured (missing API key).";
    case "provider_unavailable":
      return "AI provider is unavailable; a simpler template was used instead.";
    case "invalid_response":
    case "invalid_json":
      return "AI returned an invalid response; a simpler template was used instead.";
    case "http_error":
      return "AI provider returned an error; a simpler template was used instead.";
    case "rate_limited":
      return "AI provider rate limit reached; a simpler template was used instead.";
    case "timeout":
      return "AI request timed out; a simpler template was used instead.";
    case "input_too_large":
      return "Your shortlist has too much text for the AI in one request. A simpler version was generated—try removing a few papers and regenerating.";
    case "max_calls_exceeded":
      return "This search reached the AI call limit; a simpler template was used instead.";
    case "estimated_tokens_exceeded":
      return "This search reached the AI token budget; a simpler template was used instead.";
    case "no_abstract":
      return "Not enough abstract text for AI summary.";
    case "insufficient_content":
      return "Not enough paper content for a full AI pass.";
    case "budget_exceeded":
    case "summary_cap_exceeded":
      return "AI budget limit reached; a simpler template was used instead.";
    default:
      return "AI could not run for this step; a simpler template was used instead.";
  }
}

export function clampScore01(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

export function formatScore3dp(value: unknown): string {
  return clampScore01(value).toFixed(3);
}

export function scoreToneClass(score: number): string {
  if (score >= 0.7) {
    return "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-200";
  }
  if (score >= 0.4) {
    return "border-primary/25 text-primary bg-primary/8";
  }
  return "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100";
}

/* ── Candidate scope selection (shared with rerank tab) ─────────────── */

export type LlmCandidateScope = "top_by_score" | "shortlisted" | "active";

export const LLM_CANDIDATE_SCOPE_LABELS: Record<LlmCandidateScope, string> = {
  top_by_score: "Top by final score",
  shortlisted: "Shortlisted only",
  active: "Active candidates",
};

export function selectLlmCandidates(
  scope: LlmCandidateScope,
  candidates: CandidateItem[],
  shortlisted: CandidateItem[],
  limit: number,
): CandidateItem[] {
  const cap = Math.max(1, limit);

  if (scope === "shortlisted") {
    return shortlisted.slice(0, cap);
  }

  const pool =
    scope === "active"
      ? candidates.filter((item) => item.review_status !== "dismissed")
      : candidates;

  return [...pool]
    .sort((a, b) => {
      const aScore = a.final_score ?? Number.NEGATIVE_INFINITY;
      const bScore = b.final_score ?? Number.NEGATIVE_INFINITY;
      if (bScore !== aScore) return bScore - aScore;
      return a.id - b.id;
    })
    .slice(0, cap);
}

export function toRerankPreviewCandidate(
  candidate: CandidateItem,
): RerankPreviewCandidate {
  return {
    id: candidate.id,
    title: candidate.title || `Candidate ${candidate.id}`,
    abstract: candidate.abstract,
    source_title: candidate.source_title,
    year: candidate.year,
    citation_count: candidate.citation_count,
  };
}

export function toSummaryPreviewPaper(
  candidate: CandidateItem,
): SummaryPreviewPaper {
  return {
    title: candidate.title || `Candidate ${candidate.id}`,
    abstract: candidate.abstract,
    source_title: candidate.source_title,
    year: candidate.year,
  };
}

/** Sort rerank results by `relevance_score` descending. */
export function sortRerankResults(
  results: RerankPreviewResult[],
): RerankPreviewResult[] {
  return [...results].sort(
    (a, b) => clampScore01(b.relevance_score) - clampScore01(a.relevance_score),
  );
}

/* ── Friendly status label derived from /llm/status ─────────────────── */

export function llmStatusLabel(status: LLMStatus | null | undefined): {
  label: string;
  tone: "positive" | "warning" | "muted";
} {
  if (!status) return { label: "AI status unavailable", tone: "muted" };
  if (status.available) return { label: "AI available", tone: "positive" };
  return {
    label: status.disabled_reason || "AI disabled on this deployment",
    tone: "warning",
  };
}

/* ── Query keys ─────────────────────────────────────────────────────── */

export const LLM_QUERY_KEYS = {
  status: () => ["llm-status"] as const,
  queryPreview: (payload: QueryPreviewRequest | null) =>
    ["llm-query-preview", payload] as const,
  rerankPreview: (payload: RerankPreviewRequest | null) =>
    ["llm-rerank-preview", payload] as const,
  summaryPreview: (payload: SummaryPreviewRequest | null) =>
    ["llm-summary-preview", payload] as const,
};

/* ── Shared hooks ───────────────────────────────────────────────────── */

/**
 * Shared TanStack Query subscription for `GET /llm/status`.
 *
 * Both the session page and `LlmPreviewPanel` consume this hook so the
 * underlying network call is deduplicated against the same `queryKey` —
 * TanStack Query returns the cached entry to additional subscribers
 * without firing a second fetch.
 *
 * Cache profile:
 * - `staleTime: 5 min` — `/llm/status` is a public config readiness check
 *   (it does not validate provider credentials), so it changes very rarely.
 * - `gcTime: 30 min` — keeps the entry around across route transitions.
 * - `refetchOnWindowFocus: false` / `refetchOnReconnect: false` /
 *   `refetchOnMount: false` — avoid spurious refetches when the user tabs
 *   back into the browser, regains network, or remounts a panel.
 * - `retry: 0` — match the soft-fail semantics the page previously used.
 *
 * `queryFn` swallows network errors and resolves to `null` so consumers can
 * treat "AI status unavailable" as a benign UI state rather than an error.
 */
export function useLlmStatus() {
  return useQuery<LLMStatus | null>({
    queryKey: LLM_QUERY_KEYS.status(),
    queryFn: async () => {
      try {
        return (await apiFetch("/llm/status")) as LLMStatus;
      } catch {
        return null;
      }
    },
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 0,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
  });
}
