import {
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";

import { apiFetch } from "@/lib/axios";

export type SessionItem = {
  id: number;
  topic: string;
  raw_query: string;
  status: string;
  created_at: string;
};

export type CandidateItem = {
  id: number;
  title: string;
  abstract: string | null;
  abstract_present?: boolean;
  abstract_length?: number;
  year: number | null;
  source_title: string | null;
  doi?: string | null;
  url?: string | null;
  citation_count: number | null;
  provider: string;
  scopus_matched: boolean;
  source_match_reason?: string | null;
  source_match_confidence?: number | null;
  matched_source_title?: string | null;
  matched_source_issn?: string | null;
  matched_source_eissn?: string | null;
  final_score: number | null;
  reranker_status?: "applied" | "unavailable" | "disabled";
  reranker_relative_score?: number | null;
  created_at?: string;
  review_status?: "shortlisted" | "dismissed" | null;
  llm_relevance_score?: number | null;
  llm_relevance_reason?: string | null;
  llm_summary?: LLMSummaryItem | null;
};

export type LLMSummaryItem = {
  one_sentence_summary?: string | null;
  contribution?: string | null;
  method_or_data?: string | null;
  relevance_to_query?: string | null;
  limitations_or_caution?: string | null;
};

export type SearchQueryContext = {
  original_query?: string | null;
  optimized_query?: string | null;
  expanded_queries?: string[];
  key_terms?: string[];
  query_rewrite_notes?: string | null;
  llm_enabled?: boolean;
  llm_requested?: boolean;
  llm_available?: boolean;
  llm_query_rewrite_used?: boolean;
  llm_disabled_reason?: string | null;
  applied_filters?: Record<string, unknown> | null;
};

export type SearchResponse = {
  candidates: CandidateItem[];
  top5: CandidateItem[];
  query_context?: SearchQueryContext;
};

/** `GET /search/{session_id}/query-context` */
export type QueryContextResponse = {
  session_id: number;
  exists: boolean;
  query_context: SearchQueryContext;
  updated_at: string | null;
};

export type CandidateSummaryResponse = {
  session_id: number;
  candidate_id: number;
  summary: LLMSummaryItem;
  cached: boolean;
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
};

/**
 * Per-step LLM preference flags for `POST /search/{session_id}`
 * (Phase 2 contract — see `docs/phase2_llm_backend_handoff.md`).
 *
 * Each flag is a *preference*. The backend may still skip the step if the
 * deployment-level config disables it (`/llm/status` exposes the canonical
 * `query_rewrite_enabled` / `rerank_enabled` / `summary_enabled` switches).
 *
 * Use `null`/`undefined` to mean "let the server resolve the default for
 * this step" rather than sending an explicit `false`.
 */
export type SearchLLMFlags = {
  use_llm: boolean;
  use_query_rewrite: boolean;
  use_llm_rerank: boolean;
  use_llm_summary: boolean;
};

/** Default flag set when the user chooses AI-assisted search. */
export const DEFAULT_AI_SEARCH_FLAGS: SearchLLMFlags = {
  use_llm: true,
  use_query_rewrite: true,
  use_llm_rerank: true,
  use_llm_summary: true,
};

export type LLMStatus = {
  llm_enabled: boolean;
  llm_provider: string;
  llm_model: string;
  query_rewrite_enabled: boolean;
  rerank_enabled: boolean;
  summary_enabled: boolean;
  enabled_steps?: string[];
  api_key_configured: boolean;
  available: boolean;
  ready_for_ai_search?: boolean;
  reason?: string | null;
  disabled_reason?: string | null;
};

export type PaperQualityLevel = "strong" | "moderate" | "caution";

export type PaperQualitySignal = {
  level: PaperQualityLevel;
  label: string;
  summary: string;
  metadataComplete: boolean;
  providerLabel: string;
  sourceCue: string;
  yearCue: string;
  citationCue: string;
  hasSource: boolean;
  hasAbstract: boolean;
  hasRecentYear: boolean;
  hasCitations: boolean;
};

export const SESSION_TOPIC_PLACEHOLDER = "Any topic";

export function hasSessionTopic(topic: string | null | undefined): boolean {
  return Boolean(topic?.trim());
}

export type SessionTopicDisplay = {
  label: string;
  isPlaceholder: boolean;
};

/** Topic label for UI; empty stored topic → placeholder (reserves layout height). */
export function getSessionTopicDisplay(
  topic: string | null | undefined,
): SessionTopicDisplay {
  const trimmed = topic?.trim();
  if (trimmed) {
    return { label: trimmed, isPlaceholder: false };
  }
  return { label: SESSION_TOPIC_PLACEHOLDER, isPlaceholder: true };
}

export function formatCreatedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }

  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

/** Trim and capitalize the first character (locale-aware for e.g. Vietnamese). */
export function normalizeRawQuery(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return trimmed;
  }
  return trimmed.charAt(0).toLocaleUpperCase() + trimmed.slice(1);
}

export function summarizeQuery(value: string, maxLength = 180) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength).trim()}...`;
}

export function deriveTopCandidates(candidates: CandidateItem[], limit = 5) {
  return [...candidates]
    .sort((left, right) => {
      const leftScore = left.final_score ?? Number.NEGATIVE_INFINITY;
      const rightScore = right.final_score ?? Number.NEGATIVE_INFINITY;

      if (rightScore !== leftScore) {
        return rightScore - leftScore;
      }

      return left.id - right.id;
    })
    .slice(0, limit);
}

export function formatProviderLabel(provider: string) {
  const normalized = provider.trim().toLowerCase();

  if (normalized === "semantic_scholar") {
    return "Semantic Scholar";
  }

  if (normalized === "openalex_strict") {
    return "OpenAlex strict";
  }

  if (normalized === "mock") {
    return "Mock provider";
  }

  if (normalized === "elsevier") {
    return "Elsevier/Scopus";
  }

  return provider || "Unknown provider";
}

function deriveYearCue(year: number | null) {
  if (!year) {
    return "Year missing";
  }

  const currentYear = new Date().getFullYear();

  if (year >= currentYear - 6) {
    return "Recent publication";
  }

  if (year >= currentYear - 15) {
    return "Established publication";
  }

  return "Older publication";
}

function deriveCitationCue(citationCount: number | null) {
  if (citationCount === null) {
    return "Citation data missing";
  }

  if (citationCount >= 50) {
    return "Well-cited";
  }

  if (citationCount >= 10) {
    return "Some citation support";
  }

  if (citationCount > 0) {
    return "Early citation signal";
  }

  return "No citations yet";
}

export function getPaperQualitySignal(
  paper: CandidateItem,
): PaperQualitySignal {
  const hasSource = Boolean(paper.source_title);
  const hasAbstract = Boolean(paper.abstract);
  const hasRecentYear = Boolean(
    paper.year && paper.year >= new Date().getFullYear() - 15,
  );
  const hasCitations = Boolean((paper.citation_count ?? 0) > 0);
  const metadataComplete = hasSource && hasAbstract && Boolean(paper.year);

  let score = 0;

  if (paper.scopus_matched) {
    score += 2;
  }
  if (hasSource) {
    score += 1;
  }
  if (hasAbstract) {
    score += 1;
  }
  if (paper.year) {
    score += hasRecentYear ? 1 : 0.5;
  }
  if ((paper.citation_count ?? 0) >= 20) {
    score += 1;
  } else if ((paper.citation_count ?? 0) > 0) {
    score += 0.5;
  }
  if ((paper.final_score ?? 0) >= 0.7) {
    score += 1;
  } else if ((paper.final_score ?? 0) >= 0.45) {
    score += 0.5;
  }

  let level: PaperQualityLevel = "caution";
  let label = "Review with caution";
  let summary =
    "We have limited information for this paper (source, abstract, or year may be missing). Review it carefully before shortlisting.";

  if (score >= 5.5) {
    level = "strong";
    label = "Likely stronger candidate";
    summary =
      "Strong bibliographic signals—verified source match, abstract, recent publication, or citation support.";
  } else if (score >= 3.5) {
    level = "moderate";
    label = "Promising candidate";
    summary =
      "Reasonable metadata on file—compare with other results before you shortlist.";
  }

  return {
    level,
    label,
    summary,
    metadataComplete,
    providerLabel: formatProviderLabel(paper.provider),
    sourceCue: paper.scopus_matched
      ? "Source matched to Scopus"
      : "Source not matched",
    yearCue: deriveYearCue(paper.year),
    citationCue: deriveCitationCue(paper.citation_count),
    hasSource,
    hasAbstract,
    hasRecentYear,
    hasCitations,
  };
}

/* ── Session list query keys + shared hooks ─────────────────────────── */

/**
 * Single source of truth for the session-list `queryKey`. All call sites
 * (sidebar, sessions index page, session detail page, create/delete
 * mutations) reference this so the cache stays in lock-step.
 */
export const SESSION_QUERY_KEYS = {
  list: () => ["sessionList"] as const,
  queryContext: (sessionId: number) =>
    ["sessionQueryContext", sessionId] as const,
};

const SESSION_QUERY_CONTEXT_STALE_TIME = 60 * 1000;
const SESSION_QUERY_CONTEXT_GC_TIME = 30 * 60 * 1000;

export function normalizeQueryContextResponse(
  data: QueryContextResponse,
): SearchQueryContext | null {
  if (!data.exists) {
    return null;
  }

  const ctx = data.query_context;
  if (!ctx || Object.keys(ctx).length === 0) {
    return null;
  }

  return ctx;
}

export async function fetchSessionQueryContext(
  sessionId: number,
): Promise<SearchQueryContext | null> {
  const data = (await apiFetch(
    `/search/${sessionId}/query-context`,
  )) as QueryContextResponse;
  return normalizeQueryContextResponse(data);
}

const SESSION_LIST_STALE_TIME = 60 * 1000;
const SESSION_LIST_GC_TIME = 30 * 60 * 1000;

/**
 * Shared TanStack Query subscription for `GET /sessions`.
 *
 * `AppShell` (sidebar), the sessions index page, and the session detail
 * page all subscribe to the same `queryKey`, so the underlying network
 * request is deduplicated automatically — only one fetch fires per cache
 * window regardless of how many components are reading the list.
 *
 * Cache profile:
 * - `staleTime: 60s` — sessions don't churn during normal review use.
 * - `gcTime: 30m` — keeps the entry around across route transitions.
 * - `refetchOnWindowFocus: false` / `refetchOnReconnect: false` — avoid
 *   spurious refetches when the user tabs back into the browser.
 *
 * Trigger a fresh fetch by:
 * - calling the returned `refetch()` (manual refresh), or
 * - `queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEYS.list() })`
 *   after a server-side mutation (already wired in `useCreateSession`).
 */
export function useSessionList(options?: { enabled?: boolean }) {
  return useQuery<SessionItem[]>({
    queryKey: SESSION_QUERY_KEYS.list(),
    queryFn: () => apiFetch("/sessions") as Promise<SessionItem[]>,
    staleTime: SESSION_LIST_STALE_TIME,
    gcTime: SESSION_LIST_GC_TIME,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
    enabled: options?.enabled ?? true,
  });
}

/**
 * Look up a single session by id. Subscribes to the same `["sessionList"]`
 * cache via `select`, so this never triggers a separate network call when
 * the list is already cached.
 */
/**
 * Persisted query rewrite / filter context for a session.
 *
 * Loaded via `GET /search/{session_id}/query-context` so reopening a session
 * shows the last search context without re-running `POST /search`.
 */
export function useSessionQueryContext(
  sessionId: number,
  options?: { enabled?: boolean },
) {
  const enabled = Number.isFinite(sessionId) && (options?.enabled ?? true);
  return useQuery<SearchQueryContext | null>({
    queryKey: SESSION_QUERY_KEYS.queryContext(sessionId),
    queryFn: () => fetchSessionQueryContext(sessionId),
    enabled,
    staleTime: SESSION_QUERY_CONTEXT_STALE_TIME,
    gcTime: SESSION_QUERY_CONTEXT_GC_TIME,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
  });
}

export function useSession(sessionId: number, options?: { enabled?: boolean }) {
  const enabled = Number.isFinite(sessionId) && (options?.enabled ?? true);
  return useQuery<SessionItem[], Error, SessionItem | null>({
    queryKey: SESSION_QUERY_KEYS.list(),
    queryFn: () => apiFetch("/sessions") as Promise<SessionItem[]>,
    select: (data) => data.find((item) => item.id === sessionId) ?? null,
    enabled,
    staleTime: SESSION_LIST_STALE_TIME,
    gcTime: SESSION_LIST_GC_TIME,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
  });
}

/* ── Mutations ──────────────────────────────────────────────────────── */

export type CreateSessionPayload = {
  topic: string;
  raw_query: string;
  filters_json: Record<string, unknown>;
};

/**
 * `POST /sessions` mutation.
 *
 * On success:
 * 1. Optimistically prepends the new session to the cached list so the
 *    sidebar / index page reflect the change instantly, and so navigating
 *    to the new detail page finds it in cache (no flash of "Session not
 *    found").
 * 2. Invalidates the list query to schedule a background refetch — that
 *    way any other field the server may compute (e.g. server-generated
 *    timestamps that we didn't perfectly mirror) reconciles automatically.
 */
export function useCreateSession() {
  const queryClient = useQueryClient();
  return useMutation<SessionItem, Error, CreateSessionPayload>({
    mutationFn: (payload) =>
      apiFetch("/sessions", {
        method: "POST",
        body: JSON.stringify(payload),
      }) as Promise<SessionItem>,
    onSuccess: (created) => {
      queryClient.setQueryData<SessionItem[]>(
        SESSION_QUERY_KEYS.list(),
        (old) =>
          old
            ? [created, ...old.filter((s) => s.id !== created.id)]
            : [created],
      );
      queryClient.invalidateQueries({ queryKey: SESSION_QUERY_KEYS.list() });
    },
  });
}

/**
 * `DELETE /sessions/{id}` mutation.
 *
 * On success, removes the session from the cached list (no refetch — the
 * server already confirmed the delete via 204). Consumers can pass their
 * own `onSuccess` to chain navigation, toasts, etc.
 */
export function useDeleteSession() {
  const queryClient = useQueryClient();
  return useMutation<unknown, Error, number>({
    mutationFn: (sessionId) =>
      apiFetch(`/sessions/${sessionId}`, { method: "DELETE" }),
    onSuccess: (_data, sessionId) => {
      queryClient.setQueryData<SessionItem[]>(
        SESSION_QUERY_KEYS.list(),
        (old) => (old ? old.filter((s) => s.id !== sessionId) : []),
      );
    },
  });
}

/**
 * Patch a single session in the cached list without re-fetching.
 *
 * Use after a side-effectful action that mutates a session's local fields
 * (e.g. flipping `status` from `"created"` → `"searched"` after Run search)
 * so the sidebar / detail page reflect it immediately and consistently.
 */
export function patchSessionInCache(
  queryClient: QueryClient,
  sessionId: number,
  patch: Partial<SessionItem>,
) {
  queryClient.setQueryData<SessionItem[]>(SESSION_QUERY_KEYS.list(), (old) =>
    old ? old.map((s) => (s.id === sessionId ? { ...s, ...patch } : s)) : old,
  );
}

/** Convenience: invalidate the session list to force a background refetch. */
export function invalidateSessionList(queryClient: QueryClient) {
  return queryClient.invalidateQueries({
    queryKey: SESSION_QUERY_KEYS.list(),
  });
}

export function invalidateSessionQueryContext(
  queryClient: QueryClient,
  sessionId: number,
) {
  return queryClient.invalidateQueries({
    queryKey: SESSION_QUERY_KEYS.queryContext(sessionId),
  });
}

/** Patch query-context cache after `POST /search` without waiting for GET. */
export function setSessionQueryContextInCache(
  queryClient: QueryClient,
  sessionId: number,
  queryContext: SearchQueryContext | null | undefined,
) {
  queryClient.setQueryData<SearchQueryContext | null>(
    SESSION_QUERY_KEYS.queryContext(sessionId),
    queryContext && Object.keys(queryContext).length > 0 ? queryContext : null,
  );
}
