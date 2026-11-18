"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeftIcon,
  BookmarkCheck,
  BookmarkPlus,
  Settings2,
  Sparkles,
} from "lucide-react";
import toast from "react-hot-toast";
import AppShell from "@/components/app/AppShell";
import { Button } from "@/components/ui/Button";
import Notice from "@/components/ui/Notice";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import EmptyState from "@/components/sessions/EmptyState";
import PaperResultCard from "@/components/sessions/PaperResultCard";
import StatusBadge from "@/components/sessions/StatusBadge";
import CandidateFilterBar, {
  type SortDir,
  type SortField,
  type StatusFilter,
} from "@/components/sessions/CandidateFilterBar";
import DebateAgentPanel from "@/components/sessions/DebateAgentPanel";
import ReviewDraftPanel from "@/components/sessions/ReviewDraftPanel";
import { apiFetch, apiFetchText } from "@/lib/axios";
import { useLlmStatus } from "@/lib/llmPreview";
import {
  type CandidateSummaryResponse,
  type CandidateItem,
  DEFAULT_AI_SEARCH_FLAGS,
  deriveTopCandidates,
  formatCreatedAt,
  getSessionTopicDisplay,
  normalizeRawQuery,
  invalidateSessionQueryContext,
  patchSessionInCache,
  setSessionQueryContextInCache,
  type SearchLLMFlags,
  type SearchResponse,
  summarizeQuery,
  useSession,
  useSessionQueryContext,
} from "@/lib/session";
import { cn } from "@/lib/utils";

export default function SessionDetailPage() {
  const params = useParams<{ id: string }>();
  const rawId = params?.id;
  const sessionId = typeof rawId === "string" ? Number(rawId) : NaN;

  const queryClient = useQueryClient();

  // Read the session from the shared `["sessionList"]` cache via `select`.
  // This subscribes to the same queryKey as the sidebar and the sessions
  // index page, so opening this detail page doesn't trigger a fresh
  // `/sessions` request when the cache is already populated.
  const sessionQuery = useSession(sessionId);
  const session = sessionQuery.data ?? null;

  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [shortlistedPapers, setShortlistedPapers] = useState<CandidateItem[]>(
    [],
  );
  const queryContextQuery = useSessionQueryContext(sessionId);
  const queryContext = queryContextQuery.data ?? null;

  const [candidatesLoading, setCandidatesLoading] = useState(true);
  const [candidatesError, setCandidatesError] = useState("");
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [searchMode, setSearchMode] = useState<"standard" | "ai">("ai");

  // Per-step LLM flags for `POST /search/{id}` (Phase 2 contract). Only
  // forwarded to the server when `searchMode === "ai"`; standard mode sends
  // `use_llm=false` and omits the three step flags so the backend uses its
  // own defaults. See `docs/phase2_llm_backend_handoff.md`.
  const [aiFlags, setAiFlags] = useState<Omit<SearchLLMFlags, "use_llm">>({
    use_query_rewrite: DEFAULT_AI_SEARCH_FLAGS.use_query_rewrite,
    use_llm_rerank: DEFAULT_AI_SEARCH_FLAGS.use_llm_rerank,
    use_llm_summary: DEFAULT_AI_SEARCH_FLAGS.use_llm_summary,
  });
  const [updatingReviewId, setUpdatingReviewId] = useState<number | null>(null);
  const [exportingShortlist, setExportingShortlist] = useState(false);
  const [copyingShortlist, setCopyingShortlist] = useState(false);
  const [bulkShortlisting, setBulkShortlisting] = useState(false);
  const [generatingSummaryId, setGeneratingSummaryId] = useState<number | null>(
    null,
  );

  const [sortField, setSortField] = useState<SortField>("final_score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [scopusOnly, setScopusOnly] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("active");
  const [workspaceTab, setWorkspaceTab] = useState<
    | "top_ranked"
    | "candidate_pool"
    | "shortlisted"
    | "dismissed"
    | "review_draft"
  >("top_ranked");

  // Shared, cached /llm/status subscription (see `useLlmStatus` in `lib/llmPreview.ts`).
  // Used here for AI step toggles and the hero status pill; the AI preview console lives on `/sandbox` only.
  const llmStatusQuery = useLlmStatus();
  const llmStatus = llmStatusQuery.data ?? null;
  const llmStatusUnavailable =
    !llmStatusQuery.isLoading && llmStatusQuery.data === null;

  // Load candidates + shortlist in parallel, independent from the session
  // list cache. Re-runs only when `sessionId` changes.
  //
  // Uses an `AbortController` (not just a `cancelled` boolean) so that
  // rapidly switching sessions actually cancels the in-flight HTTP requests
  // on the network — the backend stops working on the abandoned request and
  // the browser frees the connection, instead of letting the response come
  // back only to be discarded.
  useEffect(() => {
    if (!Number.isFinite(sessionId)) {
      setCandidatesError("Invalid session id.");
      setCandidatesLoading(false);
      return;
    }

    const controller = new AbortController();
    const { signal } = controller;

    setCandidatesLoading(true);
    setCandidatesError("");

    Promise.all([
      apiFetch(`/search/${sessionId}/candidates`, { signal }) as Promise<
        CandidateItem[]
      >,
      apiFetch(`/search/${sessionId}/shortlist`, { signal }) as Promise<
        CandidateItem[]
      >,
    ])
      .then(([savedCandidates, savedShortlist]) => {
        if (signal.aborted) return;
        setCandidates(savedCandidates);
        setShortlistedPapers(savedShortlist);
      })
      .catch((err) => {
        // Ignore aborts — they are the expected path for this cleanup.
        if (signal.aborted) return;
        setCandidatesError(
          err instanceof Error
            ? err.message
            : "Unable to load saved candidates.",
        );
      })
      .finally(() => {
        if (!signal.aborted) setCandidatesLoading(false);
      });

    return () => controller.abort();
  }, [sessionId]);

  // Combined page-level loading + error state for the hero card.
  const loading = sessionQuery.isPending || candidatesLoading;
  const error = useMemo(() => {
    if (!Number.isFinite(sessionId)) return "Invalid session id.";
    if (sessionQuery.isError) {
      return sessionQuery.error instanceof Error
        ? sessionQuery.error.message
        : "Unable to load the session workspace.";
    }
    if (!sessionQuery.isPending && !session) return "Session not found.";
    return candidatesError;
  }, [
    sessionId,
    sessionQuery.isError,
    sessionQuery.error,
    sessionQuery.isPending,
    session,
    candidatesError,
  ]);

  const sessionNotFound = useMemo(
    () =>
      Number.isFinite(sessionId) &&
      !sessionQuery.isPending &&
      !sessionQuery.isError &&
      !session,
    [sessionId, sessionQuery.isPending, sessionQuery.isError, session],
  );

  // `POST /search/{id}` wrapped as a mutation.
  //
  // Cache invalidation contract (intentionally narrow):
  // - `sessionList` is patched in-place so the sidebar / index page reflect
  //   the new `status: "searched"` without a refetch.
  // - The candidate + shortlist datasets live in this component's local
  //   state (loaded by the page's mount-time `useEffect`); we replace them
  //   directly from the response, so there is nothing to invalidate via the
  //   TanStack cache. If those fetches ever migrate to `useQuery`, add an
  //   `invalidateQueries({ queryKey: ["candidates", sessionId] })` here.
  // - `sessionQueryContext` is patched from `data.query_context` and
  //   invalidated so a background GET reconciles with the persisted row.
  // - `/llm/status` is a deployment-level config snapshot and is *not*
  //   affected by running a search, so we deliberately leave that cache
  //   alone (avoids tearing down the shared subscription used by the
  //   sidebar + preview panel).
  const runSearchMutation = useMutation<SearchResponse, Error, void>({
    mutationFn: async () => {
      if (!session) throw new Error("Session is not loaded.");
      const params = new URLSearchParams({ query: session.raw_query });
      const useLlm = searchMode === "ai";
      params.set("use_llm", useLlm ? "true" : "false");
      // Only forward the per-step flags when AI is on. In standard mode we
      // intentionally omit them so the backend defaults (mirroring `use_llm`)
      // resolve to `false` for every step.
      if (useLlm) {
        params.set("use_query_rewrite", String(aiFlags.use_query_rewrite));
        params.set("use_llm_rerank", String(aiFlags.use_llm_rerank));
        params.set("use_llm_summary", String(aiFlags.use_llm_summary));
      }
      return (await apiFetch(`/search/${session.id}?${params.toString()}`, {
        method: "POST",
      })) as SearchResponse;
    },
    onSuccess: (data) => {
      if (!session) return;

      setCandidates(data.candidates);
      setShortlistedPapers([]);
      setSessionQueryContextInCache(
        queryClient,
        session.id,
        data.query_context,
      );
      void invalidateSessionQueryContext(queryClient, session.id);

      if (
        searchMode === "ai" &&
        data.query_context?.llm_requested &&
        !data.query_context.llm_available
      ) {
        toast(
          data.query_context.llm_disabled_reason ||
            "AI is not active right now — showing standard results.",
          { icon: "ℹ️" },
        );
      } else if (
        searchMode === "ai" &&
        data.query_context?.llm_requested &&
        !data.query_context.llm_query_rewrite_used &&
        !data.candidates.some((c) => typeof c.llm_relevance_score === "number")
      ) {
        toast(
          data.query_context.llm_disabled_reason ||
            "No AI signals were returned — showing standard results.",
          { icon: "ℹ️" },
        );
      }

      patchSessionInCache(queryClient, session.id, { status: "searched" });
      toast.success(
        "Search completed and the session workspace was refreshed.",
      );
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Unable to run search.");
    },
  });

  const runningSearch = runSearchMutation.isPending;
  const runSearch = () => runSearchMutation.mutate();
  const slowSearchNoticeShownRef = useRef(false);

  useEffect(() => {
    if (!runningSearch) {
      slowSearchNoticeShownRef.current = false;
      return;
    }

    const timer = window.setTimeout(() => {
      if (slowSearchNoticeShownRef.current) return;
      slowSearchNoticeShownRef.current = true;
      toast(
        "Search is still in progress. Depending on your query and filters, this can take a few minutes.",
        { icon: "ℹ️", duration: 8000 },
      );
    }, 10_000);

    return () => window.clearTimeout(timer);
  }, [runningSearch]);

  async function refreshCandidates() {
    if (!session) return;
    setLoadingCandidates(true);

    try {
      const data = (await apiFetch(
        `/search/${session.id}/candidates`,
      )) as CandidateItem[];
      const shortlist = (await apiFetch(
        `/search/${session.id}/shortlist`,
      )) as CandidateItem[];
      setCandidates(data);
      setShortlistedPapers(shortlist);
      toast.success("Saved candidates were refreshed.");
    } catch (err) {
      toast.error(
        err instanceof Error
          ? err.message
          : "Unable to refresh saved candidates.",
      );
    } finally {
      setLoadingCandidates(false);
    }
  }

  /** Top 5 non-dismissed candidates from the pool by highest final_score. */
  const topRankedPapers = useMemo(
    () =>
      deriveTopCandidates(
        candidates.filter((i) => i.review_status !== "dismissed"),
      ),
    [candidates],
  );

  const metrics = useMemo(
    () => ({
      selectedCount: topRankedPapers.length,
      shortlistedCount: shortlistedPapers.length,
      candidateCount: candidates.length,
      matchedCount: candidates.filter((i) => i.scopus_matched).length,
    }),
    [candidates, topRankedPapers, shortlistedPapers],
  );

  const dismissedCandidates = useMemo(
    () => candidates.filter((i) => i.review_status === "dismissed"),
    [candidates],
  );

  /** Non-dismissed papers — matches the pool tab default "Active" view. */
  const activeCandidatePoolCount = useMemo(
    () => candidates.filter((i) => i.review_status !== "dismissed").length,
    [candidates],
  );

  const poolBaseForFilters = useMemo(() => {
    if (statusFilter === "active") {
      return candidates.filter((i) => i.review_status !== "dismissed");
    }
    if (statusFilter === "shortlisted") {
      return candidates.filter((i) => i.review_status === "shortlisted");
    }
    return candidates;
  }, [candidates, statusFilter]);

  const filteredAndSortedCandidates = useMemo(() => {
    let pool = candidates;
    if (statusFilter === "active")
      pool = pool.filter((i) => i.review_status !== "dismissed");
    else if (statusFilter === "shortlisted")
      pool = pool.filter((i) => i.review_status === "shortlisted");
    if (scopusOnly) pool = pool.filter((i) => i.scopus_matched);

    return [...pool].sort((a, b) => {
      let aVal: number, bVal: number;
      if (sortField === "final_score") {
        aVal = a.final_score ?? Number.NEGATIVE_INFINITY;
        bVal = b.final_score ?? Number.NEGATIVE_INFINITY;
      } else if (sortField === "year") {
        aVal = a.year ?? 0;
        bVal = b.year ?? 0;
      } else if (sortField === "citation_count") {
        aVal = a.citation_count ?? 0;
        bVal = b.citation_count ?? 0;
      } else {
        aVal = a.llm_relevance_score ?? Number.NEGATIVE_INFINITY;
        bVal = b.llm_relevance_score ?? Number.NEGATIVE_INFINITY;
      }
      return sortDir === "desc" ? bVal - aVal : aVal - bVal;
    });
  }, [candidates, sortField, sortDir, scopusOnly, statusFilter]);

  const unfilteredCandidateCount = useMemo(
    () =>
      scopusOnly
        ? poolBaseForFilters.filter((i) => i.scopus_matched).length
        : poolBaseForFilters.length,
    [poolBaseForFilters, scopusOnly],
  );

  function mergeUpdatedCandidate(
    updated: Partial<CandidateItem> & { id: number },
  ) {
    const id = updated.id;
    setCandidates((cur) =>
      cur.map((i) => (i.id === id ? { ...i, ...updated } : i)),
    );
    setShortlistedPapers((cur) => {
      const inShortlist = cur.find((i) => i.id === id);
      const effectiveStatus =
        updated.review_status !== undefined
          ? updated.review_status
          : inShortlist?.review_status;

      if (effectiveStatus === "shortlisted") {
        const remaining = cur.filter((i) => i.id !== id);
        const fromCandidates = candidates.find((i) => i.id === id);
        const itemToLink = {
          ...(fromCandidates ?? inShortlist ?? {}),
          ...updated,
          review_status: "shortlisted" as const,
        } as CandidateItem;

        return [...remaining, itemToLink].sort((l, r) => {
          const ls = l.final_score ?? Number.NEGATIVE_INFINITY;
          const rs = r.final_score ?? Number.NEGATIVE_INFINITY;
          return rs - ls || l.id - r.id;
        });
      }
      return cur.filter((i) => i.id !== id);
    });
  }

  async function updateReviewStatus(
    paper: CandidateItem,
    reviewStatus: "shortlisted" | "dismissed" | null,
  ) {
    if (!session) return;
    setUpdatingReviewId(paper.id);

    try {
      const updated = reviewStatus
        ? ((await apiFetch(
            `/search/${session.id}/candidates/${paper.id}/review`,
            { method: "POST", body: JSON.stringify({ status: reviewStatus }) },
          )) as CandidateItem)
        : ((await apiFetch(
            `/search/${session.id}/candidates/${paper.id}/review`,
            { method: "DELETE" },
          )) as CandidateItem);

      // DELETE /review omits `review_status` in JSON (exclude_none); merge alone would
      // keep the previous "dismissed" / "shortlisted" flag on the client.
      mergeUpdatedCandidate(
        reviewStatus === null ? { ...updated, review_status: null } : updated,
      );
      toast.success(
        reviewStatus === "shortlisted"
          ? "Paper added to the shortlist."
          : reviewStatus === "dismissed"
            ? "Paper dismissed from the active candidate pool."
            : "Paper restored to the active candidate pool.",
      );
    } catch (err) {
      toast.error(
        err instanceof Error
          ? err.message
          : "Unable to update paper review state.",
      );
    } finally {
      setUpdatingReviewId(null);
    }
  }

  async function generateSummary(paper: CandidateItem, force: boolean) {
    if (!session) return;
    setGeneratingSummaryId(paper.id);

    try {
      const response = (await apiFetch(
        `/search/${session.id}/candidates/${paper.id}/summary?force=${force}`,
        { method: "POST" },
      )) as CandidateSummaryResponse;

      // Update only the summary field to avoid overwriting other fields with stale data from the closure
      mergeUpdatedCandidate({
        id: paper.id,
        llm_summary: response.summary || null,
      });

      if (response.fallback_reason != null) {
        let msg = response.fallback_reason;
        if (response.fallback_reason === "no_abstract") {
          msg =
            "This paper has no abstract available to generate an AI summary.";
        } else if (response.fallback_reason === "step_disabled") {
          msg = "AI summary generation is currently disabled.";
        }
        toast(msg, { icon: "ℹ️" });
      } else if (!response.cached) {
        toast.success(
          force ? "AI summary regenerated." : "AI summary generated.",
        );
      }
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Unable to generate summary.",
      );
    } finally {
      setGeneratingSummaryId(null);
    }
  }

  function papersEligibleForShortlist(papers: CandidateItem[]) {
    return papers.filter(
      (p) =>
        p.review_status !== "shortlisted" && p.review_status !== "dismissed",
    );
  }

  async function bulkShortlistPapers(
    papers: CandidateItem[],
    successNoun: string,
  ) {
    if (!session) return;
    const unshortlisted = papersEligibleForShortlist(papers);
    if (unshortlisted.length === 0) return;

    setBulkShortlisting(true);

    try {
      for (const paper of unshortlisted) {
        const updated = (await apiFetch(
          `/search/${session.id}/candidates/${paper.id}/review`,
          { method: "POST", body: JSON.stringify({ status: "shortlisted" }) },
        )) as CandidateItem;
        mergeUpdatedCandidate(updated);
      }
      toast.success(
        `${unshortlisted.length} ${successNoun}${unshortlisted.length === 1 ? "" : "s"} added to the shortlist.`,
      );
    } catch (err) {
      toast.error(
        err instanceof Error
          ? err.message
          : `Unable to shortlist all ${successNoun}s.`,
      );
    } finally {
      setBulkShortlisting(false);
    }
  }

  async function bulkShortlistTopRanked() {
    await bulkShortlistPapers(topRankedPapers, "top-ranked paper");
  }

  async function bulkShortlistFilteredPool() {
    await bulkShortlistPapers(filteredAndSortedCandidates, "paper");
  }

  async function exportShortlist() {
    if (!session) return;
    setExportingShortlist(true);

    try {
      const markdown = await apiFetchText(
        `/search/${session.id}/shortlist/export`,
      );
      const blob = new Blob([markdown], {
        type: "text/markdown;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `session-${session.id}-shortlist.md`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      // toast.success("Shortlist exported as Markdown.");
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Unable to export shortlist.",
      );
    } finally {
      setExportingShortlist(false);
    }
  }

  async function copyShortlistToClipboard() {
    if (!session) return;
    setCopyingShortlist(true);

    try {
      const markdown = await apiFetchText(
        `/search/${session.id}/shortlist/export`,
      );
      await navigator.clipboard.writeText(markdown);
      toast.success("Shortlist Markdown copied to clipboard.");
    } catch (err) {
      toast.error(
        err instanceof Error
          ? err.message
          : "Unable to copy shortlist to clipboard.",
      );
    } finally {
      setCopyingShortlist(false);
    }
  }

  const allTopRankedShortlisted =
    topRankedPapers.length > 0 &&
    topRankedPapers.every((i) => i.review_status === "shortlisted");

  const shortlistableTopCount = useMemo(
    () => papersEligibleForShortlist(topRankedPapers).length,
    [topRankedPapers],
  );

  const shortlistablePoolCount = useMemo(
    () => papersEligibleForShortlist(filteredAndSortedCandidates).length,
    [filteredAndSortedCandidates],
  );

  const allFilteredPoolShortlisted =
    filteredAndSortedCandidates.length > 0 && shortlistablePoolCount === 0;

  const sessionTopicDisplay = useMemo(
    () => (session ? getSessionTopicDisplay(session.topic) : null),
    [session],
  );

  const appliedFilters = queryContext?.applied_filters ?? null;
  const hasAiRunDetails = Boolean(
    queryContext &&
    (queryContext.llm_requested ||
      queryContext.llm_enabled ||
      queryContext.llm_disabled_reason),
  );

  const heroFilterPills = useMemo(() => {
    if (!appliedFilters) return [] as string[];
    const f = appliedFilters as Record<string, unknown>;
    const pills: string[] = [];

    if (typeof f.recent_years === "number") {
      pills.push(`Last ${f.recent_years} years`);
    } else if (f.year_from || f.year_to) {
      const from = f.year_from ? String(f.year_from) : "Any";
      const to = f.year_to ? String(f.year_to) : "Any";
      pills.push(`Year ${from} - ${to}`);
    }
    if (typeof f.open_access === "boolean") {
      pills.push(f.open_access ? "Open access" : "Any access");
    } else if (typeof f.access_type === "string") {
      pills.push(`Access: ${f.access_type}`);
    }
    if (typeof f.scopus_only === "boolean" && f.scopus_only) {
      pills.push("Scopus only");
    }
    if (typeof f.only_active_scopus === "boolean" && f.only_active_scopus) {
      pills.push("Only active Scopus");
    }
    if (typeof f.provider === "string" && f.provider) {
      pills.push(`Provider: ${f.provider}`);
    }
    if (typeof f.source === "string" && f.source) {
      pills.push(`Source: ${f.source}`);
    }
    if (typeof f.min_citations === "number") {
      pills.push(`Min citations: ${f.min_citations}`);
    }
    if (typeof f.limit === "number") {
      pills.push(`Limit: ${f.limit}`);
    }
    return pills;
  }, [appliedFilters]);

  return (
    <AppShell>
      <div className="px-6 py-6 max-w-6xl mx-auto">
        {loading && (
          <div className="grid gap-5 max-w-6xl">
            <Card>
              <CardHeader>
                <div className="h-4 w-40 animate-pulse rounded bg-muted" />
                <div className="mt-3 h-7 max-w-md animate-pulse rounded bg-muted" />
                <div className="mt-2 h-16 animate-pulse rounded-md bg-muted/80" />
              </CardHeader>
              <CardContent className="grid gap-3">
                <div className="flex flex-wrap gap-2">
                  <div className="h-9 w-32 animate-pulse rounded-md bg-muted" />
                  <div className="h-9 w-24 animate-pulse rounded-md bg-muted" />
                  <div className="h-9 w-28 animate-pulse rounded-md bg-muted" />
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="h-20 animate-pulse rounded-lg border border-border bg-muted/40"
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="py-0">
                <div className="mb-4 h-9 w-full max-w-lg animate-pulse rounded-lg bg-muted" />
                <div className="grid gap-3">
                  <div className="h-28 animate-pulse rounded-xl border bg-muted/30" />
                  <div className="h-28 animate-pulse rounded-xl border bg-muted/30" />
                  <div className="h-28 animate-pulse rounded-xl border bg-muted/30" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {!loading && sessionNotFound && (
          <Card className="border-dashed">
            <CardContent className="flex min-h-[min(24rem,50vh)] flex-col items-center justify-center gap-4 px-6 py-16 text-center">
              <div className="grid gap-2 max-w-md">
                <h2 className="font-serif text-xl font-normal tracking-tight text-foreground">
                  Something went wrong
                </h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  This session could not be loaded. Please try again later.
                </p>
              </div>
              <Button asChild className="mt-2">
                <Link href="/new-search">Start a new search</Link>
              </Button>
            </CardContent>
          </Card>
        )}

        {!loading && error && !sessionNotFound && (
          <Notice tone="error">{error}</Notice>
        )}

        {!loading && !error && session && (
          <div className="grid gap-5">
            {/* Hero card */}
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mb-2">
                      <Link
                        href="/sessions"
                        className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
                      >
                        <ArrowLeftIcon className="h-3 w-3" />
                        Sessions
                      </Link>
                      <span>·</span>
                      <span>Session #{session.id}</span>
                      <span>·</span>
                      <span>Created {formatCreatedAt(session.created_at)}</span>
                    </div>
                    <CardTitle className="text-xl font-semibold leading-snug tracking-normal">
                      {normalizeRawQuery(session.raw_query)}
                    </CardTitle>
                    {sessionTopicDisplay && (
                      <p
                        className={cn(
                          "mt-1.5 min-h-[1.25rem] text-sm leading-relaxed",
                          sessionTopicDisplay.isPlaceholder
                            ? "italic text-muted-foreground/60"
                            : "text-muted-foreground",
                        )}
                      >
                        {sessionTopicDisplay.label}
                      </p>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-2 sm:flex-row sm:items-center sm:gap-2">
                    <StatusBadge status={session.status} />
                    <Badge
                      variant="outline"
                      className={cn(
                        "rounded-full text-xs font-medium shrink-0 gap-1.5",
                        llmStatus?.available
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : llmStatusUnavailable
                            ? "border-border bg-muted text-muted-foreground"
                            : "border-amber-200 bg-amber-50 text-amber-700",
                      )}
                    >
                      <span
                        className="h-1.5 w-1.5 shrink-0 rounded-full bg-current"
                        aria-hidden
                      />
                      {llmStatus?.available
                        ? "AI available"
                        : llmStatusUnavailable
                          ? "AI status unavailable"
                          : "AI not enabled"}
                    </Badge>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="grid gap-4">
                {/* Primary actions + settings popover */}
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    size="sm"
                    onClick={runSearch}
                    disabled={runningSearch}
                  >
                    {runningSearch ? "Running search…" : "Run search"}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={refreshCandidates}
                    disabled={loadingCandidates || runningSearch}
                  >
                    {loadingCandidates ? "Refreshing…" : "Refresh candidates"}
                  </Button>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        size="sm"
                        variant="outline"
                        className="gap-2"
                        type="button"
                        disabled={runningSearch}
                      >
                        <Settings2 className="size-4 shrink-0" />
                        Search settings
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-80 sm:w-96" align="start">
                      <div className="grid gap-4">
                        <div>
                          <p className="mb-2 text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground">
                            Search mode
                          </p>
                          <div className="inline-flex items-center gap-1 rounded-full border border-border bg-background p-1">
                            {(["ai", "standard"] as const).map((mode) => (
                              <button
                                key={mode}
                                type="button"
                                onClick={() => setSearchMode(mode)}
                                disabled={runningSearch}
                                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-50 ${
                                  searchMode === mode
                                    ? "bg-primary/10 border border-primary/15 text-primary"
                                    : "text-muted-foreground hover:text-foreground"
                                }`}
                              >
                                {mode === "standard"
                                  ? "Standard search"
                                  : "AI-assisted search"}
                              </button>
                            ))}
                          </div>
                        </div>
                        {searchMode === "ai" && (
                          <div>
                            <p className="mb-2 text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground">
                              AI steps
                            </p>
                            <div className="grid gap-2">
                              {(
                                [
                                  {
                                    key: "use_query_rewrite",
                                    label: "Query rewrite",
                                    hint: "Expand and optimize your query before search.",
                                    backendEnabled:
                                      llmStatus?.query_rewrite_enabled,
                                  },
                                  {
                                    key: "use_llm_rerank",
                                    label: "Rerank",
                                    hint: "Re-score candidates by semantic relevance.",
                                    backendEnabled: llmStatus?.rerank_enabled,
                                  },
                                  {
                                    key: "use_llm_summary",
                                    label: "Summary",
                                    hint: "Generate one-sentence summaries for top papers.",
                                    backendEnabled: llmStatus?.summary_enabled,
                                  },
                                ] as const
                              ).map((step) => {
                                const checked = aiFlags[step.key];
                                const unavailable =
                                  llmStatus !== null &&
                                  step.backendEnabled === false;
                                return (
                                  <div
                                    key={step.key}
                                    className={`flex items-center justify-between gap-3 rounded-md border border-border bg-muted/30 px-3 py-2 ${
                                      unavailable ? "opacity-60" : ""
                                    }`}
                                  >
                                    <div className="min-w-0 flex-1">
                                      <Label
                                        htmlFor={`ai-flag-${step.key}`}
                                        className="cursor-pointer text-sm font-medium text-foreground"
                                      >
                                        {step.label}
                                      </Label>
                                      <p className="mt-0.5 text-[0.65rem] text-muted-foreground">
                                        {unavailable
                                          ? "Currently unavailable"
                                          : step.hint}
                                      </p>
                                    </div>
                                    <Switch
                                      id={`ai-flag-${step.key}`}
                                      checked={checked && !unavailable}
                                      onCheckedChange={(value) =>
                                        setAiFlags((cur) => ({
                                          ...cur,
                                          [step.key]: value,
                                        }))
                                      }
                                      disabled={runningSearch || unavailable}
                                    />
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </PopoverContent>
                  </Popover>
                  {hasAiRunDetails && (
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button size="sm" variant="outline">
                          Show details
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="sm:max-w-lg">
                        <DialogHeader>
                          <DialogTitle>AI run status</DialogTitle>
                          <DialogDescription>
                            Runtime diagnostics from the last search response.
                          </DialogDescription>
                        </DialogHeader>
                        <div className="grid gap-3">
                          <div className="flex flex-wrap gap-2">
                            <Badge
                              variant="outline"
                              className="text-xs rounded-full"
                            >
                              Requested:{" "}
                              {queryContext?.llm_requested ? "yes" : "no"}
                            </Badge>
                            <Badge
                              variant="outline"
                              className="text-xs rounded-full"
                            >
                              Available:{" "}
                              {queryContext?.llm_available ? "yes" : "no"}
                            </Badge>
                            <Badge
                              variant="outline"
                              className="text-xs rounded-full"
                            >
                              Rewrite used:{" "}
                              {queryContext?.llm_query_rewrite_used
                                ? "yes"
                                : "no"}
                            </Badge>
                          </div>
                          {queryContext?.llm_disabled_reason && (
                            <p className="text-xs text-muted-foreground leading-relaxed">
                              {queryContext.llm_disabled_reason}
                            </p>
                          )}
                          {appliedFilters && (
                            <div>
                              <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-1">
                                Applied filters
                              </p>
                              <p className="text-xs text-muted-foreground break-all">
                                {JSON.stringify(appliedFilters)}
                              </p>
                            </div>
                          )}
                        </div>
                      </DialogContent>
                    </Dialog>
                  )}
                </div>

                {heroFilterPills.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">
                      Last search
                    </span>
                    {heroFilterPills.map((pill) => (
                      <span
                        key={pill}
                        className="h-8 inline-flex items-center whitespace-nowrap rounded-md border border-input bg-background px-3 text-sm text-muted-foreground"
                      >
                        {pill}
                      </span>
                    ))}
                  </div>
                )}
                {queryContext?.optimized_query &&
                  queryContext.optimized_query.trim() !==
                    session.raw_query.trim() && (
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      <span className="font-medium text-foreground">
                        Optimized query:{" "}
                      </span>
                      {summarizeQuery(queryContext.optimized_query, 280)}
                    </p>
                  )}

                {/* Stats row */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    { value: metrics.selectedCount, label: "Top-ranked" },
                    { value: metrics.shortlistedCount, label: "Shortlisted" },
                    {
                      value: metrics.candidateCount,
                      label: "Saved candidates",
                    },
                    { value: metrics.matchedCount, label: "Scopus matched" },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="rounded-lg border border-primary/12 bg-primary/5 px-4 py-3"
                    >
                      <p className="font-serif text-2xl text-foreground">
                        {stat.value}
                      </p>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {stat.label}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {queryContext?.llm_query_rewrite_used && (
              <Collapsible className="rounded-lg border border-primary/20 bg-primary/5">
                <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
                  <div className="flex min-w-0 flex-1 items-start gap-2">
                    <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground leading-snug">
                        AI optimized your query for retrieval.
                      </p>
                      {queryContext.query_rewrite_notes ? (
                        <p className="mt-0.5 text-xs text-muted-foreground leading-relaxed">
                          {queryContext.query_rewrite_notes}
                        </p>
                      ) : null}
                    </div>
                  </div>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" size="sm" className="shrink-0">
                      View details
                    </Button>
                  </CollapsibleTrigger>
                </div>
                <CollapsibleContent>
                  <div className="border-t border-primary/15 px-3 pb-3 pt-2">
                    <div className="grid gap-3 sm:grid-cols-2">
                      {[
                        {
                          label: "Original query",
                          value:
                            queryContext.original_query || session.raw_query,
                        },
                        {
                          label: "Optimized query",
                          value:
                            queryContext.optimized_query || session.raw_query,
                        },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="rounded-lg border border-primary/12 bg-background/80 p-3"
                        >
                          <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-1">
                            {item.label}
                          </p>
                          <p className="text-sm text-foreground leading-snug break-words">
                            {item.value}
                          </p>
                        </div>
                      ))}
                    </div>
                    {queryContext.key_terms?.length ? (
                      <div className="mt-3">
                        <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                          Key terms
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {queryContext.key_terms.map((term) => (
                            <Badge
                              key={term}
                              variant="outline"
                              className="rounded-full text-xs"
                            >
                              {term}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    {queryContext.expanded_queries?.length ? (
                      <div className="mt-3">
                        <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                          Expanded queries
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {queryContext.expanded_queries.map((q) => (
                            <Badge
                              key={q}
                              variant="outline"
                              className="rounded-full text-xs"
                            >
                              {q}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}

            <Card>
              <CardContent className="py-0">
                <Tabs
                  value={workspaceTab}
                  onValueChange={(v) =>
                    setWorkspaceTab(
                      v as
                        | "top_ranked"
                        | "candidate_pool"
                        | "shortlisted"
                        | "dismissed"
                        | "review_draft",
                    )
                  }
                >
                  <TabsList className="h-auto min-h-9 w-full justify-start gap-1 p-1">
                    <TabsTrigger value="top_ranked">
                      Top-ranked ({topRankedPapers.length})
                    </TabsTrigger>
                    <TabsTrigger value="candidate_pool">
                      Candidate pool ({activeCandidatePoolCount})
                    </TabsTrigger>
                    <TabsTrigger value="shortlisted">
                      Shortlisted ({shortlistedPapers.length})
                    </TabsTrigger>
                    <TabsTrigger value="dismissed">
                      Dismissed ({dismissedCandidates.length})
                    </TabsTrigger>
                    <TabsTrigger value="review_draft">Review draft</TabsTrigger>
                  </TabsList>

                  <TabsContent value="top_ranked" className="mt-4">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-xs text-muted-foreground">
                        {session.status === "searched"
                          ? "Top 5 active candidates from the pool, ranked by highest final score."
                          : "Run search first to populate this section."}
                      </p>
                      {topRankedPapers.length > 0 && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="gap-1.5 shrink-0"
                          onClick={bulkShortlistTopRanked}
                          disabled={
                            bulkShortlisting ||
                            !!updatingReviewId ||
                            allTopRankedShortlisted
                          }
                        >
                          {bulkShortlisting ? (
                            "Shortlisting…"
                          ) : allTopRankedShortlisted ? (
                            <>
                              <BookmarkCheck
                                className="size-4 shrink-0"
                                aria-hidden
                              />
                              All shortlisted ({topRankedPapers.length})
                            </>
                          ) : (
                            <>
                              <BookmarkPlus
                                className="size-4 shrink-0"
                                aria-hidden
                              />
                              Shortlist all ({topRankedPapers.length})
                            </>
                          )}
                        </Button>
                      )}
                    </div>
                    {candidates.length > 0 && (
                      <p className="text-xs text-muted-foreground mb-3">
                        Found {candidates.length} candidates · showing top{" "}
                        {topRankedPapers.length}
                      </p>
                    )}
                    {topRankedPapers.length ? (
                      <div className="grid gap-2">
                        {topRankedPapers.map((paper, index) => (
                          <PaperResultCard
                            key={paper.id}
                            paper={paper}
                            variant="selected"
                            rank={index + 1}
                            actionBusy={updatingReviewId === paper.id}
                            onShortlist={(item) =>
                              updateReviewStatus(item, "shortlisted")
                            }
                            onDismiss={(item) =>
                              updateReviewStatus(item, "dismissed")
                            }
                            onGenerateSummary={generateSummary}
                            summaryBusy={generatingSummaryId === paper.id}
                          />
                        ))}
                      </div>
                    ) : (
                      <EmptyState
                        title="No top-ranked papers yet"
                        body="This session does not have saved candidates yet. Run search or refresh the saved candidates list."
                      />
                    )}
                  </TabsContent>

                  <TabsContent value="candidate_pool" className="mt-4">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <p className="text-xs text-muted-foreground">
                        Browse all saved candidates. Shortlist or dismiss any
                        paper—same actions as in Top-ranked.
                      </p>
                      {filteredAndSortedCandidates.length > 0 &&
                        shortlistablePoolCount > 0 && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-1.5 shrink-0"
                            onClick={bulkShortlistFilteredPool}
                            disabled={
                              bulkShortlisting ||
                              !!updatingReviewId ||
                              allFilteredPoolShortlisted
                            }
                          >
                            {bulkShortlisting ? (
                              "Shortlisting…"
                            ) : (
                              <>
                                <BookmarkPlus
                                  className="size-4 shrink-0"
                                  aria-hidden
                                />
                                Shortlist visible ({shortlistablePoolCount})
                              </>
                            )}
                          </Button>
                        )}
                      {filteredAndSortedCandidates.length > 0 &&
                        shortlistablePoolCount === 0 &&
                        statusFilter !== "shortlisted" && (
                          <span className="text-xs text-muted-foreground shrink-0">
                            All visible papers already shortlisted or dismissed
                          </span>
                        )}
                    </div>
                    <CandidateFilterBar
                      sortField={sortField}
                      sortDir={sortDir}
                      scopusOnly={scopusOnly}
                      statusFilter={statusFilter}
                      onSortField={setSortField}
                      onSortDir={setSortDir}
                      onScopusOnly={setScopusOnly}
                      onStatusFilter={setStatusFilter}
                      totalCount={unfilteredCandidateCount}
                      filteredCount={filteredAndSortedCandidates.length}
                    />
                    {filteredAndSortedCandidates.length ? (
                      <div className="mt-3 grid gap-2">
                        {filteredAndSortedCandidates.map((paper) => (
                          <PaperResultCard
                            key={paper.id}
                            paper={paper}
                            variant="candidate"
                            actionBusy={updatingReviewId === paper.id}
                            onShortlist={(item) =>
                              updateReviewStatus(item, "shortlisted")
                            }
                            onDismiss={(item) =>
                              updateReviewStatus(item, "dismissed")
                            }
                            onGenerateSummary={generateSummary}
                            summaryBusy={generatingSummaryId === paper.id}
                          />
                        ))}
                      </div>
                    ) : (
                      <EmptyState
                        title="No candidates match the current filters"
                        body="Try adjusting the sort or filter options above, or run search to populate the workspace."
                      />
                    )}
                  </TabsContent>

                  <TabsContent value="shortlisted" className="mt-4">
                    <div className="grid gap-4">
                      {session.status === "searched" &&
                        shortlistableTopCount > 0 && (
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <p className="text-xs text-muted-foreground min-w-0 max-w-xl leading-relaxed">
                              Add every remaining top-ranked paper to this
                              shortlist in one step ({shortlistableTopCount}{" "}
                              {shortlistableTopCount === 1 ? "paper" : "papers"}
                              ).
                            </p>
                            <Button
                              size="sm"
                              variant="outline"
                              className="gap-1.5 shrink-0"
                              onClick={bulkShortlistTopRanked}
                              disabled={
                                bulkShortlisting ||
                                !!updatingReviewId ||
                                allTopRankedShortlisted
                              }
                            >
                              {bulkShortlisting ? (
                                "Shortlisting…"
                              ) : (
                                <>
                                  <BookmarkPlus
                                    className="size-4 shrink-0"
                                    aria-hidden
                                  />
                                  Shortlist all ({shortlistableTopCount})
                                </>
                              )}
                            </Button>
                          </div>
                        )}
                      {shortlistedPapers.length > 0 && (
                        <DebateAgentPanel
                          sessionId={session.id}
                          shortlistedCount={shortlistedPapers.length}
                        />
                      )}
                      {shortlistedPapers.length > 0 ? (
                        <>
                          <div className="flex flex-wrap justify-end gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={copyShortlistToClipboard}
                              disabled={copyingShortlist}
                            >
                              {copyingShortlist ? "Copying…" : "Copy .md"}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={exportShortlist}
                              disabled={exportingShortlist}
                            >
                              {exportingShortlist ? "Exporting…" : "Export .md"}
                            </Button>
                          </div>
                          <div className="grid gap-2">
                            {shortlistedPapers.map((paper) => (
                              <PaperResultCard
                                key={paper.id}
                                paper={paper}
                                variant="selected"
                                actionBusy={updatingReviewId === paper.id}
                                onDismiss={(item) =>
                                  updateReviewStatus(item, "dismissed")
                                }
                                onRestore={(item) =>
                                  updateReviewStatus(item, null)
                                }
                                onGenerateSummary={generateSummary}
                                summaryBusy={generatingSummaryId === paper.id}
                              />
                            ))}
                          </div>
                        </>
                      ) : (
                        <EmptyState
                          title="No shortlisted papers yet"
                          body="Use Shortlist on candidate cards to build an exportable reading set for this session."
                        />
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="dismissed" className="mt-4">
                    {dismissedCandidates.length ? (
                      <div className="grid gap-2">
                        {dismissedCandidates.map((paper) => (
                          <PaperResultCard
                            key={paper.id}
                            paper={paper}
                            variant="candidate"
                            actionBusy={updatingReviewId === paper.id}
                            onRestore={(item) => updateReviewStatus(item, null)}
                          />
                        ))}
                      </div>
                    ) : (
                      <EmptyState
                        title="No dismissed papers"
                        body="Dismissed candidates appear here so you can restore them if needed."
                      />
                    )}
                  </TabsContent>

                  <TabsContent value="review_draft" className="mt-4">
                    <p className="mb-4 text-xs text-muted-foreground leading-relaxed">
                      Follow the guided workflow from your shortlist: outline,
                      then themes, then draft, then validate and export
                      Markdown.
                    </p>
                    <ReviewDraftPanel
                      sessionId={session.id}
                      shortlistedCount={shortlistedPapers.length}
                      shortlistedPapers={shortlistedPapers}
                      sessionQuery={session.raw_query}
                    />
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </AppShell>
  );
}
