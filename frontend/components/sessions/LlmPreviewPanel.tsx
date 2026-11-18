"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDownIcon, SparklesIcon } from "lucide-react";
import toast from "react-hot-toast";

import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";

import { apiFetch } from "@/lib/axios";
import { cn } from "@/lib/utils";
import type { CandidateItem } from "@/lib/session";
import {
  LLM_CANDIDATE_SCOPE_LABELS,
  LLM_QUERY_KEYS,
  RERANK_PREVIEW_MAX_CANDIDATES,
  type LlmCandidateScope,
  type LlmEstimatedUsage,
  type QueryPreviewRequest,
  type QueryPreviewResponse,
  type RerankPreviewRequest,
  type RerankPreviewResponse,
  type RerankPreviewResult,
  type SummaryPreviewRequest,
  type SummaryPreviewResponse,
  clampScore01,
  formatScore3dp,
  llmFallbackLabel,
  llmStatusLabel,
  scoreToneClass,
  selectLlmCandidates,
  sortRerankResults,
  toRerankPreviewCandidate,
  toSummaryPreviewPaper,
  useLlmStatus,
} from "@/lib/llmPreview";

type Props = {
  sessionTopic: string;
  sessionQuery: string;
  candidates: CandidateItem[];
  shortlistedCandidates: CandidateItem[];
};

type PreviewTab = "query" | "rerank" | "summary";

const TABS: { id: PreviewTab; label: string; description: string }[] = [
  {
    id: "query",
    label: "Query rewrite",
    description: "POST /llm/query-preview",
  },
  {
    id: "rerank",
    label: "Rerank",
    description: "POST /llm/rerank-preview",
  },
  {
    id: "summary",
    label: "Paper summary",
    description: "POST /llm/summary-preview",
  },
];

/* ── Small presentational helpers ────────────────────────────────────── */

function MetaPill({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "positive" | "muted" | "warning";
}) {
  const toneClass =
    tone === "positive"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-200"
      : tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100"
        : tone === "muted"
          ? "border-border bg-muted text-muted-foreground"
          : "border-border bg-background text-foreground";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.7rem]",
        toneClass,
      )}
    >
      <span className="font-semibold uppercase tracking-wide text-[0.6rem] opacity-70">
        {label}
      </span>
      <span className="font-mono tabular-nums">{value}</span>
    </div>
  );
}

function FallbackBanner({ reason }: { reason?: string | null }) {
  if (!reason) return null;
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100">
      <span className="font-semibold">Fallback active.</span>{" "}
      {llmFallbackLabel(reason)}
    </div>
  );
}

function CommonMetaRow({
  provider,
  model,
  promptVersion,
  usage,
}: {
  provider?: string | null;
  model?: string | null;
  promptVersion?: string | null;
  usage?: LlmEstimatedUsage | null;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {provider ? <MetaPill label="Provider" value={provider} /> : null}
      {model ? <MetaPill label="Model" value={model} /> : null}
      {promptVersion ? (
        <MetaPill label="Prompt" value={promptVersion} tone="muted" />
      ) : null}
      {usage ? (
        <>
          <MetaPill
            label="Tokens"
            value={`${usage.estimated_total_tokens}`}
            tone="muted"
          />
          <MetaPill
            label="Calls"
            value={`${usage.estimated_call_count}`}
            tone="muted"
          />
          {usage.input_truncated ? (
            <MetaPill label="Truncated" value="yes" tone="warning" />
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function ScoreBar({ value, className }: { value: number; className?: string }) {
  const pct = Math.round(clampScore01(value) * 100);
  return (
    <div
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-muted",
        className,
      )}
    >
      <div
        className="h-full rounded-full bg-primary transition-[width]"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function RawResponseCollapsible({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-fit [&[data-state=open]>svg]:rotate-180"
        >
          {open ? "Hide raw response" : "View raw response"}
          <ChevronDownIcon className="size-3.5 transition-transform duration-200" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ScrollArea className="mt-2 max-h-[260px] w-full rounded-lg border border-border bg-muted/40">
          <pre className="p-3 font-mono text-[0.72rem] leading-relaxed text-foreground whitespace-pre-wrap break-words">
            {JSON.stringify(value, null, 2)}
          </pre>
        </ScrollArea>
      </CollapsibleContent>
    </Collapsible>
  );
}

/* ── Main panel ─────────────────────────────────────────────────────── */

export default function LlmPreviewPanel({
  sessionTopic,
  sessionQuery,
  candidates,
  shortlistedCandidates,
}: Props) {
  const [tab, setTab] = useState<PreviewTab>("query");

  /* ── /llm/status (cached + deduped across the app) ─────────────────── */

  const statusQuery = useLlmStatus();
  const statusBadge = useMemo(
    () => llmStatusLabel(statusQuery.data ?? null),
    [statusQuery.data],
  );

  /* ── Query rewrite tab state ───────────────────────────────────────── */

  const [topic, setTopic] = useState(sessionTopic);
  const [rawQuery, setRawQuery] = useState(sessionQuery);
  const [submittedQueryPayload, setSubmittedQueryPayload] =
    useState<QueryPreviewRequest | null>(null);

  const queryPreviewQuery = useQuery<QueryPreviewResponse>({
    queryKey: LLM_QUERY_KEYS.queryPreview(submittedQueryPayload),
    queryFn: () =>
      apiFetch("/llm/query-preview", {
        method: "POST",
        body: JSON.stringify(submittedQueryPayload),
      }) as Promise<QueryPreviewResponse>,
    enabled: submittedQueryPayload !== null,
    staleTime: 5 * 60 * 1000,
    retry: 0,
    refetchOnWindowFocus: false,
  });

  const runQueryPreview = useCallback(() => {
    const trimmedQuery = rawQuery.trim();
    if (!trimmedQuery) {
      toast.error("Enter a raw query first.");
      return;
    }
    if (trimmedQuery.length > 1000) {
      toast.error("Raw query must be 1000 characters or fewer.");
      return;
    }
    setSubmittedQueryPayload({
      topic: topic.trim() || null,
      raw_query: trimmedQuery,
    });
  }, [topic, rawQuery]);

  /* ── Rerank tab state ──────────────────────────────────────────────── */

  const [rerankQuery, setRerankQuery] = useState(sessionQuery);
  const [rerankScope, setRerankScope] =
    useState<LlmCandidateScope>("top_by_score");
  const [submittedRerankPayload, setSubmittedRerankPayload] =
    useState<RerankPreviewRequest | null>(null);

  const rerankSelectedCandidates = useMemo(
    () =>
      selectLlmCandidates(
        rerankScope,
        candidates,
        shortlistedCandidates,
        RERANK_PREVIEW_MAX_CANDIDATES,
      ),
    [rerankScope, candidates, shortlistedCandidates],
  );

  const rerankPoolSize = useMemo(() => {
    if (rerankScope === "shortlisted") return shortlistedCandidates.length;
    if (rerankScope === "active") {
      return candidates.filter((c) => c.review_status !== "dismissed").length;
    }
    return candidates.length;
  }, [rerankScope, candidates, shortlistedCandidates]);

  const rerankPreviewQuery = useQuery<RerankPreviewResponse>({
    queryKey: LLM_QUERY_KEYS.rerankPreview(submittedRerankPayload),
    queryFn: () =>
      apiFetch("/llm/rerank-preview", {
        method: "POST",
        body: JSON.stringify(submittedRerankPayload),
      }) as Promise<RerankPreviewResponse>,
    enabled: submittedRerankPayload !== null,
    staleTime: 5 * 60 * 1000,
    retry: 0,
    refetchOnWindowFocus: false,
  });

  const runRerankPreview = useCallback(() => {
    const trimmed = rerankQuery.trim();
    if (!trimmed) {
      toast.error("Enter a query first.");
      return;
    }
    if (rerankSelectedCandidates.length === 0) {
      toast.error("No candidates available in the selected scope.");
      return;
    }
    setSubmittedRerankPayload({
      query: trimmed,
      candidates: rerankSelectedCandidates.map(toRerankPreviewCandidate),
    });
  }, [rerankQuery, rerankSelectedCandidates]);

  const rerankCandidateLookup = useMemo(() => {
    const m = new Map<string, CandidateItem>();
    for (const c of candidates) m.set(String(c.id), c);
    return m;
  }, [candidates]);

  const rankedRerankResults = useMemo<
    (RerankPreviewResult & { candidate: CandidateItem | null })[]
  >(() => {
    if (!rerankPreviewQuery.data) return [];
    return sortRerankResults(rerankPreviewQuery.data.results).map((item) => ({
      ...item,
      candidate:
        item.candidate_id == null
          ? null
          : (rerankCandidateLookup.get(String(item.candidate_id)) ?? null),
    }));
  }, [rerankPreviewQuery.data, rerankCandidateLookup]);

  /* ── Summary tab state ─────────────────────────────────────────────── */

  const [summaryQuery, setSummaryQuery] = useState(sessionQuery);
  const [summaryPaperId, setSummaryPaperId] = useState<string>("");
  const [submittedSummaryPayload, setSubmittedSummaryPayload] =
    useState<SummaryPreviewRequest | null>(null);

  // Keep the summary tab's paper selection in sync with the candidate pool.
  // - Preserve an active user choice when it is still valid in `candidates`.
  // - Fall back to the first candidate when no choice (or a stale one) exists.
  // - Reset to "" when the pool becomes empty.
  //
  // Uses the functional-updater form of `setState` to avoid a self-referential
  // dep on `summaryPaperId`. React bails out when the updater returns the same
  // value, so this does not cause an extra render.
  useEffect(() => {
    setSummaryPaperId((current) => {
      if (current && candidates.some((c) => String(c.id) === current)) {
        return current;
      }
      const first = candidates[0];
      return first ? String(first.id) : "";
    });
  }, [candidates]);

  const summarySelectedPaper = useMemo<CandidateItem | null>(() => {
    if (!summaryPaperId) return null;
    return candidates.find((c) => String(c.id) === summaryPaperId) ?? null;
  }, [candidates, summaryPaperId]);

  const summaryPreviewQuery = useQuery<SummaryPreviewResponse>({
    queryKey: LLM_QUERY_KEYS.summaryPreview(submittedSummaryPayload),
    queryFn: () =>
      apiFetch("/llm/summary-preview", {
        method: "POST",
        body: JSON.stringify(submittedSummaryPayload),
      }) as Promise<SummaryPreviewResponse>,
    enabled: submittedSummaryPayload !== null,
    staleTime: 5 * 60 * 1000,
    retry: 0,
    refetchOnWindowFocus: false,
  });

  const runSummaryPreview = useCallback(() => {
    if (!summarySelectedPaper) {
      toast.error("Select a paper first.");
      return;
    }
    setSubmittedSummaryPayload({
      query: summaryQuery.trim() || null,
      paper: toSummaryPreviewPaper(summarySelectedPaper),
    });
  }, [summarySelectedPaper, summaryQuery]);

  /* ── Error toasts (errorUpdatedAt only changes on new failures) ────── */

  useEffect(() => {
    if (queryPreviewQuery.isError) {
      toast.error(
        queryPreviewQuery.error instanceof Error
          ? queryPreviewQuery.error.message
          : "Unable to run query preview.",
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryPreviewQuery.errorUpdatedAt]);

  useEffect(() => {
    if (rerankPreviewQuery.isError) {
      toast.error(
        rerankPreviewQuery.error instanceof Error
          ? rerankPreviewQuery.error.message
          : "Unable to run rerank preview.",
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rerankPreviewQuery.errorUpdatedAt]);

  useEffect(() => {
    if (summaryPreviewQuery.isError) {
      toast.error(
        summaryPreviewQuery.error instanceof Error
          ? summaryPreviewQuery.error.message
          : "Unable to run summary preview.",
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summaryPreviewQuery.errorUpdatedAt]);

  /* ── Render ────────────────────────────────────────────────────────── */

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <Badge
              variant="outline"
              className="mb-2 text-xs rounded-full border-primary/15 text-primary bg-primary/8"
            >
              <SparklesIcon className="size-3" />
              LLM preview
            </Badge>
            <CardTitle className="text-base">AI preview console</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground leading-relaxed max-w-prose">
              Authenticated preview endpoints from{" "}
              <code className="font-mono">backend/app/api/llm.py</code>: query
              rewrite, rerank, and paper summary. Results are cached per input
              snapshot via TanStack Query.
            </p>
          </div>
          <div className="flex flex-col items-start gap-1 sm:items-end">
            <div
              className={cn(
                "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold",
                statusBadge.tone === "positive"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-200"
                  : statusBadge.tone === "warning"
                    ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100"
                    : "border-border bg-muted text-muted-foreground",
              )}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {statusQuery.isLoading
                ? "Checking AI status…"
                : statusBadge.label}
            </div>
            {statusQuery.data ? (
              <p className="text-[0.7rem] text-muted-foreground">
                <code className="font-mono">
                  {statusQuery.data.llm_provider}
                </code>
                {" · "}
                <code className="font-mono">{statusQuery.data.llm_model}</code>
              </p>
            ) : null}
          </div>
        </div>
      </CardHeader>

      <CardContent className="grid gap-5">
        {/* Tab switcher */}
        <div
          role="tablist"
          aria-label="LLM preview tabs"
          className="inline-flex w-fit items-center gap-1 rounded-full border border-border bg-background p-1"
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "rounded-full px-3 py-1.5 text-xs font-semibold transition-colors",
                tab === t.id
                  ? "bg-primary/10 border border-primary/15 text-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "query" && (
          <QueryRewriteTab
            topic={topic}
            rawQuery={rawQuery}
            sessionTopic={sessionTopic}
            sessionQuery={sessionQuery}
            onTopicChange={setTopic}
            onRawQueryChange={setRawQuery}
            onRun={runQueryPreview}
            queryResult={queryPreviewQuery.data}
            isFetching={queryPreviewQuery.isFetching}
            isFromCache={
              queryPreviewQuery.isSuccess && !queryPreviewQuery.isFetching
            }
          />
        )}

        {tab === "rerank" && (
          <RerankTab
            query={rerankQuery}
            sessionQuery={sessionQuery}
            scope={rerankScope}
            selectedCount={rerankSelectedCandidates.length}
            poolCount={rerankPoolSize}
            shortlistedCount={shortlistedCandidates.length}
            onQueryChange={setRerankQuery}
            onScopeChange={setRerankScope}
            onRun={runRerankPreview}
            response={rerankPreviewQuery.data}
            rankedResults={rankedRerankResults}
            isFetching={rerankPreviewQuery.isFetching}
            isFromCache={
              rerankPreviewQuery.isSuccess && !rerankPreviewQuery.isFetching
            }
          />
        )}

        {tab === "summary" && (
          <SummaryTab
            query={summaryQuery}
            paperId={summaryPaperId}
            candidates={candidates}
            shortlistedCandidates={shortlistedCandidates}
            onQueryChange={setSummaryQuery}
            onPaperChange={setSummaryPaperId}
            onRun={runSummaryPreview}
            selectedPaper={summarySelectedPaper}
            result={summaryPreviewQuery.data}
            isFetching={summaryPreviewQuery.isFetching}
            isFromCache={
              summaryPreviewQuery.isSuccess && !summaryPreviewQuery.isFetching
            }
          />
        )}
      </CardContent>
    </Card>
  );
}

/* ── Tab: Query rewrite ─────────────────────────────────────────────── */

function QueryRewriteTab({
  topic,
  rawQuery,
  sessionTopic,
  sessionQuery,
  onTopicChange,
  onRawQueryChange,
  onRun,
  queryResult,
  isFetching,
  isFromCache,
}: {
  topic: string;
  rawQuery: string;
  sessionTopic: string;
  sessionQuery: string;
  onTopicChange: (value: string) => void;
  onRawQueryChange: (value: string) => void;
  onRun: () => void;
  queryResult?: QueryPreviewResponse;
  isFetching: boolean;
  isFromCache: boolean;
}) {
  const isEdited = topic !== sessionTopic || rawQuery !== sessionQuery;

  return (
    <div className="grid gap-5">
      {/* Inputs */}
      <div className="grid gap-3 rounded-xl border border-border bg-muted/30 p-3 sm:p-4">
        <div className="flex items-center justify-between">
          <p className="text-[0.7rem] uppercase tracking-wide text-muted-foreground font-semibold">
            POST /llm/query-preview
          </p>
          {isEdited && (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              disabled={isFetching}
              onClick={() => {
                onTopicChange(sessionTopic);
                onRawQueryChange(sessionQuery);
              }}
            >
              Reset to session
            </Button>
          )}
        </div>

        <div className="grid gap-1.5">
          <Label htmlFor="llm-topic" className="text-xs">
            Topic (optional)
          </Label>
          <Input
            id="llm-topic"
            value={topic}
            onChange={(e) => onTopicChange(e.target.value)}
            placeholder="e.g. early cancer detection"
            disabled={isFetching}
            maxLength={500}
            className="bg-background"
          />
        </div>

        <div className="grid gap-1.5">
          <Label htmlFor="llm-raw-query" className="text-xs">
            Raw query
          </Label>
          <Textarea
            id="llm-raw-query"
            value={rawQuery}
            onChange={(e) => onRawQueryChange(e.target.value)}
            placeholder="Enter the user's research question…"
            disabled={isFetching}
            maxLength={1000}
            rows={2}
            className="min-h-[60px] bg-background text-sm"
          />
          <p className="text-[0.7rem] text-muted-foreground">
            Max 1000 chars. {rawQuery.length}/1000
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            onClick={onRun}
            disabled={isFetching || !rawQuery.trim()}
          >
            {isFetching ? (
              <>
                <Spinner className="size-3.5" />
                Rewriting…
              </>
            ) : queryResult ? (
              "Re-run rewrite"
            ) : (
              "Run rewrite"
            )}
          </Button>
          {isFromCache && queryResult ? (
            <Badge variant="outline" className="text-[0.65rem]">
              cached
            </Badge>
          ) : null}
        </div>
      </div>

      {/* Output */}
      {queryResult && (
        <div className="grid gap-4">
          <CommonMetaRow
            provider={queryResult.provider}
            model={queryResult.model}
            promptVersion={queryResult.prompt_version}
            usage={queryResult.estimated_usage}
          />

          <FallbackBanner reason={queryResult.fallback_reason} />

          <div className="grid gap-3 sm:grid-cols-2">
            {[
              { label: "Original query", value: queryResult.original_query },
              { label: "Optimized query", value: queryResult.optimized_query },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-lg border border-primary/12 bg-primary/5 p-3"
              >
                <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-1">
                  {item.label}
                </p>
                <p className="text-sm text-foreground leading-snug break-words">
                  {item.value || "—"}
                </p>
              </div>
            ))}
          </div>

          {queryResult.key_terms.length > 0 && (
            <div>
              <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                Key terms
              </p>
              <div className="flex flex-wrap gap-1.5">
                {queryResult.key_terms.map((term) => (
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
          )}

          {queryResult.expanded_queries.length > 0 && (
            <div>
              <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                Expanded queries
              </p>
              <div className="flex flex-wrap gap-1.5">
                {queryResult.expanded_queries.map((q) => (
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
          )}

          {queryResult.notes && (
            <div>
              <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-1">
                Notes
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {queryResult.notes}
              </p>
            </div>
          )}

          <Separator />
          <RawResponseCollapsible value={queryResult} />
        </div>
      )}
    </div>
  );
}

/* ── Tab: Rerank ────────────────────────────────────────────────────── */

function RerankTab({
  query,
  sessionQuery,
  scope,
  selectedCount,
  poolCount,
  shortlistedCount,
  onQueryChange,
  onScopeChange,
  onRun,
  response,
  rankedResults,
  isFetching,
  isFromCache,
}: {
  query: string;
  sessionQuery: string;
  scope: LlmCandidateScope;
  selectedCount: number;
  poolCount: number;
  shortlistedCount: number;
  onQueryChange: (value: string) => void;
  onScopeChange: (value: LlmCandidateScope) => void;
  onRun: () => void;
  response?: RerankPreviewResponse;
  rankedResults: (RerankPreviewResult & { candidate: CandidateItem | null })[];
  isFetching: boolean;
  isFromCache: boolean;
}) {
  const isEdited = query !== sessionQuery;

  return (
    <div className="grid gap-5">
      <div className="grid gap-3 rounded-xl border border-border bg-muted/30 p-3 sm:p-4">
        <div className="flex items-center justify-between">
          <p className="text-[0.7rem] uppercase tracking-wide text-muted-foreground font-semibold">
            POST /llm/rerank-preview
          </p>
          {isEdited && (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              disabled={isFetching}
              onClick={() => onQueryChange(sessionQuery)}
            >
              Reset to session query
            </Button>
          )}
        </div>

        <div className="grid gap-1.5">
          <Label htmlFor="llm-rerank-query" className="text-xs">
            Query
          </Label>
          <Textarea
            id="llm-rerank-query"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Enter the query to rerank against…"
            disabled={isFetching}
            maxLength={1000}
            rows={2}
            className="min-h-[60px] bg-background text-sm"
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-2 sm:items-end">
          <div className="grid gap-1.5">
            <Label htmlFor="llm-rerank-scope" className="text-xs">
              Candidate scope
            </Label>
            <Select
              value={scope}
              onValueChange={(value) =>
                onScopeChange(value as LlmCandidateScope)
              }
              disabled={isFetching}
            >
              <SelectTrigger
                id="llm-rerank-scope"
                className="w-full bg-background"
              >
                <SelectValue placeholder="Pick a scope" />
              </SelectTrigger>
              <SelectContent position="popper">
                <SelectItem value="top_by_score">
                  {LLM_CANDIDATE_SCOPE_LABELS.top_by_score}
                </SelectItem>
                <SelectItem
                  value="shortlisted"
                  disabled={shortlistedCount === 0}
                >
                  {LLM_CANDIDATE_SCOPE_LABELS.shortlisted}
                </SelectItem>
                <SelectItem value="active">
                  {LLM_CANDIDATE_SCOPE_LABELS.active}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline" className="font-mono tabular-nums">
              {selectedCount} / {poolCount}
            </Badge>
            <span>capped at {RERANK_PREVIEW_MAX_CANDIDATES} candidates.</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            onClick={onRun}
            disabled={isFetching || !query.trim() || selectedCount === 0}
          >
            {isFetching ? (
              <>
                <Spinner className="size-3.5" />
                Reranking…
              </>
            ) : response ? (
              "Re-run rerank"
            ) : (
              "Run rerank"
            )}
          </Button>
          {isFromCache && response ? (
            <Badge variant="outline" className="text-[0.65rem]">
              cached
            </Badge>
          ) : null}
        </div>
      </div>

      {response && (
        <div className="grid gap-4">
          <CommonMetaRow
            provider={response.provider}
            model={response.model}
            promptVersion={response.prompt_version}
            usage={response.estimated_usage}
          />
          <FallbackBanner reason={response.fallback_reason} />

          <div>
            <div className="flex items-center justify-between gap-2 mb-2">
              <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground">
                Ranked results
              </p>
              <span className="text-[0.7rem] text-muted-foreground">
                Sorted by <code className="font-mono">relevance_score</code>
              </span>
            </div>

            {rankedResults.length === 0 ? (
              <p className="text-xs italic text-muted-foreground">
                No results returned.
              </p>
            ) : (
              <ScrollArea className="max-h-[420px] w-full rounded-lg border border-border bg-background">
                <ol className="divide-y divide-border">
                  {rankedResults.map((result, index) => {
                    const score = clampScore01(result.relevance_score);
                    const title =
                      result.candidate?.title ??
                      (result.candidate_id != null
                        ? `Candidate ${result.candidate_id}`
                        : `Item #${result.candidate_index + 1}`);

                    return (
                      <li
                        key={`${String(result.candidate_id ?? result.candidate_index)}-${index}`}
                        className="px-3 py-3"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge
                                variant="outline"
                                className="font-mono text-[0.7rem]"
                              >
                                #{index + 1}
                              </Badge>
                              <p className="min-w-0 truncate text-sm font-medium text-foreground">
                                {title}
                              </p>
                            </div>
                            {result.candidate?.source_title ||
                            result.candidate?.year ? (
                              <p className="mt-1 text-xs text-muted-foreground truncate">
                                {[
                                  result.candidate?.source_title,
                                  result.candidate?.year
                                    ? String(result.candidate.year)
                                    : null,
                                ]
                                  .filter(Boolean)
                                  .join(" · ")}
                              </p>
                            ) : null}
                          </div>
                          <Badge
                            variant="outline"
                            className={cn(
                              "font-mono text-[0.75rem] tabular-nums shrink-0",
                              scoreToneClass(score),
                            )}
                          >
                            {formatScore3dp(score)}
                          </Badge>
                        </div>

                        <ScoreBar value={score} className="mt-2" />

                        {result.reason ? (
                          <p className="mt-2 text-xs leading-snug text-muted-foreground">
                            {result.reason}
                          </p>
                        ) : null}
                      </li>
                    );
                  })}
                </ol>
              </ScrollArea>
            )}
          </div>

          <Separator />
          <RawResponseCollapsible value={response} />
        </div>
      )}
    </div>
  );
}

/* ── Tab: Summary ───────────────────────────────────────────────────── */

function SummaryTab({
  query,
  paperId,
  candidates,
  shortlistedCandidates,
  onQueryChange,
  onPaperChange,
  onRun,
  selectedPaper,
  result,
  isFetching,
  isFromCache,
}: {
  query: string;
  paperId: string;
  candidates: CandidateItem[];
  shortlistedCandidates: CandidateItem[];
  onQueryChange: (value: string) => void;
  onPaperChange: (value: string) => void;
  onRun: () => void;
  selectedPaper: CandidateItem | null;
  result?: SummaryPreviewResponse;
  isFetching: boolean;
  isFromCache: boolean;
}) {
  const orderedCandidates = useMemo(() => {
    const seen = new Set<number>();
    const ordered: { item: CandidateItem; group: "shortlisted" | "other" }[] =
      [];
    for (const c of shortlistedCandidates) {
      if (!seen.has(c.id)) {
        seen.add(c.id);
        ordered.push({ item: c, group: "shortlisted" });
      }
    }
    const remaining = [...candidates]
      .filter((c) => !seen.has(c.id))
      .sort((a, b) => {
        const aScore = a.final_score ?? Number.NEGATIVE_INFINITY;
        const bScore = b.final_score ?? Number.NEGATIVE_INFINITY;
        if (bScore !== aScore) return bScore - aScore;
        return a.id - b.id;
      });
    for (const c of remaining) {
      ordered.push({ item: c, group: "other" });
    }
    return ordered;
  }, [candidates, shortlistedCandidates]);

  const SUMMARY_FIELDS: { key: keyof SummaryPreviewResponse; label: string }[] =
    [
      { key: "one_sentence_summary", label: "One-sentence summary" },
      { key: "contribution", label: "Contribution" },
      { key: "method_or_data", label: "Method / data" },
      { key: "relevance_to_query", label: "Relevance to query" },
      { key: "limitations_or_caution", label: "Limitations / caution" },
    ];

  return (
    <div className="grid gap-5">
      <div className="grid gap-3 rounded-xl border border-border bg-muted/30 p-3 sm:p-4">
        <p className="text-[0.7rem] uppercase tracking-wide text-muted-foreground font-semibold">
          POST /llm/summary-preview
        </p>

        <div className="grid gap-1.5">
          <Label htmlFor="llm-summary-paper" className="text-xs">
            Paper
          </Label>
          <Select
            value={paperId}
            onValueChange={onPaperChange}
            disabled={isFetching || orderedCandidates.length === 0}
          >
            <SelectTrigger
              id="llm-summary-paper"
              className="w-full bg-background"
            >
              <SelectValue placeholder="Select a candidate paper" />
            </SelectTrigger>
            <SelectContent
              position="popper"
              className="max-w-[min(36rem,calc(100vw-2rem))]"
            >
              {orderedCandidates.length === 0 ? (
                <div className="px-3 py-2 text-xs text-muted-foreground">
                  No candidates available.
                </div>
              ) : (
                orderedCandidates.map(({ item, group }) => (
                  <SelectItem key={item.id} value={String(item.id)}>
                    <span className="truncate">
                      {group === "shortlisted" ? "★ " : ""}
                      {item.title || `Candidate ${item.id}`}
                      {item.year ? ` (${item.year})` : ""}
                    </span>
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          {selectedPaper && !selectedPaper.abstract ? (
            <p className="text-[0.7rem] text-amber-700 dark:text-amber-300">
              This paper has no abstract; the backend may return a fallback
              summary.
            </p>
          ) : null}
        </div>

        <div className="grid gap-1.5">
          <Label htmlFor="llm-summary-query" className="text-xs">
            Query (optional, frames the summary)
          </Label>
          <Textarea
            id="llm-summary-query"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Optional: how to frame the summary…"
            disabled={isFetching}
            maxLength={1000}
            rows={2}
            className="min-h-[60px] bg-background text-sm"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            onClick={onRun}
            disabled={isFetching || !selectedPaper}
          >
            {isFetching ? (
              <>
                <Spinner className="size-3.5" />
                Summarizing…
              </>
            ) : result ? (
              "Re-run summary"
            ) : (
              "Run summary"
            )}
          </Button>
          {isFromCache && result ? (
            <Badge variant="outline" className="text-[0.65rem]">
              cached
            </Badge>
          ) : null}
        </div>
      </div>

      {result && (
        <div className="grid gap-4">
          <CommonMetaRow
            provider={result.provider}
            model={result.model}
            promptVersion={result.prompt_version}
            usage={result.estimated_usage}
          />
          <FallbackBanner reason={result.fallback_reason} />

          <div className="grid gap-3">
            {SUMMARY_FIELDS.map(({ key, label }) => {
              const value = result[key];
              if (typeof value !== "string" || !value.trim()) return null;
              return (
                <div
                  key={key}
                  className="rounded-lg border border-primary/12 bg-primary/5 p-3"
                >
                  <p className="text-[0.65rem] font-bold uppercase tracking-wide text-muted-foreground mb-1">
                    {label}
                  </p>
                  <p className="text-sm text-foreground leading-relaxed break-words">
                    {value}
                  </p>
                </div>
              );
            })}
          </div>

          <Separator />
          <RawResponseCollapsible value={result} />
        </div>
      )}
    </div>
  );
}
