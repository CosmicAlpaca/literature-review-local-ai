"use client";

import { Fragment, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDownIcon } from "lucide-react";
import toast from "react-hot-toast";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import GroundingCheckSummary from "@/components/sessions/GroundingCheckSummary";
import LoadingState from "@/components/ui/LoadingState";
import Notice from "@/components/ui/Notice";
import { apiFetch } from "@/lib/axios";
import { cn } from "@/lib/utils";
import {
  DEBATE_ROLE_KEYS,
  debateReviewQueryKey,
  type DebateReport,
  type DebateRoleKey,
  type DebateRoleOutput,
  type ReviewDebateResponse,
  QUALITY_SCORE_LABELS,
  finalRecommendationBadge,
  finalRecommendationLabel,
  formatScore01,
  mergeBadgeClass,
  riskLevelBadge,
  riskLevelLabel,
} from "@/lib/debate";
import { llmFallbackLabel } from "@/lib/llmPreview";
import { formatCreatedAt } from "@/lib/session";

const DEBATE_RUNNING_TITLE = "Debate agents are reviewing your draft";
const DEBATE_RUNNING_BODY =
  "The panel is critiquing your draft against shortlisted evidence. This may take a moment.";

const DEBATE_CRITIQUE_LISTS: { key: keyof DebateReport; title: string }[] = [
  { key: "strengths", title: "Strengths" },
  { key: "weaknesses", title: "Weaknesses" },
  { key: "missing_evidence", title: "Missing evidence" },
  { key: "unsupported_or_weak_claims", title: "Unsupported or weak claims" },
  { key: "source_reliability_concerns", title: "Source reliability concerns" },
  { key: "methodology_concerns", title: "Methodology concerns" },
  { key: "suggested_revisions", title: "Suggested revisions" },
];

/** Tailwind-styled elements for debate `markdown_report` (no typography plugin). */
const debateReportMarkdownComponents: Partial<Components> = {
  h1: ({ children }) => (
    <h1 className="mb-2 mt-4 first:mt-0 font-serif text-lg font-normal tracking-tight text-foreground">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-4 first:mt-0 font-serif text-base font-normal text-foreground border-b border-border pb-1">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-1.5 mt-3 first:mt-0 text-sm font-semibold text-foreground">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mb-1 mt-2 first:mt-0 text-sm font-medium text-foreground">
      {children}
    </h4>
  ),
  p: ({ children }) => (
    <p className="mb-2 last:mb-0 text-sm leading-relaxed text-foreground">
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="mb-2 list-disc space-y-1 pl-5 text-sm text-foreground last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-2 list-decimal space-y-1 pl-5 text-sm text-foreground last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }) => <em className="italic text-foreground">{children}</em>,
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-primary underline underline-offset-2 hover:text-primary/80 break-all"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-primary/30 pl-3 text-sm text-muted-foreground italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-4 border-border" />,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-md border border-border bg-background p-3 font-mono text-xs leading-relaxed text-foreground">
      {children}
    </pre>
  ),
  code: ({ className, children }) => {
    const fenced =
      typeof className === "string" && className.includes("language-");
    if (fenced) {
      return (
        <code
          className={cn(
            "block whitespace-pre font-mono text-xs text-foreground",
            className,
          )}
        >
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.8em] text-foreground">
        {children}
      </code>
    );
  },
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto rounded-md border border-border">
      <table className="w-full min-w-[12rem] border-collapse text-left text-xs">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="border-b border-border bg-muted/50">{children}</thead>
  ),
  tbody: ({ children }) => (
    <tbody className="divide-y divide-border">{children}</tbody>
  ),
  tr: ({ children }) => <tr>{children}</tr>,
  th: ({ children }) => (
    <th className="px-2 py-1.5 font-semibold text-foreground">{children}</th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1.5 text-muted-foreground align-top">{children}</td>
  ),
};

type Props = {
  sessionId: number;
  shortlistedCount: number;
};

function asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

/** Maps API / partial payloads to a safe 0..1 value (missing keys, null, non-numeric → 0). */
function toUnitInterval(raw: unknown): number {
  let n: number;
  if (typeof raw === "number") {
    n = raw;
  } else if (typeof raw === "string" && raw.trim() !== "") {
    n = Number(raw);
  } else {
    return 0;
  }
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

function normalizeRole(raw: unknown): DebateRoleOutput | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const role_name = typeof o.role_name === "string" ? o.role_name : "Role";
  const score = typeof o.score === "number" ? o.score : Number(o.score) || 0;
  return {
    ...o,
    role_name,
    score,
    findings: asStringArray(o.findings),
    concerns: asStringArray(o.concerns),
    suggestions: asStringArray(o.suggestions),
  };
}

function StringList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) {
    return (
      <div className="min-w-0">
        <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-1.5">
          {title}
        </p>
        <p className="text-xs text-muted-foreground italic">None noted.</p>
      </div>
    );
  }
  return (
    <div className="min-w-0">
      <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-1.5">
        {title}
      </p>
      <ul className="list-disc space-y-1 pl-4 text-sm text-foreground leading-snug">
        {items.map((item, i) => (
          <li key={`${title}-${i}`}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function DebateFallbackNotice({
  llmUsed,
  fallbackReason,
}: {
  llmUsed: boolean;
  fallbackReason?: string | null;
}) {
  const fallbackLabel = fallbackReason ? llmFallbackLabel(fallbackReason) : "";
  if (llmUsed && !fallbackLabel) return null;

  return (
    <Notice tone="info">
      {!llmUsed ? (
        <span>Generated without LLM or using fallback output.</span>
      ) : null}
      {fallbackLabel ? (
        <span className={!llmUsed ? "block mt-1" : undefined}>
          {fallbackLabel}
        </span>
      ) : null}
    </Notice>
  );
}

function DebateCritiqueSummary({ report }: { report: DebateReport }) {
  const overall = report.overall_assessment?.trim() ?? "";
  const hasListContent = DEBATE_CRITIQUE_LISTS.some(({ key }) => {
    const items = report[key];
    return Array.isArray(items) && items.length > 0;
  });

  return (
    <div>
      <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-3">
        Critique summary
      </p>
      {overall ? (
        <p className="text-sm text-foreground leading-relaxed mb-4">
          {overall}
        </p>
      ) : null}
      <div className="mb-4 max-w-xs">
        <ScoreBar label="Confidence" value={report.confidence_score} />
      </div>
      {hasListContent ? (
        <div className="grid gap-5 sm:grid-cols-2">
          {DEBATE_CRITIQUE_LISTS.map(({ key, title }) => {
            const items = asStringArray(report[key]);
            if (!items.length) return null;
            return <StringList key={key} title={title} items={items} />;
          })}
        </div>
      ) : null}
    </div>
  );
}

function ScoreBar({ label, value }: { label: string; value: unknown }) {
  const pct = Math.round(toUnitInterval(value) * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums text-foreground">{pct}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-[width]"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function DebateAgentPanel({
  sessionId,
  shortlistedCount,
}: Props) {
  const queryClient = useQueryClient();
  const [copying, setCopying] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [markdownDetailOpen, setMarkdownDetailOpen] = useState(false);

  const canRun = shortlistedCount > 0;
  const queryEnabled = Number.isFinite(sessionId) && sessionId > 0;

  const debateQuery = useQuery({
    queryKey: debateReviewQueryKey(sessionId),
    queryFn: () =>
      apiFetch<ReviewDebateResponse>(`/sessions/${sessionId}/review/debate`, {
        nullOn404: true,
      }),
    enabled: queryEnabled,
  });

  const debateMutation = useMutation({
    mutationFn: (force: boolean) =>
      apiFetch(
        `/sessions/${sessionId}/review/debate?${new URLSearchParams({ force: force ? "true" : "false" }).toString()}`,
        { method: "POST" },
      ) as Promise<ReviewDebateResponse>,
    onSuccess: (body, force) => {
      queryClient.setQueryData(debateReviewQueryKey(sessionId), body);
      if (force) {
        toast.success("Debate report regenerated.");
      } else {
        toast.success(
          body.cached ? "Showing saved debate report." : "Debate report ready.",
        );
      }
    },
    onError: (err: unknown) => {
      toast.error(
        err instanceof Error ? err.message : "Debate request failed.",
      );
    },
  });

  const debateData = debateQuery.data;
  const report = debateData?.debate_report;
  const hasReport = Boolean(report);
  const noSavedReport = debateQuery.isSuccess && debateData === null;
  const md = debateData?.markdown_report?.trim() ?? "";

  async function copyMarkdown() {
    if (!md) {
      toast.error("No markdown report to copy.");
      return;
    }
    setCopying(true);
    try {
      await navigator.clipboard.writeText(md);
      toast.success("Markdown copied to clipboard.");
    } catch {
      toast.error("Unable to copy to clipboard.");
    } finally {
      setCopying(false);
    }
  }

  function downloadMarkdown() {
    if (!md) {
      toast.error("No markdown report to download.");
      return;
    }
    setDownloading(true);
    try {
      const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `session-${sessionId}-debate-report.md`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      // toast.success("Debate report downloaded.");
    } catch {
      toast.error("Unable to download report.");
    } finally {
      setDownloading(false);
    }
  }

  const recBadge = report
    ? mergeBadgeClass(finalRecommendationBadge(report.final_recommendation))
    : null;
  const riskBadge = report
    ? mergeBadgeClass(riskLevelBadge(report.risk_level))
    : null;

  const actionBusy = debateMutation.isPending;
  const primaryRunning = noSavedReport ? "Reviewing draft…" : "Regenerating…";

  return (
    <Fragment>
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <Badge
                variant="outline"
                className="mb-2 text-xs rounded-full border-primary/15 text-primary bg-primary/8"
              >
                Debate agent
              </Badge>
              <CardTitle className="text-base">
                Evidence-grounded panel critique
              </CardTitle>
              <p className="mt-1 text-xs text-muted-foreground leading-relaxed max-w-prose">
                Runs a structured panel review over a generated draft using your
                shortlisted papers only.
                {!canRun
                  ? " Shortlist at least one paper to run or regenerate a review."
                  : null}
              </p>
              {hasReport && debateData && (
                <div className="mt-2 flex flex-wrap items-center gap-2 text-[0.7rem] text-muted-foreground">
                  {debateData.cached ? (
                    <Badge variant="outline" className="text-[0.65rem]">
                      Cached
                    </Badge>
                  ) : null}
                  {debateData.generated_at ? (
                    <span>
                      Generated {formatCreatedAt(debateData.generated_at)}
                    </span>
                  ) : null}
                  {debateData.updated_at &&
                  debateData.updated_at !== debateData.generated_at ? (
                    <span>
                      Updated {formatCreatedAt(debateData.updated_at)}
                    </span>
                  ) : null}
                </div>
              )}
            </div>
            <div className="flex shrink-0 flex-col items-stretch gap-2 sm:items-end">
              {debateQuery.isPending ? (
                <Button size="sm" disabled variant="outline">
                  Loading…
                </Button>
              ) : debateQuery.isError ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => debateQuery.refetch()}
                >
                  Retry load
                </Button>
              ) : hasReport ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    if (!canRun) return;
                    debateMutation.mutate(true);
                  }}
                  disabled={actionBusy || !canRun}
                  title={
                    !canRun ? "Shortlist at least one paper first." : undefined
                  }
                >
                  {actionBusy ? primaryRunning : "Regenerate debate review"}
                </Button>
              ) : noSavedReport && canRun ? (
                <Button
                  size="sm"
                  onClick={() => debateMutation.mutate(false)}
                  disabled={actionBusy}
                >
                  {actionBusy ? primaryRunning : "Run Debate Review"}
                </Button>
              ) : (
                <Button
                  size="sm"
                  disabled
                  title="Shortlist at least one paper first."
                >
                  Run Debate Review
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="grid gap-6">
          {debateQuery.isError && (
            <Notice tone="error">
              {debateQuery.error instanceof Error
                ? debateQuery.error.message
                : "Could not load the saved debate report."}
            </Notice>
          )}

          {debateQuery.isPending && (
            <LoadingState
              title="Loading debate report"
              body="Checking for a saved report for this session."
            />
          )}

          {!debateQuery.isPending &&
            !debateQuery.isError &&
            report &&
            debateData && (
              <>
                {/* Hero */}
                <div className="rounded-xl border border-primary/12 bg-primary/5 p-4 sm:p-5">
                  <div className="flex flex-wrap items-center gap-2 mb-3">
                    {recBadge ? (
                      <Badge
                        variant={recBadge.variant}
                        className={recBadge.className}
                      >
                        {finalRecommendationLabel(report.final_recommendation)}
                      </Badge>
                    ) : null}
                    {riskBadge ? (
                      <Badge
                        variant={riskBadge.variant}
                        className={riskBadge.className}
                      >
                        {riskLevelLabel(report.risk_level)}
                      </Badge>
                    ) : null}
                  </div>
                  <p className="text-sm sm:text-[0.95rem] text-foreground leading-relaxed font-serif">
                    {report.debate_summary}
                  </p>
                </div>

                {/* <DebateFallbackNotice
                  llmUsed={debateData.llm_used}
                  fallbackReason={debateData.fallback_reason}
                /> */}

                <Separator />

                <DebateCritiqueSummary report={report} />

                <Separator />

                {/* Quality scores */}
                <div>
                  <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-3">
                    Quality scores
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {QUALITY_SCORE_LABELS.map(({ key, label }) => (
                      <ScoreBar
                        key={key}
                        label={label}
                        value={report.quality_scores?.[key]}
                      />
                    ))}
                  </div>
                </div>

                <Separator />

                {debateData.grounding_check ? (
                  <div>
                    <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-3">
                      Draft grounding check
                    </p>
                    <GroundingCheckSummary check={debateData.grounding_check} />
                  </div>
                ) : null}

                <Separator />

                {/* Action items */}
                <div>
                  <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                    Action items
                  </p>
                  {report.action_items.length ? (
                    <ol className="list-decimal space-y-2 pl-5 text-sm text-foreground leading-snug">
                      {report.action_items.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ol>
                  ) : (
                    <p className="text-xs text-muted-foreground italic">
                      No action items listed.
                    </p>
                  )}
                </div>

                <Separator />

                {/* Role panels */}
                <div>
                  <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-2">
                    Panel roles
                  </p>
                  <div className="grid gap-2">
                    {DEBATE_ROLE_KEYS.map((key) => {
                      const raw =
                        debateData.agent_outputs?.[key as DebateRoleKey];
                      const role = normalizeRole(raw);
                      if (!role) return null;
                      const isArbiter = key === "final_arbiter";
                      const overall =
                        isArbiter && typeof role.overall_assessment === "string"
                          ? role.overall_assessment
                          : null;

                      return (
                        <Collapsible
                          key={key}
                          className="rounded-lg border border-border bg-background"
                        >
                          <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 px-3 py-2.5 text-left hover:bg-muted/50 transition-colors [&[data-state=open]>svg]:rotate-180">
                            <span className="flex flex-wrap items-center gap-2 min-w-0">
                              <span className="font-medium text-sm text-foreground truncate">
                                {role.role_name}
                              </span>
                              <Badge
                                variant="outline"
                                className="shrink-0 font-mono text-[0.7rem]"
                              >
                                {formatScore01(role.score)}
                              </Badge>
                            </span>
                            <ChevronDownIcon className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200" />
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <div className="border-t border-border px-3 py-3 space-y-4">
                              {overall ? (
                                <p className="text-sm text-foreground leading-relaxed">
                                  {overall}
                                </p>
                              ) : null}
                              <div className="grid gap-4 lg:grid-cols-3">
                                <StringList
                                  title="Findings"
                                  items={role.findings}
                                />
                                <StringList
                                  title="Concerns"
                                  items={role.concerns}
                                />
                                <StringList
                                  title="Suggestions"
                                  items={role.suggestions}
                                />
                              </div>
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      );
                    })}
                  </div>
                </div>

                <Separator />

                {/* Markdown */}
                <div>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-2">
                    <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground">
                      Markdown report
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setMarkdownDetailOpen(true)}
                        disabled={!md}
                      >
                        View detail
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={copyMarkdown}
                        disabled={!md || copying}
                      >
                        {copying ? "Copying…" : "Copy"}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={downloadMarkdown}
                        disabled={!md || downloading}
                      >
                        {downloading ? "Downloading…" : "Download .md"}
                      </Button>
                    </div>
                  </div>
                  {md ? (
                    <ScrollArea className="h-[220px] w-full rounded-lg border border-border bg-muted/30">
                      <div className="p-3 pr-4 [&>*:first-child]:mt-0">
                        <ReactMarkdown
                          components={debateReportMarkdownComponents}
                        >
                          {md}
                        </ReactMarkdown>
                      </div>
                    </ScrollArea>
                  ) : (
                    <p className="text-xs text-muted-foreground italic">
                      No markdown report in this response.
                    </p>
                  )}
                </div>
              </>
            )}

          {!debateQuery.isPending &&
            !debateQuery.isError &&
            noSavedReport &&
            debateMutation.isPending && (
              <LoadingState
                title={DEBATE_RUNNING_TITLE}
                body={DEBATE_RUNNING_BODY}
              />
            )}

          {!debateQuery.isPending &&
            !debateQuery.isError &&
            noSavedReport &&
            !debateMutation.isPending && (
              <p className="text-sm text-muted-foreground">
                {canRun
                  ? "No saved debate report yet. Use Run Debate Review above to generate one."
                  : "No saved debate report. Shortlist at least one paper, then run a debate review."}
              </p>
            )}
        </CardContent>
      </Card>

      <Dialog open={markdownDetailOpen} onOpenChange={setMarkdownDetailOpen}>
        <DialogContent className="flex max-h-[90vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-3xl">
          <DialogHeader className="gap-1.5 border-b border-border px-4 py-3 pr-14 text-left sm:text-left">
            <DialogTitle>Markdown report</DialogTitle>
            <DialogDescription>
              Full debate report rendered from markdown for this session run.
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="h-[min(65vh,560px)] px-4">
            <div className="pb-4 pr-3 pt-1 [&>*:first-child]:mt-0">
              {md ? (
                <ReactMarkdown components={debateReportMarkdownComponents}>
                  {md}
                </ReactMarkdown>
              ) : null}
            </div>
          </ScrollArea>
          <DialogFooter showCloseButton className="mt-0 rounded-b-xl" />
        </DialogContent>
      </Dialog>
    </Fragment>
  );
}
