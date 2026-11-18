"use client";

import { useState } from "react";
import { ChevronRightIcon, ExternalLinkIcon } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import MetadataPill from "@/components/sessions/MetadataPill";
import TrustBadge from "@/components/sessions/TrustBadge";
import {
  formatProviderLabel,
  getPaperQualitySignal,
  type CandidateItem,
} from "@/lib/session";
import { cn } from "@/lib/utils";

type Props = {
  paper: CandidateItem;
  variant?: "selected" | "candidate";
  rank?: number;
  actionBusy?: boolean;
  onShortlist?: (paper: CandidateItem) => void;
  onDismiss?: (paper: CandidateItem) => void;
  onRestore?: (paper: CandidateItem) => void;
  onGenerateSummary?: (paper: CandidateItem, force: boolean) => void;
  summaryBusy?: boolean;
};

function formatScore(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(3);
}

function formatRelevanceScore(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}%`;
}

export default function PaperResultCard({
  paper,
  variant = "candidate",
  rank,
  actionBusy = false,
  onShortlist,
  onDismiss,
  onRestore,
  onGenerateSummary,
  summaryBusy = false,
}: Props) {
  const [open, setOpen] = useState(false);
  const [summaryOpen, setSummaryOpen] = useState(true);
  const [abstractExpanded, setAbstractExpanded] = useState(false);

  const quality = getPaperQualitySignal(paper);
  const llmRelevanceScore = formatRelevanceScore(paper.llm_relevance_score);
  const summary = paper.llm_summary;
  const hasSummary = Boolean(
    summary?.one_sentence_summary ||
    summary?.contribution ||
    summary?.method_or_data ||
    summary?.relevance_to_query ||
    summary?.limitations_or_caution,
  );

  const isDismissed = paper.review_status === "dismissed";
  const hasActionRow =
    Boolean(onShortlist || onDismiss || onRestore || onGenerateSummary) ||
    Boolean(paper.doi || paper.url);

  const abstractText = paper.abstract?.trim() ?? "";
  const abstractLong = abstractText.length > 360;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <article
        className={cn(
          "rounded-xl border bg-card text-card-foreground transition-opacity",
          isDismissed && "opacity-60",
          variant === "selected" &&
            !isDismissed &&
            "border-primary/20 bg-primary/3",
        )}
      >
        {/* Header: expand on title/meta; shortlist/dismiss always visible */}
        <div className="flex w-full items-start gap-3 px-4 py-3.5">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="flex min-w-0 flex-1 items-start justify-between gap-4 text-left cursor-pointer rounded-md hover:bg-muted/30 transition-colors -mx-1 px-1"
            >
              <div className="min-w-0 flex-1">
              {/* Signal badges */}
              <div className="flex flex-wrap gap-1.5 mb-2">
                {typeof rank === "number" && (
                  <MetadataPill tone="subtle">Top {rank}</MetadataPill>
                )}
                <TrustBadge level={quality.level} label={quality.label} />
                {paper.reranker_status === "applied" && (
                  <MetadataPill tone="subtle">Locally reranked</MetadataPill>
                )}
                {paper.reranker_status === "unavailable" && (
                  <MetadataPill tone="warning">Baseline ranking</MetadataPill>
                )}
                {paper.review_status === "shortlisted" && (
                  <MetadataPill tone="success">Shortlisted</MetadataPill>
                )}
                {isDismissed && (
                  <MetadataPill tone="warning">Dismissed</MetadataPill>
                )}
                <MetadataPill
                  tone={paper.scopus_matched ? "success" : "warning"}
                >
                  {quality.sourceCue}
                </MetadataPill>
              </div>

              {/* Title as Link */}
              <div className="text-sm font-semibold leading-snug text-foreground line-clamp-2">
                {paper.url ? (
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline hover:text-primary transition-colors"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {paper.title}
                  </a>
                ) : (
                  paper.title
                )}
              </div>

              {abstractText ? (
                <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground line-clamp-3">
                  {abstractText}
                </p>
              ) : null}

              {/* Meta */}
              <p className="mt-1 text-xs text-muted-foreground">
                {[
                  paper.source_title || "Unknown source",
                  paper.year || "N/A",
                  formatProviderLabel(paper.provider),
                  `${paper.citation_count ?? 0} citations`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-3">
              {/* Score badge */}
              <div className="text-right">
                <p className="text-[0.65rem] uppercase tracking-wide text-muted-foreground">
                  Base score
                </p>
                <Badge
                  variant="secondary"
                  className="text-xs font-bold tabular-nums"
                >
                  {formatScore(paper.final_score)}
                </Badge>
              </div>

              <ChevronRightIcon
                className={cn(
                  "h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-150",
                  open && "rotate-90",
                )}
              />
            </div>
            </button>
          </CollapsibleTrigger>

          {(onShortlist || onDismiss || onRestore) && (
            <div
              className="flex shrink-0 flex-col gap-1.5 sm:flex-row sm:items-center"
              onClick={(e) => e.stopPropagation()}
            >
              {paper.review_status !== "shortlisted" && onShortlist && (
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-8 border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                  onClick={() => onShortlist(paper)}
                  disabled={actionBusy}
                >
                  Shortlist
                </Button>
              )}
              {paper.review_status !== "dismissed" && onDismiss && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8"
                  onClick={() => onDismiss(paper)}
                  disabled={actionBusy}
                >
                  Dismiss
                </Button>
              )}
              {paper.review_status && onRestore && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8"
                  onClick={() => onRestore(paper)}
                  disabled={actionBusy}
                >
                  Restore
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Expandable content */}
        <CollapsibleContent>
          <Separator />
          <div className="px-4 pt-4 pb-2 grid gap-4">
            {/* Metadata pills */}
            <div className="flex flex-wrap gap-1.5">
              <MetadataPill
                tone={quality.metadataComplete ? "success" : "warning"}
              >
                {quality.metadataComplete
                  ? "Metadata complete"
                  : "Metadata partial"}
              </MetadataPill>
              <MetadataPill tone="neutral">{quality.yearCue}</MetadataPill>
              <MetadataPill tone="neutral">{quality.citationCue}</MetadataPill>
              <MetadataPill tone="subtle">{quality.providerLabel}</MetadataPill>
              {llmRelevanceScore && (
                <MetadataPill tone="neutral">
                  AI relevance {llmRelevanceScore}
                </MetadataPill>
              )}
              {paper.doi && (
                <MetadataPill tone="subtle">DOI: {paper.doi}</MetadataPill>
              )}
            </div>

            {/* Summary line */}
            <p className="text-xs text-muted-foreground leading-relaxed">
              {quality.summary}
            </p>

            {/* AI relevance note */}
            {paper.llm_relevance_reason && (
              <div
                className={cn(
                  "rounded-lg border px-3 py-2.5",
                  paper.llm_relevance_reason.includes("HALLUCINATION DETECTED")
                    ? "border-red-200 bg-red-50"
                    : "border-primary/12 bg-primary/5",
                )}
              >
                <p
                  className={cn(
                    "text-[0.68rem] font-bold uppercase tracking-wide mb-1",
                    paper.llm_relevance_reason.includes(
                      "HALLUCINATION DETECTED",
                    )
                      ? "text-red-600"
                      : "text-primary",
                  )}
                >
                  {paper.llm_relevance_reason.includes("HALLUCINATION DETECTED")
                    ? "⚠️ AI Hallucination Warning"
                    : "AI relevance note"}
                </p>
                <p
                  className={cn(
                    "text-xs leading-relaxed",
                    paper.llm_relevance_reason.includes(
                      "HALLUCINATION DETECTED",
                    )
                      ? "text-red-700"
                      : "text-muted-foreground",
                  )}
                >
                  {paper.llm_relevance_reason}
                </p>
              </div>
            )}

            {/* AI summary collapsible */}
            {hasSummary && (
              <Collapsible open={summaryOpen} onOpenChange={setSummaryOpen}>
                <CollapsibleTrigger asChild>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:text-primary/80 transition-colors"
                  >
                    <ChevronRightIcon
                      className={cn(
                        "h-3.5 w-3.5 transition-transform duration-150",
                        summaryOpen && "rotate-90",
                      )}
                    />
                    {summaryOpen ? "Hide AI summary" : "Show AI summary"}
                  </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="mt-2 rounded-lg border border-primary/12 bg-primary/5 p-3 grid gap-2.5">
                    <p className="text-[0.68rem] font-bold uppercase tracking-wide text-primary">
                      AI-assisted summary
                    </p>
                    {summary?.one_sentence_summary && (
                      <p className="text-sm font-medium leading-snug text-foreground">
                        {summary.one_sentence_summary}
                      </p>
                    )}
                    <div className="grid gap-2">
                      {summary?.contribution && (
                        <div>
                          <p className="text-[0.7rem] font-semibold text-foreground mb-0.5">
                            Contribution
                          </p>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {summary.contribution}
                          </p>
                        </div>
                      )}
                      {summary?.method_or_data && (
                        <div>
                          <p className="text-[0.7rem] font-semibold text-foreground mb-0.5">
                            Method / data
                          </p>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {summary.method_or_data}
                          </p>
                        </div>
                      )}
                      {summary?.relevance_to_query && (
                        <div>
                          <p className="text-[0.7rem] font-semibold text-foreground mb-0.5">
                            Why it may be relevant
                          </p>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {summary.relevance_to_query}
                          </p>
                        </div>
                      )}
                      {summary?.limitations_or_caution && (
                        <div>
                          <p className="text-[0.7rem] font-semibold text-foreground mb-0.5">
                            Limitations / caution
                          </p>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {summary.limitations_or_caution}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}

            {summaryBusy && (
              <div className="space-y-2 rounded-md border border-primary/15 bg-muted/30 p-3">
                <div className="h-2.5 w-full animate-pulse rounded bg-muted" />
                <div className="h-2.5 w-full max-w-lg animate-pulse rounded bg-muted" />
                <div className="h-2.5 w-4/5 max-w-sm animate-pulse rounded bg-muted" />
                <p className="text-[0.65rem] text-muted-foreground">
                  Generating summary…
                </p>
              </div>
            )}

            {/* Abstract */}
            {abstractText && (
              <div>
                <p className="text-[0.68rem] font-bold uppercase tracking-wide text-muted-foreground mb-1.5">
                  Abstract
                </p>
                <p
                  className={cn(
                    "text-xs leading-relaxed text-muted-foreground",
                    !abstractExpanded && abstractLong && "line-clamp-3",
                  )}
                >
                  {abstractText}
                </p>
                {abstractLong && (
                  <button
                    type="button"
                    onClick={() => setAbstractExpanded((v) => !v)}
                    className="mt-1.5 text-xs font-semibold text-primary hover:text-primary/80"
                  >
                    {abstractExpanded ? "Show less" : "Read more"}
                  </button>
                )}
              </div>
            )}
          </div>

          {hasActionRow && (
            <>
              <Separator />
              <div className="flex flex-wrap items-center gap-2 border-t border-border/60 bg-muted/15 px-4 py-3">
                {paper.doi && (
                  <Button size="sm" variant="ghost" asChild>
                    <a
                      href={`https://doi.org/${paper.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1"
                    >
                      <ExternalLinkIcon className="h-3 w-3" />
                      View source
                    </a>
                  </Button>
                )}
                {!paper.doi && paper.url && (
                  <Button size="sm" variant="ghost" asChild>
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1"
                    >
                      <ExternalLinkIcon className="h-3 w-3" />
                      View source
                    </a>
                  </Button>
                )}
                {onGenerateSummary && (
                  <>
                    {!hasSummary && (
                      <Button
                        size="sm"
                        variant="secondary"
                        className="ml-auto sm:ml-0"
                        onClick={() => onGenerateSummary(paper, false)}
                        disabled={summaryBusy || actionBusy}
                      >
                        {summaryBusy
                          ? "Analyzing..."
                          : "Analyze abstract"}
                      </Button>
                    )}
                    {hasSummary && (
                      <Button
                        size="sm"
                        variant="outline"
                        className="ml-auto sm:ml-0"
                        onClick={() => onGenerateSummary(paper, true)}
                        disabled={summaryBusy || actionBusy}
                      >
                        {summaryBusy ? "Regenerating..." : "Regenerate summary"}
                      </Button>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </CollapsibleContent>
      </article>
    </Collapsible>
  );
}
