"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useMutation } from "@tanstack/react-query";
import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import {
  Check,
  ChevronDown,
  Download,
  FileCheck2,
  Loader2,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import toast from "react-hot-toast";

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
import Notice from "@/components/ui/Notice";
import { apiFetch } from "@/lib/axios";
import type { CandidateItem } from "@/lib/session";
import {
  type DraftGroundingCheck,
  type ReviewDraftResponse,
  type ReviewDraftSection,
  type ReviewMarkdownExportResponse,
  type ReviewOutlineResponse,
  type ReviewThemesResponse,
} from "@/lib/review-authoring";
import { llmFallbackLabel } from "@/lib/llmPreview";
import { cn } from "@/lib/utils";

type WizardStep = "outline" | "themes" | "draft" | "export";

const WIZARD_STEPS: { id: WizardStep; label: string; hint: string }[] = [
  { id: "outline", label: "Outline", hint: "Structure" },
  { id: "themes", label: "Themes", hint: "Synthesis" },
  { id: "draft", label: "Draft", hint: "Write-up" },
  { id: "export", label: "Export", hint: "Validate & download" },
];

const previewMarkdownComponents: Partial<Components> = {
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
};

function downloadMarkdownFile(filename: string, markdown: string) {
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.endsWith(".md") ? filename : `${filename}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function shortlistFingerprint(papers: CandidateItem[]): string {
  return [...papers.map((p) => p.id)].sort((a, b) => a - b).join(",");
}

function resolveActiveStep(
  outline: ReviewOutlineResponse | null,
  themes: ReviewThemesResponse | null,
  draft: ReviewDraftResponse | null,
): WizardStep {
  if (!outline) return "outline";
  if (!themes) return "themes";
  if (!draft) return "draft";
  return "export";
}

function stepIndex(step: WizardStep): number {
  return WIZARD_STEPS.findIndex((s) => s.id === step);
}

function GroundingSummary({ check }: { check: DraftGroundingCheck }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          Overall
        </span>
        <Badge
          variant={check.valid ? "default" : "destructive"}
          className="rounded-full"
        >
          {check.valid ? "Passed" : "Issues found"}
        </Badge>
      </div>
      <div className="text-xs text-muted-foreground sm:text-right">
        Supported sections {check.total_supported_sections}/
        {check.total_sections}
      </div>
      <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-xs sm:col-span-2">
        <p className="font-medium text-foreground">Citation and ID checks</p>
        <ul className="mt-1.5 list-inside list-disc space-y-0.5 text-muted-foreground">
          <li>Unknown citations: {check.unknown_citations}</li>
          <li>
            Unknown supporting candidate IDs:{" "}
            {check.unknown_supporting_candidate_ids}
          </li>
          <li>
            Sections without mapped evidence: {check.sections_without_evidence}
          </li>
        </ul>
      </div>
    </div>
  );
}

function AuthoringResultCollapsible({
  title,
  subtitle,
  meta,
  defaultOpen = true,
  children,
}: {
  title: string;
  subtitle?: string;
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="group flex w-full flex-col gap-1 rounded-t-xl px-4 py-0 text-left transition-colors hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <div className="flex w-full items-start justify-between gap-3">
              <div className="min-w-0 flex-1 space-y-1">
                <CardTitle className="text-base">{title}</CardTitle>
                {subtitle ? (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {subtitle}
                  </p>
                ) : null}
                {meta ? <div className="pt-0.5">{meta}</div> : null}
              </div>
              <ChevronDown
                className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180"
                aria-hidden
              />
            </div>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="pt-0">{children}</CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

function LlmModeBadge({
  llmUsed,
  fallbackReason,
}: {
  llmUsed: boolean;
  fallbackReason?: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {llmUsed ? (
        <Badge variant="secondary" className="text-[0.65rem]">
          LLM
        </Badge>
      ) : (
        <Badge variant="outline" className="text-[0.65rem]">
          Deterministic
        </Badge>
      )}
      {fallbackReason ? (
        <span className="text-[0.65rem] leading-relaxed text-amber-800 dark:text-amber-300">
          {llmFallbackLabel(fallbackReason)}
        </span>
      ) : null}
    </div>
  );
}

function SupportingPapersList({
  candidateIds,
  papersById,
}: {
  candidateIds: number[];
  papersById: Map<number, CandidateItem>;
}) {
  if (candidateIds.length === 0) return null;

  return (
    <ul className="mt-2 space-y-1.5">
      {candidateIds.map((id) => {
        const paper = papersById.get(id);
        const label = `[C${id}]`;
        return (
          <li
            key={id}
            className={cn(
              "rounded-md border px-2.5 py-1.5 text-xs",
              paper
                ? "border-border bg-muted/20"
                : "border-amber-300/60 bg-amber-50/50 dark:bg-amber-950/20",
            )}
          >
            <span className="font-mono text-[0.65rem] text-muted-foreground">
              {label}
            </span>
            {paper ? (
              <p className="mt-0.5 font-medium leading-snug text-foreground">
                {paper.title}
              </p>
            ) : (
              <p className="mt-0.5 text-amber-800 dark:text-amber-300">
                Paper not in current shortlist — re-run this step after updating
                evidence.
              </p>
            )}
            {paper ? (
              <p className="mt-0.5 text-[0.65rem] text-muted-foreground">
                {[
                  paper.year,
                  paper.source_title,
                  paper.doi ? `DOI ${paper.doi}` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

function DraftSectionEvidence({
  section,
  papersById,
}: {
  section: ReviewDraftSection;
  papersById: Map<number, CandidateItem>;
}) {
  const hasMeta =
    section.key_claims.length > 0 ||
    section.limitations_or_cautions.length > 0 ||
    section.citation_placeholders.length > 0 ||
    section.supporting_candidate_ids.length > 0;

  if (!hasMeta) return null;

  return (
    <div className="mt-3 rounded-lg border border-dashed border-border bg-muted/15 p-3 text-xs">
      {section.citation_placeholders.length > 0 ? (
        <p className="text-muted-foreground">
          <span className="font-medium text-foreground">Citations: </span>
          {section.citation_placeholders.join(", ")}
        </p>
      ) : null}
      {section.key_claims.length > 0 ? (
        <div
          className={section.citation_placeholders.length ? "mt-2" : undefined}
        >
          <span className="font-medium text-foreground">Key claims</span>
          <ul className="mt-1 list-inside list-disc space-y-0.5 text-foreground">
            {section.key_claims.map((c, i) => (
              <li key={`${section.section_id}-kc-${i}`}>{c}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {section.limitations_or_cautions.length > 0 ? (
        <div className="mt-2">
          <span className="font-medium text-foreground">
            Limitations & cautions
          </span>
          <ul className="mt-1 list-inside list-disc space-y-0.5 text-muted-foreground">
            {section.limitations_or_cautions.map((c, i) => (
              <li key={`${section.section_id}-lim-${i}`}>{c}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {section.supporting_candidate_ids.length > 0 ? (
        <div className="mt-2">
          <span className="font-medium text-foreground">
            Supporting evidence
          </span>
          <SupportingPapersList
            candidateIds={section.supporting_candidate_ids}
            papersById={papersById}
          />
        </div>
      ) : null}
    </div>
  );
}

function WizardStepper({
  activeStep,
  outline,
  themes,
  draft,
  exportReady,
}: {
  activeStep: WizardStep;
  outline: ReviewOutlineResponse | null;
  themes: ReviewThemesResponse | null;
  draft: ReviewDraftResponse | null;
  exportReady: boolean;
}) {
  const activeIdx = stepIndex(activeStep);

  function statusFor(step: WizardStep): "complete" | "current" | "upcoming" {
    const idx = stepIndex(step);
    if (idx < activeIdx) return "complete";
    if (idx === activeIdx) return "current";
    return "upcoming";
  }

  function isComplete(step: WizardStep): boolean {
    switch (step) {
      case "outline":
        return !!outline;
      case "themes":
        return !!themes;
      case "draft":
        return !!draft;
      case "export":
        return exportReady;
      default:
        return false;
    }
  }

  const progressPct = ((activeIdx + 1) / WIZARD_STEPS.length) * 100;

  return (
    <div className="grid gap-2" aria-label="Review authoring workflow">
      <div
        className="h-1 overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-valuenow={activeIdx + 1}
        aria-valuemin={1}
        aria-valuemax={WIZARD_STEPS.length}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-300 ease-out"
          style={{ width: `${progressPct}%` }}
        />
      </div>
      <ol className="grid gap-2 sm:grid-cols-4">
        {WIZARD_STEPS.map((step) => {
          const status = statusFor(step.id);
          const done = isComplete(step.id);
          return (
            <li
              key={step.id}
              className={cn(
                "flex flex-col rounded-lg border px-3 py-2.5 text-left transition-colors",
                status === "current" &&
                  "border-primary bg-primary/5 ring-1 ring-primary/20",
                status === "complete" && "border-border bg-muted/30",
                status === "upcoming" &&
                  "border-dashed border-border/80 bg-muted/10 opacity-70",
              )}
            >
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "flex size-6 shrink-0 items-center justify-center rounded-full text-[0.65rem] font-semibold",
                    done && "bg-primary text-primary-foreground",
                    status === "current" &&
                      !done &&
                      "bg-primary/15 text-primary",
                    status === "upcoming" && "bg-muted text-muted-foreground",
                  )}
                >
                  {done ? (
                    <Check className="size-3.5" aria-hidden />
                  ) : (
                    stepIndex(step.id) + 1
                  )}
                </span>
                <span className="text-sm font-medium text-foreground">
                  {step.label}
                </span>
              </div>
              <p className="mt-1 pl-8 text-[0.65rem] text-muted-foreground">
                {step.hint}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

type Props = {
  sessionId: number;
  shortlistedCount: number;
  shortlistedPapers: CandidateItem[];
  /** Session query for context before the first API response. */
  sessionQuery?: string;
};

export default function ReviewDraftPanel({
  sessionId,
  shortlistedCount,
  shortlistedPapers,
  sessionQuery,
}: Props) {
  const [outline, setOutline] = useState<ReviewOutlineResponse | null>(null);
  const [themes, setThemes] = useState<ReviewThemesResponse | null>(null);
  const [draft, setDraft] = useState<ReviewDraftResponse | null>(null);
  const [exportCache, setExportCache] =
    useState<ReviewMarkdownExportResponse | null>(null);
  const [exportStale, setExportStale] = useState(true);
  const [exportAction, setExportAction] = useState<
    "validate" | "download" | null
  >(null);

  const papersById = useMemo(
    () => new Map(shortlistedPapers.map((p) => [p.id, p])),
    [shortlistedPapers],
  );

  const fingerprint = useMemo(
    () => shortlistFingerprint(shortlistedPapers),
    [shortlistedPapers],
  );
  const prevFingerprint = useRef<string | null>(null);

  const clearExport = useCallback(() => {
    setExportCache(null);
    setExportStale(true);
    setExportAction(null);
  }, []);

  const resetWorkflow = useCallback(() => {
    setOutline(null);
    setThemes(null);
    setDraft(null);
    clearExport();
  }, [clearExport]);

  const clearDownstreamFrom = useCallback(
    (step: WizardStep) => {
      if (step === "outline") {
        setThemes(null);
        setDraft(null);
        clearExport();
      } else if (step === "themes") {
        setDraft(null);
        clearExport();
      } else if (step === "draft") {
        clearExport();
      }
    },
    [clearExport],
  );

  useEffect(() => {
    if (prevFingerprint.current === null) {
      prevFingerprint.current = fingerprint;
      return;
    }
    if (prevFingerprint.current !== fingerprint) {
      prevFingerprint.current = fingerprint;
      resetWorkflow();
      toast("Shortlist changed — review workflow reset.", { icon: "⚠️" });
    }
  }, [fingerprint, resetWorkflow]);

  const activeStep = resolveActiveStep(outline, themes, draft);
  const exportReady = !!exportCache && !exportStale;
  const canAuthor = shortlistedCount > 0;

  const researchQuery =
    draft?.research_query ??
    themes?.research_query ??
    outline?.research_query ??
    sessionQuery?.trim() ??
    "";

  const outlineMutation = useMutation({
    mutationFn: () =>
      apiFetch<ReviewOutlineResponse>(`/sessions/${sessionId}/review/outline`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (data) => {
      clearDownstreamFrom("outline");
      setOutline(data);
      toast.success("Outline ready — continue to themes.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const themesMutation = useMutation({
    mutationFn: () =>
      apiFetch<ReviewThemesResponse>(`/sessions/${sessionId}/review/themes`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (data) => {
      clearDownstreamFrom("themes");
      setThemes(data);
      toast.success("Themes ready — continue to draft.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const draftMutation = useMutation({
    mutationFn: () =>
      apiFetch<ReviewDraftResponse>(`/sessions/${sessionId}/review/draft`, {
        method: "POST",
        body: JSON.stringify({}),
      }),
    onSuccess: (data) => {
      clearDownstreamFrom("draft");
      setDraft(data);
      toast.success("Draft ready — validate and export.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const exportMutation = useMutation({
    mutationFn: () =>
      apiFetch<ReviewMarkdownExportResponse>(
        `/sessions/${sessionId}/review/export/markdown`,
        {
          method: "POST",
          body: JSON.stringify({}),
        },
      ),
    onSuccess: (data) => {
      setExportCache(data);
      setExportStale(false);
    },
    onError: (err: Error) => toast.error(err.message),
    onSettled: () => setExportAction(null),
  });

  const busyAuthoring =
    outlineMutation.isPending ||
    themesMutation.isPending ||
    draftMutation.isPending;
  const exportBusy = exportMutation.isPending;
  const validatePending = exportBusy && exportAction === "validate";
  const downloadPending = exportBusy && exportAction === "download";

  const handleValidateExport = () => {
    if (activeStep !== "export" || !draft || exportBusy) return;
    setExportAction("validate");
    exportMutation.mutate(undefined, {
      onSuccess: () => {
        toast.success("Citation validation complete.");
      },
    });
  };

  const handleExportMarkdown = () => {
    if (activeStep !== "export" || !draft) return;

    const runDownload = (data: ReviewMarkdownExportResponse) => {
      downloadMarkdownFile(data.filename, data.markdown);
      toast.success("Literature review draft downloaded.");
    };

    if (!exportStale && exportCache) {
      runDownload(exportCache);
      return;
    }

    if (exportBusy) return;
    setExportAction("download");
    exportMutation.mutate(undefined, {
      onSuccess: (data) => {
        runDownload(data);
      },
    });
  };

  const paperCount =
    outline?.paper_count ??
    themes?.paper_count ??
    draft?.paper_count ??
    shortlistedCount;

  return (
    <div className="grid gap-4">
      {!canAuthor ? (
        <Notice tone="info">
          <span className="font-semibold">Shortlist required. </span>
          Add at least one paper to the shortlist, then work through the guided
          steps: outline → themes → draft → export.
        </Notice>
      ) : null}

      {canAuthor ? (
        <>
          {researchQuery ? (
            <div className="rounded-lg border border-border bg-muted/20 px-4 py-3">
              <p className="text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground">
                Review topic
              </p>
              <p className="mt-1 text-sm leading-relaxed text-foreground">
                {researchQuery}
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                {paperCount} shortlisted {paperCount === 1 ? "paper" : "papers"}{" "}
                in evidence pack
              </p>
            </div>
          ) : null}

          <WizardStepper
            activeStep={activeStep}
            outline={outline}
            themes={themes}
            draft={draft}
            exportReady={exportReady}
          />

          {(outline || themes || draft) && (
            <div className="flex justify-end">
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="gap-1.5 text-muted-foreground"
                disabled={busyAuthoring || exportBusy}
                onClick={() => {
                  resetWorkflow();
                  toast("Workflow reset — start from outline.");
                }}
              >
                <RotateCcw className="size-3.5" aria-hidden />
                Reset workflow
              </Button>
            </div>
          )}

          <Card className="border-primary/20 bg-primary/[0.02]">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                Step {stepIndex(activeStep) + 1}:{" "}
                {WIZARD_STEPS[stepIndex(activeStep)].label}
              </CardTitle>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {activeStep === "outline" &&
                  "Generate a section outline grounded in your shortlist. Themes and draft unlock after this step."}
                {activeStep === "themes" &&
                  "Group shortlisted papers into synthesis themes. Draft unlocks after themes are ready."}
                {activeStep === "draft" &&
                  "Produce the literature-review draft with citations tied to shortlisted papers."}
                {activeStep === "export" &&
                  "Run citation validation, then download sanitized Markdown for your write-up."}
              </p>
            </CardHeader>
            <CardContent>
              {activeStep === "outline" && (
                <Button
                  size="sm"
                  className="gap-1.5"
                  disabled={!canAuthor || busyAuthoring}
                  onClick={() => outlineMutation.mutate()}
                >
                  {outlineMutation.isPending ? (
                    <Loader2 className="size-4 animate-spin" aria-hidden />
                  ) : (
                    <Sparkles className="size-4" aria-hidden />
                  )}
                  {outline ? "Regenerate outline" : "Generate outline"}
                </Button>
              )}

              {activeStep === "themes" && (
                <Button
                  size="sm"
                  className="gap-1.5"
                  disabled={!canAuthor || busyAuthoring || !outline}
                  onClick={() => themesMutation.mutate()}
                >
                  {themesMutation.isPending ? (
                    <Loader2 className="size-4 animate-spin" aria-hidden />
                  ) : (
                    <Sparkles className="size-4" aria-hidden />
                  )}
                  {themes ? "Regenerate themes" : "Generate themes"}
                </Button>
              )}

              {activeStep === "draft" && (
                <Button
                  size="sm"
                  className="gap-1.5"
                  disabled={!canAuthor || busyAuthoring || !themes}
                  onClick={() => draftMutation.mutate()}
                >
                  {draftMutation.isPending ? (
                    <Loader2 className="size-4 animate-spin" aria-hidden />
                  ) : (
                    <Sparkles className="size-4" aria-hidden />
                  )}
                  {draft ? "Regenerate draft" : "Generate draft"}
                </Button>
              )}

              {activeStep === "export" && (
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="gap-1.5"
                    disabled={!draft || exportBusy || busyAuthoring}
                    onClick={handleValidateExport}
                  >
                    {validatePending ? (
                      <Loader2 className="size-4 animate-spin" aria-hidden />
                    ) : (
                      <FileCheck2 className="size-4" aria-hidden />
                    )}
                    Validate citations
                  </Button>
                  <Button
                    size="sm"
                    className="gap-2 shadow-sm"
                    disabled={!draft || exportBusy || busyAuthoring}
                    onClick={handleExportMarkdown}
                  >
                    {downloadPending ? (
                      <Loader2 className="size-4 animate-spin" aria-hidden />
                    ) : (
                      <Download className="size-4" aria-hidden />
                    )}
                    Export Markdown
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      ) : null}

      {outline ? (
        <AuthoringResultCollapsible
          title="Outline"
          subtitle={outline.outline.title}
          meta={
            <LlmModeBadge
              llmUsed={outline.llm_used}
              fallbackReason={outline.fallback_reason}
            />
          }
          defaultOpen={activeStep === "outline" || activeStep === "themes"}
        >
          <div className="grid gap-3">
            {outline.outline.sections.map((sec) => (
              <div
                key={sec.section_id}
                className="rounded-lg border border-border bg-card/50 p-3 text-sm"
              >
                <p className="font-semibold text-foreground">{sec.title}</p>
                <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                  {sec.purpose}
                </p>
                {sec.key_points.length > 0 ? (
                  <ul className="mt-2 list-inside list-disc space-y-0.5 text-xs text-foreground">
                    {sec.key_points.map((pt, i) => (
                      <li key={`${sec.section_id}-kp-${i}`}>{pt}</li>
                    ))}
                  </ul>
                ) : null}
                {sec.supporting_candidate_ids.length > 0 ? (
                  <div className="mt-2">
                    <p className="text-[0.65rem] font-medium text-muted-foreground">
                      Supporting evidence
                    </p>
                    <SupportingPapersList
                      candidateIds={sec.supporting_candidate_ids}
                      papersById={papersById}
                    />
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </AuthoringResultCollapsible>
      ) : null}

      {themes ? (
        <AuthoringResultCollapsible
          title="Themes"
          meta={
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="text-[0.65rem]">
                {themes.themes.length} groups
              </Badge>
              <LlmModeBadge
                llmUsed={themes.llm_used}
                fallbackReason={themes.fallback_reason}
              />
            </div>
          }
          defaultOpen={activeStep === "themes" || activeStep === "draft"}
        >
          <div className="grid gap-3 sm:grid-cols-2">
            {themes.themes.map((t) => (
              <div
                key={t.theme_id}
                className="rounded-lg border border-border bg-muted/20 p-3"
              >
                <p className="text-sm font-semibold text-foreground">
                  {t.name}
                </p>
                <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                  {t.rationale}
                </p>
                {t.keywords.length ? (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {t.keywords.map((kw, i) => (
                      <Badge
                        key={`${t.theme_id}-kw-${i}`}
                        variant="outline"
                        className="rounded-full text-[0.65rem]"
                      >
                        {kw}
                      </Badge>
                    ))}
                  </div>
                ) : null}
                {t.supporting_candidate_ids.length ? (
                  <div className="mt-2">
                    <p className="text-[0.65rem] font-medium text-muted-foreground">
                      Supporting evidence
                    </p>
                    <SupportingPapersList
                      candidateIds={t.supporting_candidate_ids}
                      papersById={papersById}
                    />
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </AuthoringResultCollapsible>
      ) : null}

      {draft ? (
        <AuthoringResultCollapsible
          title="Draft"
          subtitle={draft.draft.title}
          meta={
            <LlmModeBadge
              llmUsed={draft.llm_used}
              fallbackReason={draft.fallback_reason}
            />
          }
          defaultOpen={activeStep === "draft" || activeStep === "export"}
        >
          <div className="space-y-4">
            {draft.markdown_preview ? (
              <ScrollArea className="h-[min(24rem,50vh)] rounded-md border border-border bg-background p-4">
                <ReactMarkdown components={previewMarkdownComponents}>
                  {draft.markdown_preview}
                </ReactMarkdown>
              </ScrollArea>
            ) : null}

            {draft.draft.sections.length > 0 ? (
              <Collapsible defaultOpen={!draft.markdown_preview}>
                <CollapsibleTrigger className="group flex w-full items-center justify-between rounded-md border border-border bg-muted/30 px-3 py-2 text-left text-sm font-medium text-foreground hover:bg-muted/50">
                  <span>
                    {draft.markdown_preview
                      ? "Section evidence & claims"
                      : "Draft sections"}
                  </span>
                  <ChevronDown
                    className="size-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180"
                    aria-hidden
                  />
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-3 space-y-4">
                  {draft.draft.sections.map((sec) => (
                    <div
                      key={sec.section_id}
                      className="rounded-lg border border-border bg-card/40 p-3"
                    >
                      <h4 className="text-sm font-semibold text-foreground">
                        {sec.title}
                      </h4>
                      {!draft.markdown_preview ? (
                        <div className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                          {sec.content}
                        </div>
                      ) : null}
                      <DraftSectionEvidence
                        section={sec}
                        papersById={papersById}
                      />
                    </div>
                  ))}
                </CollapsibleContent>
              </Collapsible>
            ) : null}
          </div>
        </AuthoringResultCollapsible>
      ) : null}

      {activeStep === "export" || exportCache ? (
        exportCache ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Citation validation</CardTitle>
              <p className="text-xs text-muted-foreground">
                From the last export/validation pass
                {exportStale
                  ? " (stale — re-validate after regenerating draft)"
                  : ""}
                .
              </p>
            </CardHeader>
            <CardContent>
              <GroundingSummary check={exportCache.grounding_check} />
              {exportCache.fallback_reason ? (
                <p className="mt-3 text-xs leading-relaxed text-amber-800 dark:text-amber-300">
                  {llmFallbackLabel(exportCache.fallback_reason)}
                </p>
              ) : null}
            </CardContent>
          </Card>
        ) : (
          <Card className="border-dashed">
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              Validate citations or export Markdown to complete this step.
            </CardContent>
          </Card>
        )
      ) : null}

      <Separator />

      <p className="text-[0.65rem] text-muted-foreground">
        Debate Agent critique lives in the Shortlisted tab once you have
        evidence in place.
      </p>
    </div>
  );
}
