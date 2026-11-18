"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FlaskConicalIcon } from "lucide-react";

import AppShell from "@/components/app/AppShell";
import LlmPreviewPanel from "@/components/sessions/LlmPreviewPanel";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";

import { apiFetch } from "@/lib/axios";
import { useLlmStatus } from "@/lib/llmPreview";
import {
  type CandidateItem,
  getSessionTopicDisplay,
  useSessionList,
} from "@/lib/session";

const NO_SESSION = "__none__";

/**
 * Standalone preview console for the Phase 2 authenticated LLM endpoints:
 *   POST /llm/query-preview
 *   POST /llm/rerank-preview
 *   POST /llm/summary-preview
 *
 * Reuses `LlmPreviewPanel` (the same component embedded in the session
 * detail page) but lets the user optionally pick *any* session as the
 * candidate source for the Rerank / Summary tabs. The Query rewrite tab
 * works without a session.
 *
 * The panel reads its initial topic / raw-query state from props on mount,
 * so we use `key={selectedSessionId}` to force a remount whenever the user
 * changes the picker — that guarantees the form re-seeds from the newly
 * loaded session.
 */
export default function SandboxPage() {
  const sessionsQuery = useSessionList();
  const llmStatusQuery = useLlmStatus();
  const llmStatus = llmStatusQuery.data ?? null;
  const sessions = useMemo(
    () => sessionsQuery.data ?? [],
    [sessionsQuery.data],
  );

  const [selectedSessionId, setSelectedSessionId] =
    useState<string>(NO_SESSION);
  const sessionId =
    selectedSessionId === NO_SESSION ? null : Number(selectedSessionId);
  const session = useMemo(
    () =>
      sessionId !== null
        ? (sessions.find((s) => s.id === sessionId) ?? null)
        : null,
    [sessions, sessionId],
  );

  // Load saved candidates + shortlist for the picked session. Cached per
  // session id so switching back and forth is instant. We deliberately use
  // a separate query key (`["sandbox", ...]`) instead of piggy-backing on
  // the session detail page's local-state fetches — those aren't TanStack
  // queries, and we don't want to disturb their lifecycle.
  const candidatesQuery = useQuery<CandidateItem[]>({
    queryKey: ["sandbox", "candidates", sessionId],
    queryFn: () =>
      apiFetch(`/search/${sessionId}/candidates`) as Promise<CandidateItem[]>,
    enabled: sessionId !== null,
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
    retry: 0,
  });

  const shortlistQuery = useQuery<CandidateItem[]>({
    queryKey: ["sandbox", "shortlist", sessionId],
    queryFn: () =>
      apiFetch(`/search/${sessionId}/shortlist`) as Promise<CandidateItem[]>,
    enabled: sessionId !== null,
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
    retry: 0,
  });

  const candidates = candidatesQuery.data ?? [];
  const shortlistedCandidates = shortlistQuery.data ?? [];
  const sessionLoading =
    sessionId !== null &&
    (candidatesQuery.isLoading || shortlistQuery.isLoading);

  return (
    <AppShell>
      <div className="px-6 py-6 max-w-6xl mx-auto grid gap-5">
        <Card>
          <CardHeader>
            <Badge
              variant="outline"
              className="mb-2 w-fit text-xs rounded-full border-primary/15 text-primary bg-primary/8"
            >
              <FlaskConicalIcon className="size-3" />
              AI sandbox
            </Badge>
            <CardTitle className="text-xl font-serif font-normal tracking-tight">
              AI preview console
            </CardTitle>
            <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed max-w-prose">
              Try the LLM query rewrite, rerank, and paper summary endpoints
              with your own input. Pick a session below to load its candidates
              into the rerank and summary tabs, or leave it as None for a blank
              slate.
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 max-w-md">
              <Label htmlFor="sandbox-session">Session context</Label>
              <Select
                value={selectedSessionId}
                onValueChange={setSelectedSessionId}
              >
                <SelectTrigger id="sandbox-session">
                  <SelectValue placeholder="None — query rewrite only" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_SESSION}>
                    None — start with a blank form
                  </SelectItem>
                  {sessions.map((s) => {
                    const topicDisplay = getSessionTopicDisplay(s.topic);
                    return (
                      <SelectItem key={s.id} value={String(s.id)}>
                        #{s.id} · {topicDisplay.label}
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>

              {sessionsQuery.isLoading && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Spinner className="size-3" />
                  Loading sessions…
                </div>
              )}

              {sessionId !== null && sessionLoading && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Spinner className="size-3" />
                  Loading session candidates…
                </div>
              )}

              {sessionId !== null && !sessionLoading && session && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Loaded{" "}
                  <span className="font-semibold text-foreground">
                    {candidates.length}
                  </span>{" "}
                  candidate{candidates.length === 1 ? "" : "s"} ·{" "}
                  <span className="font-semibold text-foreground">
                    {shortlistedCandidates.length}
                  </span>{" "}
                  shortlisted from{" "}
                  <span
                    className={
                      getSessionTopicDisplay(session.topic).isPlaceholder
                        ? "italic text-muted-foreground/60"
                        : "italic text-foreground"
                    }
                  >
                    {getSessionTopicDisplay(session.topic).label}
                  </span>
                  .
                </p>
              )}

              {sessionId === null && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  The rerank and summary tabs need a session&apos;s candidates
                  to score against.
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Badge
              variant="outline"
              className="mb-2 w-fit text-xs rounded-full border-primary/15 text-primary bg-primary/8"
            >
              LLM config
            </Badge>
            <CardTitle className="text-base">Deployment AI status</CardTitle>
          </CardHeader>
          <CardContent>
            {llmStatusQuery.isLoading && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Spinner className="size-3" />
                Checking /llm/status...
              </div>
            )}
            {!llmStatusQuery.isLoading && !llmStatus && (
              <p className="text-xs text-muted-foreground">
                Unable to read `/llm/status`.
              </p>
            )}
            {llmStatus && (
              <div className="grid gap-2 sm:grid-cols-2">
                <p className="text-xs text-muted-foreground">
                  Provider:{" "}
                  <span className="text-foreground">{llmStatus.llm_provider}</span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Model:{" "}
                  <span className="text-foreground">{llmStatus.llm_model}</span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Available:{" "}
                  <span className="text-foreground">
                    {llmStatus.available ? "yes" : "no"}
                  </span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Ready for AI search:{" "}
                  <span className="text-foreground">
                    {llmStatus.ready_for_ai_search ? "yes" : "no"}
                  </span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Query rewrite enabled:{" "}
                  <span className="text-foreground">
                    {llmStatus.query_rewrite_enabled ? "yes" : "no"}
                  </span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Rerank enabled:{" "}
                  <span className="text-foreground">
                    {llmStatus.rerank_enabled ? "yes" : "no"}
                  </span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Summary enabled:{" "}
                  <span className="text-foreground">
                    {llmStatus.summary_enabled ? "yes" : "no"}
                  </span>
                </p>
                <p className="text-xs text-muted-foreground">
                  API key configured:{" "}
                  <span className="text-foreground">
                    {llmStatus.api_key_configured ? "yes" : "no"}
                  </span>
                </p>
                {llmStatus.disabled_reason && (
                  <p className="text-xs text-muted-foreground sm:col-span-2">
                    Disabled reason:{" "}
                    <span className="text-foreground">
                      {llmStatus.disabled_reason}
                    </span>
                  </p>
                )}
                {llmStatus.enabled_steps?.length ? (
                  <p className="text-xs text-muted-foreground sm:col-span-2">
                    Enabled steps:{" "}
                    <span className="text-foreground">
                      {llmStatus.enabled_steps.join(", ")}
                    </span>
                  </p>
                ) : null}
              </div>
            )}
          </CardContent>
        </Card>

        <LlmPreviewPanel
          key={selectedSessionId}
          sessionTopic={session?.topic ?? ""}
          sessionQuery={session?.raw_query ?? ""}
          candidates={candidates}
          shortlistedCandidates={shortlistedCandidates}
        />
      </div>
    </AppShell>
  );
}
