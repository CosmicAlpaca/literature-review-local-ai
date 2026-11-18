"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowRightIcon } from "lucide-react";
import AppShell from "@/components/app/AppShell";
import EmptyState from "@/components/sessions/EmptyState";
import { Button } from "@/components/ui/Button";
import LoadingState from "@/components/ui/LoadingState";
import Notice from "@/components/ui/Notice";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import CandidatePreview from "@/components/sessions/CandidatePreview";
import StatusBadge from "@/components/sessions/StatusBadge";
import { apiFetch } from "@/lib/axios";
import {
  type CandidateItem,
  formatCreatedAt,
  getSessionTopicDisplay,
  normalizeRawQuery,
  patchSessionInCache,
  type SessionItem,
  useDeleteSession,
  useSessionList,
} from "@/lib/session";
import { cn } from "@/lib/utils";

export default function SessionsPage() {
  const queryClient = useQueryClient();

  // Shared, deduped session-list cache — same `queryKey` as the sidebar
  // and the detail page, so opening this view does not re-fetch when the
  // sidebar already loaded the list.
  const sessionsQuery = useSessionList();
  const sessions = useMemo(() => sessionsQuery.data ?? [], [sessionsQuery.data]);
  const loadingSessions = sessionsQuery.isPending;
  const loadError = sessionsQuery.isError
    ? sessionsQuery.error instanceof Error
      ? sessionsQuery.error.message
      : "Unable to load sessions."
    : "";

  const deleteMutation = useDeleteSession();

  const [resultsBySession, setResultsBySession] = useState<Record<number, CandidateItem[]>>({});
  const [loadingSessionId, setLoadingSessionId] = useState<number | null>(null);
  const [loadingCandidatesId, setLoadingCandidatesId] = useState<number | null>(null);
  const [actionErrors, setActionErrors] = useState<Record<number, string>>({});

  async function runSearch(session: SessionItem) {
    setLoadingSessionId(session.id);
    setActionErrors((cur) => ({ ...cur, [session.id]: "" }));

    try {
      const data = await apiFetch(
        `/search/${session.id}?query=${encodeURIComponent(session.raw_query)}`,
        { method: "POST" },
      );
      const response = data as { candidates: CandidateItem[] };
      setResultsBySession((cur) => ({ ...cur, [session.id]: response.candidates }));
      patchSessionInCache(queryClient, session.id, { status: "searched" });
    } catch (err) {
      setActionErrors((cur) => ({
        ...cur,
        [session.id]: err instanceof Error ? err.message : "Unable to run search for this session.",
      }));
    } finally {
      setLoadingSessionId(null);
    }
  }

  async function loadCandidates(sessionId: number) {
    setLoadingCandidatesId(sessionId);
    setActionErrors((cur) => ({ ...cur, [sessionId]: "" }));

    try {
      const data = await apiFetch(`/search/${sessionId}/candidates`);
      const candidates = data as CandidateItem[];
      setResultsBySession((cur) => ({ ...cur, [sessionId]: candidates }));
    } catch (err) {
      setActionErrors((cur) => ({
        ...cur,
        [sessionId]: err instanceof Error ? err.message : "Unable to load saved candidates.",
      }));
    } finally {
      setLoadingCandidatesId(null);
    }
  }

  function removeSession(sessionId: number) {
    if (!window.confirm("Are you sure you want to delete this session?")) return;
    deleteMutation.mutate(sessionId, {
      onError: (err) => {
        alert(err instanceof Error ? err.message : "Unable to delete session.");
      },
    });
  }

  const sortedSessions = useMemo(
    () =>
      [...sessions].sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      ),
    [sessions],
  );

  const searchedCount = sessions.filter((s) => s.status === "searched").length;
  const createdCount = sessions.length - searchedCount;

  return (
    <AppShell>
      <div className="px-6 py-6 max-w-4xl mx-auto">
        {/* Page header */}
        <div className="mb-6">
          <div className="inline-flex items-center rounded-full border border-primary/15 bg-primary/8 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-primary mb-3">
            Session registry
          </div>
          <h1 className="font-serif text-3xl font-normal tracking-tight text-foreground">
            Review sessions
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Manage saved review sessions, trigger search execution, and inspect the current persisted candidate set.
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          {[
            { value: sessions.length, label: "Total sessions" },
            { value: searchedCount, label: "Searched" },
            { value: createdCount, label: "Awaiting search" },
          ].map((stat) => (
            <div
              key={stat.label}
              className="rounded-xl border border-primary/12 bg-primary/5 px-4 py-3"
            >
              <p className="font-serif text-2xl text-foreground">{stat.value}</p>
              <p className="mt-0.5 text-xs text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Loading / error / empty states */}
        {loadingSessions && (
          <Card>
            <CardContent className="py-6">
              <LoadingState
                title="Loading sessions"
                body="Fetching saved review workspaces from the backend."
              />
            </CardContent>
          </Card>
        )}

        {loadError && <Notice tone="error">{loadError}</Notice>}

        {!loadingSessions && !loadError && sortedSessions.length === 0 && (
          <Card>
            <CardContent className="py-6">
              <EmptyState
                title="No sessions yet"
                body="Create the first review session from the dashboard, then return here to run search and inspect saved candidates."
              />
            </CardContent>
          </Card>
        )}

        {/* Session list */}
        <div className="grid gap-4">
          {sortedSessions.map((session) => {
            const topicDisplay = getSessionTopicDisplay(session.topic);
            return (
            <Card key={session.id}>
              <CardHeader>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground mb-1.5">
                      <span className="font-semibold text-primary">Session #{session.id}</span>
                      <span>·</span>
                      <span>Created {formatCreatedAt(session.created_at)}</span>
                    </div>
                    <CardTitle className="text-base font-semibold truncate">
                      {normalizeRawQuery(session.raw_query)}
                    </CardTitle>
                    <p
                      className={cn(
                        "mt-1 min-h-[1.25rem] text-sm leading-relaxed line-clamp-2",
                        topicDisplay.isPlaceholder
                          ? "italic text-muted-foreground/60"
                          : "text-muted-foreground",
                      )}
                    >
                      {topicDisplay.label}
                    </p>
                  </div>
                  <StatusBadge status={session.status} />
                </div>
              </CardHeader>

              <CardContent className="grid gap-4">
                {/* Detail chips */}
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {[
                    {
                      label: "Topic",
                      value: topicDisplay.label,
                    },
                    { label: "Status", value: session.status },
                    { label: "Saved candidates", value: resultsBySession[session.id]?.length ?? 0 },
                  ].map((item) => (
                    <div
                      key={item.label}
                      className="rounded-lg border border-border bg-muted/30 px-3 py-2"
                    >
                      <p className="text-[0.65rem] uppercase tracking-wide text-muted-foreground mb-0.5">{item.label}</p>
                      <p className="font-semibold text-foreground truncate">{item.value}</p>
                    </div>
                  ))}
                </div>

                {/* Actions */}
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    onClick={() => runSearch(session)}
                    disabled={loadingSessionId === session.id}
                  >
                    {loadingSessionId === session.id ? "Running search…" : "Run search"}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => loadCandidates(session.id)}
                    disabled={loadingCandidatesId === session.id}
                  >
                    {loadingCandidatesId === session.id ? "Loading…" : "Load saved candidates"}
                  </Button>
                  <Button size="sm" variant="ghost" asChild>
                    <Link href={`/sessions/${session.id}`} className="inline-flex items-center gap-1">
                      Open workspace
                      <ArrowRightIcon className="h-3 w-3" />
                    </Link>
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive hover:text-destructive"
                    onClick={() => removeSession(session.id)}
                  >
                    Delete
                  </Button>
                </div>

                {actionErrors[session.id] && (
                  <Notice tone="error">{actionErrors[session.id]}</Notice>
                )}

                {resultsBySession[session.id]?.length ? (
                  <>
                    <Separator />
                    <CandidatePreview candidates={resultsBySession[session.id]} />
                  </>
                ) : null}
              </CardContent>
            </Card>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}
