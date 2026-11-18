"use client";

import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import AppShell from "@/components/app/AppShell";
import SearchForm, { type SearchPayload } from "@/components/SearchForm";
import { useCreateSession } from "@/lib/session";

export default function NewSearchPage() {
  const router = useRouter();

  // Shared mutation: optimistically updates the `["sessionList"]` cache and
  // invalidates it for a background refresh. The detail page we navigate
  // into therefore finds the new session already in cache.
  const createSession = useCreateSession();

  function handleSubmit(payload: SearchPayload) {
    createSession.mutate(payload, {
      onSuccess: (data) => {
        router.push(`/sessions/${data.id}`);
      },
      onError: (err) => {
        toast.error(
          err instanceof Error ? err.message : "Unable to create the session.",
        );
      },
    });
  }

  return (
    <AppShell>
      <div className="flex min-h-screen flex-col items-center justify-center px-6 py-16">
        <div className="w-full max-w-2xl">
          <div className="mb-8">
            <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-3">
              New search
            </p>
            <h1 className="font-serif text-3xl font-normal tracking-tight text-foreground sm:text-4xl">
              Start a new literature review
            </h1>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-lg">
              Enter your research question below, then narrow by topic, year
              range, or access type. Ranking indicates relevance to your query;
              inspect the paper and its sources before drawing conclusions.
            </p>
          </div>

          <SearchForm
            onSubmit={handleSubmit}
            submitting={createSession.isPending}
          />
        </div>
      </div>
    </AppShell>
  );
}
