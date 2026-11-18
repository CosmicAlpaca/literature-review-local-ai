import { Badge } from "@/components/ui/badge";
import type { CandidateItem } from "@/lib/session";
import { formatProviderLabel } from "@/lib/session";

type Props = {
  candidates: CandidateItem[];
};

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength).trim()}…`;
}

export default function CandidatePreview({ candidates }: Props) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-foreground">Saved candidate preview</p>
        <span className="text-xs text-muted-foreground">
          {candidates.length} candidate{candidates.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="grid gap-2">
        {candidates.map((candidate) => (
          <div
            key={candidate.id}
            className="rounded-lg border border-border bg-white/60 px-3 py-2.5"
          >
            <div className="flex items-start justify-between gap-3">
              <p className="text-xs font-semibold text-foreground leading-snug">
                {candidate.url ? (
                  <a
                    href={candidate.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline hover:text-primary transition-colors"
                  >
                    {candidate.title}
                  </a>
                ) : (
                  candidate.title
                )}
              </p>
              <Badge variant="secondary" className="shrink-0 text-xs tabular-nums">
                {candidate.final_score?.toFixed(3) ?? "N/A"}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {[
                candidate.source_title || "Unknown source",
                candidate.year || "N/A",
                formatProviderLabel(candidate.provider),
                `${candidate.citation_count ?? 0} citations`,
                candidate.scopus_matched ? "Source matched to Scopus" : "Source not matched",
              ].join(" · ")}
            </p>
            {candidate.abstract && (
              <p className="mt-1.5 text-xs text-muted-foreground leading-relaxed">
                {truncateText(candidate.abstract, 220)}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
