import { Badge } from "@/components/ui/badge";
import type { DraftGroundingCheck } from "@/lib/debate";

type Props = {
  check: DraftGroundingCheck;
};

/** Shared draft/debate grounding check summary (matches review export panel). */
export default function GroundingCheckSummary({ check }: Props) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Overall</span>
        <Badge
          variant={check.valid ? "default" : "destructive"}
          className="rounded-full"
        >
          {check.valid ? "Passed" : "Issues found"}
        </Badge>
      </div>
      <div className="text-xs text-muted-foreground sm:text-right">
        Supported sections {check.total_supported_sections}/{check.total_sections}
      </div>
      <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-xs sm:col-span-2">
        <p className="font-medium text-foreground">Citation and ID checks</p>
        <ul className="mt-1.5 list-inside list-disc space-y-0.5 text-muted-foreground">
          <li>Unknown citations: {check.unknown_citations}</li>
          <li>
            Unknown supporting candidate IDs: {check.unknown_supporting_candidate_ids}
          </li>
          <li>Sections without mapped evidence: {check.sections_without_evidence}</li>
        </ul>
      </div>
    </div>
  );
}
