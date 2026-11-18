import { cn } from "@/lib/utils";

/** Matches shadcn Badge variants in `components/ui/badge.tsx`. */
export type DebateBadgeVariant =
  | "default"
  | "secondary"
  | "destructive"
  | "outline"
  | "ghost"
  | "link";

export type FinalRecommendation =
  | "strong_draft"
  | "acceptable_with_minor_revision"
  | "revise_before_use"
  | "insufficient_evidence";

export type RiskLevel = "low" | "medium" | "high";

export type DebateQualityScores = {
  relevance: number;
  grounding: number;
  methodology: number;
  source_reliability: number;
  overall: number;
};

export type DraftGroundingCheck = {
  valid: boolean;
  unknown_citations: number;
  unknown_supporting_candidate_ids: number;
  sections_without_evidence: number;
  total_sections: number;
  total_supported_sections: number;
};

export type DebateReport = {
  overall_assessment: string;
  strengths: string[];
  weaknesses: string[];
  missing_evidence: string[];
  unsupported_or_weak_claims: string[];
  source_reliability_concerns: string[];
  methodology_concerns: string[];
  suggested_revisions: string[];
  final_recommendation: FinalRecommendation;
  confidence_score: number;
  action_items: string[];
  risk_level: RiskLevel;
  quality_scores: DebateQualityScores;
  debate_summary: string;
};

/** Generic role fields present on every agent_outputs entry (final_arbiter merges full report). */
export type DebateRoleOutput = {
  role_name: string;
  score: number;
  findings: string[];
  concerns: string[];
  suggestions: string[];
} & Record<string, unknown>;

export type DebateAgentOutputs = {
  relevance_reviewer: DebateRoleOutput;
  evidence_grounding_critic: DebateRoleOutput;
  methodology_critic: DebateRoleOutput;
  source_reliability_critic: DebateRoleOutput;
  final_arbiter: DebateRoleOutput;
};

export type ReviewDebateResponse = {
  session_id: number;
  research_query: string;
  paper_count: number;
  debate_report: DebateReport;
  agent_outputs: DebateAgentOutputs;
  grounding_check: DraftGroundingCheck;
  markdown_report?: string | null;
  debate_mode: "structured_panel";
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
  /** Present on GET/POST from persisted or fresh generation. */
  cached?: boolean;
  generated_at?: string | null;
  updated_at?: string | null;
};

/** TanStack Query key for GET/POST debate review. */
export function debateReviewQueryKey(sessionId: number) {
  return ["sessions", sessionId, "review", "debate"] as const;
}

export const DEBATE_ROLE_KEYS = [
  "relevance_reviewer",
  "evidence_grounding_critic",
  "methodology_critic",
  "source_reliability_critic",
  "final_arbiter",
] as const;

export type DebateRoleKey = (typeof DEBATE_ROLE_KEYS)[number];

export const QUALITY_SCORE_LABELS: { key: keyof DebateQualityScores; label: string }[] = [
  { key: "relevance", label: "Relevance" },
  { key: "grounding", label: "Grounding" },
  { key: "methodology", label: "Methodology" },
  { key: "source_reliability", label: "Source reliability" },
  { key: "overall", label: "Overall" },
];

export function finalRecommendationLabel(v: FinalRecommendation): string {
  switch (v) {
    case "strong_draft":
      return "Strong draft";
    case "acceptable_with_minor_revision":
      return "Acceptable, minor revision";
    case "revise_before_use":
      return "Revise before use";
    case "insufficient_evidence":
      return "Insufficient evidence";
    default:
      return v;
  }
}

export function finalRecommendationBadge(v: FinalRecommendation): {
  variant: DebateBadgeVariant;
  className?: string;
} {
  switch (v) {
    case "strong_draft":
      return {
        variant: "secondary",
        className:
          "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
      };
    case "acceptable_with_minor_revision":
      return { variant: "outline", className: "border-primary/25 text-primary" };
    case "revise_before_use":
      return { variant: "outline", className: "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100" };
    case "insufficient_evidence":
      return { variant: "destructive" };
    default:
      return { variant: "outline" };
  }
}

export function riskLevelLabel(level: RiskLevel): string {
  switch (level) {
    case "low":
      return "Risk: low";
    case "medium":
      return "Risk: medium";
    case "high":
      return "Risk: high";
    default:
      return `Risk: ${level}`;
  }
}

export function riskLevelBadge(level: RiskLevel): {
  variant: DebateBadgeVariant;
  className?: string;
} {
  switch (level) {
    case "low":
      return {
        variant: "outline",
        className:
          "border-emerald-200 bg-emerald-50/80 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-200",
      };
    case "medium":
      return {
        variant: "outline",
        className:
          "border-amber-200 bg-amber-50/80 text-amber-900 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-100",
      };
    case "high":
      return { variant: "destructive" };
    default:
      return { variant: "outline" };
  }
}

export function formatScore01(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(2);
}

export function mergeBadgeClass(
  base: { variant: DebateBadgeVariant; className?: string },
  extra?: string,
): { variant: DebateBadgeVariant; className: string } {
  return { variant: base.variant, className: cn(base.className, extra) };
}
