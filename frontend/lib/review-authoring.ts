/** Shapes aligned with FastAPI `app/schemas/review.py` (authoring endpoints). */

export type ReviewOutlineSection = {
  section_id: string;
  title: string;
  purpose: string;
  supporting_candidate_ids: number[];
  key_points: string[];
};

export type ReviewOutlinePayload = {
  title: string;
  sections: ReviewOutlineSection[];
};

export type ReviewOutlineResponse = {
  session_id: number;
  research_query: string;
  paper_count: number;
  outline: ReviewOutlinePayload;
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
};

export type ReviewTheme = {
  theme_id: string;
  name: string;
  rationale: string;
  keywords: string[];
  supporting_candidate_ids: number[];
};

export type ReviewThemesResponse = {
  session_id: number;
  research_query: string;
  paper_count: number;
  themes: ReviewTheme[];
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
};

export type ReviewDraftSection = {
  section_id: string;
  title: string;
  content: string;
  supporting_candidate_ids: number[];
  citation_placeholders: string[];
  key_claims: string[];
  limitations_or_cautions: string[];
};

export type ReviewDraftPayload = {
  title: string;
  sections: ReviewDraftSection[];
};

export type ReviewDraftResponse = {
  session_id: number;
  research_query: string;
  paper_count: number;
  draft: ReviewDraftPayload;
  markdown_preview?: string | null;
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
};

export type DraftGroundingCheck = {
  valid: boolean;
  unknown_citations: number;
  unknown_supporting_candidate_ids: number;
  sections_without_evidence: number;
  total_sections: number;
  total_supported_sections: number;
};

export type ReviewMarkdownExportResponse = {
  session_id: number;
  filename: string;
  markdown: string;
  paper_count: number;
  grounding_check: DraftGroundingCheck;
  llm_used: boolean;
  fallback_reason?: string | null;
  prompt_version?: string | null;
};
