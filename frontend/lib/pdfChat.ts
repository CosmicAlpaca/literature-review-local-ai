/**
 * lib/pdfChat.ts – API client for the PDF Chat feature.
 * Completely isolated from the search/session pipeline.
 */
import { apiFetch } from "@/lib/axios";

export type UploadResult = {
  session_key: string;
  doc_name: string;
  chunks_added: number;
  total_chunks: number;
  documents: string[];
};

export type DocumentList = {
  session_key: string;
  documents: string[];
  total_chunks: number;
};

export type CitedChunk = {
  doc_name: string;
  page_num: number;
  text_preview: string;
};

export type AskResult = {
  answer: string;
  cited_chunks: CitedChunk[];
  error?: string | null;
};

export type SummaryResult = {
  summary: string;
  doc_name: string | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  cited_chunks?: CitedChunk[];
  isLoading?: boolean;
};

// Create a new isolated PDF chat session and return its session key
export async function createPdfSession(): Promise<string> {
  const res = await apiFetch<{ session_key: string }>("/pdf-chat/sessions", {
    method: "POST",
  });
  return res.session_key;
}

// Upload a PDF file to the session
export async function uploadPdf(
  sessionKey: string,
  file: File
): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<UploadResult>(
    `/pdf-chat/sessions/${encodeURIComponent(sessionKey)}/upload`,
    { method: "POST", body: formData }
  );
}

// List documents in a session
export async function listDocuments(sessionKey: string): Promise<DocumentList> {
  return apiFetch<DocumentList>(
    `/pdf-chat/sessions/${encodeURIComponent(sessionKey)}/docs`
  );
}

// Ask a question
export async function askQuestion(
  sessionKey: string,
  question: string,
  history: { role: "user" | "assistant"; content: string }[] = []
): Promise<AskResult> {
  return apiFetch<AskResult>(
    `/pdf-chat/sessions/${encodeURIComponent(sessionKey)}/ask`,
    { method: "POST", body: JSON.stringify({ question, history }) }
  );
}

// Summarize a document
export async function summarizeDocument(
  sessionKey: string,
  docName: string | null
): Promise<SummaryResult> {
  return apiFetch<SummaryResult>(
    `/pdf-chat/sessions/${encodeURIComponent(sessionKey)}/summary`,
    { method: "POST", body: JSON.stringify({ doc_name: docName }) }
  );
}

// Delete the session
export async function deletePdfSession(sessionKey: string): Promise<void> {
  await apiFetch(`/pdf-chat/sessions/${encodeURIComponent(sessionKey)}`, {
    method: "DELETE",
  });
}

export type KnowledgeGraph = {
  nodes: { id: string; label: string; weight: number; sources: CitedChunk[] }[];
  edges: { source: string; target: string; weight: number; source_ref: CitedChunk }[];
};
export async function getKnowledgeGraph(sessionKey: string): Promise<KnowledgeGraph> {
  return apiFetch(`/pdf-chat/sessions/${encodeURIComponent(sessionKey)}/graph`);
}
