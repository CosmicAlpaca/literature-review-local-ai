"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  FileTextIcon,
  SendIcon,
  TrashIcon,
  UploadCloudIcon,
  XIcon,
  SparklesIcon,
  BookOpenCheckIcon,
  Loader2Icon,
  FileIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import KnowledgeMap from "@/components/pdf-chat/KnowledgeMap";
import {
  type ChatMessage,
  type CitedChunk,
  createPdfSession,
  uploadPdf,
  askQuestion,
  summarizeDocument,
  deletePdfSession,
  getKnowledgeGraph,
  type KnowledgeGraph,
} from "@/lib/pdfChat";

// ── Helpers ────────────────────────────────────────────────────────────────────
let msgCounter = 0;
const nextId = () => `m${++msgCounter}`;

// ── Sub-components ─────────────────────────────────────────────────────────────

function CitationCard({ chunk }: { chunk: CitedChunk }) {
  return (
    <div className="mt-1 rounded-md border border-border bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-semibold text-foreground">{chunk.doc_name}</span>{" "}
      · trang {chunk.page_num}
      <p className="mt-0.5 line-clamp-2 italic">{chunk.text_preview}</p>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex gap-2", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <SparklesIcon className="h-3.5 w-3.5" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
          isUser
            ? "rounded-tr-none bg-primary text-primary-foreground"
            : "rounded-tl-none bg-card border border-border text-foreground"
        )}
      >
        {msg.isLoading ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2Icon className="h-4 w-4 animate-spin" />
            <span>Đang phân tích tài liệu...</span>
          </div>
        ) : (
          <>
            <div className="whitespace-pre-wrap">{msg.content}</div>
            {msg.cited_chunks && msg.cited_chunks.length > 0 && (
              <div className="mt-3 space-y-1 border-t border-border/50 pt-2">
                <p className="text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground">
                  Nguồn trích dẫn
                </p>
                {msg.cited_chunks.map((c, i) => (
                  <CitationCard key={i} chunk={c} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function PdfChatPage() {
  const [sessionKey, setSessionKey] = useState<string | null>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  // Init session on mount
  useEffect(() => {
    createPdfSession()
      .then(setSessionKey)
      .catch(() => setUploadError("Không thể tạo phiên chat. Vui lòng tải lại trang."));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionKey) deletePdfSession(sessionKey).catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey]);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      if (!sessionKey) return;
      const pdfs = Array.from(files).filter((f) =>
        f.name.toLowerCase().endsWith(".pdf")
      );
      if (pdfs.length === 0) {
        setUploadError("Chỉ chấp nhận file PDF.");
        return;
      }

      setIsUploading(true);
      setUploadError(null);

      for (const file of pdfs) {
        try {
          const res = await uploadPdf(sessionKey, file);
          setDocuments(res.documents);
          try { setGraph(await getKnowledgeGraph(sessionKey)); } catch { setGraph(null); }
          setMessages((prev) => [
            ...prev,
            {
              id: nextId(),
              role: "assistant",
              content: `✅ Đã tải lên **${res.doc_name}** (${res.chunks_added} đoạn văn bản). Bạn có thể hỏi về tài liệu này.`,
            },
          ]);
        } catch (e: any) {
          setUploadError(e?.message ?? "Lỗi khi tải lên file.");
        }
      }
      setIsUploading(false);
    },
    [sessionKey]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleSend = async () => {
    if (!input.trim() || !sessionKey || isSending) return;

    const question = input.trim();
    setInput("");
    setIsSending(true);

    const userMsg: ChatMessage = {
      id: nextId(),
      role: "user",
      content: question,
    };
    const placeholderId = nextId();
    const placeholder: ChatMessage = {
      id: placeholderId,
      role: "assistant",
      content: "",
      isLoading: true,
    };
    setMessages((prev) => [...prev, userMsg, placeholder]);

    try {
      const history = messages
        .filter((m) => !m.isLoading && (m.role === "user" || Boolean(m.cited_chunks?.length)))
        .slice(-4)
        .map((m) => ({ role: m.role, content: m.content.slice(0, 500) }));
      const res = await askQuestion(sessionKey, question, history);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? {
                id: placeholderId,
                role: "assistant",
                content: res.answer,
                cited_chunks: res.cited_chunks,
              }
            : m
        )
      );
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? {
                id: placeholderId,
                role: "assistant",
                content: "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
              }
            : m
        )
      );
    } finally {
      setIsSending(false);
    }
  };

  const handleSummarize = async (docName: string | null) => {
    if (!sessionKey || isSending) return;
    setIsSending(true);

    const label = docName ? `"${docName}"` : "tất cả tài liệu";
    const userMsg: ChatMessage = {
      id: nextId(),
      role: "user",
      content: `Hãy tóm tắt ${label}`,
    };
    const placeholderId = nextId();
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: placeholderId, role: "assistant", content: "", isLoading: true },
    ]);

    try {
      const res = await summarizeDocument(sessionKey, docName);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? { id: placeholderId, role: "assistant", content: res.summary }
            : m
        )
      );
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? { id: placeholderId, role: "assistant", content: "Lỗi khi tạo tóm tắt." }
            : m
        )
      );
    } finally {
      setIsSending(false);
    }
  };

  const handleClearSession = async () => {
    if (!sessionKey) return;
    await deletePdfSession(sessionKey).catch(() => {});
    const newKey = await createPdfSession().catch(() => null);
    setSessionKey(newKey);
    setDocuments([]);
    setGraph(null);
    setMessages([]);
    setUploadError(null);
  };

  const isEmpty = documents.length === 0;

  return (
    <div className="flex h-full">
      {/* ── Sidebar: document list ─────────────────────────────────── */}
      <aside className="hidden w-56 shrink-0 flex-col border-r border-border bg-card md:flex">
        <div className="flex items-center gap-2 border-b border-border px-4 py-4">
          <FileTextIcon className="h-4 w-4 text-primary shrink-0" />
          <span className="text-sm font-semibold text-foreground">Tài liệu</span>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3">
          {documents.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center mt-4">
              Chưa có tài liệu
            </p>
          ) : (
            <div className="space-y-1">
              {documents.map((d) => (
                <div
                  key={d}
                  className="group flex items-start gap-2 rounded-lg px-2 py-2 hover:bg-accent"
                >
                  <FileIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-foreground">{d}</p>
                    <button
                      onClick={() => handleSummarize(d)}
                      disabled={isSending}
                      className="mt-0.5 text-[0.65rem] text-primary hover:underline disabled:opacity-50"
                    >
                      Tóm tắt bằng AI
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {documents.length > 1 && (
          <div className="border-t border-border px-3 py-3">
            <button
              onClick={() => handleSummarize(null)}
              disabled={isSending}
              className="flex w-full items-center gap-1.5 rounded-lg px-2 py-2 text-xs font-medium text-primary hover:bg-accent disabled:opacity-50"
            >
              <BookOpenCheckIcon className="h-3.5 w-3.5" />
              Tóm tắt bằng AI
            </button>
          </div>
        )}
      </aside>

      {/* ── Main chat area ──────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col min-w-0 bg-background">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border px-5 py-3">
          <div className="flex items-center gap-2">
            <SparklesIcon className="h-4 w-4 text-primary" />
            <h1 className="text-sm font-semibold text-foreground">
              Chat với PDF
            </h1>
            {documents.length > 0 && (
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[0.65rem] font-medium text-primary">
                {documents.length} tài liệu
              </span>
            )}
          </div>
          <button
            onClick={handleClearSession}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
          >
            <TrashIcon className="h-3.5 w-3.5" />
            Xóa phiên
          </button>
        </header>

        {graph && graph.nodes.length > 0 && <KnowledgeMap graph={graph} />}
        <p className="px-5 py-1 text-xs text-muted-foreground">Local AI chat · Answers use retrieved passages and may contain errors; check the sources. Sessions are temporary and lost when the server restarts.</p>

        {/* Drop zone / messages */}
        <div
          className={cn(
            "flex-1 overflow-y-auto",
            isDragging && "ring-2 ring-inset ring-primary/40 bg-primary/5"
          )}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {isEmpty && messages.length === 0 ? (
            /* ── Empty state: big upload zone ── */
            <div className="flex h-full flex-col items-center justify-center px-6 text-center gap-6">
              <div className="rounded-2xl border-2 border-dashed border-border bg-muted/30 p-12 flex flex-col items-center gap-4 max-w-md w-full">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                  <UploadCloudIcon className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <p className="font-semibold text-foreground">Tải lên file PDF</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Kéo thả file hoặc nhấn nút bên dưới
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Tối đa 20 MB / file · Tối đa 10 file
                  </p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading || !sessionKey}
                  className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
                >
                  {isUploading ? (
                    <Loader2Icon className="h-4 w-4 animate-spin" />
                  ) : (
                    <UploadCloudIcon className="h-4 w-4" />
                  )}
                  {isUploading ? "Đang tải lên..." : "Chọn file PDF"}
                </button>
              </div>
              {uploadError && (
                <p className="text-sm text-destructive">{uploadError}</p>
              )}
            </div>
          ) : (
            /* ── Message list ── */
            <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        {!isEmpty && (
          <div className="border-t border-border bg-background px-4 py-3">
            {uploadError && (
              <div className="mb-2 flex items-center gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <XIcon className="h-3.5 w-3.5 shrink-0" />
                {uploadError}
                <button onClick={() => setUploadError(null)} className="ml-auto">
                  <XIcon className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
            <div className="flex items-end gap-2 rounded-xl border border-border bg-card px-3 py-2 shadow-sm focus-within:ring-1 focus-within:ring-primary/50">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading || !sessionKey}
                className="mb-0.5 flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50 transition-colors"
                title="Tải thêm PDF"
              >
                {isUploading ? (
                  <Loader2Icon className="h-4 w-4 animate-spin" />
                ) : (
                  <UploadCloudIcon className="h-4 w-4" />
                )}
              </button>

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Hỏi về nội dung tài liệu... (Enter để gửi)"
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none"
                disabled={isSending}
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || isSending}
                className="mb-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
              >
                {isSending ? (
                  <Loader2Icon className="h-4 w-4 animate-spin" />
                ) : (
                  <SendIcon className="h-4 w-4" />
                )}
              </button>
            </div>
            <p className="mt-1.5 text-center text-[0.65rem] text-muted-foreground">
              AI có thể mắc lỗi. Hãy kiểm tra các trích dẫn.
            </p>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
