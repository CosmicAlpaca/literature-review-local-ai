"use client";
import { useState } from "react";
import type { KnowledgeGraph } from "@/lib/pdfChat";

type Props = { graph: KnowledgeGraph };
export default function KnowledgeMap({ graph }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const nodes = graph.nodes.slice(0, 14);
  const positions = new Map(nodes.map((n, i) => {
    const angle = (2 * Math.PI * i) / Math.max(nodes.length, 1) - Math.PI / 2;
    return [n.id, { x: 250 + 175 * Math.cos(angle), y: 160 + 115 * Math.sin(angle) }];
  }));
  const current = nodes.find((n) => n.id === selected);
  return (
    <details className="border-b border-border bg-muted/20 px-5 py-2">
      <summary className="cursor-pointer text-sm font-semibold">Bản đồ khái niệm · Concept map</summary>
      <p className="mt-1 text-xs text-muted-foreground">Terms linked here occur in the same passage. A link does not establish a scientific relationship. Select a term to inspect its sources.</p>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
        <svg viewBox="0 0 500 320" role="img" aria-label="Concept co-occurrence map" className="w-full max-h-80">
          {graph.edges.map((e, i) => {
            const a = positions.get(e.source), b = positions.get(e.target);
            return a && b ? <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="currentColor" className="text-border" strokeWidth={Math.min(3, e.weight)} /> : null;
          })}
          {nodes.map((n) => {
            const p = positions.get(n.id)!;
            return <g key={n.id} role="button" tabIndex={0} aria-label={`Show sources for ${n.label}`} onClick={() => setSelected(n.id)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelected(n.id); } }} className="cursor-pointer">
              <circle cx={p.x} cy={p.y} r={selected === n.id ? 8 : 6} fill="currentColor" className="text-primary" />
              <text x={p.x} y={p.y - 11} textAnchor="middle" fontSize="10" fill="currentColor">{n.label.slice(0, 26)}</text>
            </g>;
          })}
        </svg>
        <div className="py-3 text-xs text-muted-foreground" aria-live="polite">
          {current ? <><strong className="text-foreground">{current.label}</strong>{current.sources.map((s, i) => <div key={i} className="mt-2 rounded border p-2"><b>{s.doc_name} · page {s.page_num}</b><p className="mt-1">{s.text_preview}…</p></div>)}</> : "Select a term to see the document and page where it appears."}
        </div>
      </div>
    </details>
  );
}
