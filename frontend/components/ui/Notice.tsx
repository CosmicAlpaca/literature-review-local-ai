import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type Props = {
  tone?: "success" | "error" | "info";
  children: ReactNode;
};

const toneStyles: Record<string, string> = {
  success: "bg-emerald-50 border-emerald-200 text-emerald-800",
  error: "bg-red-50 border-red-200 text-red-800",
  info: "bg-blue-50 border-blue-200 text-blue-800",
};

export default function Notice({ tone = "info", children }: Props) {
  return (
    <div
      className={cn(
        "rounded-lg border px-4 py-3 text-sm leading-relaxed",
        toneStyles[tone],
      )}
    >
      {children}
    </div>
  );
}
