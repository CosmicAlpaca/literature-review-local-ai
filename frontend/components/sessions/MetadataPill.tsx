import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type Tone = "neutral" | "success" | "warning" | "subtle";

type Props = {
  children: ReactNode;
  tone?: Tone;
};

const toneClass: Record<Tone, string> = {
  neutral: "bg-primary/10 text-primary border-primary/15",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  subtle: "bg-muted text-muted-foreground border-border",
};

export default function MetadataPill({ children, tone = "neutral" }: Props) {
  return (
    <Badge
      variant="outline"
      className={cn("rounded-full text-xs font-medium", toneClass[tone])}
    >
      {children}
    </Badge>
  );
}
