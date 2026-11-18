import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { PaperQualityLevel } from "@/lib/session";

type Props = {
  level: PaperQualityLevel;
  label: string;
};

const levelClass: Record<PaperQualityLevel, string> = {
  strong: "bg-emerald-50 text-emerald-700 border-emerald-200",
  moderate: "bg-primary/10 text-primary border-primary/15",
  caution: "bg-red-50 text-red-700 border-red-200",
};

export default function TrustBadge({ level, label }: Props) {
  return (
    <Badge
      variant="outline"
      className={cn("rounded-full text-xs font-medium", levelClass[level])}
    >
      {label}
    </Badge>
  );
}
