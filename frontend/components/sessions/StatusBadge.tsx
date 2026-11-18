import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type Props = {
  status: string;
};

function normalizeStatus(status: string) {
  const normalized = status.trim().toLowerCase();

  if (normalized === "searched") {
    return { label: "Searched", className: "bg-emerald-50 text-emerald-700 border-emerald-200" };
  }

  if (normalized === "created") {
    return { label: "Created", className: "bg-primary/10 text-primary border-primary/15" };
  }

  return { label: status || "Unknown", className: "bg-muted text-muted-foreground border-border" };
}

export default function StatusBadge({ status }: Props) {
  const { label, className } = normalizeStatus(status);

  return (
    <Badge
      variant="outline"
      className={cn("rounded-full text-xs font-medium shrink-0", className)}
    >
      {label}
    </Badge>
  );
}
