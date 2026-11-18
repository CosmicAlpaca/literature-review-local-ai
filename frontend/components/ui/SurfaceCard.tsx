import type { HTMLAttributes, ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Props = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
  compact?: boolean;
};

export default function SurfaceCard({
  children,
  className,
  compact = false,
  ...props
}: Props) {
  return (
    <Card
      size={compact ? "sm" : "default"}
      className={cn("overflow-visible", className)}
      {...props}
    >
      {children}
    </Card>
  );
}
