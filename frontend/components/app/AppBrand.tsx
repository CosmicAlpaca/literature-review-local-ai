import Link from "next/link";
import { BookOpen } from "lucide-react";

import { cn } from "@/lib/utils";

export const APP_BRAND_NAME = "Literature Review";

type AppBrandSize = "sm" | "md";

type AppBrandProps = {
  /** Hide the wordmark (e.g. collapsed sidebar). */
  showName?: boolean;
  /** Icon box + type scale. */
  size?: AppBrandSize;
  /** Light text/icon for dark panels (login). */
  tone?: "default" | "inverse";
  className?: string;
  href?: string;
};

const SIZE_STYLES: Record<
  AppBrandSize,
  { box: string; icon: number; name: string }
> = {
  sm: {
    box: "size-9 rounded-lg",
    icon: 18,
    name: "text-base font-bold tracking-tight",
  },
  md: {
    box: "size-10 rounded-[11px]",
    icon: 20,
    name: "text-lg font-extrabold tracking-[-0.02em]",
  },
};

export default function AppBrand({
  showName = true,
  size = "md",
  tone = "default",
  className,
  href,
}: AppBrandProps) {
  const scale = SIZE_STYLES[size];
  const inverse = tone === "inverse";

  const mark = (
    <div
      className={cn(
        "grid shrink-0 place-items-center",
        scale.box,
        inverse ? "bg-white/12 text-white" : "bg-primary/12 text-primary",
      )}
      aria-hidden
    >
      <BookOpen size={scale.icon} />
    </div>
  );

  const content = (
    <div className={cn("flex min-w-0 items-center gap-3", className)}>
      {mark}
      {showName ? (
        <span
          className={cn(
            scale.name,
            "truncate leading-tight",
            inverse ? "text-white" : "text-foreground",
          )}
        >
          {APP_BRAND_NAME}
        </span>
      ) : null}
    </div>
  );

  if (href) {
    return (
      <Link
        href={href}
        className="inline-flex min-w-0 rounded-md outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
      >
        {content}
      </Link>
    );
  }

  return content;
}
