"use client";

import { useRouter } from "next/navigation";
import AppBrand, { APP_BRAND_NAME } from "@/components/app/AppBrand";
import { useAccessToken } from "@/components/auth/queries";
import { cn } from "@/lib/utils";
import {
  Sparkles,
  ShieldCheck,
  BarChart3,
  Search,
  ArrowRight,
  Zap,
  BookOpen,
  Database,
  Clock,
  ChevronRight,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Animated typing placeholder for the hero search bar               */
/* ------------------------------------------------------------------ */
const PLACEHOLDER_QUERIES = [
  "How does transformer architecture improve NLP tasks?",
  "Impact of climate change on marine biodiversity",
  "Machine learning approaches for drug discovery",
  "Renewable energy storage solutions in developing countries",
  "CRISPR gene editing ethical implications",
];

function useTypingPlaceholder() {
  const [text, setText] = useState("");
  const [queryIndex, setQueryIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const current = PLACEHOLDER_QUERIES[queryIndex];
    let timeout: NodeJS.Timeout;

    if (!isDeleting && charIndex < current.length) {
      timeout = setTimeout(() => {
        setText(current.slice(0, charIndex + 1));
        setCharIndex((c) => c + 1);
      }, 45);
    } else if (!isDeleting && charIndex === current.length) {
      timeout = setTimeout(() => setIsDeleting(true), 2200);
    } else if (isDeleting && charIndex > 0) {
      timeout = setTimeout(() => {
        setText(current.slice(0, charIndex - 1));
        setCharIndex((c) => c - 1);
      }, 25);
    } else if (isDeleting && charIndex === 0) {
      setIsDeleting(false);
      setQueryIndex((i) => (i + 1) % PLACEHOLDER_QUERIES.length);
    }

    return () => clearTimeout(timeout);
  }, [charIndex, isDeleting, queryIndex]);

  return text;
}

/* ------------------------------------------------------------------ */
/*  Animated counter for statistics                                    */
/* ------------------------------------------------------------------ */
function AnimatedCounter({
  target,
  suffix = "",
  duration = 2000,
}: {
  target: number;
  suffix?: string;
  duration?: number;
}) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const startTime = performance.now();
          const animate = (now: number) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(2, -10 * progress);
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.3 },
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target, duration]);

  return (
    <span ref={ref}>
      {count.toLocaleString()}
      {suffix}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Quick-tag chips                                                    */
/* ------------------------------------------------------------------ */
const QUICK_TAGS = [
  "Computer Science",
  "Medicine",
  "Engineering",
  "Q1 Journals",
  "Open Access",
  "Social Sciences",
];

/* ------------------------------------------------------------------ */
/*  Feature cards data                                                 */
/* ------------------------------------------------------------------ */
const FEATURES = [
  {
    icon: Sparkles,
    eyebrow: "Step 1",
    title: "AI Query Transformation",
    body: "Your natural-language research question is intelligently expanded into a precise, multi-faceted academic query — capturing synonyms, related concepts, and field-specific terminology.",
    accent: "var(--color-primary)",
  },
  {
    icon: ShieldCheck,
    eyebrow: "Step 2",
    title: "Scopus-Exclusive Filter",
    body: "Every result is cross-referenced against the Scopus journal index, ensuring you only see papers from reputable, peer-reviewed sources that meet rigorous quality standards.",
    accent: "var(--color-success)",
  },
  {
    icon: BarChart3,
    eyebrow: "Step 3",
    title: "Semantic Reranking",
    body: "Results are reordered by contextual relevance using AI embeddings — not just keyword matching — so the most meaningful papers surface first.",
    accent: "#8d5a12",
  },
];

const pageWrap = "w-[min(1120px,calc(100vw-32px))] mx-auto";

const btnBase =
  "inline-flex items-center justify-center gap-2 min-h-12 px-[18px] rounded-full border border-transparent font-bold cursor-pointer transition-all duration-[120ms] ease-out disabled:cursor-wait disabled:opacity-[0.68]";

const btnPrimary =
  "bg-[var(--color-primary)] text-[var(--color-text-inverse)] shadow-[0_10px_22px_rgba(0,117,149,0.24)] hover:bg-[var(--color-primary-hover)] hover:-translate-y-px";

const btnSecondary =
  "bg-[rgba(255,255,255,0.64)] text-[var(--color-text)] border-[var(--color-border)] hover:bg-[var(--color-surface-strong)]";

const eyebrowLight =
  "inline-flex items-center gap-2 w-fit rounded-full border px-3 py-2 text-[0.84rem] tracking-[0.04em] uppercase";

const eyebrowPrimaryTint =
  "bg-[rgba(0,117,149,0.08)] border-[rgba(0,117,149,0.14)] text-primary";

/* ================================================================== */
/*  Landing Page Component                                             */
/* ================================================================== */
export default function LandingPage() {
  const router = useRouter();
  const { data: token, isPending } = useAccessToken();
  const typingText = useTypingPlaceholder();
  const isLoggedIn = !isPending && !!token;

  function handleCTA() {
    router.push(isLoggedIn ? "/new-search" : "/login");
  }

  const heroSearchLabel = isLoggedIn
    ? "Start a new literature search. Opens your search dashboard."
    : "Sign in to start a literature search. Opens the sign-in page.";

  return (
    <div className="min-h-screen overflow-x-hidden">
      <nav
        className={cn(
          "sticky top-0 z-[100] border-b border-[var(--color-border)] backdrop-blur-md",
          "bg-[rgba(247,244,237,0.82)]",
        )}
      >
        <div className={cn(pageWrap, "flex h-16 items-center justify-between")}>
          <AppBrand href="/" />
          <div className="flex items-center gap-2.5">
            {isLoggedIn ? (
              <button
                type="button"
                className={cn(btnBase, btnPrimary)}
                onClick={() => router.push("/new-search")}
              >
                Go to Dashboard
                <ArrowRight size={16} />
              </button>
            ) : (
              <>
                <button
                  type="button"
                  className={cn(
                    btnBase,
                    "min-h-0 border-0 bg-transparent p-0 font-bold text-primary hover:text-[var(--color-primary-hover)] max-[640px]:hidden",
                  )}
                  onClick={() => router.push("/login")}
                >
                  Sign in
                </button>
                <button
                  type="button"
                  className={cn(btnBase, btnPrimary)}
                  onClick={() => router.push("/login")}
                >
                  Get Started
                  <ArrowRight size={16} />
                </button>
              </>
            )}
          </div>
        </div>
      </nav>

      <section
        className={cn(
          "relative overflow-hidden py-20 pb-16 text-center max-[960px]:py-14 max-[960px]:pb-12",
        )}
      >
        <div className={cn(pageWrap, "relative z-[2] flex flex-col items-center")}>
          <div
            className="pointer-events-none absolute top-[-120px] left-1/2 z-0 h-[700px] w-[700px] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(0,117,149,0.1)_0%,rgba(0,117,149,0.04)_40%,transparent_70%)]"
            aria-hidden
          />
          <div
            className="pointer-events-none absolute inset-0 z-0 bg-[length:48px_48px] [background-image:linear-gradient(rgba(23,33,43,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(23,33,43,0.04)_1px,transparent_1px)] [-webkit-mask-image:radial-gradient(ellipse_60%_50%_at_50%_40%,black,transparent)] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_40%,black,transparent)]"
            aria-hidden
          />

          <div
            className={cn(eyebrowLight, eyebrowPrimaryTint, "mb-6")}
          >
            <Zap size={14} />
            Powered by AI &amp; OpenAlex
          </div>

          <h1
            className={cn(
              "mb-5 max-w-[16ch] text-[clamp(2.6rem,6vw,4.6rem)] leading-[0.97] font-normal tracking-[-0.035em] text-[var(--color-text)] max-[960px]:max-w-full [font-family:var(--font-heading)]",
            )}
          >
            Elevate your literature review with artificial intelligence.
          </h1>

          <p
            className={cn(
              "mx-auto mb-9 max-w-[44rem] text-[1.08rem] leading-[1.75] text-[var(--color-text-soft)]",
            )}
          >
            More than keyword search.{" "}
            <strong>{APP_BRAND_NAME.replace(" ", "\u00a0")}</strong>{" "}
            understands your research intent, scans the OpenAlex database, and
            surfaces only the most valuable results from the Scopus index.
          </p>

          <div
            role="button"
            tabIndex={0}
            aria-label={heroSearchLabel}
            onClick={handleCTA}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleCTA();
              }
            }}
            className={cn(
              "group relative flex w-[min(640px,100%)] cursor-pointer items-center gap-3.5 rounded-full border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-5 py-4 shadow-[0_4px_24px_rgba(16,33,51,0.08),0_0_0_4px_rgba(0,117,149,0.06)] transition-[box-shadow,border-color,transform] duration-200 ease-out",
              "hover:-translate-y-px hover:border-primary hover:shadow-[0_8px_32px_rgba(16,33,51,0.12),0_0_0_4px_rgba(0,117,149,0.1)]",
            )}
          >
            <Search
              size={20}
              className="shrink-0 text-[var(--color-text-soft)]"
              aria-hidden
            />
            <span
              className="min-h-[1.5em] flex-1 overflow-hidden text-left text-base whitespace-nowrap text-[var(--color-text-soft)]"
              aria-hidden
            >
              {typingText}
              <span
                className="ml-px inline-block h-[1.2em] w-0.5 align-text-bottom bg-primary [animation:cursor-blink_1s_step-end_infinite]"
                aria-hidden
              />
            </span>
            <span
              className={cn(
                "grid size-9 shrink-0 place-items-center rounded-full bg-[var(--color-primary)] text-[var(--color-text-inverse)] transition-colors duration-150",
                "group-hover:bg-[var(--color-primary-hover)]",
              )}
              aria-hidden
            >
              <ArrowRight size={16} aria-hidden />
            </span>
          </div>

          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {QUICK_TAGS.map((tag) => (
              <button
                type="button"
                key={tag}
                className={cn(
                  "inline-flex cursor-pointer items-center rounded-full border border-[var(--color-border)] bg-white/60 px-3.5 py-[7px] text-[0.88rem] font-semibold text-[var(--color-text-soft)] transition-[border-color,color,background-color] duration-150",
                  "hover:border-[rgba(0,117,149,0.22)] hover:bg-[rgba(0,117,149,0.06)] hover:text-primary",
                )}
                onClick={handleCTA}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="py-[72px]">
        <div className={pageWrap}>
          <div className="mb-12 text-center">
            <span className={cn(eyebrowLight, eyebrowPrimaryTint)}>
              <Sparkles size={14} />
              How It Works
            </span>
            <h2
              className={cn(
                "mx-0 my-[18px] mb-3.5 text-[clamp(2rem,4.5vw,3.2rem)] leading-none font-normal tracking-[-0.03em] [font-family:var(--font-heading)]",
              )}
            >
              Three steps to smarter research.
            </h2>
            <p
              className={cn(
                "mx-auto max-w-[36rem] text-[1.05rem] leading-[1.72] text-[var(--color-text-soft)]",
              )}
            >
              Our AI-powered pipeline transforms how you discover and evaluate
              academic literature.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-[18px] max-[960px]:grid-cols-1">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className={cn(
                  "relative rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-8 px-7 shadow-[var(--shadow-soft)] backdrop-blur-[14px] transition-[transform,box-shadow,border-color] duration-300 ease-out",
                  "hover:-translate-y-1 hover:border-[var(--color-border-strong)] hover:shadow-[var(--shadow-panel)]",
                )}
              >
                <div
                  className="mb-[18px] grid size-12 place-items-center rounded-[14px] border border-solid"
                  style={{
                    color: f.accent,
                    background: `color-mix(in srgb, ${f.accent} 10%, transparent)`,
                    borderColor: `color-mix(in srgb, ${f.accent} 16%, transparent)`,
                  }}
                >
                  <f.icon size={22} />
                </div>
                <span className="mb-2 block text-[0.78rem] font-extrabold tracking-[0.06em] text-[var(--color-text-soft)] uppercase">
                  {f.eyebrow}
                </span>
                <h3 className="mb-2.5 text-xl leading-tight font-extrabold">
                  {f.title}
                </h3>
                <p className="m-0 text-[0.96rem] leading-[1.72] text-[var(--color-text-soft)]">
                  {f.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-[72px]">
        <div className={pageWrap}>
          <div className="grid grid-cols-2 items-center gap-10 max-[960px]:grid-cols-1 max-[960px]:gap-8">
            <div className="flex flex-col items-start">
              <span className={cn(eyebrowLight, eyebrowPrimaryTint)}>
                <Database size={14} />
                Data Integrity
              </span>
              <h2
                className={cn(
                  "my-[18px] mb-3.5 text-left text-[clamp(2rem,4.5vw,3.2rem)] leading-none font-normal tracking-[-0.03em] [font-family:var(--font-heading)]",
                )}
              >
                Trusted sources, verified data.
              </h2>
              <p
                className={cn(
                  "m-0 max-w-[36rem] text-left text-[1.05rem] leading-[1.72] text-[var(--color-text-soft)]",
                )}
              >
                We draw from the world&apos;s most comprehensive open scholarly
                databases to ensure breadth, and apply Scopus indexing to
                guarantee quality.
              </p>

              <div className="mt-8 flex items-center gap-6 max-[640px]:flex-col max-[640px]:items-start max-[640px]:gap-3.5">
                <div className="flex flex-col gap-1">
                  <span className="text-[1.1rem] font-extrabold tracking-[-0.01em] text-[var(--color-text)] opacity-70">
                    OpenAlex
                  </span>
                  <span className="text-[0.88rem] text-[var(--color-text-soft)]">
                    Open scholarly metadata
                  </span>
                </div>
                <div className="h-9 w-px bg-[var(--color-border-strong)] max-[640px]:h-px max-[640px]:w-10" />
                <div className="flex flex-col gap-1">
                  <span className="text-[1.1rem] font-extrabold tracking-[-0.01em] text-[var(--color-text)] opacity-70">
                    Scopus
                  </span>
                  <span className="text-[0.88rem] text-[var(--color-text-soft)]">
                    Curated journal index
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3.5 max-[640px]:grid-cols-1">
              <div
                className={cn(
                  "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-[var(--shadow-soft)] backdrop-blur-[14px] transition-[transform,box-shadow] duration-[250ms] ease-out",
                  "hover:-translate-y-0.5 hover:shadow-[var(--shadow-panel)]",
                )}
              >
                <div className="mb-3.5 grid size-10 place-items-center rounded-xl bg-[rgba(0,117,149,0.08)] text-primary">
                  <Database size={20} />
                </div>
                <div className="mb-1.5 text-[1.6rem] leading-none font-extrabold tracking-[-0.02em] [font-family:var(--font-heading)]">
                  <AnimatedCounter target={250} suffix="M+" />
                </div>
                <div className="text-[0.9rem] leading-[1.55] text-[var(--color-text-soft)]">
                  Academic works indexed
                </div>
              </div>
              <div
                className={cn(
                  "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-[var(--shadow-soft)] backdrop-blur-[14px] transition-[transform,box-shadow] duration-[250ms] ease-out",
                  "hover:-translate-y-0.5 hover:shadow-[var(--shadow-panel)]",
                )}
              >
                <div className="mb-3.5 grid size-10 place-items-center rounded-xl bg-[rgba(0,117,149,0.08)] text-primary">
                  <BookOpen size={20} />
                </div>
                <div className="mb-1.5 text-[1.6rem] leading-none font-extrabold tracking-[-0.02em] [font-family:var(--font-heading)]">
                  <AnimatedCounter target={48} suffix="K+" />
                </div>
                <div className="text-[0.9rem] leading-[1.55] text-[var(--color-text-soft)]">
                  Scopus-listed journals
                </div>
              </div>
              <div
                className={cn(
                  "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-[var(--shadow-soft)] backdrop-blur-[14px] transition-[transform,box-shadow] duration-[250ms] ease-out",
                  "hover:-translate-y-0.5 hover:shadow-[var(--shadow-panel)]",
                )}
              >
                <div className="mb-3.5 grid size-10 place-items-center rounded-xl bg-[rgba(0,117,149,0.08)] text-primary">
                  <Clock size={20} />
                </div>
                <div className="mb-1.5 text-[1.6rem] leading-none font-extrabold tracking-[-0.02em] [font-family:var(--font-heading)]">
                  {"<"}800ms
                </div>
                <div className="text-[0.9rem] leading-[1.55] text-[var(--color-text-soft)]">
                  Average response time
                </div>
              </div>
              <div
                className={cn(
                  "rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-[var(--shadow-soft)] backdrop-blur-[14px] transition-[transform,box-shadow] duration-[250ms] ease-out",
                  "hover:-translate-y-0.5 hover:shadow-[var(--shadow-panel)]",
                )}
              >
                <div className="mb-3.5 grid size-10 place-items-center rounded-xl bg-[rgba(0,117,149,0.08)] text-primary">
                  <Zap size={20} />
                </div>
                <div className="mb-1.5 text-[1.6rem] leading-none font-extrabold tracking-[-0.02em] [font-family:var(--font-heading)]">
                  <AnimatedCounter target={3} />
                </div>
                <div className="text-[0.9rem] leading-[1.55] text-[var(--color-text-soft)]">
                  AI models in pipeline
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="pt-12 pb-[72px]">
        <div className={pageWrap}>
          <div
            className={cn(
              "relative overflow-hidden rounded-[var(--radius-lg)] px-12 py-16 text-center text-[var(--color-text-inverse)] shadow-[var(--shadow-panel)] max-[960px]:px-6 max-[960px]:py-10",
              "bg-[radial-gradient(circle_at_0%_0%,rgba(0,117,149,0.18),transparent_34%),radial-gradient(circle_at_100%_100%,rgba(250,235,199,0.1),transparent_28%),linear-gradient(160deg,#102133_0%,#0f2431_58%,#183e4b_100%)]",
            )}
          >
            <div
              className="pointer-events-none absolute -top-20 -right-20 size-60 rounded-full border border-white/[0.06] bg-white/[0.03]"
              aria-hidden
            />
            <h2
              className={cn(
                "mx-auto mb-4 max-w-[16ch] text-[clamp(2rem,4.5vw,3rem)] leading-none font-normal tracking-[-0.03em] [font-family:var(--font-heading)]",
              )}
            >
              Start your research journey today.
            </h2>
            <p className="mx-auto mb-8 max-w-[34rem] text-[1.05rem] leading-[1.72] text-[rgba(246,247,249,0.78)]">
              Join researchers who are already discovering higher-quality papers
              faster with AI-powered semantic search and Scopus-verified
              filtering.
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              <button
                type="button"
                className={cn(
                  btnBase,
                  btnPrimary,
                  "min-h-[52px] px-6 text-base",
                )}
                onClick={handleCTA}
              >
                {isLoggedIn ? "Go to Dashboard" : "Get Started — It's Free"}
                <ArrowRight size={16} />
              </button>
              <button
                type="button"
                className={cn(
                  btnBase,
                  btnSecondary,
                  "min-h-[52px] border-white/16 bg-white/10 px-6 text-base text-[var(--color-text-inverse)] hover:bg-white/16",
                )}
                onClick={handleCTA}
              >
                Learn More
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-[var(--color-border)] py-6">
        <div
          className={cn(
            pageWrap,
            "flex items-center justify-between max-[960px]:flex-col max-[960px]:gap-3 max-[960px]:text-center",
          )}
        >
          <AppBrand size="sm" href="/" />
          <p className="m-0 text-[0.88rem] text-[var(--color-text-soft)]">
            © {new Date().getFullYear()} {APP_BRAND_NAME}. Built for
            researchers, by researchers.
          </p>
        </div>
      </footer>
    </div>
  );
}
