"use client";

import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { CalendarIcon, ChevronDownIcon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { normalizeRawQuery } from "@/lib/session";
import { cn } from "@/lib/utils";

const CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = Array.from({ length: 30 }, (_, i) => CURRENT_YEAR - i);

const topics = [
  "Engineering",
  "Medicine",
  "Social Sciences",
  "Arts and Humanities",
  "Computer Science",
  "Biochemistry, Genetics and Molecular Biology",
  "Agricultural and Biological Sciences",
  "Environmental Science",
  "Economics, Econometrics and Finance",
  "Physics and Astronomy",
];

const schema = z
  .object({
    raw_query: z.string().min(1, "Research query is required"),
    topic: z.string().optional(),
    year_from: z.number().nullable().optional(),
    year_to: z.number().nullable().optional(),
    open_access: z.boolean(),
  })
  .refine(
    (d) => {
      if (d.year_from != null && d.year_to != null) {
        return d.year_from <= d.year_to;
      }
      return true;
    },
    { message: "Start year must not exceed end year", path: ["year_from"] },
  );

type FormValues = z.infer<typeof schema>;

export type SearchPayload = {
  topic: string;
  raw_query: string;
  filters_json: Record<string, unknown>;
};

type Props = {
  onSubmit: (payload: SearchPayload) => void;
  submitting?: boolean;
};

function yearRangeLabel(
  from: number | null | undefined,
  to: number | null | undefined,
): string {
  if (!from && !to) return "Any year";
  if (from && to) return `${from} – ${to}`;
  if (from) return `From ${from}`;
  return `Until ${to}`;
}

export default function SearchForm({ onSubmit, submitting = false }: Props) {
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    control,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      raw_query: "",
      topic: "",
      year_from: null,
      year_to: null,
      open_access: false,
    },
  });

  const openAccess = watch("open_access");
  const yearFrom = watch("year_from");
  const yearTo = watch("year_to");
  const hasYearRange = yearFrom != null || yearTo != null;

  function applyPreset(years: number | null) {
    if (years === null) {
      setValue("year_from", null);
      setValue("year_to", null);
    } else {
      setValue("year_from", CURRENT_YEAR - years + 1);
      setValue("year_to", CURRENT_YEAR);
    }
  }

  function isActivePreset(years: number) {
    return yearFrom === CURRENT_YEAR - years + 1 && yearTo === CURRENT_YEAR;
  }

  function onValid(values: FormValues) {
    const trimmedTopic = (values.topic ?? "").trim();
    const trimmedQuery = normalizeRawQuery(values.raw_query);

    onSubmit({
      topic: trimmedTopic,
      raw_query: trimmedQuery,
      filters_json: {
        pages: 3,
        only_active_scopus: true,
        year_from: values.year_from ?? null,
        year_to: values.year_to ?? null,
        open_access: values.open_access,
      },
    });
  }

  return (
    <form onSubmit={handleSubmit(onValid)} className="grid gap-3">
      {/* Research query */}
      <div className="grid gap-1.5">
        <Textarea
          id="raw_query"
          rows={5}
          placeholder="Describe your research question, inclusion focus, or search objective…"
          className="resize-none text-sm leading-relaxed"
          {...register("raw_query", {
            onBlur: (e) => {
              const normalized = normalizeRawQuery(e.target.value);
              if (normalized !== e.target.value) {
                setValue("raw_query", normalized, { shouldValidate: true });
              }
            },
          })}
        />
        {errors.raw_query && (
          <p className="text-xs text-destructive">{errors.raw_query.message}</p>
        )}
      </div>

      {/* Filter bar — one line on large screens */}
      <div className="flex flex-wrap items-center gap-2 lg:flex-nowrap">
        {/* Topic */}
        <Controller
          name="topic"
          control={control}
          render={({ field }) => (
            <Select value={field.value ?? ""} onValueChange={field.onChange}>
              <SelectTrigger
                className={cn(
                  "h-9 flex-1 min-w-40 text-sm text-muted-foreground hover:text-foreground",
                  field.value
                    ? "border-primary bg-background text-primary"
                    : "",
                )}
                xIcon={!!field.value}
                xIconClassName="text-muted-foreground"
                xIconOnClick={() => field.onChange("")}
              >
                <SelectValue
                  placeholder={
                    <span className="text-inherit">Topic (any)</span>
                  }
                  customValueItem={
                    <span>
                      {field.value && field.value.length > 20
                        ? String(field.value).substring(0, 20) + "..."
                        : field.value}
                    </span>
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {topics.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        />

        {/* Year range picker */}
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={cn(
                "h-8 inline-flex flex-1 min-w-36 items-center justify-between gap-1.5 rounded-md border px-3 text-sm transition-colors",
                hasYearRange
                  ? "border-primary bg-background text-primary"
                  : "border-input bg-background text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="flex items-center gap-1.5">
                <CalendarIcon className="size-4 shrink-0" />
                {yearRangeLabel(yearFrom, yearTo)}
              </span>
              <div className="flex items-center gap-1.5">
                {hasYearRange && (
                  <XIcon
                    className="size-4 shrink-0 text-muted-foreground"
                    onClick={(e) => {
                      setValue("year_from", null);
                      setValue("year_to", null);
                      e.stopPropagation();
                      e.preventDefault();
                    }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                    }}
                  />
                )}
                <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground" />
              </div>
            </button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-60 p-3 grid gap-3">
            {/* Quick presets */}
            <div className="flex flex-wrap gap-1.5">
              {(
                [
                  { label: "Any", years: null },
                  { label: "Last 3 yrs", years: 3 },
                  { label: "Last 5 yrs", years: 5 },
                ] as const
              ).map(({ label, years }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => applyPreset(years)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                    (years === null && !hasYearRange) ||
                      (years !== null && isActivePreset(years))
                      ? "border-primary bg-background text-primary"
                      : "border-border text-muted-foreground hover:border-primary/20 hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>

            <Separator />

            {/* Custom from / to */}
            <div className="grid grid-cols-2 gap-2">
              <div className="grid gap-1">
                <span className="text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground">
                  From
                </span>
                <Controller
                  name="year_from"
                  control={control}
                  render={({ field }) => (
                    <Select
                      value={field.value != null ? String(field.value) : "any"}
                      onValueChange={(v) =>
                        field.onChange(v === "any" ? null : Number(v))
                      }
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue placeholder="Any" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        {YEAR_OPTIONS.map((y) => (
                          <SelectItem key={y} value={String(y)}>
                            {y}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
              <div className="grid gap-1">
                <span className="text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground">
                  To
                </span>
                <Controller
                  name="year_to"
                  control={control}
                  render={({ field }) => (
                    <Select
                      value={field.value != null ? String(field.value) : "any"}
                      onValueChange={(v) =>
                        field.onChange(v === "any" ? null : Number(v))
                      }
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue placeholder="Any" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="any">Any</SelectItem>
                        {YEAR_OPTIONS.map((y) => (
                          <SelectItem key={y} value={String(y)}>
                            {y}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>
        {errors.year_from && (
          <p className="text-xs text-destructive">
            {errors.year_from.message}
          </p>
        )}

        {/* Open access toggle */}
        <button
          type="button"
          onClick={() => setValue("open_access", !openAccess)}
          className={cn(
            "h-8 inline-flex items-center whitespace-nowrap rounded-md border px-3 text-sm transition-colors",
            openAccess
              ? "border-primary bg-background text-primary"
              : "border-input bg-background text-muted-foreground hover:text-foreground",
          )}
        >
          Open access only
        </button>
      </div>

      <Button
        type="submit"
        disabled={submitting}
        className="mt-1 w-full sm:w-auto"
      >
        {submitting ? "Creating session…" : "Create review session →"}
      </Button>
    </form>
  );
}
