type SortField = "final_score" | "year" | "citation_count" | "llm_relevance_score";
type SortDir = "desc" | "asc";
type StatusFilter = "all" | "active" | "shortlisted";

type Props = {
  sortField: SortField;
  sortDir: SortDir;
  scopusOnly: boolean;
  statusFilter: StatusFilter;
  onSortField: (field: SortField) => void;
  onSortDir: (dir: SortDir) => void;
  onScopusOnly: (value: boolean) => void;
  onStatusFilter: (status: StatusFilter) => void;
  totalCount: number;
  filteredCount: number;
};

const SORT_FIELD_LABELS: Record<SortField, string> = {
  final_score: "Score",
  year: "Year",
  citation_count: "Citations",
  llm_relevance_score: "AI relevance",
};

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "active", label: "Active" },
  { value: "shortlisted", label: "Shortlisted" },
];

export type { SortField, SortDir, StatusFilter };

export default function CandidateFilterBar({
  sortField,
  sortDir,
  scopusOnly,
  statusFilter,
  onSortField,
  onSortDir,
  onScopusOnly,
  onStatusFilter,
  totalCount,
  filteredCount,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3 px-4 py-3 mb-3.5 rounded-xl border border-black/8 bg-black/3">

      {/* Sort field + direction */}
      <div className="flex items-center gap-2">
        <span className="text-[0.75rem] font-bold uppercase tracking-widest text-(--color-text-soft) whitespace-nowrap">
          Sort
        </span>
        <select
          value={sortField}
          onChange={(e) => onSortField(e.target.value as SortField)}
          aria-label="Sort by"
          className="border border-black/10 rounded-lg bg-white/90 text-(--color-text) px-2.5 py-1.5 text-[0.875rem] font-bold outline-none cursor-pointer focus:border-(--color-primary)/50 focus:ring-2 focus:ring-(--color-primary)/10"
        >
          {(Object.keys(SORT_FIELD_LABELS) as SortField[]).map((field) => (
            <option key={field} value={field}>
              {SORT_FIELD_LABELS[field]}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => onSortDir(sortDir === "desc" ? "asc" : "desc")}
          aria-label={sortDir === "desc" ? "Currently descending, click for ascending" : "Currently ascending, click for descending"}
          className="border border-black/10 rounded-lg bg-white/90 text-(--color-text) px-2.5 py-1.5 text-[0.875rem] font-bold cursor-pointer whitespace-nowrap hover:bg-(--color-surface-strong) hover:border-black/18 transition-colors duration-100"
        >
          {sortDir === "desc" ? "↓ Desc" : "↑ Asc"}
        </button>
      </div>

      {/* Status filter chips */}
      {/* <div className="flex items-center gap-2">
        <span className="text-[0.75rem] font-bold uppercase tracking-widest text-(--color-text-soft) whitespace-nowrap">
          Status
        </span>
        <div className="flex items-center gap-1.5">
          {STATUS_OPTIONS.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => onStatusFilter(value)}
              className={[
                "border rounded-full px-3 py-1 text-[0.875rem] font-bold cursor-pointer transition-colors duration-100",
                statusFilter === value
                  ? "bg-(--color-primary)/10 border-(--color-primary)/20 text-(--color-primary)"
                  : "bg-white/72 border-black/10 text-(--color-text-soft) hover:bg-white hover:border-black/16",
              ].join(" ")}
            >
              {label}
            </button>
          ))}
        </div>
      </div> */}

      {/* Scopus toggle */}
      <label className="flex items-center gap-1.5 text-[0.875rem] font-bold text-(--color-text) cursor-pointer select-none">
        <input
          type="checkbox"
          checked={scopusOnly}
          onChange={(e) => onScopusOnly(e.target.checked)}
          className="w-4 h-4 cursor-pointer accent-(--color-primary)"
        />
        Scopus matched only
      </label>

      {/* Result count */}
      <span className="ml-auto text-[0.875rem] font-bold text-(--color-text-soft) whitespace-nowrap">
        {filteredCount === totalCount
          ? `${totalCount} candidates`
          : `${filteredCount} of ${totalCount}`}
      </span>
    </div>
  );
}
