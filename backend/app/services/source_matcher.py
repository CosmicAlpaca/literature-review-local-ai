from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.scopus_source import ScopusSource
from app.services.source_registry import (
    RegistryEntry,
    _normalize_publisher,
    get_primary_registry_workbook_path,
    load_registry_workbook,
)
from app.utils.normalize import normalize_text


@dataclass
class SourceMatchResult:
    source: ScopusSource | None
    confidence: float
    reason: str
    matched_value: str | None = None
    normalized_input_title: str | None = None
    input_identifiers: dict[str, str] | None = None
    registry_metadata: dict[str, Any] | None = None


def _normalize_issn(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_issn(item)
            if normalized:
                return normalized
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    elif not isinstance(value, str):
        return ""
    normalized = re.sub(r"[^0-9xX]", "", value).upper()
    return normalized


def _normalize_all_issns(value: Any) -> list[str]:
    """Extract ALL ISSNs from a value (string, list, nested)."""
    results: list[str] = []
    if not value:
        return results
    if isinstance(value, (list, tuple)):
        for item in value:
            for issn in _normalize_all_issns(item):
                if issn not in results:
                    results.append(issn)
        return results
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    elif not isinstance(value, str):
        return results
    normalized = re.sub(r"[^0-9xX]", "", value).upper()
    if normalized and normalized not in results:
        results.append(normalized)
    return results


def _iter_identifier_values(value: Any):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_identifier_values(item)
        return
    yield value


def _strip_leading_article(text: str) -> str:
    tokens = text.split()
    if tokens[:1] and tokens[0] in {"the", "a", "an"}:
        return " ".join(tokens[1:])
    return text


def _singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _singularize_title(text: str) -> str:
    return " ".join(_singularize_token(token) for token in text.split())


def _extract_identifiers(raw_metadata: dict[str, Any] | None) -> dict[str, str]:
    """Extract ALL identifier variants from raw_metadata for matching."""
    if not isinstance(raw_metadata, dict):
        return {}

    identifiers: dict[str, str] = {}
    seen_values: set[str] = set()

    def add_identifier(key: str, raw_value: Any) -> None:
        all_issns = _normalize_all_issns(raw_value)
        for issn in all_issns:
            if not issn or issn in seen_values:
                continue
            seen_values.add(issn)
            identifier_key = key
            index = 2
            while identifier_key in identifiers:
                identifier_key = f"{key}_{index}"
                index += 1
            identifiers[identifier_key] = issn

    # Standard keys
    for key in ("issn", "issn_l", "eissn", "print_issn", "electronic_issn"):
        add_identifier(key, raw_metadata.get(key))

    # External IDs block
    external_ids = raw_metadata.get("external_ids")
    if isinstance(external_ids, dict):
        for key in ("issn", "issn_l", "eissn"):
            add_identifier(f"external_{key}", external_ids.get(key))

    return identifiers


def _token_set(text: str) -> set[str]:
    return {token for token in text.split() if token}


def _fuzzy_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0

    sequence_ratio = SequenceMatcher(None, left, right).ratio()
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union if union else 0.0
    containment = intersection / min(len(left_tokens), len(right_tokens))

    return round((0.5 * sequence_ratio) + (0.3 * jaccard) + (0.2 * containment), 4)


def _category_adjustment(metadata: dict[str, Any] | None) -> float:
    if not isinstance(metadata, dict):
        return 1.0
    category = str(metadata.get("registry_category") or "").strip().lower()
    if category == "active_indexed_journal":
        return 1.0
    if category == "accepted_title":
        return 0.80  # Reduced penalty for accepted
    if category == "discontinued_title":
        return 0.20  # Increased penalty for discontinued
    if category == "conference_proceeding":
        return 0.65
    if category == "non_journal_source":
        return 0.62
    if category == "inactive_indexed_source":
        return 0.50
    return 0.75


def _medline_bonus(metadata: dict[str, Any] | None) -> float:
    if isinstance(metadata, dict) and metadata.get("medline_sourced"):
        return 0.03
    return 0.0


def _finalize_confidence(base_confidence: float, metadata: dict[str, Any] | None) -> float:
    adjusted = (base_confidence * _category_adjustment(metadata)) + _medline_bonus(metadata)
    return round(max(0.0, min(adjusted, 1.0)), 4)


def _registry_metadata(entry: RegistryEntry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "registry_category": entry.registry_category,
        "active_status": entry.active_status,
        "source_type": entry.source_type,
        "is_accepted": entry.is_accepted,
        "is_discontinued": entry.is_discontinued,
        "discontinued_due_to_quality": entry.discontinued_due_to_quality,
        "medline_sourced": entry.medline_sourced,
        "publisher": entry.publisher,
        "main_publisher": entry.main_publisher,
        "asjc_codes": entry.asjc_codes,
        "top_level_fields": list(entry.top_level_fields),
        "related_titles": list(entry.related_titles),
        "open_access_status": entry.open_access_status,
        "coverage": entry.coverage,
        "source_record_id": entry.source_record_id,
        "issn": entry.issn,
        "eissn": entry.eissn,
    }


def _build_registry_maps(
    sources: list[ScopusSource],
) -> tuple[dict[str, ScopusSource], dict[str, ScopusSource], dict[str, ScopusSource]]:
    by_record_id: dict[str, ScopusSource] = {}
    by_title: dict[str, ScopusSource] = {}
    by_identifier: dict[str, ScopusSource] = {}
    for source in sources:
        if source.source_record_id:
            by_record_id[str(source.source_record_id)] = source
        if source.normalized_source_title:
            by_title[source.normalized_source_title] = source
        for identifier in (_normalize_issn(source.issn), _normalize_issn(source.eissn)):
            if identifier:
                by_identifier[identifier] = source
    return by_record_id, by_title, by_identifier


def _match_db_source_for_entry(
    entry: RegistryEntry,
    by_record_id: dict[str, ScopusSource],
    by_title: dict[str, ScopusSource],
    by_identifier: dict[str, ScopusSource],
) -> ScopusSource | None:
    if entry.source_record_id and entry.source_record_id in by_record_id:
        return by_record_id[entry.source_record_id]
    for identifier in (entry.issn, entry.eissn):
        normalized = _normalize_issn(identifier)
        if normalized and normalized in by_identifier:
            return by_identifier[normalized]
    return by_title.get(entry.normalized_source_title)


def _publisher_match_bonus(
    raw_metadata: dict[str, Any] | None,
    source: ScopusSource,
) -> float:
    """Return a bonus score if the paper's publisher matches the source's publisher."""
    if not isinstance(raw_metadata, dict):
        return 0.0
    input_pub = _normalize_publisher(raw_metadata.get("publisher") or "")
    source_pub = _normalize_publisher(source.publisher or "")
    if not input_pub or not source_pub:
        return 0.0
    # Exact normalized match
    if input_pub == source_pub:
        return 0.10
    # Containment (one is substring of the other)
    if input_pub in source_pub or source_pub in input_pub:
        return 0.06
    return 0.0


def resolve_scopus_match(
    db: Session,
    source_title: str | None,
    raw_metadata: dict[str, Any] | None = None,
) -> SourceMatchResult:
    normalized = normalize_text(source_title)
    identifiers = _extract_identifiers(raw_metadata)

    if not normalized and not identifiers:
        return SourceMatchResult(
            source=None,
            confidence=0.0,
            reason="no_source_metadata",
            normalized_input_title=normalized or None,
            input_identifiers=identifiers or None,
            registry_metadata=None,
        )

    sources = db.query(ScopusSource).all()
    by_record_id, by_title, by_identifier = _build_registry_maps(sources)
    try:
        registry = load_registry_workbook(str(get_primary_registry_workbook_path()))
    except Exception:
        registry = None

    def registry_for_source(source: ScopusSource | None) -> RegistryEntry | None:
        if source is None or registry is None:
            return None
        if source.source_record_id and source.source_record_id in registry.by_record_id:
            return registry.by_record_id[source.source_record_id]
        matches = registry.by_normalized_title.get(source.normalized_source_title or "", ())
        return matches[0] if matches else None

    # ---- STEP 1: Match by all identifiers (ISSN, EISSN, ISSN-L) ----
    for source in sources:
        source_issn = _normalize_issn(source.issn)
        source_eissn = _normalize_issn(source.eissn)
        source_ids = {v for v in (source_issn, source_eissn) if v}
        for input_key, input_value in identifiers.items():
            if input_value and input_value in source_ids:
                registry_entry = registry_for_source(source)
                return SourceMatchResult(
                    source=source,
                    confidence=_finalize_confidence(1.0, _registry_metadata(registry_entry)),
                    reason=f"identifier_match:{input_key}",
                    matched_value=input_value,
                    normalized_input_title=normalized or None,
                    input_identifiers=identifiers,
                    registry_metadata=_registry_metadata(registry_entry),
                )

    # ---- STEP 1b: Match identifiers against registry by_all_identifiers ----
    if registry is not None:
        for input_key, input_value in identifiers.items():
            if input_value and input_value in registry.by_all_identifiers:
                for entry in registry.by_all_identifiers[input_value]:
                    db_source = _match_db_source_for_entry(
                        entry, by_record_id, by_title, by_identifier,
                    )
                    if db_source:
                        metadata = _registry_metadata(entry)
                        return SourceMatchResult(
                            source=db_source,
                            confidence=_finalize_confidence(0.98, metadata),
                            reason=f"registry_identifier_match:{input_key}",
                            matched_value=input_value,
                            normalized_input_title=normalized or None,
                            input_identifiers=identifiers,
                            registry_metadata=metadata,
                        )

    # ---- STEP 2: Exact normalized title match ----
    if normalized:
        for source in sources:
            if source.normalized_source_title == normalized:
                registry_entry = registry_for_source(source)
                return SourceMatchResult(
                    source=source,
                    confidence=_finalize_confidence(0.99, _registry_metadata(registry_entry)),
                    reason="exact_normalized_title",
                    matched_value=source.source_title,
                    normalized_input_title=normalized,
                    input_identifiers=identifiers or None,
                    registry_metadata=_registry_metadata(registry_entry),
                )

        # ---- STEP 3: Strip leading article ----
        stripped = _strip_leading_article(normalized)
        if stripped and stripped != normalized:
            for source in sources:
                if _strip_leading_article(source.normalized_source_title) == stripped:
                    registry_entry = registry_for_source(source)
                    return SourceMatchResult(
                        source=source,
                        confidence=_finalize_confidence(0.96, _registry_metadata(registry_entry)),
                        reason="exact_title_ignoring_leading_article",
                        matched_value=source.source_title,
                        normalized_input_title=normalized,
                        input_identifiers=identifiers or None,
                        registry_metadata=_registry_metadata(registry_entry),
                    )

        # ---- STEP 4: Related titles from registry ----
        if registry is not None and normalized in registry.by_related_title:
            for entry in registry.by_related_title[normalized]:
                db_source = _match_db_source_for_entry(
                    entry,
                    by_record_id,
                    by_title,
                    by_identifier,
                )
                if db_source:
                    metadata = _registry_metadata(entry)
                    return SourceMatchResult(
                        source=db_source,
                        confidence=_finalize_confidence(0.9, metadata),
                        reason="related_title_match",
                        matched_value=entry.source_title,
                        normalized_input_title=normalized,
                        input_identifiers=identifiers or None,
                        registry_metadata=metadata,
                    )

        # ---- STEP 5: Light singularization ----
        singularized = _singularize_title(normalized)
        if singularized:
            for source in sources:
                source_singularized = _singularize_title(source.normalized_source_title)
                if source_singularized == singularized and (
                    source.normalized_source_title != normalized
                    or source_singularized != source.normalized_source_title
                ):
                    registry_entry = registry_for_source(source)
                    return SourceMatchResult(
                        source=source,
                        confidence=_finalize_confidence(0.94, _registry_metadata(registry_entry)),
                        reason="exact_title_after_light_singularization",
                        matched_value=source.source_title,
                        normalized_input_title=normalized,
                        input_identifiers=identifiers or None,
                        registry_metadata=_registry_metadata(registry_entry),
                    )

        # ---- STEP 6: Fuzzy title match with publisher bonus ----
        best_source: ScopusSource | None = None
        best_entry: RegistryEntry | None = None
        best_score = 0.0
        for source in sources:
            score = _fuzzy_similarity(normalized, source.normalized_source_title)
            # Publisher-aware matching bonus
            score += _publisher_match_bonus(raw_metadata, source)

            if score > best_score:
                best_score = score
                best_source = source
                best_entry = registry_for_source(source)

        if best_source and best_score >= 0.78:  # Tuned fuzzy threshold
            metadata = _registry_metadata(best_entry)
            return SourceMatchResult(
                source=best_source,
                confidence=_finalize_confidence(min(best_score, 1.0), metadata),
                reason="high_confidence_fuzzy_title",
                matched_value=best_source.source_title,
                normalized_input_title=normalized,
                input_identifiers=identifiers or None,
                registry_metadata=metadata,
            )

    return SourceMatchResult(
        source=None,
        confidence=0.0,
        reason="no_match",
        normalized_input_title=normalized or None,
        input_identifiers=identifiers or None,
        registry_metadata=None,
    )


def match_scopus_source(
    db: Session,
    source_title: str | None,
    raw_metadata: dict[str, Any] | None = None,
) -> ScopusSource | None:
    return resolve_scopus_match(db=db, source_title=source_title, raw_metadata=raw_metadata).source
