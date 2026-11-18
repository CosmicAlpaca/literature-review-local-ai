from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class PromptInjectionSignal:
    suspicious: bool
    categories: list[str]
    matched_patterns_count: int


@dataclass(frozen=True)
class PromptInjectionScanResult:
    suspicious: bool
    categories: list[str]
    fields_flagged: list[str]
    matched_patterns_count: int


_PROMPT_INJECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "instruction_override": (
        re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
        re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
        re.compile(r"\bforget\s+all\s+instructions\b", re.I),
        re.compile(r"\boverride\s+(?:the\s+)?instructions\b", re.I),
    ),
    "secret_exfiltration": (
        re.compile(r"\breveal\s+(?:the\s+)?system\s+prompt\b", re.I),
        re.compile(r"\bshow\s+(?:the\s+)?system\s+prompt\b", re.I),
        re.compile(r"\bshow\s+(?:me\s+)?(?:your\s+)?api\s+key\b", re.I),
        re.compile(r"\breveal\s+(?:the\s+)?api\s+key\b", re.I),
        re.compile(r"\bprint\s+secrets?\b", re.I),
        re.compile(r"\benvironment\s+variables\b", re.I),
    ),
    "role_manipulation": (
        re.compile(r"\byou\s+are\s+now\b", re.I),
        re.compile(r"\bdeveloper\s+message\b", re.I),
        re.compile(r"\bsystem\s+message\b", re.I),
        re.compile(r"\bact\s+as\b", re.I),
    ),
    "schema_bypass": (
        re.compile(r"\bdo\s+not\s+return\s+json\b", re.I),
        re.compile(r"\bignore\s+(?:the\s+)?schema\b", re.I),
        re.compile(r"\boutput\s+(?:the\s+)?raw\s+prompt\b", re.I),
        re.compile(r"\braw\s+prompt\b", re.I),
    ),
    "jailbreak": (
        re.compile(r"\bjailbreak\b", re.I),
        re.compile(r"\bdo\s+anything\s+now\b", re.I),
        re.compile(r"\bDAN\b"),
    ),
    "scoring_manipulation": (
        re.compile(r"\bmark\s+this\s+paper\s+as\s+highly\s+relevant\b", re.I),
        re.compile(r"\bgive\s+this\s+paper\s+maximum\s+score\b", re.I),
        re.compile(r"\balways\s+rank\s+this\s+first\b", re.I),
        re.compile(r"\brank\s+(?:it|this)\s+first\b", re.I),
    ),
}

_PUBLIC_REDACTION_PATTERNS = tuple(
    pattern
    for patterns in _PROMPT_INJECTION_PATTERNS.values()
    for pattern in patterns
) + (
    re.compile(r"\bapi[_\s-]?key\b", re.I),
    re.compile(r"\bstack\s+trace\b", re.I),
    re.compile(r"\btraceback\b", re.I),
    re.compile(r"\braw\s+model\s+response\b", re.I),
    re.compile(r"\b_llm_diagnostics\b", re.I),
    re.compile(r"\b_ranking\b", re.I),
    re.compile(r"\b_trust\b", re.I),
)


def detect_prompt_injection(text: str | None) -> PromptInjectionSignal:
    if not isinstance(text, str) or not text.strip():
        return PromptInjectionSignal(
            suspicious=False,
            categories=[],
            matched_patterns_count=0,
        )

    categories: list[str] = []
    matched_count = 0
    for category, patterns in _PROMPT_INJECTION_PATTERNS.items():
        category_matched = False
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                matched_count += len(matches)
                category_matched = True
        if category_matched:
            categories.append(category)

    return PromptInjectionSignal(
        suspicious=bool(categories),
        categories=categories,
        matched_patterns_count=matched_count,
    )


def scan_untrusted_payload(payload: Any) -> PromptInjectionScanResult:
    categories: list[str] = []
    fields_flagged: list[str] = []
    matched_count = 0

    def _scan(value: Any, path: str) -> None:
        nonlocal matched_count
        if isinstance(value, str):
            signal = detect_prompt_injection(value)
            if signal.suspicious:
                fields_flagged.append(path)
                matched_count += signal.matched_patterns_count
                for category in signal.categories:
                    if category not in categories:
                        categories.append(category)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                safe_key = str(key)
                child_path = f"{path}.{safe_key}" if path else safe_key
                _scan(item, child_path)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                _scan(item, f"{path}[{index}]")

    _scan(payload, "")
    return PromptInjectionScanResult(
        suspicious=bool(fields_flagged),
        categories=categories,
        fields_flagged=fields_flagged[:20],
        matched_patterns_count=matched_count,
    )


def truncate_for_llm(text: str, max_chars: int) -> tuple[str, bool]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    limit = max(int(max_chars), 0)
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def sanitize_public_text(text: str | None, *, replacement: str = "[redacted unsafe instruction]") -> str:
    if not isinstance(text, str):
        return ""
    sanitized = text
    for pattern in _PUBLIC_REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return " ".join(sanitized.split())


def sanitize_public_text_list(values: list[str], *, limit: int = 8) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = sanitize_public_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:limit]
