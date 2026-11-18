from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any

import httpx

from app.core.config import settings
from app.services.llm_prompts import build_chat_messages
from app.services.llm_prompts import current_prompt_version as prompt_registry_version


SAFE_LLM_FALLBACK_REASONS = {
    "llm_disabled",
    "provider_unavailable",
    "missing_api_key",
    "timeout",
    "http_error",
    "invalid_json",
    "step_disabled",
    "no_candidates",
    "no_abstract",
    "summary_cap_exceeded",
    "insufficient_content",
    "budget_exceeded",
    "input_too_large",
    "max_calls_exceeded",
    "estimated_tokens_exceeded",
}

SAFE_LLM_CACHE_MISS_REASONS = {
    "cache_disabled",
    "no_cached_artifact",
    "prompt_version_changed",
    "query_changed",
    "candidate_changed",
    "malformed_cached_artifact",
}


def safe_fallback_reason(reason: str | None) -> str:
    if reason in SAFE_LLM_FALLBACK_REASONS:
        return reason
    return "provider_unavailable"


def safe_cache_miss_reason(reason: str | None) -> str:
    if reason in SAFE_LLM_CACHE_MISS_REASONS:
        return reason
    return "malformed_cached_artifact"


def fallback_reason_from_exception(exc: Exception) -> str:
    if isinstance(exc, LLMUnavailableError):
        return exc.reason
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "provider_unavailable"


class LLMUnavailableError(RuntimeError):
    """Raised when the optional LLM layer is disabled or cannot be used."""

    def __init__(self, reason: str = "provider_unavailable"):
        self.reason = safe_fallback_reason(reason)
        super().__init__(self.reason)


@dataclass(frozen=True)
class LLMUsageEstimate:
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_call_count: int = 1
    input_truncated: bool = False

    def to_metadata(self) -> dict[str, int | bool]:
        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_call_count": self.estimated_call_count,
            "input_truncated": self.input_truncated,
        }


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def current_prompt_version(settings_obj: Any | None = None) -> str:
    return prompt_registry_version(settings_obj)


def normalize_for_fingerprint(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def _stable_hash(parts: list[str]) -> str:
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def query_fingerprint(query: str | None, topic: str | None = None) -> str:
    return _stable_hash(
        [
            normalize_for_fingerprint(query),
            normalize_for_fingerprint(topic),
        ]
    )


def _value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _raw_metadata(candidate: Any) -> dict:
    metadata = _value(candidate, "raw_metadata_json", _value(candidate, "raw_metadata", {}))
    return metadata if isinstance(metadata, dict) else {}


def candidate_fingerprint(candidate: Any) -> str:
    metadata = _raw_metadata(candidate)
    provider_id = (
        metadata.get("paper_id")
        or metadata.get("paperId")
        or metadata.get("openalex_id")
        or metadata.get("id")
        or metadata.get("doi")
    )
    return _stable_hash(
        [
            normalize_for_fingerprint(_value(candidate, "title")),
            normalize_for_fingerprint(_value(candidate, "doi")),
            normalize_for_fingerprint(_value(candidate, "source_title")),
            normalize_for_fingerprint(_value(candidate, "year")),
            normalize_for_fingerprint(_value(candidate, "provider")),
            normalize_for_fingerprint(provider_id),
        ]
    )


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def estimate_payload_tokens(payload: Any) -> int:
    return estimate_text_tokens(_json_text(payload))


def payload_char_count(payload: Any) -> int:
    return len(_json_text(payload))


def estimate_llm_usage(
    *,
    input_payload: Any,
    output_payload: Any | None = None,
    default_output_tokens: int = 512,
    input_truncated: bool = False,
) -> LLMUsageEstimate:
    estimated_input_tokens = estimate_payload_tokens(input_payload)
    estimated_output_tokens = (
        estimate_payload_tokens(output_payload)
        if output_payload is not None
        else max(int(default_output_tokens), 0)
    )
    return LLMUsageEstimate(
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_tokens=estimated_input_tokens + estimated_output_tokens,
        estimated_call_count=1,
        input_truncated=input_truncated,
    )


def _truncate_strings(value: Any, max_string_chars: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) > max_string_chars:
            return value[:max_string_chars].rstrip(), True
        return value, False
    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            truncated_item, item_changed = _truncate_strings(item, max_string_chars)
            changed = changed or item_changed
            items.append(truncated_item)
        return items, changed
    if isinstance(value, dict):
        changed = False
        data = {}
        for key, item in value.items():
            truncated_item, item_changed = _truncate_strings(item, max_string_chars)
            changed = changed or item_changed
            data[key] = truncated_item
        return data, changed
    return value, False


def compact_payload_for_llm(payload: dict[str, Any], max_chars: int | None = None) -> tuple[dict[str, Any], bool]:
    limit = max(int(max_chars or getattr(settings, "llm_max_input_chars_per_call", 12000)), 1)
    if payload_char_count(payload) <= limit:
        return payload, False

    for max_string_chars in (4000, 2000, 1000, 500, 250):
        compacted, truncated = _truncate_strings(payload, max_string_chars)
        if isinstance(compacted, dict) and payload_char_count(compacted) <= limit:
            return compacted, truncated

    raise LLMUnavailableError("input_too_large")


@dataclass(frozen=True)
class LLMGeneration:
    provider: str
    model: str
    task: str
    content: dict[str, Any]
    used_llm: bool
    prompt_version: str | None = None
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_call_count: int = 1
    input_truncated: bool = False


class LLMClient:
    """Small boundary for optional structured LLM calls.

    The production provider integration is intentionally thin. Phase 2 services
    can be tested through the deterministic mock provider without requiring a
    real API key or adding a framework dependency.
    """

    def __init__(self, settings_obj=settings, http_client: httpx.Client | None = None):
        self.settings = settings_obj
        self._http_client = http_client

    @property
    def provider(self) -> str:
        return str(getattr(self.settings, "llm_provider", "mock") or "mock").strip().lower()

    @property
    def model(self) -> str:
        return str(getattr(self.settings, "llm_model", "mock-literature-review-v1") or "")

    @property
    def prompt_version(self) -> str:
        return current_prompt_version(self.settings)

    def is_enabled(self) -> bool:
        return bool(getattr(self.settings, "llm_enabled", False))

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        if not self.is_enabled():
            return "llm_disabled"
        if self.provider == "mock":
            return None
        if self.provider == "openai_compatible":
            if not getattr(self.settings, "llm_api_key", None):
                return "missing_api_key"
            return None
        return "provider_unavailable"

    def generate_structured_text(self, task: str, payload: dict[str, Any]) -> LLMGeneration:
        availability_reason = self.availability_reason()
        if availability_reason:
            raise LLMUnavailableError(availability_reason)

        payload, input_truncated = compact_payload_for_llm(
            payload,
            max_chars=getattr(self.settings, "llm_max_input_chars_per_call", 12000),
        )

        if self.provider == "mock":
            content = {
                "task": task,
                "payload": payload,
                "mock": True,
            }
            usage = estimate_llm_usage(
                input_payload=payload,
                output_payload=content,
                input_truncated=input_truncated,
            )
            return LLMGeneration(
                provider=self.provider,
                model=self.model,
                task=task,
                content=content,
                used_llm=True,
                prompt_version=self.prompt_version,
                **usage.to_metadata(),
            )

        if self.provider == "openai_compatible":
            return self._generate_openai_compatible(
                task=task,
                payload=payload,
                input_truncated=input_truncated,
            )

        raise LLMUnavailableError("provider_unavailable")

    def _generate_openai_compatible(
        self,
        task: str,
        payload: dict[str, Any],
        *,
        input_truncated: bool = False,
    ) -> LLMGeneration:
        base_url = str(getattr(self.settings, "llm_base_url", "") or "").rstrip("/")
        api_key = str(getattr(self.settings, "llm_api_key", "") or "")
        timeout = float(getattr(self.settings, "llm_timeout_seconds", 8.0) or 8.0)
        temperature = float(getattr(self.settings, "llm_temperature", 0.0) or 0.0)

        if not base_url:
            raise LLMUnavailableError("provider_unavailable")

        body = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": build_chat_messages(task=task, payload=payload, settings_obj=self.settings),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        close_client = self._http_client is None
        client = self._http_client or httpx.Client(timeout=timeout)
        try:
            response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            parsed_response = response.json()
            content_text = (
                parsed_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            content = self._parse_json_content(content_text)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("timeout") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError("http_error") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("http_error") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError("invalid_json") from exc
        finally:
            if close_client:
                client.close()

        usage = estimate_llm_usage(
            input_payload=payload,
            output_payload=content,
            input_truncated=input_truncated,
        )
        return LLMGeneration(
            provider=self.provider,
            model=self.model,
            task=task,
            content=content,
            used_llm=True,
            prompt_version=self.prompt_version,
            **usage.to_metadata(),
        )

    @staticmethod
    def _parse_json_content(content_text: str) -> dict[str, Any]:
        if not isinstance(content_text, str) or not content_text.strip():
            raise ValueError("empty model content")

        cleaned = content_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("model content JSON must be an object")
        return parsed


def get_llm_client() -> LLMClient:
    return LLMClient()
