from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.llm_prompts import (
    PAPER_SUMMARY_TASK,
    PROMPT_TEMPLATES,
    QUERY_REWRITE_TASK,
    RERANK_TASK,
    REVIEW_DEBATE_TASK,
    REVIEW_DRAFT_TASK,
    REVIEW_OUTLINE_TASK,
    SECURITY_SYSTEM_INSTRUCTION,
    THEME_GROUPING_TASK,
    build_chat_messages,
    current_prompt_version,
)


def _prompt_text(task: str) -> str:
    messages = build_chat_messages(
        task=task,
        payload={"raw_query": "flood detection", "candidates": [{"title": "Flood mapping"}]},
        settings_obj=SimpleNamespace(llm_prompt_version="prompt-test-v2"),
    )
    return "\n".join(message["content"] for message in messages)


def test_query_rewrite_prompt_includes_contract_fields_and_version():
    text = _prompt_text(QUERY_REWRITE_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "optimized_query" in text
    assert "expanded_queries" in text
    assert "key_terms" in text
    assert "notes" in text
    assert "non-English" in text
    assert "English academic search query" in text
    assert "UNTRUSTED_USER_QUERY" in text


def test_rerank_prompt_includes_contract_fields_and_version():
    text = _prompt_text(RERANK_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "candidate_index" in text
    assert "candidate_id" in text
    assert "relevance_score" in text
    assert "reason" in text
    assert "UNTRUSTED_CANDIDATE_METADATA" in text
    assert "UNTRUSTED_PAPER_ABSTRACT" in text


def test_summary_prompt_includes_contract_fields_and_version():
    text = _prompt_text(PAPER_SUMMARY_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "one_sentence_summary" in text
    assert "contribution" in text
    assert "method_or_data" in text
    assert "relevance_to_query" in text
    assert "limitations_or_caution" in text
    assert "UNTRUSTED_PAPER_TITLE" in text
    assert "UNTRUSTED_PAPER_ABSTRACT" in text


def test_theme_grouping_prompt_includes_contract_fields_and_version():
    text = _prompt_text(THEME_GROUPING_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "themes" in text
    assert "theme_id" in text
    assert "name" in text
    assert "rationale" in text
    assert "keywords" in text
    assert "supporting_candidate_ids" in text


def test_review_draft_prompt_includes_contract_fields_and_version():
    text = _prompt_text(REVIEW_DRAFT_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "draft" in text
    assert "sections" in text
    assert "section_id" in text
    assert "content" in text
    assert "supporting_candidate_ids" in text
    assert "citation_placeholders" in text
    assert "key_claims" in text
    assert "limitations_or_cautions" in text
    assert "UNTRUSTED_EVIDENCE_PACK" in text


def test_debate_prompt_includes_security_and_untrusted_labels():
    text = _prompt_text(REVIEW_DEBATE_TASK)

    assert "Prompt version: prompt-test-v2" in text
    assert "agent_outputs" in text
    assert "debate_report" in text
    assert "UNTRUSTED_DRAFT_TEXT" in text
    assert "UNTRUSTED_EVIDENCE_PACK" in text


def test_all_prompt_templates_include_shared_security_instruction():
    for task in PROMPT_TEMPLATES:
        text = _prompt_text(task)
        assert SECURITY_SYSTEM_INSTRUCTION in text
        assert "return the required JSON schema" in text
        assert "Payload JSON" in text
        assert "untrusted data" in text


def test_prompt_version_defaults_to_v1_when_empty():
    assert current_prompt_version(SimpleNamespace(llm_prompt_version="")) == "v1"
