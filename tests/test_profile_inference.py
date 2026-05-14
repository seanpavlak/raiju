"""Tests for profile LLM enrichment helpers (no live HTTP)."""

import pytest
from pydantic import ValidationError
from raiju.inference.chat import parse_llm_json_object
from raiju.inference.llm_schemas import (
    LLMTokenUsage,
    ProfileEnrichmentColumn,
    ProfileEnrichmentResponse,
    RaijuLLMUsageWarning,
)
from raiju.inference.token_count import build_llm_token_usage


def test_parse_llm_json_object_fenced():
    raw = 'Here:\n```json\n{"columns": [{"column": "a"}]}\n```'
    j = parse_llm_json_object(raw)
    assert j == {"columns": [{"column": "a"}]}


def test_parse_llm_json_object_plain():
    j = parse_llm_json_object('  {"x": 1} trailing ')
    assert j == {"x": 1}


def test_profile_enrichment_column_strips_bad_regex():
    with pytest.warns(UserWarning, match="invalid regex"):
        col = ProfileEnrichmentColumn(
            column="x",
            suggested_validation_regex="(unclosed",
            human_summary="hi",
        )
    assert col.suggested_validation_regex is None
    assert col.human_summary == "hi"


def test_profile_enrichment_column_keeps_valid_regex():
    col = ProfileEnrichmentColumn(
        column="d",
        suggested_validation_regex=r"^\d{4}-\d{2}-\d{2}$",
    )
    assert col.suggested_validation_regex == r"^\d{4}-\d{2}-\d{2}$"


def test_profile_enrichment_response_roundtrip():
    raw = {
        "columns": [
            {
                "column": "event_ts",
                "human_summary": "ISO-like timestamps",
                "java_simple_date_format": "yyyy-MM-dd",
                "extra_ignored": 1,
            }
        ]
    }
    p = ProfileEnrichmentResponse.model_validate(raw)
    assert len(p.columns) == 1
    assert p.columns[0].column == "event_ts"


def test_profile_enrichment_response_rejects_missing_column_field():
    with pytest.raises(ValidationError):
        ProfileEnrichmentResponse.model_validate({"columns": [{"human_summary": "x"}]})


def test_build_llm_token_usage_tiktoken_and_api_blob():
    u = build_llm_token_usage(
        provider="openrouter",
        model="gpt-4o-mini",
        system="hello",
        user="world",
        assistant="reply",
        raw_api_response={"usage": {"prompt_tokens": 99, "completion_tokens": 1}},
    )
    assert u.provider == "openrouter"
    assert u.total_tokens == (u.prompt_tokens or 0) + (u.completion_tokens or 0)
    assert u.raw_usage["api"]["prompt_tokens"] == 99
    assert "tiktoken" in u.raw_usage
    assert u.prompt_tokens is not None and u.prompt_tokens > 0


def test_build_llm_token_usage_ollama_preserves_api_fields():
    u = build_llm_token_usage(
        provider="ollama",
        model="llama3.2",
        system="a",
        user="b",
        assistant="c",
        raw_api_response={"prompt_eval_count": 10, "eval_count": 3},
    )
    assert u.raw_usage["api"]["prompt_eval_count"] == 10
    assert u.total_tokens == (u.prompt_tokens or 0) + (u.completion_tokens or 0)


def test_token_usage_warns_with_total():
    u = LLMTokenUsage(
        provider="openrouter",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    with pytest.warns(RaijuLLMUsageWarning, match="tiktoken estimate"):
        u.warn_if_known()


def test_profile_options_inference_provider():
    from raiju.profile import ProfileOptions

    o = ProfileOptions(inference_provider="OLLAMA ")
    assert o.inference_provider == "ollama"
    with pytest.raises(ValueError):
        ProfileOptions(inference_provider="azure")
