"""Tests for :mod:`raiju.inference.profile_enrichment` (no live HTTP)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from raiju.inference.profile_enrichment import (
    _collect_column_samples,
    _collect_samples_batch,
    _pick_enrichment_columns,
    attach_profile_llm_enrichment,
)
from raiju.inference.settings import InferenceSettings, OllamaConfig


def _mock_df_collect(rows: list[dict]):
    df = MagicMock()
    chain = MagicMock()
    df.select.return_value = chain
    chain.limit.return_value = chain

    def _row(d):
        m = MagicMock()

        def as_dict(recursive=True):  # noqa: ARG001
            return d

        m.asDict = as_dict
        return m

    chain.collect.return_value = [_row(r) for r in rows]
    return df


def test_collect_samples_batch_empty_names():
    out = _collect_samples_batch(
        MagicMock(), [], scan_limit=10, max_distinct_values=3, max_value_chars=50
    )
    assert out == {}


def test_collect_samples_batch_str_bytes_and_truncation():
    df = _mock_df_collect(
        [
            {"c": "hello", "b": b"ab"},
            {"c": "hello", "b": None},
            {"c": "x" * 100, "b": b"x"},
        ]
    )
    out = _collect_samples_batch(
        df,
        ["c", "b"],
        scan_limit=10,
        max_distinct_values=5,
        max_value_chars=8,
    )
    assert "hello"[:8] in out["c"][0] or out["c"][0].startswith("hello")
    assert len(out["c"]) >= 1
    df.select.assert_called_once()


def test_collect_column_samples_delegates():
    df = _mock_df_collect([{"x": 1}])
    s = _collect_column_samples(
        df, "x", scan_limit=5, max_distinct_values=2, max_value_chars=10
    )
    assert s == ["1"]


def test_pick_enrichment_columns_priority_and_max():
    fields = [
        SimpleNamespace(name="n_num"),
        SimpleNamespace(name="s_str"),
        SimpleNamespace(name="t_temp"),
    ]
    by_column = {
        "n_num": {"category": "numeric", "approx_distinct": 4},
        "s_str": {"category": "string"},
        "t_temp": {"category": "temporal"},
    }
    names = _pick_enrichment_columns(fields, by_column, max_columns=2)
    assert names[0] == "s_str"
    assert set(names) == {"s_str", "t_temp"}


def test_pick_enrichment_columns_bad_approx_distinct_falls_back():
    fields = [SimpleNamespace(name="n")]
    by_column = {"n": {"category": "numeric", "approx_distinct": "nope"}}
    names = _pick_enrichment_columns(fields, by_column, max_columns=5)
    assert names == ["n"]


def test_attach_profile_llm_enrichment_columns_not_dict():
    out: dict = {"columns": []}
    attach_profile_llm_enrichment(
        MagicMock(),
        [],
        out,
        InferenceSettings(ollama=OllamaConfig()),
        provider="auto",
        http_timeout_s=30.0,
        max_columns=5,
        sample_scan_limit=10,
        max_sample_values=3,
        max_value_chars=50,
    )
    assert "llm_enrichment" not in out


def test_attach_profile_llm_enrichment_skipped_no_columns():
    out = {"columns": {}, "approximate_spark_actions": 1}
    attach_profile_llm_enrichment(
        MagicMock(),
        [],
        out,
        InferenceSettings(ollama=OllamaConfig()),
        provider="auto",
        http_timeout_s=30.0,
        max_columns=0,
        sample_scan_limit=10,
        max_sample_values=3,
        max_value_chars=50,
    )
    assert out["llm_enrichment"] == {"status": "skipped", "reason": "no_columns"}


def test_attach_profile_llm_enrichment_http_failure():
    df = _mock_df_collect([{"c1": "a"}])
    fields = [SimpleNamespace(name="c1")]
    out = {
        "columns": {"c1": {"category": "string"}},
        "approximate_spark_actions": 1,
    }
    with patch(
        "raiju.inference.profile_enrichment.inference_chat",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.warns(UserWarning, match="profile LLM enrichment failed"):
            attach_profile_llm_enrichment(
                df,
                fields,
                out,
                InferenceSettings(ollama=OllamaConfig()),
                provider="auto",
                http_timeout_s=30.0,
                max_columns=5,
                sample_scan_limit=10,
                max_sample_values=3,
                max_value_chars=50,
            )
    assert out["llm_enrichment"]["status"] == "failed"
    assert out["llm_enrichment"]["error"] == "boom"
    assert out["columns"]["c1"]["llm"] is None


def test_attach_profile_llm_enrichment_unparseable_model_json():
    df = _mock_df_collect([{"c1": "a"}])
    fields = [SimpleNamespace(name="c1")]
    out = {"columns": {"c1": {"category": "string"}}, "approximate_spark_actions": 1}
    with patch(
        "raiju.inference.profile_enrichment.inference_chat",
        return_value=("ollama:m", "not-json-at-all", None),
    ):
        with pytest.warns(UserWarning, match="unparseable"):
            attach_profile_llm_enrichment(
                df,
                fields,
                out,
                InferenceSettings(ollama=OllamaConfig()),
                provider="auto",
                http_timeout_s=30.0,
                max_columns=5,
                sample_scan_limit=10,
                max_sample_values=3,
                max_value_chars=50,
            )
    assert out["llm_enrichment"]["status"] == "failed"
    assert out["llm_enrichment"]["error"] == "unparseable_model_output"
    assert out["columns"]["c1"]["llm"] is None


def test_attach_profile_llm_enrichment_pydantic_validation_fails():
    df = _mock_df_collect([{"c1": "a"}])
    fields = [SimpleNamespace(name="c1")]
    out = {"columns": {"c1": {"category": "string"}}, "approximate_spark_actions": 1}
    bad = '{"columns":[{"human_summary":"missing column field"}]}'
    with patch(
        "raiju.inference.profile_enrichment.inference_chat",
        return_value=("ollama:m", bad, None),
    ):
        with pytest.warns(UserWarning, match="Pydantic validation"):
            attach_profile_llm_enrichment(
                df,
                fields,
                out,
                InferenceSettings(ollama=OllamaConfig()),
                provider="auto",
                http_timeout_s=30.0,
                max_columns=5,
                sample_scan_limit=10,
                max_sample_values=3,
                max_value_chars=50,
            )
    assert out["llm_enrichment"]["status"] == "failed"
    assert out["llm_enrichment"]["error"] == "pydantic_validation_error"
    assert out["columns"]["c1"]["llm"] is None


def test_attach_profile_llm_enrichment_success():
    df = _mock_df_collect([{"c1": "alpha"}])
    fields = [SimpleNamespace(name="c1")]
    out = {
        "columns": {"c1": {"category": "string", "null_count": 0}},
        "approximate_spark_actions": 1,
    }
    body = '{"columns":[{"column":"c1","human_summary":"text column"}]}'
    with patch(
        "raiju.inference.profile_enrichment.inference_chat",
        return_value=("ollama:llama3.2", body, None),
    ):
        attach_profile_llm_enrichment(
            df,
            fields,
            out,
            InferenceSettings(ollama=OllamaConfig()),
            provider="auto",
            http_timeout_s=30.0,
            max_columns=5,
            sample_scan_limit=10,
            max_sample_values=3,
            max_value_chars=50,
        )
    assert out["llm_enrichment"]["status"] == "ok"
    assert out["llm_enrichment"]["provider"] == "ollama:llama3.2"
    assert out["columns"]["c1"]["llm"]["human_summary"] == "text column"


def test_attach_profile_llm_enrichment_column_missing_from_model_response():
    """Model omits requested column name → null ``llm`` for that column."""
    df = _mock_df_collect([{"c1": "x"}])
    fields = [SimpleNamespace(name="c1")]
    out = {"columns": {"c1": {"category": "string"}}, "approximate_spark_actions": 1}
    body = '{"columns":[{"column":"other","human_summary":"o"}]}'
    with patch(
        "raiju.inference.profile_enrichment.inference_chat",
        return_value=("ollama:m", body, None),
    ):
        attach_profile_llm_enrichment(
            df,
            fields,
            out,
            InferenceSettings(ollama=OllamaConfig()),
            provider="auto",
            http_timeout_s=30.0,
            max_columns=5,
            sample_scan_limit=10,
            max_sample_values=3,
            max_value_chars=50,
        )
    assert out["llm_enrichment"]["status"] == "ok"
    assert out["columns"]["c1"]["llm"] is None
