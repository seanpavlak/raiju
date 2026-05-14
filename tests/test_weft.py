"""Weft schema mapping: Pydantic models and guardrail resolution (no live HTTP)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from raiju.inference.llm_schemas import WeftColumnMapping, WeftResponse
from raiju.session import Raiju
from raiju.weft import _scan_column_evidence, resolve_weft_mappings


def _resp(mappings: list[WeftColumnMapping]) -> WeftResponse:
    return WeftResponse(mappings=mappings)


def test_weft_column_mapping_map_requires_target():
    with pytest.raises(ValidationError):
        WeftColumnMapping(
            source_column="x",
            target_column=None,
            confidence=0.9,
            reason="test",
            action="map",
        )


def test_weft_response_rejects_duplicate_sources():
    with pytest.raises(ValidationError):
        WeftResponse(
            mappings=[
                WeftColumnMapping(
                    source_column="a",
                    target_column="t",
                    confidence=1.0,
                    reason="",
                    action="map",
                ),
                WeftColumnMapping(
                    source_column="a",
                    target_column="t",
                    confidence=1.0,
                    reason="",
                    action="map",
                ),
            ]
        )


def test_resolve_accepts_high_confidence_maps():
    allowed = {"payee_name", "payment_amount"}
    m = [
        WeftColumnMapping(
            source_column="Vendor",
            target_column="payee_name",
            confidence=0.97,
            reason="name-like",
            action="map",
        ),
        WeftColumnMapping(
            source_column="Amt",
            target_column="payment_amount",
            confidence=0.90,
            reason="numeric",
            action="map",
        ),
    ]
    r = resolve_weft_mappings(
        _resp(m),
        ["Vendor", "Amt"],
        allowed,
        min_confidence=0.85,
        require_review_below=0.95,
        allow_unmapped=False,
        allow_many_to_one=False,
    )
    assert r["accepted_mappings"] == {
        "Vendor": "payee_name",
        "Amt": "payment_amount",
    }
    assert r["review_suggested"] == {"Amt": "payment_amount"}
    assert set(r["accepted_specs"]) == {"Vendor", "Amt"}
    assert r["accepted_specs"]["Vendor"].target_spark_type == "string"


def test_weft_spark_datatype_skips_when_pyspark_stubbed():
    try:
        from pyspark.sql.types import DecimalType, LongType
        from raiju.weft_types import weft_spark_datatype
    except (ImportError, AttributeError, ModuleNotFoundError):
        pytest.skip("pyspark.sql.types unavailable in this environment")

    m = WeftColumnMapping(
        source_column="x",
        target_column="amt",
        confidence=1.0,
        reason="",
        action="map",
        target_spark_type="long",
    )
    assert isinstance(weft_spark_datatype(m), LongType)
    m2 = m.model_copy(
        update={
            "target_spark_type": "decimal",
            "decimal_precision": 12,
            "decimal_scale": 2,
        }
    )
    d = weft_spark_datatype(m2)
    assert isinstance(d, DecimalType)
    assert d.precision == 12 and d.scale == 2


def test_weft_decimal_scale_requires_precision_on_map():
    with pytest.raises(ValidationError):
        WeftColumnMapping(
            source_column="x",
            target_column="amt",
            confidence=1.0,
            reason="",
            action="map",
            target_spark_type="decimal",
            decimal_scale=2,
        )
    m = [
        WeftColumnMapping(
            source_column="Vendor",
            target_column="payee_name",
            confidence=0.5,
            reason="weak",
            action="map",
        ),
        WeftColumnMapping(
            source_column="Amt",
            target_column="payment_amount",
            confidence=0.99,
            reason="ok",
            action="map",
        ),
    ]
    with pytest.raises(ValueError, match="allow_unmapped=False"):
        resolve_weft_mappings(
            _resp(m),
            ["Vendor", "Amt"],
            {"payee_name", "payment_amount"},
            min_confidence=0.85,
            require_review_below=0.95,
            allow_unmapped=False,
            allow_many_to_one=False,
        )


def test_resolve_collision_blocks_without_many_to_one():
    m = [
        WeftColumnMapping(
            source_column="A",
            target_column="payee_name",
            confidence=0.99,
            reason="",
            action="map",
        ),
        WeftColumnMapping(
            source_column="B",
            target_column="payee_name",
            confidence=0.99,
            reason="",
            action="map",
        ),
    ]
    r = resolve_weft_mappings(
        _resp(m),
        ["A", "B"],
        {"payee_name"},
        min_confidence=0.85,
        require_review_below=0.95,
        allow_unmapped=True,
        allow_many_to_one=False,
    )
    assert r["accepted_mappings"] == {}
    assert "payee_name" in r["needs_review"]["A"]


def test_resolve_many_to_one_keeps_best_confidence():
    m = [
        WeftColumnMapping(
            source_column="A",
            target_column="payee_name",
            confidence=0.9,
            reason="",
            action="map",
        ),
        WeftColumnMapping(
            source_column="B",
            target_column="payee_name",
            confidence=0.99,
            reason="",
            action="map",
        ),
    ]
    r = resolve_weft_mappings(
        _resp(m),
        ["A", "B"],
        {"payee_name"},
        min_confidence=0.85,
        require_review_below=0.95,
        allow_unmapped=True,
        allow_many_to_one=True,
    )
    assert r["accepted_mappings"] == {"B": "payee_name"}
    assert "payee_name" in r["needs_review"]["A"]


def test_scan_column_evidence():
    field = SimpleNamespace(
        name="a",
        dataType=SimpleNamespace(typeName=lambda: "string"),
    )
    schema = SimpleNamespace(fields=[field])
    row = MagicMock()
    row.asDict = lambda recursive=True: {"a": "hello"}  # noqa: ARG005

    df = MagicMock()
    df.schema = schema
    chain = df.select.return_value.limit.return_value
    chain.collect.return_value = [row]

    ev, n = _scan_column_evidence(
        df, scan_limit=10, max_distinct_samples=5, max_value_chars=100
    )
    assert n == 1
    assert len(ev) == 1
    assert ev[0]["column"] == "a"
    assert ev[0]["sample_values"] == ["hello"]


def test_raiju_weft_requires_inference():
    spark = MagicMock()
    r = Raiju(spark)
    with pytest.raises(TypeError, match="weft requires inference"):
        r.weft(MagicMock(), {"x": "desc"})
