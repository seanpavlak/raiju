"""Extra :mod:`raiju.weft_types` coverage using conftest PySpark stubs."""

import sys
import types
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from raiju.inference.llm_schemas import WeftColumnMapping, WeftWarning
from raiju.weft_types import (
    _coalesce_try_timestamps,
    _try_to_timestamp,
    weft_canonical_select,
    weft_coerce_column,
    weft_spark_datatype,
)


def _spec(**kwargs):
    d = dict(
        source_column="src",
        target_column="tgt",
        confidence=1.0,
        reason="",
        action="map",
        target_spark_type="string",
    )
    d.update(kwargs)
    return WeftColumnMapping(**d)


@contextmanager
def _swap_pyspark_functions(fn_obj: Any):
    """``from pyspark.sql import functions`` must match ``sys.modules`` entry."""
    sql = sys.modules["pyspark.sql"]
    old_sys = sys.modules["pyspark.sql.functions"]
    old_attr = sql.functions
    sys.modules["pyspark.sql.functions"] = fn_obj
    sql.functions = fn_obj
    try:
        yield
    finally:
        sys.modules["pyspark.sql.functions"] = old_sys
        sql.functions = old_attr


def test_weft_spark_datatype_all_literal_types():
    for t in (
        "string",
        "boolean",
        "byte",
        "short",
        "int",
        "long",
        "float",
        "double",
        "date",
        "timestamp",
        "timestamp_ntz",
    ):
        dt = weft_spark_datatype(_spec(target_spark_type=t))
        assert dt is not None

    dec = weft_spark_datatype(
        _spec(
            target_spark_type="decimal",
            decimal_precision=8,
            decimal_scale=2,
        )
    )
    assert dec.precision == 8 and dec.scale == 2


def test_weft_canonical_select_flat_without_typing():
    df = MagicMock()
    select_out = MagicMock()
    df.select.return_value = select_out
    spec = _spec(source_column="c1", target_column="payee_name")
    out, st, adv = weft_canonical_select(
        df,
        {"payee_name": "who"},
        {"c1": "payee_name"},
        {"c1": spec},
        {"c1": "string"},
        apply_typing=False,
        output="flat",
        struct_name="weft",
        keep_extra_columns=False,
        emit_warnings=False,
        ignored_sources=frozenset(),
    )
    assert out is select_out
    assert st is None
    assert isinstance(adv, list)
    df.select.assert_called_once()


def test_weft_canonical_select_struct_output():
    df = MagicMock()
    df.select.return_value = MagicMock()
    spec = _spec(source_column="c1", target_column="payee_name")
    out, st, adv = weft_canonical_select(
        df,
        {"payee_name": "who"},
        {"c1": "payee_name"},
        {"c1": spec},
        {"c1": "string"},
        apply_typing=False,
        output="struct",
        struct_name="bundle",
        keep_extra_columns=False,
        emit_warnings=False,
        ignored_sources=frozenset(),
    )
    assert st is not None
    assert st.simpleString() == "struct<>"
    assert not adv or isinstance(adv, list)


def test_weft_canonical_select_keep_extra_columns():
    df = MagicMock()
    df.columns = ["c1", "extra1"]
    df.select.return_value = MagicMock()
    spec = _spec(source_column="c1", target_column="payee_name")
    weft_canonical_select(
        df,
        {"payee_name": "who"},
        {"c1": "payee_name"},
        {"c1": spec},
        {"c1": "string"},
        apply_typing=False,
        output="flat",
        struct_name="weft",
        keep_extra_columns=True,
        emit_warnings=False,
        ignored_sources=frozenset(),
    )
    call_args = df.select.call_args[0]
    assert len(call_args) >= 2


def test_weft_canonical_select_missing_mapping_warns():
    df = MagicMock()
    df.select.return_value = MagicMock()
    with pytest.warns(WeftWarning, match="no accepted source mapping"):
        _out, _st, adv = weft_canonical_select(
            df,
            {"only_canon": "desc"},
            {},
            {},
            {},
            apply_typing=False,
            output="flat",
            struct_name="weft",
            keep_extra_columns=False,
            emit_warnings=True,
            ignored_sources=frozenset(),
        )
    assert any("missing mapping" in x for x in adv)


def test_weft_canonical_select_apply_typing_coercion():
    df = MagicMock()
    df.select.return_value = MagicMock()
    spec_long = _spec(
        source_column="c1",
        target_column="n",
        target_spark_type="long",
    )
    spec_bool = _spec(
        source_column="c2",
        target_column="b",
        target_spark_type="boolean",
    )
    weft_canonical_select(
        df,
        {"n": "a", "b": "b"},
        {"c1": "n", "c2": "b"},
        {"c1": spec_long, "c2": spec_bool},
        {"c1": "string", "c2": "string"},
        apply_typing=True,
        output="flat",
        struct_name="weft",
        keep_extra_columns=False,
        emit_warnings=False,
        ignored_sources=frozenset(),
    )
    df.select.assert_called_once()


def test_weft_coerce_column_numeric_from_string_and_native_cast():
    col = MagicMock()
    col.cast.return_value = col
    col.isNull.return_value = MagicMock()
    weft_coerce_column(
        col,
        _spec(target_spark_type="long"),
        source_spark_type="string",
        emit_warnings=False,
    )
    weft_coerce_column(
        col,
        _spec(target_spark_type="double"),
        source_spark_type="string",
        emit_warnings=False,
    )
    weft_coerce_column(
        col,
        _spec(
            target_spark_type="decimal",
            decimal_precision=10,
            decimal_scale=2,
        ),
        source_spark_type="string",
        emit_warnings=False,
    )
    weft_coerce_column(
        col,
        _spec(target_spark_type="boolean"),
        source_spark_type="string",
        emit_warnings=False,
    )
    weft_coerce_column(
        col,
        _spec(target_spark_type="long"),
        source_spark_type="bigint",
        emit_warnings=False,
    )


def test_weft_coerce_column_temporal_spark_formats():
    col = MagicMock()
    col.cast.return_value = col
    spec = _spec(
        target_spark_type="timestamp",
        temporal_parse_strategy="spark_formats",
        spark_timestamp_formats=["yyyy-MM-dd"],
    )
    weft_coerce_column(
        col,
        spec,
        source_spark_type="string",
        emit_warnings=False,
    )


def test_weft_coerce_column_temporal_python_dateutil_inner():
    col = MagicMock()
    fn_mod = sys.modules["pyspark.sql.functions"]
    fn_mod.udf.reset_mock()
    spec = _spec(
        target_spark_type="timestamp",
        temporal_parse_strategy="python_dateutil",
        python_dateutil_fuzzy=True,
    )
    weft_coerce_column(
        col,
        spec,
        source_spark_type="string",
        emit_warnings=False,
    )
    assert fn_mod.udf.called
    inner = fn_mod.udf.call_args[0][0]
    assert inner(None) is None
    dt = datetime(2019, 6, 15, 12, 0, 0)
    assert inner(dt) is dt
    d = date(2020, 3, 1)
    out_d = inner(d)
    assert out_d.year == 2020 and out_d.month == 3 and out_d.day == 1
    assert inner("") is None
    assert inner("zzzz-no-parseable-date-zzzz") is None


def test_try_to_timestamp_falls_back_when_try_to_timestamp_missing():
    fn = types.SimpleNamespace()
    fn.lit = lambda x: ("lit", x)
    fn.to_timestamp = lambda c, fmt: ("to_ts", c, fmt)
    col = object()
    with _swap_pyspark_functions(fn):
        out = _try_to_timestamp(col, "yyyy-MM-dd")
    assert out == ("to_ts", col, "yyyy-MM-dd")


def test_coalesce_try_timestamps_empty_and_multi():
    class _Cast:
        def __init__(self, inner):
            self._inner = inner

        def cast(self, _t):
            return ("cast_ts", self._inner)

    fn = types.SimpleNamespace()
    fn.lit = lambda x: _Cast(x)
    fn.coalesce = lambda *parts: ("coalesce", parts)
    fn.to_timestamp = lambda c, fmt: ("to_ts", c, fmt)
    fn.try_to_timestamp = lambda c, fmt: ("try_ts", c, fmt)
    col = "C"
    with _swap_pyspark_functions(fn):
        empty = _coalesce_try_timestamps(col, ())
        one = _coalesce_try_timestamps(col, ("yyyy-MM-dd",))
        multi = _coalesce_try_timestamps(col, ("a", "b"))
    assert empty[0] == "cast_ts" and empty[1] is None
    assert one[0] == "try_ts" and one[1] == col
    assert multi[0] == "coalesce"
    assert len(multi[1]) == 2
