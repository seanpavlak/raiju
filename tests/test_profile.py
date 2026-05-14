"""Tests for raiju.profile pure helpers and option validation."""

import pytest
from raiju.profile import (
    ProfileOptions,
    _column_category,
    _heuristic_string_pattern,
    _json_safe,
    _normalize_percentiles,
    _type_name,
    profile_dataframe,
    profile_to_describe_rows,
)


class _DT:
    def __init__(self, name: str):
        self._name = name

    def typeName(self) -> str:  # noqa: N802
        return self._name


def test_normalize_percentiles_default():
    p = _normalize_percentiles(None)
    assert 0.5 in p
    assert p == tuple(sorted(set(p)))


def test_normalize_percentiles_rejects_oob():
    with pytest.raises(ValueError):
        _normalize_percentiles([1.5])


def test_column_category():
    assert _column_category("integer") == "numeric"
    assert _column_category("string") == "string"
    assert _column_category("boolean") == "boolean"
    assert _column_category("timestamp") == "temporal"
    assert _column_category("array") == "array"
    assert _column_category("foo") == "other"


def test_type_name():
    assert _type_name(_DT("long")) == "long"


def test_json_safe_nan():
    import math

    assert _json_safe(float("nan")) is None
    assert _json_safe(math.inf) is None
    assert _json_safe(1.25) == 1.25


def test_heuristic_string_pattern():
    h = _heuristic_string_pattern(
        len_min=5, len_max=5, len_avg=5.0, approx_distinct=100
    )
    assert h.get("shape") == "fixed_width"
    assert h.get("width") == 5


def test_profile_options_validation_freq():
    with pytest.raises(ValueError, match="freq_items_support"):
        ProfileOptions(freq_items_support=0.0)


def test_profile_dataframe_rejects_unknown_option():
    df = object()
    with pytest.raises(TypeError, match="ProfileOptions"):
        profile_dataframe(df, options=ProfileOptions(), no_such_field=1)


def test_profile_to_describe_rows():
    prof = {
        "columns": {
            "a": {
                "describe": {
                    "count": 10,
                    "mean": 2.0,
                    "stddev": 1.0,
                    "min": 0,
                    "max": 4,
                }
            }
        }
    }
    rows = profile_to_describe_rows(prof)
    kinds = {r["summary"] for r in rows}
    assert kinds == {"count", "mean", "stddev", "min", "max"}
