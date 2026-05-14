"""High-throughput DataFrame profiling using Spark-native aggregations.

Design notes
------------
Row-at-a-time Python UDFs are a poor fit for whole-column statistics: each row
crosses the Python boundary and materializes Python objects. This module
instead builds a **single tree aggregate** (one narrow stage over the input)
plus an optional **freqItems** pass for approximate frequent values / mode
candidates — the same execution pattern Spark uses internally for summaries.

When ``inference`` is :class:`raiju.inference.InferenceSettings` and
``ProfileOptions.inference_enrichment`` is true, a **bounded** driver-side
LLM call adds per-column ``llm`` blocks (regex, date formats, semantics) from
aggregates plus capped samples — never the full DataFrame. Failures emit
``UserWarning`` and null ``llm`` payloads.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_PERCENTILES: tuple[float, ...] = (
    0.0,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    1.0,
)


def _normalize_percentiles(values: Sequence[float] | None) -> tuple[float, ...]:
    if not values:
        return DEFAULT_PERCENTILES
    out: list[float] = []
    for p in values:
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError("percentiles must be numbers")
        pf = float(p)
        if not math.isfinite(pf) or pf < 0 or pf > 1:
            raise ValueError(f"percentile must be finite and in [0, 1], got {p!r}")
        out.append(pf)
    return tuple(sorted(set(out)))


def _type_name(dt: Any) -> str:
    fn = getattr(dt, "typeName", None)
    if callable(fn):
        return str(fn())
    return type(dt).__name__


def _column_category(type_name: str) -> str:
    if type_name in (
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "decimal",
    ):
        return "numeric"
    if type_name == "boolean":
        return "boolean"
    if type_name in ("string", "varchar", "char"):
        return "string"
    if type_name in ("date", "timestamp", "timestamp_ntz"):
        return "temporal"
    if type_name == "binary":
        return "binary"
    if type_name == "array":
        return "array"
    if type_name == "map":
        return "map"
    if type_name == "struct":
        return "struct"
    return "other"


def _spec(col: str, metric: str, tn: str, cat: str, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "column": col,
        "metric": metric,
        "dtype": tn,
        "category": cat,
    }
    meta.update(extra)
    return meta


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _heuristic_string_pattern(
    *,
    len_min: int | None,
    len_max: int | None,
    len_avg: float | None,
    approx_distinct: int | None,
) -> dict[str, Any]:
    """Cheap driver-side hint; no regex over cell contents."""
    hint: dict[str, Any] = {"kind": "heuristic"}
    if len_min is not None and len_max is not None:
        if len_min == len_max == 0:
            hint["shape"] = "empty_or_null_trimmed"
        elif len_min == len_max:
            hint["shape"] = "fixed_width"
            hint["width"] = len_min
        elif len_max is not None and len_max <= 64:
            hint["shape"] = "short_text"
        else:
            hint["shape"] = "variable_length"
    if len_avg is not None:
        hint["avg_length"] = round(len_avg, 4)
    if approx_distinct is not None and len_max is not None and len_max > 0:
        ratio = approx_distinct / float(len_max)
        if ratio > 10:
            hint["cardinality_vs_max_len"] = "high_cardinality"
        elif ratio < 0.25:
            hint["cardinality_vs_max_len"] = "low_cardinality"
        else:
            hint["cardinality_vs_max_len"] = "mixed"
    return hint


def _regex_catalog() -> list[tuple[str, re.Pattern[str]]]:
    return [
        (
            "uuid_v4",
            re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-"
                r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
            ),
        ),
        ("iso8601_date", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
        ("integer_text", re.compile(r"^-?\d+$")),
        ("decimal_text", re.compile(r"^-?\d+\.\d+$")),
        ("email_like", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
    ]


def _inference_column_notes(
    col_name: str,
    category: str,
    metrics: Mapping[str, Any],
    inference_obj: Any,
) -> dict[str, Any]:
    """Structured notes when inference settings exist."""
    notes: dict[str, Any] = {
        "column": col_name,
        "category": category,
        "inference_attached": True,
        "providers": [],
    }
    if inference_obj is None:
        return notes
    ollama = getattr(inference_obj, "ollama", None)
    openrouter = getattr(inference_obj, "openrouter", None)
    if ollama is not None:
        notes["providers"].append("ollama")
    if openrouter is not None:
        notes["providers"].append("openrouter")
    if category == "string":
        lex_min = metrics.get("lex_min")
        lex_max = metrics.get("lex_max")
        samples: list[str] = []
        for v in (lex_min, lex_max):
            if isinstance(v, str) and v:
                samples.append(v)
        catalog_hits: list[str] = []
        for label, pat in _regex_catalog():
            if samples and all(pat.match(s) for s in samples):
                catalog_hits.append(label)
        notes["regex_catalog_hits_on_min_max"] = catalog_hits
    if isinstance(metrics.get("llm"), Mapping):
        notes["llm_field_keys"] = sorted(metrics["llm"].keys())
    return notes


@dataclass
class ProfileOptions:
    """Tunables for :func:`profile_dataframe`."""

    percentiles: Sequence[float] | None = None
    percentile_accuracy: int = 10_000
    approx_count_distinct_relative_sd: float = 0.05
    include_skew_kurtosis: bool = True
    include_sum: bool = True
    include_variance: bool = True
    include_freq_items: bool = False
    freq_items_support: float = 0.01
    columns: Sequence[str] | None = None
    collect: bool = True
    inference_enrichment: bool = False
    inference_provider: str = "auto"
    inference_http_timeout_s: float = 120.0
    inference_max_columns: int = 28
    inference_sample_scan_limit: int = 120
    inference_max_sample_values: int = 14
    inference_max_value_chars: int = 280

    def __post_init__(self) -> None:
        if int(self.percentile_accuracy) < 1:
            raise ValueError("percentile_accuracy must be at least 1")
        if not 0 < float(self.freq_items_support) <= 1:
            raise ValueError("freq_items_support must be in (0, 1]")
        p = str(self.inference_provider).strip().lower()
        self.inference_provider = p
        if p not in ("auto", "ollama", "openrouter"):
            raise ValueError(
                "inference_provider must be one of: auto, ollama, openrouter"
            )
        if int(self.inference_max_columns) < 1:
            raise ValueError("inference_max_columns must be at least 1")
        if float(self.inference_http_timeout_s) <= 0:
            raise ValueError("inference_http_timeout_s must be positive")


def profile_dataframe(
    df: Any,
    *,
    inference: Any | None = None,
    options: ProfileOptions | None = None,
    **option_kwargs: Any,
) -> dict[str, Any]:
    """Compute rich per-column metrics in as few Spark jobs as practical.

    Parameters
    ----------
    df
        A PySpark ``DataFrame``.
    inference
        Optional :class:`raiju.inference.InferenceSettings`. When
        ``ProfileOptions.inference_enrichment`` is true and this is real
        ``InferenceSettings``, a bounded LLM request adds per-column ``llm``
        metadata (regex, date formats, etc.) from aggregates and capped samples.
        Otherwise only lightweight ``inference_notes`` are added.
    options
        Profile tuning; keyword arguments override fields on ``options`` when
        both are supplied.

    Returns
    -------
    dict
        Nested structure suitable for JSON logging or UI: ``row_count``,
        ``columns`` (name → metrics), optional ``freq_items``, optional
        ``inference_notes``, optional ``llm_enrichment`` summary, optional
        per-column ``llm`` when enrichment runs, and optional ``llm_token_usage``
        after a successful LLM HTTP call.
    """
    from pyspark.sql import functions as F  # noqa: N812

    opts = options or ProfileOptions()
    if option_kwargs:
        for k, v in option_kwargs.items():
            if not hasattr(opts, k):
                raise TypeError(f"ProfileOptions has no field {k!r}")
            setattr(opts, k, v)

    if int(opts.percentile_accuracy) < 1:
        raise ValueError("percentile_accuracy must be at least 1")
    if not 0 < float(opts.freq_items_support) <= 1:
        raise ValueError("freq_items_support must be in (0, 1]")
    if int(opts.inference_max_columns) < 1:
        raise ValueError("inference_max_columns must be at least 1")
    if float(opts.inference_http_timeout_s) <= 0:
        raise ValueError("inference_http_timeout_s must be positive")

    percentiles = _normalize_percentiles(
        None if opts.percentiles is None else list(opts.percentiles)
    )

    schema = df.schema
    selected = None if opts.columns is None else [str(c) for c in opts.columns]
    fields = list(schema.fields)
    if selected is not None:
        want = set(selected)
        fields = [f for f in fields if f.name in want]
        missing = want.difference(f.name for f in fields)
        if missing:
            raise ValueError(f"Unknown columns: {sorted(missing)}")

    specs: list[dict[str, Any]] = []
    exprs: list[Any] = []

    def add_expr(meta: dict[str, Any], expr: Any) -> None:
        alias = f"_rj_m_{len(exprs)}"
        meta = {**meta, "alias": alias}
        specs.append(meta)
        exprs.append(expr.alias(alias))

    add_expr({"column": "__dataset__", "metric": "row_count"}, F.count(F.lit(1)))

    for f in fields:
        name = f.name
        c = df[name]
        tn = _type_name(f.dataType)
        cat = _column_category(tn)

        add_expr(
            _spec(name, "null_count", tn, cat),
            F.sum(F.when(c.isNull(), F.lit(1)).otherwise(F.lit(0))).cast("long"),
        )
        add_expr(
            _spec(name, "non_null_count", tn, cat),
            F.count(c),
        )
        add_expr(
            _spec(name, "approx_distinct", tn, cat),
            F.approx_count_distinct(
                c,
                float(opts.approx_count_distinct_relative_sd),
            ),
        )

        if cat == "numeric":
            add_expr(
                _spec(name, "mean", tn, cat),
                F.avg(c),
            )
            add_expr(
                _spec(name, "stddev_samp", tn, cat),
                F.stddev_samp(c),
            )
            add_expr(
                _spec(name, "stddev_pop", tn, cat),
                F.stddev_pop(c),
            )
            add_expr(
                _spec(name, "min", tn, cat),
                F.min(c),
            )
            add_expr(
                _spec(name, "max", tn, cat),
                F.max(c),
            )
            if opts.include_sum:
                add_expr(
                    _spec(name, "sum", tn, cat),
                    F.sum(c),
                )
            if opts.include_variance:
                add_expr(
                    _spec(name, "var_samp", tn, cat),
                    F.var_samp(c),
                )
                add_expr(
                    _spec(name, "var_pop", tn, cat),
                    F.var_pop(c),
                )
            if opts.include_skew_kurtosis:
                add_expr(
                    _spec(name, "skewness", tn, cat),
                    F.skewness(c),
                )
                add_expr(
                    _spec(name, "kurtosis", tn, cat),
                    F.kurtosis(c),
                )
            pct_array = F.array(*[F.lit(p) for p in percentiles])
            add_expr(
                _spec(
                    name,
                    "percentiles_approx",
                    tn,
                    cat,
                    percentile_values=list(percentiles),
                ),
                F.percentile_approx(c, pct_array, F.lit(int(opts.percentile_accuracy))),
            )

        elif cat == "boolean":
            add_expr(
                _spec(name, "true_count", tn, cat),
                F.sum(F.when(c == F.lit(True), F.lit(1)).otherwise(F.lit(0))).cast(
                    "long"
                ),
            )
            add_expr(
                _spec(name, "false_count", tn, cat),
                F.sum(F.when(c == F.lit(False), F.lit(1)).otherwise(F.lit(0))).cast(
                    "long"
                ),
            )

        elif cat == "string":
            trimmed = F.trim(c)
            lens = F.length(trimmed)
            add_expr(
                _spec(name, "lex_min", tn, cat),
                F.min(c),
            )
            add_expr(
                _spec(name, "lex_max", tn, cat),
                F.max(c),
            )
            add_expr(
                _spec(name, "char_len_min", tn, cat),
                F.min(lens),
            )
            add_expr(
                _spec(name, "char_len_max", tn, cat),
                F.max(lens),
            )
            add_expr(
                _spec(name, "char_len_mean", tn, cat),
                F.avg(lens),
            )
            add_expr(
                _spec(name, "blank_or_empty_trimmed_count", tn, cat),
                F.sum(
                    F.when(c.isNull(), F.lit(0))
                    .when(trimmed == F.lit(""), F.lit(1))
                    .otherwise(F.lit(0))
                ).cast("long"),
            )

        elif cat == "temporal":
            add_expr(
                _spec(name, "min", tn, cat),
                F.min(c),
            )
            add_expr(
                _spec(name, "max", tn, cat),
                F.max(c),
            )

        elif cat == "binary":
            blen = F.length(c)
            add_expr(
                _spec(name, "byte_len_min", tn, cat),
                F.min(blen),
            )
            add_expr(
                _spec(name, "byte_len_max", tn, cat),
                F.max(blen),
            )
            add_expr(
                _spec(name, "byte_len_mean", tn, cat),
                F.avg(blen),
            )

        elif cat == "array":
            sz = F.size(c)
            add_expr(
                _spec(name, "element_count_min", tn, cat),
                F.min(sz),
            )
            add_expr(
                _spec(name, "element_count_max", tn, cat),
                F.max(sz),
            )
            add_expr(
                _spec(name, "element_count_mean", tn, cat),
                F.avg(sz),
            )

        elif cat == "map":
            msz = F.size(c)
            add_expr(
                _spec(name, "entry_count_min", tn, cat),
                F.min(msz),
            )
            add_expr(
                _spec(name, "entry_count_max", tn, cat),
                F.max(msz),
            )
            add_expr(
                _spec(name, "entry_count_mean", tn, cat),
                F.avg(msz),
            )

        elif cat == "struct":
            # Whole-struct distinctness; detailed field stats need explode/select.
            add_expr(
                _spec(name, "min", tn, cat),
                F.min(c),
            )
            add_expr(
                _spec(name, "max", tn, cat),
                F.max(c),
            )

    agg_row = df.agg(*exprs).first()
    if agg_row is None:
        raise RuntimeError("profile_dataframe: aggregate returned no row")

    by_column: dict[str, dict[str, Any]] = {}
    row_count: int | None = None

    for spec in specs:
        col = spec["column"]
        metric = spec["metric"]
        alias = spec["alias"]
        val = agg_row[alias]
        if col == "__dataset__" and metric == "row_count":
            row_count = int(val) if val is not None else 0
            continue
        by_column.setdefault(
            col,
            {"spark_type": spec.get("dtype"), "category": spec.get("category")},
        )
        bucket: MutableMapping[str, Any] = by_column[col]
        if metric == "percentiles_approx":
            keys = spec.get("percentile_values") or []
            if val is None:
                bucket["percentiles"] = {str(k): None for k in keys}
            elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                lst = list(val)
                bucket["percentiles"] = {
                    str(keys[i]): lst[i] if i < len(lst) else None
                    for i in range(len(keys))
                }
                try:
                    idx = next(i for i, k in enumerate(keys) if float(k) == 0.5)
                except StopIteration:
                    idx = None
                if idx is not None:
                    bucket["median_approx"] = lst[idx] if idx < len(lst) else None
            else:
                bucket["percentiles"] = {"error": "unexpected_percentile_result"}
        else:
            bucket[metric] = val

    # Derived metrics + describe-style aliases
    for col_name, metrics in by_column.items():
        cat = metrics.get("category")
        nn = metrics.get("non_null_count")
        if row_count is not None and nn is not None:
            metrics["null_fraction"] = (
                (row_count - int(nn)) / float(row_count) if row_count else None
            )
            metrics["completeness"] = int(nn) / float(row_count) if row_count else None
        if cat == "numeric" and "mean" in metrics:
            metrics["describe"] = {
                "count": metrics.get("non_null_count"),
                "mean": metrics.get("mean"),
                "stddev": metrics.get("stddev_samp"),
                "min": metrics.get("min"),
                "max": metrics.get("max"),
            }
            if row_count and metrics.get("approx_distinct") is not None:
                try:
                    ad = float(metrics["approx_distinct"])
                    metrics["approx_distinct_fraction_of_rows"] = ad / float(row_count)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        if cat == "string":
            metrics["describe"] = {
                "count": metrics.get("non_null_count"),
                "mean": metrics.get("char_len_mean"),
                "stddev": None,
                "min": metrics.get("lex_min"),
                "max": metrics.get("lex_max"),
            }
            metrics["pattern_hint"] = _heuristic_string_pattern(
                len_min=_as_int(metrics.get("char_len_min")),
                len_max=_as_int(metrics.get("char_len_max")),
                len_avg=_as_float(metrics.get("char_len_mean")),
                approx_distinct=_as_int(metrics.get("approx_distinct")),
            )
            if row_count and metrics.get("approx_distinct") is not None:
                try:
                    ad = float(metrics["approx_distinct"])
                    metrics["approx_distinct_fraction_of_rows"] = ad / float(row_count)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        if cat == "boolean":
            tc = metrics.get("true_count")
            fc = metrics.get("false_count")
            try:
                ti = int(tc) if tc is not None else None
                fi = int(fc) if fc is not None else None
            except (TypeError, ValueError):
                ti = fi = None
            if ti is not None and fi is not None:
                pop = ti + fi
                if pop:
                    metrics["true_fraction"] = ti / float(pop)
                    metrics["false_fraction"] = fi / float(pop)

    out: dict[str, Any] = {
        "row_count": row_count,
        "percentiles_requested": list(percentiles),
        "columns": by_column,
        "aggregate_expression_count": len(exprs),
    }

    extra_jobs = 0
    if opts.include_freq_items:
        freq_targets = [
            f.name
            for f in fields
            if _column_category(_type_name(f.dataType))
            in ("string", "numeric", "boolean")
        ]
        if freq_targets:
            fi_row = df.stat.freqItems(freq_targets, opts.freq_items_support).first()
            freq_map: dict[str, Any] = {}
            if fi_row is not None:
                row_dict = (
                    fi_row.asDict(recursive=True) if hasattr(fi_row, "asDict") else {}
                )
                for n in freq_targets:
                    key = f"{n}_freq"
                    items = row_dict.get(key)
                    if items is None and hasattr(fi_row, key):
                        items = getattr(fi_row, key)
                    freq_map[n] = items
                    if items and isinstance(items, (list, tuple)) and n in by_column:
                        by_column[n]["approx_frequent_values"] = list(items)
                        by_column[n]["approx_mode_candidates"] = list(items)[:3]
            out["freq_items"] = freq_map
            extra_jobs = 1

    out["approximate_spark_actions"] = 1 + extra_jobs

    if opts.inference_enrichment:
        from raiju.inference.profile_enrichment import attach_profile_llm_enrichment
        from raiju.inference.settings import InferenceSettings

        if not isinstance(inference, InferenceSettings):
            warnings.warn(
                "inference_enrichment=True requires inference=InferenceSettings; "
                "skipping LLM enrichment.",
                UserWarning,
                stacklevel=2,
            )
        elif inference is not None:
            attach_profile_llm_enrichment(
                df,
                fields,
                out,
                inference,
                provider=opts.inference_provider,
                http_timeout_s=float(opts.inference_http_timeout_s),
                max_columns=int(opts.inference_max_columns),
                sample_scan_limit=int(opts.inference_sample_scan_limit),
                max_sample_values=int(opts.inference_max_sample_values),
                max_value_chars=int(opts.inference_max_value_chars),
            )

    if inference is not None:
        notes: list[dict[str, Any]] = []
        for col_name, metrics in by_column.items():
            cat = str(metrics.get("category") or "other")
            notes.append(_inference_column_notes(col_name, cat, metrics, inference))
        out["inference_notes"] = notes

    if opts.collect:
        return _json_safe(out)
    return out


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def profile_to_describe_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build rows compatible with the usual ``summary`` / ``describe`` layout.

    One row per summary statistic (count, mean, stddev, min, max) when present.
    """
    rows: list[dict[str, Any]] = []
    cols = profile.get("columns") or {}
    if not isinstance(cols, Mapping):
        return rows
    metric_names = ("count", "mean", "stddev", "min", "max")
    for col_name in cols:
        if col_name == "__dataset__":
            continue
        col_metrics = cols[col_name]
        desc = col_metrics.get("describe") if isinstance(col_metrics, Mapping) else None
        if not isinstance(desc, Mapping):
            continue
        for m in metric_names:
            if m not in desc:
                continue
            row = {"summary": m, "column": col_name, "value": desc[m]}
            rows.append(row)
    return rows
