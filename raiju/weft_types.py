"""Spark-native coercion helpers for Weft (single-select friendly, minimal UDFs).

PySpark imports are **lazy** so test environments that stub ``pyspark`` can still
import :mod:`raiju.weft` for pure helpers.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from raiju.inference.llm_schemas import WeftColumnMapping, WeftWarning

_DEFAULT_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "yyyy-MM-dd",
    "MM/dd/yyyy",
    "M/d/yyyy",
    "yyyy-MM-dd HH:mm:ss",
    "MM/dd/yyyy HH:mm:ss",
    "yyyyMMdd",
    "dd/MM/yyyy",
    "MMM d, yyyy",
)


def weft_spark_datatype(spec: WeftColumnMapping) -> Any:
    """Map Weft typing hints to a concrete Spark ``DataType``."""
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        TimestampType,
    )

    ts_ntz_cls: Any = None
    try:
        from pyspark.sql.types import TimestampNTZType

        ts_ntz_cls = TimestampNTZType
    except ImportError:  # pragma: no cover
        pass

    t = spec.target_spark_type
    if t == "string":
        return StringType()
    if t == "boolean":
        return BooleanType()
    if t == "byte":
        return ByteType()
    if t == "short":
        return ShortType()
    if t == "int":
        return IntegerType()
    if t == "long":
        return LongType()
    if t == "float":
        return FloatType()
    if t == "double":
        return DoubleType()
    if t == "decimal":
        p = int(spec.decimal_precision or 38)
        s = int(spec.decimal_scale or 18)
        return DecimalType(p, s)
    if t == "date":
        return DateType()
    if t == "timestamp":
        return TimestampType()
    if t == "timestamp_ntz":
        if ts_ntz_cls is not None:
            return ts_ntz_cls()
        warnings.warn(
            "timestamp_ntz requested but TimestampNTZType is unavailable; "
            "using timestamp.",
            WeftWarning,
            stacklevel=2,
        )
        return TimestampType()
    return StringType()


def _try_to_timestamp(col: Any, fmt: str) -> Any:
    from pyspark.sql import functions

    fn = getattr(functions, "try_to_timestamp", None)
    if fn is not None:
        return fn(col, functions.lit(fmt))
    return functions.to_timestamp(col, fmt)


def _coalesce_try_timestamps(col: Any, formats: tuple[str, ...]) -> Any:
    from pyspark.sql import functions

    if not formats:
        from pyspark.sql.types import TimestampType

        return functions.lit(None).cast(TimestampType())
    parts = [_try_to_timestamp(col, f) for f in formats]
    if len(parts) == 1:
        return parts[0]
    return functions.coalesce(*parts)


def _dateutil_udf(fuzzy: bool) -> Any:
    from datetime import date, datetime

    from dateutil import parser as dup
    from pyspark.sql import functions
    from pyspark.sql.types import TimestampType

    def _inner(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)
        s = str(value).strip()
        if not s:
            return None
        try:
            return dup.parse(s, fuzzy=fuzzy, default=datetime(2000, 1, 1))
        except (ValueError, OverflowError, TypeError, AttributeError):
            return None

    return functions.udf(_inner, TimestampType())


def _string_to_long(col: Any) -> Any:
    from pyspark.sql import functions
    from pyspark.sql.types import LongType

    s = functions.trim(col.cast("string"))
    return (
        functions.when(s.isNull() | (s == ""), functions.lit(None).cast(LongType()))
        .when(s.rlike(r"^-?\d+$"), s.cast(LongType()))
        .otherwise(functions.lit(None).cast(LongType()))
    )


def _string_to_double(col: Any) -> Any:
    from pyspark.sql import functions
    from pyspark.sql.types import DoubleType

    s = functions.trim(col.cast("string"))
    cleaned = functions.regexp_replace(s, r"[\$,]", "")
    null_d = functions.lit(None).cast(DoubleType())
    return (
        functions.when(cleaned.isNull() | (cleaned == ""), null_d)
        .when(
            cleaned.rlike(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$"),
            cleaned.cast(DoubleType()),
        )
        .otherwise(null_d)
    )


def _string_to_decimal(col: Any, prec: int, scale: int) -> Any:
    from pyspark.sql import functions
    from pyspark.sql.types import DecimalType

    s = functions.trim(col.cast("string"))
    cleaned = functions.regexp_replace(s, r"[\$,]", "")
    dt = DecimalType(int(prec), int(scale))
    return (
        functions.when(cleaned.isNull() | (cleaned == ""), functions.lit(None).cast(dt))
        .when(cleaned.rlike(r"^-?\d+(\.\d+)?$"), cleaned.cast(dt))
        .otherwise(functions.lit(None).cast(dt))
    )


def _bool_column(col: Any) -> Any:
    from pyspark.sql import functions
    from pyspark.sql.types import BooleanType

    s = functions.lower(functions.trim(col.cast("string")))
    return (
        functions.when(col.isNull(), functions.lit(None).cast(BooleanType()))
        .when(s.isin("true", "t", "1", "yes", "y"), functions.lit(True))
        .when(s.isin("false", "f", "0", "no", "n"), functions.lit(False))
        .otherwise(functions.lit(None).cast(BooleanType()))
    )


def _temporal_column(
    col: Any,
    spec: WeftColumnMapping,
    *,
    source_spark_type: str,
    emit_warnings: bool,
    report_lines: list[str] | None,
) -> Any:
    from pyspark.sql import functions
    from pyspark.sql.types import DateType, TimestampType

    ts_ntz_cls: Any = None
    try:
        from pyspark.sql.types import TimestampNTZType

        ts_ntz_cls = TimestampNTZType
    except ImportError:  # pragma: no cover
        pass

    strategy = spec.temporal_parse_strategy
    tgt = spec.target_spark_type
    st_lower = source_spark_type.lower()

    if strategy == "python_dateutil":
        if emit_warnings:
            msg = (
                f"Weft column {spec.source_column!r} → {spec.target_column!r} "
                "uses python_dateutil (Python UDF); expect lower throughput than "
                "native Spark."
            )
            warnings.warn(msg, WeftWarning, stacklevel=3)
            if report_lines is not None:
                report_lines.append(msg)
        ts = _dateutil_udf(spec.python_dateutil_fuzzy)(col)
    elif strategy == "spark_formats":
        fmts = tuple(spec.spark_timestamp_formats) or _DEFAULT_TIMESTAMP_FORMATS
        if (
            emit_warnings
            and not spec.spark_timestamp_formats
            and report_lines is not None
        ):
            report_lines.append(
                f"Weft: no spark_timestamp_formats for {spec.source_column!r}; "
                f"using default set starting with {fmts[0]!r}."
            )
        ts = _coalesce_try_timestamps(col, fmts)
    elif strategy == "native" and st_lower in (
        "date",
        "timestamp",
        "timestamp_ntz",
    ):
        ts = col.cast(TimestampType()) if st_lower != "date" else col.cast(DateType())
        if st_lower == "date" and tgt != "date":
            ts = ts.cast(TimestampType())
        if st_lower == "date" and tgt == "date":
            return ts
    else:
        if (
            strategy == "native"
            and st_lower in ("string", "varchar", "char")
            and emit_warnings
            and report_lines is not None
        ):
            report_lines.append(
                f"Weft: temporal_parse_strategy=native on string column "
                f"{spec.source_column!r}; trying common date formats."
            )
        fmts = _DEFAULT_TIMESTAMP_FORMATS
        ts = _coalesce_try_timestamps(col, fmts)

    if tgt == "date":
        return ts.cast(DateType())
    if tgt == "timestamp_ntz" and ts_ntz_cls is not None:
        ntz_fn = getattr(functions, "to_timestamp_ntz", None)
        if ntz_fn is not None:
            return ntz_fn(ts)
        return ts.cast(ts_ntz_cls())
    if tgt == "timestamp_ntz":
        return ts
    return ts


def weft_coerce_column(
    col: Any,
    spec: WeftColumnMapping,
    *,
    source_spark_type: str = "string",
    emit_warnings: bool = True,
    report_lines: list[str] | None = None,
) -> Any:
    """Return a single Column expression (no ``withColumn``) for one Weft mapping."""
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
    )

    tgt = spec.target_spark_type
    st = source_spark_type.lower()
    string_like = st in ("string", "varchar", "char")

    if tgt in ("date", "timestamp", "timestamp_ntz"):
        return _temporal_column(
            col,
            spec,
            source_spark_type=st,
            emit_warnings=emit_warnings,
            report_lines=report_lines,
        )

    if tgt == "long":
        return _string_to_long(col) if string_like else col.cast(LongType())

    if tgt == "int":
        base = weft_coerce_column(
            col,
            spec.model_copy(update={"target_spark_type": "long"}),
            source_spark_type=source_spark_type,
            emit_warnings=False,
            report_lines=None,
        )
        return base.cast(IntegerType())

    if tgt == "short":
        base = weft_coerce_column(
            col,
            spec.model_copy(update={"target_spark_type": "long"}),
            source_spark_type=source_spark_type,
            emit_warnings=False,
            report_lines=None,
        )
        return base.cast(ShortType())

    if tgt == "byte":
        base = weft_coerce_column(
            col,
            spec.model_copy(update={"target_spark_type": "long"}),
            source_spark_type=source_spark_type,
            emit_warnings=False,
            report_lines=None,
        )
        return base.cast(ByteType())

    if tgt == "double":
        return _string_to_double(col) if string_like else col.cast(DoubleType())

    if tgt == "float":
        base = weft_coerce_column(
            col,
            spec.model_copy(update={"target_spark_type": "double"}),
            source_spark_type=source_spark_type,
            emit_warnings=False,
            report_lines=None,
        )
        return base.cast(FloatType())

    if tgt == "decimal":
        p = int(spec.decimal_precision or 38)
        sc = int(spec.decimal_scale or 18)
        dt = DecimalType(p, sc)
        return _string_to_decimal(col, p, sc) if string_like else col.cast(dt)

    if tgt == "boolean":
        return _bool_column(col) if string_like else col.cast(BooleanType())

    return col.cast(StringType())


def weft_canonical_select(
    df: Any,
    structure: dict[str, str],
    accepted_mappings: dict[str, str],
    accepted_specs: dict[str, WeftColumnMapping],
    source_dtypes: Mapping[str, str],
    *,
    apply_typing: bool,
    output: str,
    struct_name: str,
    keep_extra_columns: bool,
    emit_warnings: bool,
    ignored_sources: frozenset[str],
) -> tuple[Any, Any | None, list[str]]:
    """One ``DataFrame.select`` from the **original** frame (rename + cast + order).

    Returns ``(dataframe, struct_type_or_none, advisory_lines)``.
    """
    from pyspark.sql import functions
    from pyspark.sql.types import StringType, StructField, StructType

    target_to_source = {tgt: src for src, tgt in accepted_mappings.items()}
    advisory: list[str] = []
    fields: list[StructField] = []
    exprs: list[Any] = []

    for canon in structure:
        src = target_to_source.get(canon)
        if src is None:
            if emit_warnings:
                warnings.warn(
                    f"Weft: canonical field {canon!r} has no accepted source mapping; "
                    f"filling null string column.",
                    WeftWarning,
                    stacklevel=3,
                )
            advisory.append(f"missing mapping for canonical field {canon!r}")
            null_t = StringType()
            fields.append(StructField(canon, null_t, True))
            exprs.append(functions.lit(None).cast(null_t).alias(canon))
            continue

        spec = accepted_specs[src]
        spark_t = weft_spark_datatype(spec)
        fields.append(StructField(canon, spark_t, spec.nullable))
        base = functions.col(src)
        src_dtype = source_dtypes.get(src, "string")
        if apply_typing:
            inner = weft_coerce_column(
                base,
                spec,
                source_spark_type=src_dtype,
                emit_warnings=emit_warnings,
                report_lines=advisory,
            )
        else:
            inner = base
        exprs.append(inner.alias(canon))

    extra_exprs: list[Any] = []
    if keep_extra_columns:
        consumed = set(accepted_mappings.keys()) | set(ignored_sources)
        for name in df.columns:
            if name not in consumed:
                extra_exprs.append(functions.col(name))
                advisory.append(f"kept extra column {name!r} after canonical block")

    if output == "struct":
        stype = StructType(fields)
        inner = functions.struct(*exprs).alias(struct_name)
        out = df.select(inner, *extra_exprs) if extra_exprs else df.select(inner)
        return out, stype, advisory

    all_exprs = [*exprs, *extra_exprs]
    out = df.select(*all_exprs)
    return out, None, advisory
