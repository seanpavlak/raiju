"""Pytest configuration. Mock PySpark so tests run without starting the JVM."""

import sys
import types
from typing import Any
from unittest.mock import MagicMock

# Install mock PySpark before raiju (or any test) imports the real one.
if "pyspark" not in sys.modules:
    # __instancecheck__ is invoked on the metaclass, so use a custom metaclass
    _meta = type(
        "Meta",
        (type,),
        {"__instancecheck__": lambda cls, inst: True},
    )
    _mock_spark_session = _meta("SparkSession", (), {})
    _mock_builder = MagicMock()
    _mock_spark_session.builder = _mock_builder
    _pyspark_sql = MagicMock()
    _pyspark_sql.SparkSession = _mock_spark_session
    _pyspark = MagicMock()
    _pyspark.sql = _pyspark_sql
    sys.modules["pyspark"] = _pyspark
    sys.modules["pyspark.sql"] = _pyspark_sql

    # Stubs for ``pyspark.sql.types`` / ``functions`` (``pyspark.sql`` is a MagicMock).
    _ts = types.ModuleType("pyspark.sql.types")

    def _simple(nm: str):
        return type(nm, (), {"__repr__": lambda self, _n=nm: _n})

    for _nm in (
        "StringType",
        "BooleanType",
        "ByteType",
        "ShortType",
        "IntegerType",
        "LongType",
        "FloatType",
        "DoubleType",
        "DateType",
        "TimestampType",
        "TimestampNTZType",
    ):
        setattr(_ts, _nm, _simple(_nm))

    class _DecimalType:
        __slots__ = ("precision", "scale")

        def __init__(self, p: int = 38, s: int = 18):
            self.precision = int(p)
            self.scale = int(s)

    _ts.DecimalType = _DecimalType

    class _StructField:
        def __init__(self, name: str, spark_data_type: Any, nullable: bool = True):
            self.name = name
            self.dataType = spark_data_type
            self.nullable = nullable

    class _StructType:
        def __init__(self, fields: list | None = None):
            self.fields = fields or []

        def simpleString(self) -> str:  # noqa: N802 — Spark API name
            return "struct<>"

    _ts.StructField = _StructField
    _ts.StructType = _StructType

    sys.modules["pyspark.sql.types"] = _ts
    _fn_mock = MagicMock()
    sys.modules["pyspark.sql.functions"] = _fn_mock
    _pyspark_sql.functions = _fn_mock
