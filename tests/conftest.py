"""Pytest configuration. Mock PySpark so tests run without starting the JVM."""

import sys
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
