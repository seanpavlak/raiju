"""Tests for Raiju session and builder."""

import builtins
from unittest.mock import MagicMock, patch

import pytest

# Import after conftest has mocked pyspark
from raiju import Raiju
from raiju import session as session_mod
from raiju.session import SparkSession


class TestRaiju:
    """Raiju wrapper delegation and validation."""

    def test_requires_spark_session(self):
        # Patch isinstance only for SparkSession so invalid inputs raise
        real_isinstance = builtins.isinstance

        def strict_isinstance(obj, cls):
            if cls is session_mod.SparkSession:
                if obj is None or type(obj) is str:
                    return False
                return real_isinstance(obj, cls)
            return real_isinstance(obj, cls)

        with patch.object(builtins, "isinstance", strict_isinstance):
            with pytest.raises(TypeError, match="requires a pyspark.sql.SparkSession"):
                Raiju(None)
            with pytest.raises(TypeError, match="requires a pyspark.sql.SparkSession"):
                Raiju("not a session")

    def test_delegates_getattr(self):
        spark = MagicMock()
        spark.read = MagicMock()
        spark.sql.return_value = "dataframe"
        r = Raiju(spark)
        assert r.read is spark.read
        assert r.sql("SELECT 1") == "dataframe"
        spark.sql.assert_called_once_with("SELECT 1")

    def test_delegates_setattr(self):
        spark = MagicMock()
        r = Raiju(spark)
        r.some_attr = 42
        assert spark.some_attr == 42

    def test_repr(self):
        spark = MagicMock()
        r = Raiju(spark)
        assert "Raiju" in repr(r)
        assert repr(spark) in repr(r)


class TestRaijuBuilder:
    """Raiju.builder forwarding and getOrCreate."""

    def test_builder_forwards_attributes(self):
        assert hasattr(Raiju.builder, "appName")
        assert hasattr(Raiju.builder, "master")
        assert hasattr(Raiju.builder, "config")
        assert callable(Raiju.builder.getOrCreate)

    def test_get_or_create_returns_raiju_wrapping_session(self):
        mock_spark = MagicMock()
        with patch.object(SparkSession.builder, "getOrCreate", return_value=mock_spark):
            result = Raiju.builder.getOrCreate()
        assert isinstance(result, Raiju)
        assert result._spark is mock_spark
