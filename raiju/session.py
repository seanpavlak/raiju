"""
Raiju session: a thin proxy over PySpark's SparkSession.

All PySpark functionality is exposed implicitly via __getattr__ delegation.
No explicit method listing — any SparkSession API (current or future) works.
"""

from pyspark.sql import SparkSession


class _RaijuBuilder:
    """Proxy for SparkSession.builder; getOrCreate() returns a Raiju session."""

    def __getattr__(self, name: str):
        attr = getattr(SparkSession.builder, name)
        if name == "getOrCreate":

            def getOrCreate(*args, **kwargs):
                spark = SparkSession.builder.getOrCreate(*args, **kwargs)
                return Raiju(spark)

            return getOrCreate
        return attr


class Raiju:
    """
    Wrapper around PySpark's SparkSession that forwards all attribute and
    method access to the underlying session. All PySpark functionality
    is available through this instance without hardcoding.
    """

    builder = _RaijuBuilder()

    def __init__(self, spark: SparkSession):
        if not isinstance(spark, SparkSession):
            raise TypeError("Raiju requires a pyspark.sql.SparkSession")
        object.__setattr__(self, "_spark", spark)

    def __getattr__(self, name: str):
        return getattr(self._spark, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_spark":
            object.__setattr__(self, name, value)
        else:
            setattr(self._spark, name, value)

    def __repr__(self) -> str:
        return f"Raiju({self._spark!r})"
