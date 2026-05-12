"""
Raiju — distributed PySpark execution utilities with a Spark-native entry point.

The public `Raiju` type wraps `SparkSession` and forwards all attributes and
methods so PySpark APIs stay available without a duplicated surface area.
"""

from raiju.session import Raiju

__all__ = ["Raiju"]
