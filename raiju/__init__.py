"""
Raiju — a PySpark wrapper that implicitly exposes all PySpark functionality.

Use Raiju as a drop-in entry point for SparkSession. All SparkSession methods
and attributes are forwarded automatically; nothing is hardcoded.
"""

from raiju.session import Raiju

__all__ = ["Raiju"]
