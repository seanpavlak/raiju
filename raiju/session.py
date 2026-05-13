"""
Raiju session: a thin proxy over PySpark's SparkSession.

All PySpark functionality is exposed implicitly via __getattr__ delegation.
No explicit method listing — any SparkSession API (current or future) works.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from raiju.inference.settings import InferenceSettings
from raiju.joins import BroadcastJoinPolicy, BroadcastSideSpec
from raiju.joins import weave as weave_fn


class _RaijuBuilder:
    """Proxy for SparkSession.builder; getOrCreate() returns a Raiju session."""

    def __getattr__(self, name: str):
        attr = getattr(SparkSession.builder, name)
        if name == "getOrCreate":

            def _wrapped_get_or_create(*args, **kwargs):
                spark = SparkSession.builder.getOrCreate(*args, **kwargs)
                return Raiju(spark)

            return _wrapped_get_or_create
        return attr


class Raiju:
    """
    Wrapper around PySpark's SparkSession that forwards all attribute and
    method access to the underlying session. All PySpark functionality
    is available through this instance without hardcoding.

    Optional ``inference`` settings attach Ollama / OpenRouter configuration
    for future enrichment execution; they do not start inference by themselves.
    """

    builder = _RaijuBuilder()

    def __init__(
        self,
        spark: SparkSession,
        inference: InferenceSettings | None = None,
    ):
        if not isinstance(spark, SparkSession):
            raise TypeError("Raiju requires a pyspark.sql.SparkSession")
        object.__setattr__(self, "_spark", spark)
        object.__setattr__(self, "_inference", inference)

    @property
    def inference(self) -> InferenceSettings | None:
        """Ollama/OpenRouter configuration for future execution hooks, or ``None``."""
        return self._inference

    def weave(
        self,
        left,
        right,
        on=None,
        how=None,
        *,
        policy: BroadcastJoinPolicy | None = None,
        broadcast_side: BroadcastSideSpec = "auto",
    ):
        """
        Weave two DataFrames with optional broadcast of the smaller side.

        Uses bounded row-count inference when ``broadcast_side`` is ``"auto"``;
        see :func:`raiju.joins.weave`.
        """
        return weave_fn(
            left,
            right,
            on=on,
            how=how,
            policy=policy,
            broadcast_side=broadcast_side,
        )

    def with_inference(self, inference: InferenceSettings) -> Raiju:
        """
        Return a new ``Raiju`` wrapping the same Spark session with inference settings.

        Use after ``Raiju.builder...getOrCreate()`` when the builder path cannot
        attach settings directly.
        """
        if not isinstance(inference, InferenceSettings):
            raise TypeError("inference must be an InferenceSettings instance")
        return Raiju(self._spark, inference=inference)

    def __getattr__(self, name: str):
        return getattr(self._spark, name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("_spark", "_inference"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._spark, name, value)

    def __repr__(self) -> str:
        return f"Raiju({self._spark!r})"
