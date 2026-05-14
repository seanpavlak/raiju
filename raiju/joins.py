"""Broadcast join helpers with bounded row-count inference.

``weave`` combines two DataFrames with optional ``broadcast()`` hints when
one side is much smaller than the other.

Spark already auto-broadcasts when estimated plan size is below
``spark.sql.autoBroadcastJoinThreshold``. These helpers add an explicit
row-count heuristic when estimates are missing or conservative, using at
most ``bounded_count_cap + 1`` rows scanned per side for the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BroadcastSide = Literal["left", "right"]
BroadcastSideSpec = Literal["left", "right", "auto", "none"]


@dataclass(frozen=True)
class BroadcastJoinPolicy:
    """Controls when a side is considered small enough to broadcast.

    ``max_small_to_large_ratio`` enforces that the smaller bounded count is at
    most this fraction of the larger (e.g. ``0.2`` → small is ≤ 20% of large).
    """

    bounded_count_cap: int = 100_000
    max_small_to_large_ratio: float = 0.2
    ambiguous_when_both_at_cap: bool = True

    def __post_init__(self) -> None:
        if self.bounded_count_cap < 1:
            raise ValueError("bounded_count_cap must be at least 1")
        if not 0 < self.max_small_to_large_ratio <= 1:
            raise ValueError("max_small_to_large_ratio must be in (0, 1]")


def bounded_row_count(df: Any, cap: int) -> int:
    """Return ``min(actual_row_count, cap + 1)`` by scanning at most ``cap + 1`` rows.

    A return value of ``cap + 1`` means "at least ``cap + 1`` rows" without a
    full table count.
    """
    if cap < 1:
        raise ValueError("cap must be at least 1")
    return int(df.limit(cap + 1).count())


def infer_broadcast_side(
    left: Any,
    right: Any,
    policy: BroadcastJoinPolicy | None = None,
) -> BroadcastSide | None:
    """Decide which side to broadcast using bounded counts and a size ratio.

    Returns ``None`` when broadcasting is not recommended (similar sizes,
    both hit the cap without a clear winner, or ratio threshold not met).
    """
    policy = policy or BroadcastJoinPolicy()
    cap = policy.bounded_count_cap
    lc = bounded_row_count(left, cap)
    rc = bounded_row_count(right, cap)

    at_cap = cap + 1
    if policy.ambiguous_when_both_at_cap and lc == at_cap and rc == at_cap:
        return None

    if lc == 0 and rc == 0:
        return None
    if lc == 0:
        return "left"
    if rc == 0:
        return "right"

    small_side: BroadcastSide = "left" if lc <= rc else "right"
    small_n, large_n = (lc, rc) if small_side == "left" else (rc, lc)
    if small_n > large_n * policy.max_small_to_large_ratio:
        return None
    return small_side


def _normalize_how(how: str) -> str:
    h = how.lower().replace("_", "")
    aliases = {
        "leftouter": "leftouter",
        "left": "leftouter",
        "rightouter": "rightouter",
        "right": "rightouter",
        "fullouter": "fullouter",
        "outer": "fullouter",
        "full": "fullouter",
        "leftsemi": "leftsemi",
        "leftanti": "leftanti",
        "anti": "leftanti",
        "semi": "leftsemi",
    }
    return aliases.get(h, h)


def _broadcast(df: Any) -> Any:
    from pyspark.sql.functions import broadcast as spark_broadcast

    return spark_broadcast(df)


def _physical_join(
    left: Any,
    right: Any,
    on: str | list[str] | Any | None,
    how: str,
    broadcast_side: BroadcastSide | None,
) -> Any:
    """Apply broadcast hint while preserving user ``left``/``right`` join semantics."""
    if broadcast_side is None:
        return left.join(right, on, how)

    h = _normalize_how(how)

    if h == "inner":
        if broadcast_side == "right":
            return left.join(_broadcast(right), on, "inner")
        return right.join(_broadcast(left), on, "inner")

    if h == "cross":
        if broadcast_side == "right":
            return left.crossJoin(_broadcast(right))
        return right.crossJoin(_broadcast(left))

    if h == "leftouter":
        if broadcast_side == "right":
            return left.join(_broadcast(right), on, "left_outer")
        return _broadcast(left).join(right, on, "left_outer")

    if h == "rightouter":
        if broadcast_side == "left":
            return _broadcast(left).join(right, on, "right_outer")
        return left.join(_broadcast(right), on, "right_outer")

    if h == "fullouter":
        if broadcast_side == "right":
            return left.join(_broadcast(right), on, "outer")
        return _broadcast(left).join(right, on, "outer")

    if h == "leftsemi":
        if broadcast_side == "right":
            return left.join(_broadcast(right), on, "left_semi")
        return _broadcast(left).join(right, on, "left_semi")

    if h == "leftanti":
        if broadcast_side == "right":
            return left.join(_broadcast(right), on, "left_anti")
        return _broadcast(left).join(right, on, "left_anti")

    # Unknown join type: avoid surprising broadcast behavior.
    return left.join(right, on, how)


def weave(
    left: Any,
    right: Any,
    on: str | list[str] | Any | None = None,
    how: str | None = None,
    *,
    policy: BroadcastJoinPolicy | None = None,
    broadcast_side: BroadcastSideSpec = "auto",
) -> Any:
    """Weave two ``DataFrame``s together, optionally broadcasting the smaller side.

    When ``broadcast_side`` is ``"auto"`` (default), bounded row counts and
    ``BroadcastJoinPolicy`` pick a side; no broadcast is applied when inference
    returns ``None``. Use ``"none"`` to skip broadcasting, or ``"left"`` /
    ``"right"`` to force a side.

    ``how`` defaults to Spark's join default (``inner``) when omitted.
    """
    if how is None:
        how = "inner"
    policy = policy or BroadcastJoinPolicy()

    side: BroadcastSide | None
    if broadcast_side == "auto":
        side = infer_broadcast_side(left, right, policy)
    elif broadcast_side == "none":
        side = None
    elif broadcast_side in ("left", "right"):
        side = broadcast_side
    else:
        raise ValueError(
            "broadcast_side must be 'auto', 'none', 'left', or 'right'",
        )

    return _physical_join(left, right, on, how, side)
