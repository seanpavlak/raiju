"""Tests for broadcast join inference and weave."""

from unittest.mock import MagicMock, patch

import pytest
from raiju import BroadcastJoinPolicy, Raiju, weave
from raiju.joins import bounded_row_count, infer_broadcast_side


def _df_with_bounded_count(count_after_limit: int):
    df = MagicMock()
    lim = MagicMock()
    lim.count.return_value = count_after_limit
    df.limit.return_value = lim
    return df


class TestBroadcastJoinPolicy:
    def test_invalid_cap(self):
        with pytest.raises(ValueError, match="bounded_count_cap"):
            BroadcastJoinPolicy(bounded_count_cap=0)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError, match="max_small_to_large_ratio"):
            BroadcastJoinPolicy(max_small_to_large_ratio=0)
        with pytest.raises(ValueError, match="max_small_to_large_ratio"):
            BroadcastJoinPolicy(max_small_to_large_ratio=1.5)


class TestBoundedRowCount:
    def test_returns_count(self):
        df = _df_with_bounded_count(42)
        assert bounded_row_count(df, 100) == 42
        df.limit.assert_called_once_with(101)

    def test_cap_minimum(self):
        with pytest.raises(ValueError, match="cap"):
            bounded_row_count(MagicMock(), 0)


class TestInferBroadcastSide:
    def test_broadcast_smaller_when_ratio_ok(self):
        left = _df_with_bounded_count(10)
        right = _df_with_bounded_count(500)
        p = BroadcastJoinPolicy(bounded_count_cap=1000, max_small_to_large_ratio=0.2)
        assert infer_broadcast_side(left, right, p) == "left"

    def test_no_broadcast_when_similar_size(self):
        left = _df_with_bounded_count(400)
        right = _df_with_bounded_count(500)
        p = BroadcastJoinPolicy(bounded_count_cap=1000, max_small_to_large_ratio=0.2)
        assert infer_broadcast_side(left, right, p) is None

    def test_ambiguous_when_both_at_cap(self):
        cap = 50
        left = _df_with_bounded_count(cap + 1)
        right = _df_with_bounded_count(cap + 1)
        p = BroadcastJoinPolicy(
            bounded_count_cap=cap,
            ambiguous_when_both_at_cap=True,
        )
        assert infer_broadcast_side(left, right, p) is None

    def test_empty_both(self):
        left = _df_with_bounded_count(0)
        right = _df_with_bounded_count(0)
        assert infer_broadcast_side(left, right) is None

    def test_one_side_zero(self):
        left = _df_with_bounded_count(0)
        right = _df_with_bounded_count(5)
        assert infer_broadcast_side(left, right) == "left"

    def test_right_side_zero_broadcasts_right(self):
        left = _df_with_bounded_count(8)
        right = _df_with_bounded_count(0)
        assert infer_broadcast_side(left, right) == "right"


class TestWeave:
    def test_inner_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            out = weave(
                left,
                right,
                on="k",
                how="inner",
                broadcast_side="right",
            )
        left.join.assert_called_once()
        args, kwargs = left.join.call_args
        assert args[0].startswith("bc(")
        assert args[1] == "k" and args[2] == "inner"
        assert out == "out"

    def test_inner_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        right.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            out = weave(
                left,
                right,
                on="k",
                how="inner",
                broadcast_side="left",
            )
        right.join.assert_called_once()
        assert out == "out"

    def test_left_outer_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, on="k", how="left_outer", broadcast_side="right")
        left.join.assert_called_once_with(
            f"bc({id(right)})",
            "k",
            "left_outer",
        )

    def test_left_outer_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        bc_left = MagicMock()
        bc_left.join.return_value = "out"

        def broadcast_side_effect(d):
            return bc_left if d is left else d

        with patch("raiju.joins._broadcast", side_effect=broadcast_side_effect):
            out = weave(left, right, on="k", how="left_outer", broadcast_side="left")
        bc_left.join.assert_called_once_with(right, "k", "left_outer")
        assert out == "out"

    def test_none_skips_broadcast(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast") as bc:
            weave(left, right, on="k", how="inner", broadcast_side="none")
        bc.assert_not_called()
        left.join.assert_called_once_with(right, "k", "inner")

    def test_invalid_broadcast_side(self):
        with pytest.raises(ValueError, match="broadcast_side"):
            weave(MagicMock(), MagicMock(), broadcast_side="maybe")  # type: ignore[arg-type]

    def test_raiju_weave_delegates(self):
        spark = MagicMock()
        r = Raiju(spark)
        left, right = MagicMock(), MagicMock()
        with patch("raiju.session.weave_fn", return_value="joined") as sj:
            assert r.weave(left, right, on="x", how="inner") == "joined"
        sj.assert_called_once_with(
            left,
            right,
            on="x",
            how="inner",
            policy=None,
            broadcast_side="auto",
        )

    def test_cross_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.crossJoin.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, how="cross", broadcast_side="right")
        left.crossJoin.assert_called_once()

    def test_cross_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        right.crossJoin.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            assert weave(left, right, how="cross", broadcast_side="left") == "out"

    def test_right_outer_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        bc_left = MagicMock()
        bc_left.join.return_value = "out"

        def broadcast_side_effect(d):
            return bc_left if d is left else d

        with patch("raiju.joins._broadcast", side_effect=broadcast_side_effect):
            out = weave(left, right, on="k", how="right_outer", broadcast_side="left")
        bc_left.join.assert_called_once_with(right, "k", "right_outer")
        assert out == "out"

    def test_right_outer_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, on="k", how="right", broadcast_side="right")
        left.join.assert_called_once()

    def test_full_outer_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, on="k", how="outer", broadcast_side="right")
        left.join.assert_called_once_with(
            f"bc({id(right)})",
            "k",
            "outer",
        )

    def test_full_outer_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        bc_left = MagicMock()
        bc_left.join.return_value = "out"

        def broadcast_side_effect(d):
            return bc_left if d is left else d

        with patch("raiju.joins._broadcast", side_effect=broadcast_side_effect):
            out = weave(left, right, on="k", how="full", broadcast_side="left")
        bc_left.join.assert_called_once_with(right, "k", "outer")
        assert out == "out"

    def test_left_semi_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, on="k", how="semi", broadcast_side="right")
        left.join.assert_called_once_with(
            f"bc({id(right)})",
            "k",
            "left_semi",
        )

    def test_left_semi_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        bc_left = MagicMock()
        bc_left.join.return_value = "out"

        def broadcast_side_effect(d):
            return bc_left if d is left else d

        with patch("raiju.joins._broadcast", side_effect=broadcast_side_effect):
            out = weave(left, right, on="k", how="left_semi", broadcast_side="left")
        bc_left.join.assert_called_once_with(right, "k", "left_semi")
        assert out == "out"

    def test_left_anti_broadcast_right(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"):
            weave(left, right, on="k", how="anti", broadcast_side="right")
        left.join.assert_called_once_with(
            f"bc({id(right)})",
            "k",
            "left_anti",
        )

    def test_left_anti_broadcast_left(self):
        left, right = MagicMock(), MagicMock()
        bc_left = MagicMock()
        bc_left.join.return_value = "out"

        def broadcast_side_effect(d):
            return bc_left if d is left else d

        with patch("raiju.joins._broadcast", side_effect=broadcast_side_effect):
            out = weave(left, right, on="k", how="left_anti", broadcast_side="left")
        bc_left.join.assert_called_once_with(right, "k", "left_anti")
        assert out == "out"

    def test_unknown_how_skips_broadcast(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with patch("raiju.joins._broadcast") as bc:
            weave(left, right, on="k", how="bogus_join", broadcast_side="right")
        bc.assert_not_called()
        left.join.assert_called_once_with(right, "k", "bogus_join")

    def test_auto_uses_infer_broadcast_side(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        with (
            patch("raiju.joins.infer_broadcast_side", return_value="right") as inf,
            patch("raiju.joins._broadcast", side_effect=lambda d: f"bc({id(d)})"),
        ):
            weave(left, right, on="k", how="inner", broadcast_side="auto")
        inf.assert_called_once()
        left.join.assert_called_once()

    def test_how_none_defaults_to_inner(self):
        left, right = MagicMock(), MagicMock()
        left.join.return_value = "out"
        weave(left, right, on="k", broadcast_side="none")
        left.join.assert_called_once_with(right, "k", "inner")
