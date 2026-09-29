"""src/selectrag/evaluation/abstain_audit.py:文字皆為中性占位符。"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from selectrag.evaluation.abstain_audit import is_abstain, truth_rank

MARK = "<棄答字樣>"


def test_is_abstain_true_when_marker_present() -> None:
    assert is_abstain(f"{MARK}。", MARK) is True


def test_is_abstain_true_when_marker_is_second_sentence() -> None:
    # 發現 48 的形態:第一句作答、第二句宣稱缺席。整段含字樣即算含棄答成分
    assert is_abstain(f"<有據的一句>[1]。{MARK}其他限制。", MARK) is True


def test_is_abstain_false_without_marker() -> None:
    assert is_abstain("<純作答句>[1]。", MARK) is False


def test_is_abstain_false_on_empty_answer() -> None:
    assert is_abstain("", MARK) is False


def test_is_abstain_raises_on_empty_marker() -> None:
    # 空字樣會讓任何回答都算棄答 —— 設定錯誤必須 raise,不得靜默(決策 4)
    with pytest.raises(ValueError):
        is_abstain("<任意回答>", "")


def test_truth_rank_returns_one_based_position() -> None:
    assert truth_rank({"c1"}, ["c9", "c1", "c3"]) == 2


def test_truth_rank_returns_first_of_several_truths() -> None:
    assert truth_rank({"c1", "c3"}, ["c9", "c3", "c1"]) == 2


def test_truth_rank_none_when_absent() -> None:
    assert truth_rank({"c1"}, ["c9", "c3"]) is None


def test_truth_rank_none_on_empty_hits() -> None:
    assert truth_rank({"c1"}, []) is None


def test_truth_rank_raises_on_empty_relevant() -> None:
    # 壞題目一律 raise(2026-08-27 決策 4:靜默回 None 會與「真的沒取到」混淆)
    with pytest.raises(ValueError):
        truth_rank(set(), ["c1"])


@given(st.lists(st.text(min_size=1), min_size=1, max_size=8, unique=True))
def test_property_truth_rank_is_one_when_truth_is_top(hits: list[str]) -> None:
    assert truth_rank({hits[0]}, hits) == 1


@given(st.lists(st.text(min_size=1), min_size=1, max_size=8, unique=True))
def test_property_truth_rank_within_bounds_or_none(hits: list[str]) -> None:
    got = truth_rank(set(hits), hits)
    assert got is not None and 1 <= got <= len(hits)


@given(st.text(min_size=1))
def test_property_is_abstain_marker_in_itself(marker: str) -> None:
    assert is_abstain(marker, marker) is True


@given(st.text(min_size=1), st.text(), st.text())
def test_property_is_abstain_true_with_any_surrounding_text(
    marker: str, before: str, after: str
) -> None:
    """棄答字樣**出現在任何位置**都算棄答,不限整段相等。

    補這條的理由(2026-09-22 第 4b 層):`is_abstain(marker, marker)` 在
    「含」與「等於」兩種實作下都回 True,所以把判定改成整段相等的變異在
    property 子集存活。而發現 48 的 b4-q15 正是第一句作答、第二句宣稱缺席
    ——整段相等會漏掉那一型,那是這個函式存在的主要理由之一。
    """
    assert is_abstain(before + marker + after, marker) is True
