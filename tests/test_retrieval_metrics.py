"""檢索指標測試 —— 場景為 2026-08-27 定案的驗收條件。

所有真值分數皆為手算,不由被測程式產生。
"""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from selectrag.evaluation.retrieval_metrics import (
    mean_recall_at_k,
    mrr,
    recall_at_k,
    reciprocal_rank,
)

# --------------------------------------------------------------------------
# recall@k — 手算案例
# --------------------------------------------------------------------------


def test_recall_partial_hit() -> None:
    """top-3 只含 d2,真值有 3 筆 → 1/3。"""
    assert recall_at_k(["d1", "d2", "d3", "d4"], {"d2", "d4", "d9"}, k=3) == pytest.approx(1 / 3)


def test_recall_k_exceeds_retrieved_length() -> None:
    """k 大於檢索長度時等同全取,不補零也不報錯。"""
    assert recall_at_k(["d1", "d2"], {"d1", "d5"}, k=10) == pytest.approx(0.5)


def test_recall_no_hit() -> None:
    assert recall_at_k(["d1"], {"d9"}, k=1) == 0.0


def test_recall_counts_relevant_not_k() -> None:
    """分母必須是 |relevant|,不是 k —— 失效模式 1 的直接防線。"""
    # top-5 命中 2 筆,真值共 4 筆 → 0.5;若誤用 k 當分母會得 0.4
    assert recall_at_k(["a", "b", "x", "y", "z"], {"a", "b", "c", "d"}, k=5) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# reciprocal_rank — 手算案例
# --------------------------------------------------------------------------


def test_rr_second_position() -> None:
    """只看第一個命中(d2 在 rank 2)→ 0.5,不因 d3 也命中而改變。"""
    assert reciprocal_rank(["d1", "d2", "d3"], {"d2", "d3"}) == pytest.approx(0.5)


def test_rr_first_position() -> None:
    assert reciprocal_rank(["d1", "d2"], {"d1"}) == 1.0


def test_rr_no_hit() -> None:
    assert reciprocal_rank(["d1", "d2"], {"d9"}) == 0.0


def test_rr_is_order_sensitive() -> None:
    """順序敏感 —— 失效模式 4:誤用 set 運算會讓兩者相同。"""
    forward = reciprocal_rank(["d1", "d2", "d3"], {"d3"})
    reversed_ = reciprocal_rank(["d3", "d2", "d1"], {"d3"})
    assert forward == pytest.approx(1 / 3)
    assert reversed_ == 1.0
    assert forward != reversed_


# --------------------------------------------------------------------------
# 聚合 — 手算案例
# --------------------------------------------------------------------------


def test_mean_recall_divides_by_query_count() -> None:
    """per-query recall = 1.0, 0.0, 0.5 → 平均 0.5(除以 3,不是除以文件數)。"""
    cases = [
        (["a"], {"a"}),  # 1.0
        (["x"], {"a"}),  # 0.0
        (["a", "x"], {"a", "b"}),  # 0.5
    ]
    assert mean_recall_at_k(cases, k=2) == pytest.approx(0.5)


def test_mrr_divides_by_query_count() -> None:
    """per-query RR = 1.0, 0.0, 0.25 → 平均 5/12。"""
    cases = [
        (["a"], {"a"}),  # 1.0
        (["x"], {"a"}),  # 0.0
        (["w", "x", "y", "a"], {"a"}),  # 0.25
    ]
    assert mrr(cases) == pytest.approx(5 / 12)


def test_mean_recall_rejects_empty_query_set() -> None:
    with pytest.raises(ValueError, match="no queries"):
        mean_recall_at_k([], k=5)


def test_mrr_rejects_empty_query_set() -> None:
    with pytest.raises(ValueError, match="no queries"):
        mrr([])


# --------------------------------------------------------------------------
# 錯誤案例 —— 壞資料一律 raise
# --------------------------------------------------------------------------


def test_recall_rejects_empty_relevant_set() -> None:
    with pytest.raises(ValueError, match="relevant set is empty"):
        recall_at_k(["d1"], set(), k=1)


def test_rr_rejects_empty_relevant_set() -> None:
    with pytest.raises(ValueError, match="relevant set is empty"):
        reciprocal_rank(["d1"], set())


@pytest.mark.parametrize("bad_k", [0, -1])
def test_recall_rejects_non_positive_k(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["d1"], {"d1"}, k=bad_k)


def test_recall_rejects_duplicate_retrieved_ids() -> None:
    """重複 id 會讓 recall 虛高 —— 失效模式 5。"""
    with pytest.raises(ValueError, match="duplicate"):
        recall_at_k(["d1", "d1", "d2"], {"d1"}, k=3)


def test_rr_rejects_duplicate_retrieved_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank(["d1", "d1"], {"d1"})


def test_aggregate_propagates_bad_case_errors() -> None:
    """聚合不得吞掉單一 case 的資料錯誤。"""
    with pytest.raises(ValueError, match="relevant set is empty"):
        mean_recall_at_k([(["a"], {"a"}), (["b"], set())], k=1)


# --------------------------------------------------------------------------
# Property-based(Tier 3)
# --------------------------------------------------------------------------

doc_ids = st.text(alphabet="abcdefg", min_size=1, max_size=2)
retrieved_lists = st.lists(doc_ids, min_size=0, max_size=8, unique=True)
relevant_sets = st.sets(doc_ids, min_size=1, max_size=8)


@given(retrieved=retrieved_lists, relevant=relevant_sets, k=st.integers(min_value=1, max_value=12))
def test_property_recall_within_unit_interval(
    retrieved: list[str], relevant: set[str], k: int
) -> None:
    assert 0.0 <= recall_at_k(retrieved, relevant, k) <= 1.0


@given(retrieved=retrieved_lists, relevant=relevant_sets, k=st.integers(min_value=1, max_value=12))
def test_property_recall_monotonic_in_k(retrieved: list[str], relevant: set[str], k: int) -> None:
    """k 放大,recall 不可能下降。"""
    assert recall_at_k(retrieved, relevant, k) <= recall_at_k(retrieved, relevant, k + 1)


@given(retrieved=retrieved_lists, relevant=relevant_sets)
def test_property_rr_is_zero_or_unit_fraction(retrieved: list[str], relevant: set[str]) -> None:
    """RR 必為 0.0 或某個 1/n(n 為正整數)。"""
    value = reciprocal_rank(retrieved, relevant)
    if value == 0.0:
        return
    inverse = 1.0 / value
    assert math.isclose(inverse, round(inverse), rel_tol=1e-9)
    assert round(inverse) >= 1


@given(retrieved=retrieved_lists, relevant=relevant_sets)
def test_property_rr_matches_first_hit_position(retrieved: list[str], relevant: set[str]) -> None:
    """RR 必須等於第一個命中位置的倒數(獨立重算,不呼叫被測邏輯)。"""
    expected = 0.0
    for index, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            expected = 1.0 / index
            break
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(expected)


@given(
    cases=st.lists(st.tuples(retrieved_lists, relevant_sets), min_size=1, max_size=5),
    k=st.integers(min_value=1, max_value=12),
)
def test_property_mean_recall_within_per_query_bounds(
    cases: list[tuple[list[str], set[str]]], k: int
) -> None:
    """平均值必落在各 per-query 值的 min~max 之間 —— 失效模式 3 的防線。"""
    per_query = [recall_at_k(r, rel, k) for r, rel in cases]
    assert min(per_query) - 1e-9 <= mean_recall_at_k(cases, k) <= max(per_query) + 1e-9


@given(retrieved=retrieved_lists, relevant=relevant_sets)
def test_property_full_k_recall_equals_set_intersection(
    retrieved: list[str], relevant: set[str]
) -> None:
    """k 涵蓋全部檢索結果時,recall 等於集合交集比例。"""
    assume(len(retrieved) > 0)
    expected = len(set(retrieved) & relevant) / len(relevant)
    assert recall_at_k(retrieved, relevant, len(retrieved)) == pytest.approx(expected)


@given(retrieved=retrieved_lists, relevant=relevant_sets, k=st.integers(min_value=1, max_value=12))
def test_property_recall_matches_independent_computation(
    retrieved: list[str], relevant: set[str], k: int
) -> None:
    """recall 必等於「前 k 筆與真值交集 / 真值數」(獨立重算,含截斷語意)。

    補洞:單調性與值域不變量在「忽略 k 截斷」的實作下依然成立,
    唯有明寫截斷的不變量能抓到它(property-only 變異測試證實)。
    """
    expected = len(set(retrieved[:k]) & relevant) / len(relevant)
    assert recall_at_k(retrieved, relevant, k) == pytest.approx(expected)


# --------------------------------------------------------------------------
# citation 正確率
# --------------------------------------------------------------------------


def test_citation_precision_counts_cited_truth_share() -> None:
    from selectrag.evaluation.retrieval_metrics import citation_precision

    assert citation_precision(["c1", "c2", "x1", "x2"], {"c1", "c2", "c9"}) == 0.5
    assert citation_precision(["c1"], {"c1"}) == 1.0
    assert citation_precision(["x1"], {"c1"}) == 0.0


def test_citation_precision_rejects_bad_inputs() -> None:
    import pytest

    from selectrag.evaluation.retrieval_metrics import citation_precision

    with pytest.raises(ValueError, match="no citations"):
        citation_precision([], {"c1"})
    with pytest.raises(ValueError, match="duplicate"):
        citation_precision(["c1", "c1"], {"c1"})
    with pytest.raises(ValueError, match="relevant set is empty"):
        citation_precision(["c1"], set())
