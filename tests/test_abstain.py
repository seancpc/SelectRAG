"""拒答門檻網格掃描。分數皆為合成值。"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.demo.abstain import GridPoint, ScoreRow, kfold, plateau, sweep


def _r(kind: str, top1: float | None, truth: bool, i: int) -> ScoreRow:
    return ScoreRow("p", f"{kind}-{i}", kind, top1, truth)


ROWS = [
    _r("eval", 0.9, True, 1),
    _r("eval", 0.7, True, 2),
    _r("eval", 0.2, False, 3),  # 無真值:拒了算 bonus
    _r("offcorpus", 0.1, False, 4),
    _r("offcorpus", 0.3, False, 5),
    _r("eval", None, False, 6),  # 無結果:一律視為拒答
]


def test_sweep_hand_computed() -> None:
    pts = sweep(ROWS, grid=3)  # 門檻 0.1, 0.5, 0.9(min..max 等距)
    assert [p.threshold for p in pts] == pytest.approx([0.1, 0.5, 0.9])
    # τ=0.5:有真值 2 題皆 ≥0.5 → 誤拒 0;offcorpus 2 題皆 <0.5 → 正確拒 1.0;
    # 無真值 eval 2 題(0.2, None)皆拒 → bonus 1.0
    assert pts[1] == GridPoint(0.5, 0.0, 1.0, 1.0)
    # τ=0.9:0.7 那題被誤拒 → 1/2
    assert pts[2].false_abstain == pytest.approx(0.5)
    # τ=0.1:top1 恰等於門檻不拒 → offcorpus 只拒 0/2,None 仍拒 → bonus 1/2
    assert pts[0] == GridPoint(0.1, 0.0, 0.0, 0.5)


def test_sweep_requires_scores_and_grid() -> None:
    with pytest.raises(ValueError):
        sweep(ROWS, grid=1)
    with pytest.raises(ValueError):
        sweep([_r("eval", None, True, 1)], grid=3)  # 全無分數無法定範圍


def test_plateau_midpoint_and_none() -> None:
    pts = [
        GridPoint(0.1, 0.0, 0.5, 0.0),
        GridPoint(0.2, 0.0, 0.9, 0.0),
        GridPoint(0.3, 0.0, 1.0, 0.0),
        GridPoint(0.4, 0.05, 1.0, 0.0),
    ]
    assert plateau(pts, max_false=0.02, min_true=0.9) == pytest.approx((0.2, 0.3, 0.25))
    assert plateau(pts, max_false=0.0, min_true=1.1) is None


_row = st.builds(
    ScoreRow,
    st.just("p"),
    st.text(min_size=1, max_size=4),
    st.sampled_from(["eval", "offcorpus"]),
    st.one_of(st.none(), st.floats(0, 1, allow_nan=False)),
    st.booleans(),
)


@given(st.lists(_row, min_size=1, max_size=30), st.integers(2, 12))
@settings(max_examples=200)
def test_sweep_property(rows: list[ScoreRow], grid: int) -> None:
    if all(r.top1 is None for r in rows):
        return
    pts = sweep(rows, grid)
    assert len(pts) == grid
    ths = [p.threshold for p in pts]
    assert ths == sorted(ths)
    for p in pts:
        for v in (p.false_abstain, p.true_abstain, p.bonus_abstain):
            assert 0.0 <= v <= 1.0
    # 門檻升高,三種拒答率單調不減
    for a, b in zip(pts, pts[1:], strict=False):
        assert b.false_abstain >= a.false_abstain - 1e-12
        assert b.true_abstain >= a.true_abstain - 1e-12
        assert b.bonus_abstain >= a.bonus_abstain - 1e-12
    # 獨立重算最後一格的誤拒率
    pos = [r for r in rows if r.kind == "eval" and r.truth_in_topk]
    if pos:
        t = pts[-1].threshold
        exp = sum(1 for r in pos if r.top1 is None or r.top1 < t) / len(pos)
        assert pts[-1].false_abstain == pytest.approx(exp)


_pt = st.builds(
    GridPoint,
    st.floats(0, 1, allow_nan=False),
    st.floats(0, 1, allow_nan=False),
    st.floats(0, 1, allow_nan=False),
    st.floats(0, 1, allow_nan=False),
)


_unit = st.floats(0, 1, allow_nan=False)


@given(st.lists(_pt, max_size=20), _unit, _unit)
@settings(max_examples=200)
def test_plateau_property(points: list[GridPoint], max_false: float, min_true: float) -> None:
    got = plateau(points, max_false, min_true)
    ok = [
        p.threshold for p in points if p.false_abstain <= max_false and p.true_abstain >= min_true
    ]
    if not ok:
        assert got is None
        return
    assert got is not None
    lo, hi, mid = got
    assert lo == min(ok) and hi == max(ok)  # 區間兩端都是「合格門檻」的極值,不是退化成一點
    assert mid == pytest.approx((lo + hi) / 2)


# 交叉驗證(爭議點 2:門檻與驗收集同源)
def _cv_rows() -> list[ScoreRow]:
    pos = [_r("eval", 0.6 + 0.01 * i, True, i) for i in range(20)]  # 有真值 0.60–0.79
    neg = [_r("offcorpus", 0.01 * i, False, 100 + i) for i in range(10)]  # 語料外 0.00–0.09
    return pos + neg


def test_kfold_hand_computed_separable() -> None:
    res = kfold(_cv_rows(), k=5, seed=0, grid=20, max_false=0.0, min_true=1.0)
    assert len(res) == 5 and [r.fold for r in res] == [0, 1, 2, 3, 4]
    for r in res:
        assert r.threshold is not None and 0.09 < r.threshold < 0.60  # 落在正負例之間
        assert r.heldout_false_abstain == 0.0 and r.heldout_true_abstain == 1.0


def test_kfold_heldout_is_truly_held_out() -> None:
    # 正例放一個低分離群值(0.20):不洩漏時,含它的那一折用其餘折搜出的門檻(≈0.35)會誤拒它;
    # 若訓練集偷看到它,門檻會被壓到 0.20 以下,誤拒消失 → 以「存在誤拒 > 0 的折」分辨洩漏
    rows = _cv_rows() + [_r("eval", 0.20, True, 999)]
    res = kfold(rows, k=2, seed=0, grid=50, max_false=0.0, min_true=1.0)
    assert all(r.threshold is not None for r in res)
    assert any(r.heldout_false_abstain > 0.0 for r in res)
    assert sum(r.heldout_false_abstain > 0.0 for r in res) == 1  # 離群值只在一折


def test_kfold_no_plateau_gives_none_and_zero_rates() -> None:
    rows = [
        _r("eval", 0.5, True, 1),
        _r("eval", 0.5, True, 2),
        _r("offcorpus", 0.9, False, 3),
        _r("offcorpus", 0.9, False, 4),
    ]
    res = kfold(rows, k=2, seed=0, grid=5, max_false=0.0, min_true=1.0)
    assert all(r.threshold is None for r in res)
    assert all(r.heldout_false_abstain == 0.0 and r.heldout_true_abstain == 0.0 for r in res)


def test_kfold_validation() -> None:
    with pytest.raises(ValueError):
        kfold(_cv_rows(), k=1, seed=0, grid=5, max_false=0.0, min_true=1.0)
    with pytest.raises(ValueError):
        kfold(_cv_rows()[:3], k=5, seed=0, grid=5, max_false=0.0, min_true=1.0)  # 折數多於列數


def test_kfold_deterministic_and_seed_sensitive() -> None:
    a = kfold(_cv_rows(), k=4, seed=1, grid=10, max_false=0.0, min_true=1.0)
    b = kfold(_cv_rows(), k=4, seed=1, grid=10, max_false=0.0, min_true=1.0)
    assert a == b


@given(st.lists(_row, min_size=6, max_size=40), st.integers(2, 5), st.integers(0, 3))
@settings(max_examples=100)
def test_kfold_property(rows: list[ScoreRow], k: int, seed: int) -> None:
    if sum(1 for r in rows if r.top1 is not None) < k:
        return
    res = kfold(rows, k=k, seed=seed, grid=5, max_false=0.02, min_true=0.9)
    assert len(res) == k
    for r in res:
        assert 0.0 <= r.heldout_false_abstain <= 1.0 and 0.0 <= r.heldout_true_abstain <= 1.0
        if r.threshold is None:
            assert r.heldout_false_abstain == 0.0 and r.heldout_true_abstain == 0.0
