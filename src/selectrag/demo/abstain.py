"""拒答門檻網格掃描(2026-09-19 定案):純函式,無 I/O。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreRow:
    preset: str
    item_id: str
    kind: str  # "eval" | "offcorpus"
    top1: float | None  # 無結果為 None
    truth_in_topk: bool


@dataclass(frozen=True)
class GridPoint:
    threshold: float
    false_abstain: float  # eval 且 truth_in_topk 卻拒答
    true_abstain: float  # offcorpus 拒答
    bonus_abstain: float  # eval 且無真值時拒答


def _rate(rows: Sequence[ScoreRow], threshold: float) -> float:
    """rows 中「會被拒答」的比例:無結果(top1 None)一律拒,否則 top1 < threshold 才拒(恰等於不拒)。"""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.top1 is None or r.top1 < threshold) / len(rows)


def sweep(rows: Sequence[ScoreRow], grid: int) -> list[GridPoint]:
    """在 [min top1, max top1] 等距取 grid 個門檻,回每格三種拒答率。"""
    if grid < 2:
        raise ValueError("grid must be >= 2")
    scores = [r.top1 for r in rows if r.top1 is not None]
    if not scores:
        raise ValueError("no scored rows: cannot derive threshold range")
    lo, hi = min(scores), max(scores)
    pos = [r for r in rows if r.kind == "eval" and r.truth_in_topk]
    neg = [r for r in rows if r.kind == "offcorpus"]
    bonus = [r for r in rows if r.kind == "eval" and not r.truth_in_topk]
    out = []
    for i in range(grid):
        t = lo + (hi - lo) * i / (grid - 1)
        out.append(GridPoint(t, _rate(pos, t), _rate(neg, t), _rate(bonus, t)))
    return out


def plateau(
    points: Sequence[GridPoint], max_false: float, min_true: float
) -> tuple[float, float, float] | None:
    """同時滿足誤拒 ≤ max_false 且正確拒 ≥ min_true 的門檻區間 (lo, hi, 中點);無則 None。"""
    ok = [
        p.threshold for p in points if p.false_abstain <= max_false and p.true_abstain >= min_true
    ]
    if not ok:
        return None
    return min(ok), max(ok), (min(ok) + max(ok)) / 2


@dataclass(frozen=True)
class FoldResult:
    fold: int
    threshold: float | None  # 訓練折無高原 → None
    heldout_false_abstain: float
    heldout_true_abstain: float


def kfold(
    rows: Sequence[ScoreRow], k: int, seed: int, grid: int, max_false: float, min_true: float
) -> list[FoldResult]:
    """k 折交叉驗證:每折用其餘 k-1 折搜門檻(高原中點),在該折量 held-out 誤拒率 / 正確拒率。

    eval 與 offcorpus 各自分層切折(每折都有正負例);訓練折無高原 → threshold None、兩率記 0.0。
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    scored = [r for r in rows if r.top1 is not None]
    if len(scored) < k:
        raise ValueError(f"need at least k={k} scored rows, got {len(scored)}")
    rng = random.Random(seed)
    folds: list[list[ScoreRow]] = [[] for _ in range(k)]
    for kind in ("eval", "offcorpus"):
        group = [r for r in rows if r.kind == kind]
        rng.shuffle(group)
        for i, r in enumerate(group):
            folds[i % k].append(r)
    out = []
    for f in range(k):
        test = folds[f]
        train = [r for g in range(k) if g != f for r in folds[g]]
        train_scored = [r for r in train if r.top1 is not None]
        pl = plateau(sweep(train, grid), max_false, min_true) if train_scored else None
        if pl is None:
            out.append(FoldResult(f, None, 0.0, 0.0))
            continue
        t = pl[2]
        pos = [r for r in test if r.kind == "eval" and r.truth_in_topk]
        neg = [r for r in test if r.kind == "offcorpus"]
        out.append(FoldResult(f, t, _rate(pos, t), _rate(neg, t)))
    return out
