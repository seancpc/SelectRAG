"""檢索評測指標:recall@k 與 MRR。

設計原則:壞資料一律 raise,不靜默容錯——錯誤的評測數字比中斷的實驗危險。
"""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from statistics import fmean
from typing import TypeAlias

# 單一 query 的檢索結果與其真值集合
QueryCase: TypeAlias = tuple[Sequence[str], AbstractSet[str]]


def _check_inputs(retrieved: Sequence[str], relevant: AbstractSet[str]) -> None:
    """拒絕會讓分數失真的輸入:空真值集、重複檢索 id。"""
    if not relevant:
        raise ValueError("relevant set is empty: cannot score a query without ground truth")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("retrieved contains duplicate document ids")


def recall_at_k(retrieved: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """top-k 命中的真值比例:|relevant ∩ retrieved[:k]| / |relevant|。

    k 大於檢索長度時等同全取。分母恆為 |relevant|,不是 k。
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    _check_inputs(retrieved, relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: AbstractSet[str]) -> float:
    """第一個命中文件的排名倒數;無命中回傳 0.0。"""
    _check_inputs(retrieved, relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(cases: Sequence[QueryCase], k: int) -> float:
    """對 query 取 recall@k 平均(除以 query 數)。"""
    if not cases:
        raise ValueError("no queries to aggregate")
    return fmean(recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases)


def mrr(cases: Sequence[QueryCase]) -> float:
    """對 query 取 reciprocal rank 平均(除以 query 數)。"""
    if not cases:
        raise ValueError("no queries to aggregate")
    return fmean(reciprocal_rank(retrieved, relevant) for retrieved, relevant in cases)


def hit_at_k(retrieved: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """top-k 內至少命中一個真值即 1.0,否則 0.0。

    與 recall@k 互補:真值集含多個等價 chunk 時,recall 會因「只找到其中
    一個」被稀釋;hit 回答的是 RAG 的實際問題——答案到底有沒有被拿到。
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    _check_inputs(retrieved, relevant)
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def citation_precision(cited: Sequence[str], relevant: AbstractSet[str]) -> float:
    """生成端引用的正確率:|cited ∩ relevant| / |cited|。

    「答案完全沒有引用」是另一種失效(無據),不是 0 分——由呼叫端分開計數,
    本函式對空引用一律 raise,沿用壞資料不靜默的專案決策。
    """
    if not cited:
        raise ValueError("no citations: an uncited answer must be counted separately, not scored 0")
    if len(set(cited)) != len(cited):
        raise ValueError("cited contains duplicate document ids")
    _check_inputs(cited, relevant)
    return sum(1 for doc in cited if doc in relevant) / len(cited)
