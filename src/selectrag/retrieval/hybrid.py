"""RRF hybrid 融合(A 軸;2026-09-14 核准)。

設計決策:
  融合  — Reciprocal Rank Fusion:score(id) = Σ_r 1/(rrf_k + rank_r(id)),rank 從 1 起。
          只用名次不用分數:BM25 與 cosine 的分數尺度不可比,正規化本身會成為變因
  平手  — 分數相同依「最早出現」(先列出的檢索器、再依名次)——跨次執行 bit-for-bit 可重現
  預算  — pool(每個 base 取回的候選數)與 top_k(輸出長度)分離:pool 是實驗變因,
          輸出永遠 ≤ top_k,與單一檢索器的對照組在同一 k 預算上比較
"""

from __future__ import annotations

from collections.abc import Sequence

from selectrag.retrieval.base import Retriever


def rrf_fuse(rankings: Sequence[Sequence[str]], rrf_k: float) -> list[str]:
    """把多份名次清單融合成一份;空 rankings、rrf_k ≤ 0、清單內重複 id 一律 raise。"""
    if not rankings:
        raise ValueError("rankings must not be empty")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for r_idx, ranking in enumerate(rankings):
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"duplicate id in ranking {r_idx}")
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(doc_id, (r_idx, rank))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))


class HybridRRF:
    """每個 base 各取 pool 筆 → RRF 融合 → 截回 top_k。"""

    def __init__(self, retrievers: Sequence[Retriever], pool: int, rrf_k: float) -> None:
        if len(retrievers) < 2:
            raise ValueError("hybrid requires at least 2 retrievers")
        if pool < 1:
            raise ValueError("pool must be >= 1")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        self._retrievers = list(retrievers)
        self._pool = pool
        self._rrf_k = rrf_k

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        rankings = [list(r.retrieve(question, self._pool)) for r in self._retrievers]
        return rrf_fuse(rankings, self._rrf_k)[:top_k]
