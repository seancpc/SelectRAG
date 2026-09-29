"""檢索期鄰塊擴展(A 軸 chunking ablation;2026-09-14 核准)。

背景:採樣偏差抽驗發現 7 筆「落點正確、chunk 邊界差一格」的漏檢。eval set 真值是內文 hash,
重新切塊會讓全部真值失效,因此對策放在檢索期:命中後把同文件的前後鄰塊接在其後。

設計決策:
  順序來源 — 以 chunks.jsonl 的實際行序建表(doc_id 分組),不解析 chunk_id 字串
  插入順序 — 命中、前 1、後 1、前 2、後 2 …(距離近者優先,同距離先前後後)
  k 預算   — base 以同一 top_k 取回;擴展後去重(先出現者保留位置)並截回 top_k。
             輸出永不超過 top_k,對照組數字不會因「看了更多塊」而虛高
  window=0 — runner 不包裝,行為與現況 bit-for-bit 相同
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from selectrag.retrieval.base import Retriever


class ChunkOrder:
    """每份文件的 chunk 有序清單;neighbors 只在同一文件內取,不跨文件。"""

    def __init__(self, docs: Mapping[str, Sequence[str]]) -> None:
        self._docs = {doc: list(ids) for doc, ids in docs.items()}
        self._pos: dict[str, tuple[str, int]] = {}
        for doc, ids in self._docs.items():
            for i, cid in enumerate(ids):
                if cid in self._pos:
                    raise ValueError(f"duplicate chunk id {cid!r}")
                self._pos[cid] = (doc, i)
        if not self._pos:
            raise ValueError("empty chunk order")

    def neighbors(self, chunk_id: str, window: int) -> list[str]:
        """依距離遞增回傳同文件鄰塊:前 1、後 1、前 2、後 2 …;越界者略過。"""
        if chunk_id not in self._pos:
            raise ValueError(f"unknown chunk id {chunk_id!r}")
        doc, i = self._pos[chunk_id]
        ids = self._docs[doc]
        out: list[str] = []
        for d in range(1, window + 1):
            if i - d >= 0:
                out.append(ids[i - d])
            if i + d < len(ids):
                out.append(ids[i + d])
        return out


def load_chunk_order(path: Path) -> ChunkOrder:
    """從 chunks.jsonl 依行序建表;缺 chunk_id / doc_id 即 raise。"""
    docs: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("chunk_id", "doc_id"):
                if field not in row:
                    raise ValueError(f"{path.name} line {line_no}: missing field '{field}'")
            docs.setdefault(row["doc_id"], []).append(row["chunk_id"])
    return ChunkOrder(docs)


class NeighborExpand:
    """把 base 的每個命中擴展成「命中 + 同文件 ±window 鄰塊」,去重後截回 top_k。"""

    def __init__(self, base: Retriever, order: ChunkOrder, window: int) -> None:
        if window < 0:
            raise ValueError("window must be >= 0")
        self._base = base
        self._order = order
        self._window = window

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        hits = self._base.retrieve(question, top_k)
        seen: set[str] = set()
        out: list[str] = []
        for hit in hits:
            for cid in (hit, *self._order.neighbors(hit, self._window)):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out[:top_k]
