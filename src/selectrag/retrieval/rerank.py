"""Cross-encoder reranker(A 軸步驟 2;2026-09-14 核准)。

設計決策:
  流程  — base 取 candidates 筆 → 每筆以 index_text 規則組出重排文本(context_fields 與索引端同語意)
          → reranker 打分 → 分數遞減、同分依 base 名次(stable sort)→ 截回 top_k
  預算  — candidates 與 top_k 分離;輸出永遠 ≤ top_k
  防護  — 候選 id 不在 chunk 表、分數筆數與候選數不符,一律 raise(分數錯位會無聲把名次全打亂)
  模型  — 本模組不 import sentence_transformers;CrossEncoderReranker 內 lazy import,只在桌機實例化
  NIM   — NIMReranker 打本機 NeMo Retriever reranking NIM 的 /v1/ranking(標準庫 urllib,不走 proxy)
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from selectrag.retrieval.base import Retriever
from selectrag.retrieval.bm25 import index_text


class Reranker(Protocol):
    """對 (question, texts) 回傳與 texts 等長的相關性分數,越高越相關。"""

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]: ...


def load_chunk_rows(path: Path) -> dict[str, dict[str, Any]]:
    """chunks.jsonl → chunk_id → row;缺 chunk_id / text 即 raise。"""
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("chunk_id", "text"):
                if field not in row:
                    raise ValueError(f"{path.name} line {line_no}: missing field '{field}'")
            rows[row["chunk_id"]] = row
    return rows


class RerankRetriever:
    """把 base 的前 candidates 筆交給 cross-encoder 重排。"""

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        rows: Mapping[str, Mapping[str, Any]],
        candidates: int,
        context_fields: tuple[str, ...] = (),
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._base = base
        self._reranker = reranker
        self._rows = rows
        self._candidates = candidates
        self._context_fields = context_fields

    def _text(self, chunk_id: str) -> str:
        if chunk_id not in self._rows:
            raise ValueError(f"unknown chunk id {chunk_id!r} (not in chunks table)")
        row = dict(self._rows[chunk_id])
        return index_text(row, self._context_fields, chunk_id)

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        return [cid for cid, _ in self.retrieve_scored(question, top_k)]

    def retrieve_scored(self, question: str, top_k: int) -> Sequence[tuple[str, float]]:
        """與 retrieve 同序,附 cross-encoder 分數(拒答門檻用)。"""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        ids = list(self._base.retrieve(question, self._candidates))
        if not ids:
            return []
        scores = list(self._reranker.score(question, [self._text(c) for c in ids]))
        if len(scores) != len(ids):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(ids)} candidates")
        order = sorted(range(len(ids)), key=lambda i: -scores[i])  # stable:同分依 base 名次
        return [(ids[i], float(scores[i])) for i in order[:top_k]]


class CrossEncoderReranker:  # pragma: no cover — 需真模型與 GPU 套件,僅桌機真跑
    """sentence-transformers CrossEncoder(bge-reranker-v2-m3 等)。"""

    def __init__(self, model: str, revision: str | None, device: str | None = None) -> None:
        from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]  # 僅桌機安裝;桌機有型別時此 ignore 多餘,故允許 unused
            CrossEncoder,
        )

        self._model = CrossEncoder(model, revision=revision, device=device)

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        pairs = [(question, t) for t in texts]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]


class NIMReranker:
    """NeMo Retriever text reranking NIM 的 HTTP 客戶端(POST <base_url>/v1/ranking)。

    請求:{"model", "query": {"text"}, "passages": [{"text"}], "truncate": "END"}
    回應:{"rankings": [{"index", "logit"}]}(依分數排序,index 指回 passages 位置)
    分數依 index 還原成與 texts 等長的序列;index 缺漏 / 重複 / 越界一律 raise。
    NIM 一律跑在本機 docker,故不使用環境 proxy。
    """

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self._url = base_url.rstrip("/") + "/v1/ranking"
        self._model = model
        self._timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        if not texts:
            return []
        body = json.dumps(
            {
                "model": self._model,
                "query": {"text": question},
                "passages": [{"text": t} for t in texts],
                "truncate": "END",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with self._opener.open(req, timeout=self._timeout) as resp:  # 非 2xx 由 urllib 直接 raise
            payload = json.loads(resp.read().decode("utf-8"))
        scores: list[float | None] = [None] * len(texts)
        for item in payload["rankings"]:
            idx = int(item["index"])
            if not 0 <= idx < len(texts) or scores[idx] is not None:
                raise ValueError(f"NIM ranking returned bad or duplicate index {idx}")
            scores[idx] = float(item["logit"])
        if any(v is None for v in scores):
            got = len(payload["rankings"])
            raise ValueError(f"NIM ranking returned {got} of {len(texts)} scores")
        return [float(v) for v in scores if v is not None]
