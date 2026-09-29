"""Dense(embedding)檢索器 — A 軸對照組(2026-09-13 核准)。

設計決策:
  索引  — 桌機離線建置(tools/build_dense_index.py),存成 .npz:ids、float32 向量矩陣、
          metadata JSON(model / revision / dim / context_fields / chunks_sha256 / normalized /
          revision_pinned)。runner 載入時比對 model 與 revision,錯配即 raise——
          防「舊索引配新模型」無聲產生錯數字
  相似度 — cosine:索引與查詢向量皆 L2 normalize 後做內積;normalize 具冪等性,
          建索引端是否已 normalize 不影響結果
  平手  — 分數相同依插入順序(stable argsort),跨次執行 bit-for-bit 可重現
  模型  — 本模組不 import torch / sentence-transformers;真模型只在 SentenceTransformerEmbedder
          內 lazy import,雲端測試以替身覆蓋檢索邏輯
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

_METADATA_FIELDS = (
    "model",
    "revision",
    "dim",
    "context_fields",
    "chunks_sha256",
    "normalized",
    "revision_pinned",
)


class QueryEmbedder(Protocol):
    """把查詢文字轉成 (n, dim) 的向量;與索引向量必須來自同一模型。"""

    def embed(self, texts: Sequence[str]) -> Any: ...


@dataclass(frozen=True)
class IndexMetadata:
    """索引的身分證——結果檔靠它把數字對回「哪個模型、哪組脈絡欄位、哪份語料」。"""

    model: str
    revision: str | None
    dim: int
    context_fields: tuple[str, ...]
    chunks_sha256: str
    normalized: bool
    revision_pinned: bool


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized: NDArray[np.float32] = (matrix / norms).astype(np.float32)
    return normalized


class DenseRetriever:
    """對 (id, vector) 建索引;retrieve 回傳 cosine 遞減的 id。"""

    def __init__(self, ids: Sequence[str], vectors: Any, embedder: QueryEmbedder) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("empty index: nothing to search")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"ids ({len(ids)}) and vector rows ({matrix.shape[0]}) differ")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in index")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("index vectors must be finite (found NaN/Inf)")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("index contains a zero vector (cosine undefined)")
        self._ids = list(ids)
        self._matrix = _l2_normalize(matrix)
        self._embedder = embedder

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1])

    def scored(self, question: str, top_k: int) -> list[tuple[str, float]]:
        """回傳 (id, cosine) 依分數遞減,至多 top_k 筆;同分依插入順序。"""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query = np.asarray(self._embedder.embed([question]), dtype=np.float32)
        if query.ndim != 2 or query.shape != (1, self.dim):
            raise ValueError(f"query dim mismatch: got {query.shape}, index dim {self.dim}")
        if not np.all(np.isfinite(query)) or np.linalg.norm(query) == 0:
            raise ValueError("query vector must be finite and non-zero")
        scores = self._matrix @ _l2_normalize(query)[0]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        return [doc_id for doc_id, _ in self.scored(question, top_k)]


def save_index(path: Path, ids: Sequence[str], vectors: Any, metadata: IndexMetadata) -> None:
    """寫出索引;既有檔案一律拒絕覆蓋(實驗產物不可無聲取代)。"""
    if path.exists():
        raise FileExistsError(f"index already exists: {path}")
    matrix = np.asarray(vectors, dtype=np.float32)
    meta = asdict(metadata)
    meta["context_fields"] = list(metadata.context_fields)
    np.savez(path, ids=np.asarray(list(ids)), vectors=matrix, metadata=json.dumps(meta))


def load_index(path: Path) -> tuple[list[str], NDArray[np.float32], IndexMetadata]:
    with np.load(path, allow_pickle=False) as data:
        if not {"ids", "vectors", "metadata"} <= set(data.files):
            raise ValueError(f"{path.name}: not a dense index (missing arrays)")
        ids = [str(x) for x in data["ids"]]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        raw = json.loads(str(data["metadata"]))
    missing = [f for f in _METADATA_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"{path.name}: metadata missing fields {missing}")
    raw["context_fields"] = tuple(raw["context_fields"])
    return ids, vectors, IndexMetadata(**{k: raw[k] for k in _METADATA_FIELDS})


def load_dense_from_index(
    path: Path, embedder: QueryEmbedder, model: str, revision: str | None
) -> DenseRetriever:
    """載入索引並核對身分;model / revision 與設定不符即 raise。"""
    ids, vectors, meta = load_index(path)
    if meta.model != model:
        raise ValueError(f"index model {meta.model!r} != configured model {model!r}")
    if meta.revision != revision:
        raise ValueError(f"index revision {meta.revision!r} != configured revision {revision!r}")
    return DenseRetriever(ids, vectors, embedder)


class SentenceTransformerEmbedder:  # pragma: no cover — 需真模型與 GPU 套件,僅桌機 gauntlet 5e 真跑
    """bge-m3 等 sentence-transformers 模型的查詢端;只在桌機(有 GPU 套件)實例化。"""

    def __init__(self, model: str, revision: str | None, device: str | None = None) -> None:
        from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]  # 僅桌機安裝;桌機有型別時此 ignore 多餘,故允許 unused
            SentenceTransformer,
        )

        self._model = SentenceTransformer(model, revision=revision, device=device)

    def embed(self, texts: Sequence[str]) -> Any:
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
