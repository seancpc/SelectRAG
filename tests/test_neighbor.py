"""鄰塊擴展測試 —— 場景為 2026-09-14 定案的驗收條件。id / 文字皆為中性占位符。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.retrieval.neighbor import ChunkOrder, NeighborExpand, load_chunk_order
from selectrag.retrieval.scripted import ScriptedRetriever
from selectrag.runner.config import RetrieverConfig

DOC_A = [f"a{i}" for i in range(10)]
DOC_B = [f"b{i}" for i in range(3)]
ORDER = ChunkOrder({"A": DOC_A, "B": DOC_B})
Q = "<問題>"


def _expand(base: Sequence[str], window: int, top_k: int = 10) -> list[str]:
    return list(NeighborExpand(ScriptedRetriever({Q: base}), ORDER, window).retrieve(Q, top_k))


class RecordingBase:
    def __init__(self, result: Sequence[str]) -> None:
        self.calls: list[int] = []
        self._result = list(result)

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        self.calls.append(top_k)
        return self._result[:top_k]


# 1~10 手算案例


def test_window_zero_is_identity() -> None:
    assert _expand(["a5", "b1", "a2"], 0) == ["a5", "b1", "a2"]
    assert _expand(["a5", "b1", "a2"], 0, top_k=2) == ["a5", "b1"]


def test_window_one_inserts_prev_then_next() -> None:
    assert _expand(["a5"], 1) == ["a5", "a4", "a6"]


def test_window_two_ordering() -> None:
    assert _expand(["a5"], 2) == ["a5", "a4", "a6", "a3", "a7"]


def test_respects_doc_start_and_end() -> None:
    assert _expand(["a0"], 1) == ["a0", "a1"]
    assert _expand(["a9"], 1) == ["a9", "a8"]
    assert _expand(["a0"], 2) == ["a0", "a1", "a2"]


def test_no_cross_document_neighbors() -> None:
    assert _expand(["a9"], 2) == ["a9", "a8", "a7"]
    assert _expand(["b0"], 1) == ["b0", "b1"]


def test_dedupes_and_keeps_first_position() -> None:
    assert _expand(["a5", "a6"], 1) == ["a5", "a4", "a6", "a7"]


def test_truncates_to_top_k() -> None:
    assert _expand(["a5", "b1"], 1, top_k=3) == ["a5", "a4", "a6"]
    assert len(_expand(["a5", "b1"], 2, top_k=4)) == 4


def test_base_called_with_same_top_k() -> None:
    base = RecordingBase(["a5"])
    NeighborExpand(base, ORDER, 1).retrieve(Q, 7)
    assert base.calls == [7]


def test_unknown_chunk_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown chunk"):
        _expand(["zz"], 1)


def test_negative_window_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        NeighborExpand(ScriptedRetriever({Q: ["a5"]}), ORDER, -1)


def test_top_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _expand(["a5"], 1, top_k=0)


# 11 ChunkOrder 建表


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return path


def test_chunk_order_from_jsonl(tmp_path: Path) -> None:
    rows = [
        {"chunk_id": "x1", "doc_id": "X", "text": "<甲>"},
        {"chunk_id": "y1", "doc_id": "Y", "text": "<乙>"},
        {"chunk_id": "x2", "doc_id": "X", "text": "<丙>"},
        {"chunk_id": "x3", "doc_id": "X", "text": "<丁>"},
    ]
    order = load_chunk_order(_write_jsonl(tmp_path / "c.jsonl", rows))
    assert order.neighbors("x2", 1) == ["x1", "x3"]
    assert order.neighbors("y1", 1) == []
    assert order.neighbors("x1", 5) == ["x2", "x3"]


def test_chunk_order_rejects_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="doc_id"):
        load_chunk_order(_write_jsonl(tmp_path / "c.jsonl", [{"chunk_id": "x1", "text": "t"}]))


def test_chunk_order_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [{"chunk_id": "x1", "doc_id": "X"}, {"chunk_id": "x1", "doc_id": "X"}]
    with pytest.raises(ValueError, match="duplicate"):
        load_chunk_order(_write_jsonl(tmp_path / "c.jsonl", rows))


def test_chunk_order_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_chunk_order(_write_jsonl(tmp_path / "c.jsonl", []))


# 12 property

_hits = st.lists(st.sampled_from(DOC_A + DOC_B), min_size=1, max_size=6, unique=True)


def _neighborhood(cid: str, window: int) -> set[str]:
    doc = DOC_A if cid.startswith("a") else DOC_B
    i = doc.index(cid)
    return set(doc[max(0, i - window) : i + window + 1])


@settings(max_examples=100)
@given(hits=_hits, window=st.integers(0, 3), data=st.data())
def test_property_prefix_consistency(hits: list[str], window: int, data: st.DataObject) -> None:
    n = data.draw(st.integers(1, 13))
    m = data.draw(st.integers(1, n))
    assert _expand(hits, window, top_k=m) == _expand(hits, window, top_k=n)[:m]


@settings(max_examples=100)
@given(hits=_hits, window=st.integers(0, 3))
def test_property_output_within_window(hits: list[str], window: int) -> None:
    out = _expand(hits, window, top_k=13)
    allowed = set().union(*(_neighborhood(h, window) for h in hits))
    assert set(out) <= allowed
    assert len(out) == len(set(out))
    assert len(out) <= 13


@settings(max_examples=100)
@given(hits=_hits, k=st.integers(1, 13))
def test_property_window_zero_identity(hits: list[str], k: int) -> None:
    assert _expand(hits, 0, top_k=k) == hits[:k]


@settings(max_examples=100)
@given(hits=_hits, window=st.integers(0, 3))
def test_property_hits_keep_relative_order(hits: list[str], window: int) -> None:
    """驗收條件的顯性修訂(2026-09-14 GREEN):後面的命中若已在前面命中的鄰域內,依條目 6
    會以鄰塊身分提前;因此只對「未被前面任何命中鄰域吸收」的命中要求相對順序不變。"""
    out = _expand(hits, window, top_k=13)
    independent: list[str] = []
    covered: set[str] = set()
    for h in hits:
        if h not in covered:
            independent.append(h)
        covered |= _neighborhood(h, window)
    assert [h for h in out if h in independent] == [h for h in independent if h in out]
    assert out[0] == hits[0]


# 13 runner 整合


def _chunks_file(tmp_path: Path) -> Path:
    rows = [
        {"chunk_id": f"c{i}", "doc_id": "D", "text": "<甲乙丙>" if i == 5 else "<丁戊己>"}
        for i in range(8)
    ]
    return _write_jsonl(tmp_path / "chunks.jsonl", rows)


def test_runner_neighbor_window_bm25(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    chunks = _chunks_file(tmp_path)
    plain = build_retriever(RetrieverConfig("bm25", {"chunks": str(chunks)}))
    wrapped = build_retriever(
        RetrieverConfig("bm25", {"chunks": str(chunks), "neighbor_window": 1})
    )
    assert list(plain.retrieve("甲乙丙", 3)) == ["c5"]
    assert list(wrapped.retrieve("甲乙丙", 3)) == ["c5", "c4", "c6"]
    assert isinstance(wrapped, NeighborExpand)


def test_runner_window_zero_unwrapped(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    chunks = _chunks_file(tmp_path)
    r = build_retriever(RetrieverConfig("bm25", {"chunks": str(chunks), "neighbor_window": 0}))
    assert not isinstance(r, NeighborExpand)


def test_runner_neighbor_window_dense(tmp_path: Path) -> None:
    import numpy as np

    from selectrag.retrieval.dense import IndexMetadata, save_index
    from selectrag.runner.__main__ import build_retriever

    chunks = _chunks_file(tmp_path)
    ids = [f"c{i}" for i in range(8)]
    vecs = np.eye(8, dtype=np.float32)
    meta = IndexMetadata("<m>", "<r>", 8, (), "0" * 64, True, True)
    idx = tmp_path / "idx.npz"
    save_index(idx, ids, vecs, meta)

    class Emb:
        def embed(self, texts: Sequence[str]) -> np.ndarray:
            return np.eye(8, dtype=np.float32)[[5]]

    cfg = RetrieverConfig(
        "dense",
        {
            "index": str(idx),
            "model": "<m>",
            "revision": "<r>",
            "chunks": str(chunks),
            "neighbor_window": 1,
        },
    )
    r = build_retriever(cfg, embedder_factory=lambda m, rev: Emb())
    assert list(r.retrieve(Q, 3)) == ["c5", "c4", "c6"]


def test_dense_window_requires_chunks_param(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    cfg = RetrieverConfig("dense", {"index": "x", "model": "<m>", "neighbor_window": 1})
    with pytest.raises(ValueError, match="chunks"):
        build_retriever(cfg, embedder_factory=lambda m, rev: None)  # type: ignore[arg-type,return-value]
