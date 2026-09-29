"""Dense 檢索器測試 —— 場景為 2026-09-13 定案的驗收條件。

所有向量為合成數值,所有 id / 文字皆為中性占位符;不載入任何真實模型。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from selectrag.retrieval.dense import (
    DenseRetriever,
    IndexMetadata,
    load_dense_from_index,
    load_index,
    save_index,
)
from selectrag.runner.config import RetrieverConfig

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 測試替身:查詢向量由字典直接給定(只 mock 模型邊界,不 mock 檢索邏輯)
# --------------------------------------------------------------------------


class TableEmbedder:
    def __init__(self, table: dict[str, Sequence[float]]) -> None:
        self._table = table

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._table[t] for t in texts], dtype=np.float32)


def _meta(dim: int = 3, fields: tuple[str, ...] = ()) -> IndexMetadata:
    return IndexMetadata(
        model="<model>",
        revision="<rev>",
        dim=dim,
        context_fields=fields,
        chunks_sha256="0" * 64,
        normalized=True,
        revision_pinned=True,
    )


# --------------------------------------------------------------------------
# 1~7 DenseRetriever 手算案例
# --------------------------------------------------------------------------

IDS = ["c1", "c2", "c3"]
VECS = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)


def test_top1_is_identical_vector() -> None:
    r = DenseRetriever(IDS, VECS, TableEmbedder({"q": [0, 1, 0]}))
    assert r.retrieve("q", 1) == ["c2"]


def test_ranking_by_cosine_desc() -> None:
    # q=(2,1,0):cos(c3)=3/(√5·√2)=0.949,cos(c1)=2/√5=0.894,cos(c2)=1/√5=0.447
    r = DenseRetriever(IDS, VECS, TableEmbedder({"q": [2, 1, 0]}))
    assert list(r.retrieve("q", 3)) == ["c3", "c1", "c2"]
    scores = [s for _, s in r.scored("q", 3)]
    assert scores == pytest.approx([0.9487, 0.8944, 0.4472], abs=1e-3)


def test_scale_invariance() -> None:
    scaled = VECS * np.asarray([[7.0], [0.01], [300.0]], dtype=np.float32)
    r_a = DenseRetriever(IDS, VECS, TableEmbedder({"q": [2, 1, 0]}))
    r_b = DenseRetriever(IDS, scaled, TableEmbedder({"q": [2, 1, 0]}))
    assert list(r_a.retrieve("q", 3)) == list(r_b.retrieve("q", 3))


def test_top_k_truncates() -> None:
    ids = [f"c{i}" for i in range(5)]
    vecs = np.eye(5, dtype=np.float32)
    r = DenseRetriever(ids, vecs, TableEmbedder({"q": [5, 4, 3, 2, 1]}))
    assert list(r.retrieve("q", 2)) == ["c0", "c1"]
    assert len(r.retrieve("q", 50)) == 5


def test_tie_breaks_by_insertion_order() -> None:
    vecs = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    r = DenseRetriever(["b", "a", "z"], vecs, TableEmbedder({"q": [1, 0]}))
    assert list(r.retrieve("q", 2)) == ["b", "a"]


def test_rejects_empty_index() -> None:
    with pytest.raises(ValueError, match="empty"):
        DenseRetriever([], np.zeros((0, 3), dtype=np.float32), TableEmbedder({}))


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DenseRetriever(["a", "a"], np.eye(2, dtype=np.float32), TableEmbedder({}))


def test_rejects_shape_mismatch_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        DenseRetriever(["a", "b", "c"], np.eye(2, dtype=np.float32), TableEmbedder({}))


def test_rejects_shape_mismatch_query_dim() -> None:
    r = DenseRetriever(["a", "b"], np.eye(2, dtype=np.float32), TableEmbedder({"q": [1, 0, 0]}))
    with pytest.raises(ValueError, match="dim"):
        r.retrieve("q", 1)


def test_rejects_nonfinite_rows() -> None:
    bad = np.asarray([[1, 0], [np.nan, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        DenseRetriever(["a", "b"], bad, TableEmbedder({}))


def test_rejects_zero_vector_row() -> None:
    bad = np.asarray([[1, 0], [0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero"):
        DenseRetriever(["a", "b"], bad, TableEmbedder({}))


def test_rejects_zero_query_vector() -> None:
    r = DenseRetriever(IDS, VECS, TableEmbedder({"q": [0, 0, 0]}))
    with pytest.raises(ValueError, match="non-zero"):
        r.retrieve("q", 1)


def test_rejects_top_k_below_one() -> None:
    r = DenseRetriever(IDS, VECS, TableEmbedder({"q": [1, 0, 0]}))
    with pytest.raises(ValueError, match="top_k"):
        r.retrieve("q", 0)


# --------------------------------------------------------------------------
# 8 property
# --------------------------------------------------------------------------

_small_matrix = hnp.arrays(
    dtype=np.float32,
    shape=st.tuples(st.integers(2, 8), st.integers(2, 5)),
    elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False, width=32),
).filter(lambda m: bool(np.all(np.linalg.norm(m, axis=1) > 1e-3)))


def _retriever(m: np.ndarray, q: Sequence[float]) -> DenseRetriever:
    return DenseRetriever([f"d{i}" for i in range(m.shape[0])], m, TableEmbedder({"q": q}))


@settings(max_examples=60)
@given(m=_small_matrix, data=st.data())
def test_property_topk_prefix(m: np.ndarray, data: st.DataObject) -> None:
    n = m.shape[0]
    q = data.draw(st.lists(st.floats(-3, 3, width=32), min_size=m.shape[1], max_size=m.shape[1]))
    if np.linalg.norm(q) <= 1e-3:
        q = [1.0] * m.shape[1]
    small = data.draw(st.integers(1, n))
    r = _retriever(m, q)
    assert list(r.retrieve("q", small)) == list(r.retrieve("q", n))[:small]


@settings(max_examples=60)
@given(m=_small_matrix, data=st.data())
def test_property_scores_monotone(m: np.ndarray, data: st.DataObject) -> None:
    q = data.draw(st.lists(st.floats(-3, 3, width=32), min_size=m.shape[1], max_size=m.shape[1]))
    if np.linalg.norm(q) <= 1e-3:
        q = [1.0] * m.shape[1]
    scores = [s for _, s in _retriever(m, q).scored("q", m.shape[0])]
    assert all(a >= b - 1e-6 for a, b in zip(scores, scores[1:], strict=False))
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


_scales = st.lists(st.floats(0.015625, 64.0, width=32), min_size=8, max_size=8)


@settings(max_examples=60)
@given(m=_small_matrix, scales=_scales)
def test_property_scale_invariance(m: np.ndarray, scales: list[float]) -> None:
    """縮放不變性成立到浮點精度:分數差距明顯(>1e-4)的任兩筆,縮放後相對順序不變。

    驗收條件的第二次顯性修訂(2026-09-14 gauntlet 隨機序 seed 22015 抓到):
    先前以 1e-5 分組比「同分群序」,近平手對落在容差邊界時,縮放引入的 1 ulp 會讓它
    一邊同群、一邊分開——那是分組法的脆弱,不是排序錯。改為只對「明顯分開」的對做斷言。
    """
    q = [1.0] * m.shape[1]
    s = np.asarray(scales[: m.shape[0]], dtype=np.float32)[:, None]
    base = dict(_retriever(m, q).scored("q", m.shape[0]))
    scaled_rank = {d: i for i, (d, _) in enumerate(_retriever(m * s, q).scored("q", m.shape[0]))}
    ids = list(base)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if abs(base[a] - base[b]) > 1e-4:
                assert (base[a] > base[b]) == (scaled_rank[a] < scaled_rank[b]), (a, b)


# --------------------------------------------------------------------------
# 9~11 索引檔
# --------------------------------------------------------------------------


def test_index_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    meta = _meta(fields=("doc_title",))
    save_index(path, IDS, VECS, meta)
    ids, vecs, meta_back = load_index(path)
    assert ids == IDS
    assert vecs.dtype == np.float32 and np.array_equal(vecs, VECS)
    assert meta_back == meta


def test_save_index_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    save_index(path, IDS, VECS, _meta())
    with pytest.raises(FileExistsError):
        save_index(path, IDS, VECS, _meta())


def test_load_dense_rejects_model_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    save_index(path, IDS, VECS, _meta())
    with pytest.raises(ValueError, match="model"):
        load_dense_from_index(path, TableEmbedder({}), model="<other>", revision="<rev>")
    with pytest.raises(ValueError, match="revision"):
        load_dense_from_index(path, TableEmbedder({}), model="<model>", revision="<other>")


def test_load_dense_from_index_retrieves(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    save_index(path, IDS, VECS, _meta())
    r = load_dense_from_index(path, TableEmbedder({"q": [0, 1, 0]}), "<model>", "<rev>")
    assert r.retrieve("q", 1) == ["c2"]


def test_load_rejects_file_without_index_arrays(tmp_path: Path) -> None:
    path = tmp_path / "not_index.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(ValueError, match="not a dense index"):
        load_index(path)


def test_load_rejects_missing_metadata_fields(tmp_path: Path) -> None:
    path = tmp_path / "idx.npz"
    np.savez(path, ids=np.asarray(IDS), vectors=VECS, metadata=json.dumps({"model": "x"}))
    with pytest.raises(ValueError, match="metadata"):
        load_index(path)


# --------------------------------------------------------------------------
# 12~14 建索引腳本純邏輯
# --------------------------------------------------------------------------


def _write_chunks(path: Path, n: int = 4) -> Path:
    rows = [
        {
            "chunk_id": f"c{i}",
            "text": f"<正文 {i}>",
            "doc_title": f"<標題 {i}>",
            "heading_path": [f"<節 {i}>", "<小節>"],
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return path


def test_index_text_matches_bm25_context_rule() -> None:
    from selectrag.retrieval.bm25 import index_text

    mod = _load_tool("build_dense_index")
    row = {
        "chunk_id": "c",
        "text": "<正文>",
        "doc_title": "<標題>",
        "heading_path": ["<甲>", "<乙>"],
    }
    for fields in ((), ("doc_title",), ("doc_title", "heading_path")):
        assert mod.index_text(row, fields, "row") == index_text(row, fields, "row")
    assert index_text(row, ("doc_title", "heading_path"), "row") == "<標題> <甲> <乙> <正文>"


def test_build_index_with_fake_embedder_end_to_end(tmp_path: Path) -> None:
    mod = _load_tool("build_dense_index")
    chunks = _write_chunks(tmp_path / "chunks.jsonl")
    out = tmp_path / "idx.npz"
    rc = mod.main(
        [
            "--chunks",
            str(chunks),
            "--out",
            str(out),
            "--embedder",
            "fake",
            "--context-fields",
            "doc_title",
            "heading_path",
        ]
    )
    assert rc == 0
    ids, vecs, meta = load_index(out)
    assert ids == ["c0", "c1", "c2", "c3"]
    assert vecs.shape == (4, mod.FAKE_DIM) and vecs.dtype == np.float32
    assert meta.context_fields == ("doc_title", "heading_path")
    assert meta.model == "fake" and meta.revision_pinned is False
    assert meta.chunks_sha256 == mod.sha256_of(chunks)
    assert meta.normalized is True
    # 同名輸出拒絕覆蓋
    assert mod.main(["--chunks", str(chunks), "--out", str(out), "--embedder", "fake"]) != 0


def test_fake_embedder_is_deterministic_and_unit_norm() -> None:
    mod = _load_tool("build_dense_index")
    a = mod.FakeHashEmbedder().embed(["<甲>", "<乙>"])
    b = mod.FakeHashEmbedder().embed(["<甲>", "<乙>"])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(a[0], a[1])


def test_build_index_skips_when_no_chunks(tmp_path: Path) -> None:
    mod = _load_tool("build_dense_index")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "idx.npz"
    assert mod.main(["--chunks", str(empty), "--out", str(out), "--embedder", "fake"]) != 0
    assert not out.exists()


# --------------------------------------------------------------------------
# 15~17 runner 整合
# --------------------------------------------------------------------------


def _fake_factory(model: str, revision: str | None) -> TableEmbedder:
    return TableEmbedder({"<問題文字 1>": [0, 1, 0], "<問題文字 2>": [1, 0, 0]})


def test_build_retriever_dense(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    path = tmp_path / "idx.npz"
    save_index(path, IDS, VECS, _meta())
    cfg = RetrieverConfig(
        type="dense", params={"index": str(path), "model": "<model>", "revision": "<rev>"}
    )
    r = build_retriever(cfg, embedder_factory=_fake_factory)
    assert isinstance(r, DenseRetriever)
    assert r.retrieve("<問題文字 1>", 1) == ["c2"]


def test_dense_requires_index_and_model_params() -> None:
    from selectrag.runner.__main__ import build_retriever

    with pytest.raises(ValueError, match="index"):
        build_retriever(RetrieverConfig(type="dense", params={"model": "m"}), _fake_factory)
    with pytest.raises(ValueError, match="model"):
        build_retriever(RetrieverConfig(type="dense", params={"index": "x"}), _fake_factory)


def test_runner_end_to_end_dense_fake(tmp_path: Path) -> None:
    from selectrag.evaluation.eval_set import EvalItem, EvalSet
    from selectrag.runner.config import ExperimentConfig
    from selectrag.runner.experiment import run_experiment

    path = tmp_path / "idx.npz"
    save_index(path, IDS, VECS, _meta())
    cfg = ExperimentConfig(
        experiment_id="dense-fake",
        eval_set=tmp_path / "unused.json",
        retriever=RetrieverConfig(
            type="dense", params={"index": str(path), "model": "<model>", "revision": "<rev>"}
        ),
        k_values=(1, 3),
        results_dir=tmp_path,
    )
    from selectrag.runner.__main__ import build_retriever

    retriever = build_retriever(cfg.retriever, embedder_factory=_fake_factory)
    eval_set = EvalSet(
        version="v0",
        frozen_at="2026-09-13",
        items=(
            EvalItem("q1", "<問題文字 1>", frozenset({"c2"})),
            EvalItem("q2", "<問題文字 2>", frozenset({"c9"})),
        ),
    )
    result = run_experiment(cfg, eval_set, retriever, run_id="r")
    assert result.metrics["recall@1"] == pytest.approx(0.5)
    assert result.metrics["hit@1"] == pytest.approx(0.5)
    assert {"recall@3", "hit@3", "mrr"} <= set(result.metrics)
    assert len(result.per_query) == 2
