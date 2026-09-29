"""RRF hybrid 測試 —— 場景為 2026-09-14 定案的驗收條件。id / 文字皆為中性占位符。"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.retrieval.hybrid import HybridRRF, rrf_fuse
from selectrag.retrieval.scripted import ScriptedRetriever
from selectrag.runner.config import RetrieverConfig

TOOLS = Path(__file__).resolve().parent.parent / "tools"
Q = "<問題>"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingBase:
    def __init__(self, result: Sequence[str]) -> None:
        self.calls: list[int] = []
        self._result = list(result)

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        self.calls.append(top_k)
        return self._result[:top_k]


def _hybrid(a: Sequence[str], b: Sequence[str], pool: int = 10, rrf_k: float = 60.0) -> HybridRRF:
    return HybridRRF([ScriptedRetriever({Q: a}), ScriptedRetriever({Q: b})], pool, rrf_k)


# 1~5 rrf_fuse


def test_rrf_single_ranking_is_identity() -> None:
    assert rrf_fuse([["x", "y", "z"]], 60) == ["x", "y", "z"]


def test_rrf_hand_computed() -> None:
    # y: 1/61 + 1/62 = 0.03252;z: 1/62 + 1/63 = 0.03200;x: 1/61 = 0.01639;w: 1/63 = 0.01587
    assert rrf_fuse([["x", "y", "z"], ["y", "z", "w"]], 60) == ["y", "z", "x", "w"]


def test_rrf_constant_changes_ranking() -> None:
    """p 名次 (1, 10)、q 名次 (3, 3):小常數獎勵「一邊極前」,大常數獎勵「兩邊都中等」。

    rrf_k=1:   p = 1/2 + 1/11 = 0.591;q = 1/4 + 1/4 = 0.500 → p 先
    rrf_k=1000:p = 1/1001 + 1/1010 = 0.001989;q = 2/1003 = 0.001994 → q 先
    """
    a = ["p", "x1", "q"]
    b = ["x2", "x3", "q", "x4", "x5", "x6", "x7", "x8", "x9", "p"]
    assert rrf_fuse([a, b], 1)[:2] == ["p", "q"]
    assert rrf_fuse([a, b], 1000)[:2] == ["q", "p"]


def test_rrf_tie_breaks_by_first_appearance() -> None:
    a, b = ["m", "n"], ["n", "m"]
    assert rrf_fuse([a, b], 60) == ["m", "n"]
    assert rrf_fuse([b, a], 60) == ["n", "m"]


def test_rrf_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="rankings"):
        rrf_fuse([], 60)
    with pytest.raises(ValueError, match="rrf_k"):
        rrf_fuse([["x"]], 0)
    with pytest.raises(ValueError, match="duplicate"):
        rrf_fuse([["x", "x"]], 60)


# 6~10 HybridRRF


def test_hybrid_calls_each_base_with_pool() -> None:
    a, b = RecordingBase([f"a{i}" for i in range(30)]), RecordingBase([f"b{i}" for i in range(30)])
    out = HybridRRF([a, b], pool=20, rrf_k=60).retrieve(Q, 5)
    assert a.calls == [20] and b.calls == [20]
    assert len(out) == 5


def test_hybrid_truncates_to_top_k() -> None:
    assert len(_hybrid(["x", "y", "z"], ["y", "z", "w"]).retrieve(Q, 2)) == 2
    assert list(_hybrid(["x", "y", "z"], ["y", "z", "w"]).retrieve(Q, 2)) == ["y", "z"]


def test_hybrid_top_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _hybrid(["x"], ["y"]).retrieve(Q, 0)


def test_hybrid_requires_two_retrievers() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        HybridRRF([ScriptedRetriever({Q: ["x"]})], 10, 60)


def test_hybrid_rejects_bad_pool_and_k() -> None:
    with pytest.raises(ValueError, match="pool"):
        _hybrid(["x"], ["y"], pool=0)
    with pytest.raises(ValueError, match="rrf_k"):
        _hybrid(["x"], ["y"], rrf_k=0)


_ids = st.lists(st.sampled_from([f"d{i}" for i in range(12)]), min_size=1, max_size=8, unique=True)


@settings(max_examples=100)
@given(a=_ids, b=_ids, k=st.integers(1, 10), pool=st.integers(1, 8))
def test_hybrid_union_property(a: list[str], b: list[str], k: int, pool: int) -> None:
    out = list(_hybrid(a, b, pool=pool).retrieve(Q, k))
    assert set(out) <= set(a[:pool]) | set(b[:pool])
    assert len(out) == len(set(out)) and len(out) <= k


@settings(max_examples=100)
@given(a=_ids, b=_ids, data=st.data())
def test_hybrid_prefix_property(a: list[str], b: list[str], data: st.DataObject) -> None:
    n = data.draw(st.integers(1, 12))
    m = data.draw(st.integers(1, n))
    h = _hybrid(a, b)
    assert list(h.retrieve(Q, m)) == list(h.retrieve(Q, n))[:m]


@settings(max_examples=150)
@given(a=_ids, b=_ids)
def test_hybrid_dominance_property(a: list[str], b: list[str]) -> None:
    """RRF 支配性:x 在每份清單的名次都不劣於 y(缺席 = 無限深)且至少一處嚴格較前 → x 在 y 前。

    對任何遞減的名次函數都成立,是 RRF 的核心語意;殺「排序方向反轉」類變異。
    """
    out = list(_hybrid(a, b, pool=8).retrieve(Q, 16))
    pos = {d: i for i, d in enumerate(out)}
    inf = 10**9

    def rank(lst: list[str], d: str) -> int:
        return lst.index(d) if d in lst else inf

    for x in out:
        for y in out:
            rx, ry = (rank(a, x), rank(b, x)), (rank(a, y), rank(b, y))
            if rx[0] <= ry[0] and rx[1] <= ry[1] and rx != ry:
                assert pos[x] < pos[y], (x, y, rx, ry)


@settings(max_examples=50)
@given(a=_ids, k=st.integers(1, 12))
def test_hybrid_symmetric_when_rankings_identical(a: list[str], k: int) -> None:
    assert list(_hybrid(a, a).retrieve(Q, k)) == a[:k]


# 11 runner


def test_build_retriever_hybrid_nested(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    sa = tmp_path / "a.json"
    sb = tmp_path / "b.json"
    sa.write_text(json.dumps({Q: ["x", "y", "z"]}), encoding="utf-8")
    sb.write_text(json.dumps({Q: ["y", "z", "w"]}), encoding="utf-8")
    cfg = RetrieverConfig(
        "hybrid",
        {
            "pool": 10,
            "rrf_k": 60,
            "retrievers": [
                {"type": "scripted", "params": {"script": str(sa)}},
                {"type": "scripted", "params": {"script": str(sb)}},
            ],
        },
    )
    r = build_retriever(cfg)
    assert isinstance(r, HybridRRF)
    assert list(r.retrieve(Q, 4)) == ["y", "z", "x", "w"]


def test_hybrid_requires_pool_rrf_k_and_retrievers() -> None:
    from selectrag.runner.__main__ import build_retriever

    base = [{"type": "scripted", "params": {"script": "x"}}] * 2
    with pytest.raises(ValueError, match="retrievers"):
        build_retriever(RetrieverConfig("hybrid", {"pool": 10, "rrf_k": 60}))
    with pytest.raises(ValueError, match="pool"):
        build_retriever(RetrieverConfig("hybrid", {"rrf_k": 60, "retrievers": base}))
    with pytest.raises(ValueError, match="rrf_k"):
        build_retriever(RetrieverConfig("hybrid", {"pool": 10, "retrievers": base}))


def test_hybrid_nested_dense_uses_factory(tmp_path: Path) -> None:
    import numpy as np

    from selectrag.retrieval.dense import IndexMetadata, save_index
    from selectrag.runner.__main__ import build_retriever

    idx = tmp_path / "idx.npz"
    save_index(
        idx,
        ["c0", "c1", "c2"],
        np.eye(3, dtype=np.float32),
        IndexMetadata("<m>", "<r>", 3, (), "0" * 64, True, True),
    )
    sa = tmp_path / "a.json"
    sa.write_text(json.dumps({Q: ["c2", "c0"]}), encoding="utf-8")

    class Emb:
        def embed(self, texts: Sequence[str]) -> np.ndarray:
            return np.eye(3, dtype=np.float32)[[1]]

    cfg = RetrieverConfig(
        "hybrid",
        {
            "pool": 3,
            "rrf_k": 60,
            "retrievers": [
                {"type": "scripted", "params": {"script": str(sa)}},
                {"type": "dense", "params": {"index": str(idx), "model": "<m>", "revision": "<r>"}},
            ],
        },
    )
    r = build_retriever(cfg, embedder_factory=lambda m, rev: Emb())
    # dense 回 [c1, c0, c2](c1 為 1,其餘 0 同分依插入序);scripted 回 [c2, c0]
    # c0: 1/62 + 1/62;c2: 1/61 + 1/63;c1: 1/61 → c2 > c0 > c1
    assert list(r.retrieve(Q, 3)) == ["c2", "c0", "c1"]


# 12 tools/hybrid_grid.py


def _result_file(path: Path, retrieved: dict[str, list[str]]) -> Path:
    path.write_text(
        json.dumps({"per_query": [{"item_id": k, "retrieved": v} for k, v in retrieved.items()]}),
        encoding="utf-8",
    )
    return path


def test_union_bound_hand_computed(tmp_path: Path) -> None:
    mod = _load_tool("hybrid_grid")
    truth = {"q1": {"t1", "t2"}, "q2": {"t3"}}
    a = _result_file(tmp_path / "a.json", {"q1": ["t1", "x", "y"], "q2": ["x", "y", "z"]})
    b = _result_file(tmp_path / "b.json", {"q1": ["x", "t2", "y"], "q2": ["t3", "x", "y"]})
    rows = mod.union_bound(truth, a, b, depths=(1, 2, 3))
    # N=1:A: q1 1/2, q2 0 → 0.25;B: q1 0, q2 1 → 0.5;聯集: q1 1/2, q2 1 → 0.75
    # N=2:A 0.25;B: q1 1/2, q2 1 → 0.75;聯集: q1 1, q2 1 → 1.0
    assert rows[1] == {
        "a_recall": 0.25,
        "b_recall": 0.5,
        "union_recall": 0.75,
        "a_hit": 0.5,
        "b_hit": 0.5,
        "union_hit": 1.0,
    }
    assert rows[2]["union_recall"] == 1.0 and rows[2]["b_recall"] == 0.75
    assert rows[3]["a_recall"] == 0.25


def test_grid_runs_with_scripted_bases_and_refuses_overwrite(tmp_path: Path) -> None:
    from selectrag.evaluation.eval_set import EvalItem, freeze

    mod = _load_tool("hybrid_grid")
    eval_path = tmp_path / "e.json"
    items = [EvalItem("q1", "<q1>", frozenset({"t1"})), EvalItem("q2", "<q2>", frozenset({"t3"}))]
    eval_path.write_text(json.dumps(freeze(items, "t", "t")), encoding="utf-8")
    sa = tmp_path / "a.json"
    sb = tmp_path / "b.json"
    sa.write_text(json.dumps({"<q1>": ["t1", "x"], "<q2>": ["x", "y"]}), encoding="utf-8")
    sb.write_text(json.dumps({"<q1>": ["x", "t1"], "<q2>": ["t3", "x"]}), encoding="utf-8")
    for name, script in (("a", sa), ("b", sb)):
        (tmp_path / f"{name}.yaml").write_text(
            f"experiment_id: {name}\neval_set: {eval_path}\nretriever:\n  type: scripted\n"
            f"  params:\n    script: {script}\nk_values: [1, 2]\nresults_dir: {tmp_path / 'r'}\n",
            encoding="utf-8",
        )
    out = tmp_path / "grid"
    rc = mod.main(
        [
            str(tmp_path / "a.yaml"),
            str(tmp_path / "b.yaml"),
            str(out),
            "--pools",
            "1",
            "2",
            "--rrf-ks",
            "10",
            "60",
        ]
    )
    assert rc == 0
    files = sorted(p.name for p in out.glob("pool*.json"))
    assert files == ["pool1_k10.json", "pool1_k60.json", "pool2_k10.json", "pool2_k60.json"]
    got = json.loads((out / "pool2_k60.json").read_text(encoding="utf-8"))
    # q1:t1 與 x 同分(1/61+1/62),t1 先出現 → recall@1 = 1;q2:x 兩邊都有勝過只在 B 的 t3 → 0
    assert got["metrics"]["recall@1"] == pytest.approx(0.5)
    assert got["metrics"]["recall@2"] == pytest.approx(1.0)
    # 聯集上限:各 pool 深度的 A / B / 聯集 recall、hit 一併落地
    ub = json.loads((out / "union_bound.json").read_text(encoding="utf-8"))
    assert set(ub) == {"1", "2"}
    assert ub["2"]["union_recall"] == pytest.approx(1.0) and ub["1"][
        "union_recall"
    ] == pytest.approx(1.0)
    assert ub["1"]["a_recall"] == pytest.approx(0.5) and ub["1"]["b_recall"] == pytest.approx(0.5)
    assert mod.main([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"), str(out)]) != 0
