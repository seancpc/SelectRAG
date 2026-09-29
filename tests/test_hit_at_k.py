"""hit@k 指標測試 —— 場景為 2026-09-01 定案的驗收條件。"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from selectrag.evaluation.retrieval_metrics import hit_at_k, recall_at_k


def test_hit_semantics_any_match_counts() -> None:
    """A1:5 個真值只找到 1 個 → recall=0.2 但 hit=1.0(b2-q03 實況)。"""
    retrieved = ["d1", "x1", "x2"]
    relevant = {"d1", "d2", "d3", "d4", "d5"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.2)
    assert hit_at_k(retrieved, relevant, 3) == 1.0


def test_hit_zero_when_truth_beyond_k() -> None:
    """A2:真值在第 k+1 名 → hit@k = 0.0。"""
    assert hit_at_k(["x1", "x2", "d1"], {"d1"}, 2) == 0.0
    assert hit_at_k(["x1", "x2", "d1"], {"d1"}, 3) == 1.0


def test_hit_bad_inputs_raise() -> None:
    """A3:與 recall 同套輸入驗證。"""
    with pytest.raises(ValueError, match="k must be >= 1"):
        hit_at_k(["d1"], {"d1"}, 0)
    with pytest.raises(ValueError, match="relevant set is empty"):
        hit_at_k(["d1"], set(), 1)
    with pytest.raises(ValueError, match="duplicate"):
        hit_at_k(["d1", "d1"], {"d1"}, 2)


_IDS = st.lists(st.sampled_from([f"d{i}" for i in range(8)]), unique=True, max_size=8)


@given(
    retrieved=_IDS,
    relevant=st.sets(st.sampled_from([f"d{i}" for i in range(8)]), min_size=1),
    k=st.integers(1, 10),
)
def test_property_hit_dominates_recall_and_is_binary(
    retrieved: list[str], relevant: set[str], k: int
) -> None:
    """A5:hit ≥ recall;hit 對 k 單調不減;單題值域 {0.0, 1.0}。"""
    hit = hit_at_k(retrieved, relevant, k)
    assert hit in (0.0, 1.0)
    assert hit >= recall_at_k(retrieved, relevant, k)
    assert hit_at_k(retrieved, relevant, k + 1) >= hit
    # 雙邊鎖定:hit 恰為「recall 是否非零」——殺死「恆報命中」與「忽略截斷」
    assert hit == (1.0 if recall_at_k(retrieved, relevant, k) > 0 else 0.0)


def test_experiment_metrics_include_hit(tmp_path) -> None:
    """A4:實驗結果 metrics 含 hit@k,值為各題平均。"""
    from selectrag.evaluation.eval_set import EvalItem, EvalSet
    from selectrag.retrieval.scripted import ScriptedRetriever
    from selectrag.runner.config import ExperimentConfig, RetrieverConfig
    from selectrag.runner.experiment import run_experiment

    items = (
        EvalItem(id="q1", question="alpha", relevant_doc_ids=frozenset({"d1", "d2"})),
        EvalItem(id="q2", question="beta", relevant_doc_ids=frozenset({"d9"})),
    )
    eval_set = EvalSet(version="t", frozen_at="t", items=items)
    retriever = ScriptedRetriever({"alpha": ["d1", "x1"], "beta": ["x2", "x3"]})
    config = ExperimentConfig(
        experiment_id="t",
        eval_set=tmp_path / "e.json",
        retriever=RetrieverConfig(type="scripted", params={}),
        k_values=(1, 2),
        results_dir=tmp_path,
    )
    result = run_experiment(config, eval_set, retriever, run_id="r1")
    # q1: hit@1=1, q2: hit@1=0 → 平均 0.5;q2 的 d9 不在腳本結果 → hit@2 也是 0
    assert result.metrics["hit@1"] == 0.5
    assert result.metrics["hit@2"] == 0.5
    assert result.per_query[0].hit == {"hit@1": 1.0, "hit@2": 1.0}
