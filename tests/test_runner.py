"""runner 與檢索器介面測試 —— 場景為 2026-08-27 定案的驗收條件。

所有題目與 doc id 皆為中性占位符,非真實醫療內容。
"""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.evaluation.eval_set import EvalItem, EvalSet, compute_content_hash, freeze
from selectrag.retrieval.scripted import ScriptedRetriever
from selectrag.runner.config import ExperimentConfig, RetrieverConfig, load_config
from selectrag.runner.experiment import read_result, run_experiment, write_result

# --------------------------------------------------------------------------
# 測試資料(合成)
# --------------------------------------------------------------------------

Q1, Q2, Q3 = "<問題文字 1>", "<問題文字 2>", "<問題文字 3>"


def make_eval_set() -> EvalSet:
    items = (
        EvalItem("q001", Q1, frozenset({"doc_b", "doc_x"})),
        EvalItem("q002", Q2, frozenset({"doc_f"})),
        EvalItem("q003", Q3, frozenset({"doc_z"})),
    )
    return EvalSet(version="v0-synthetic", frozen_at="2026-08-27", items=items)


def make_script() -> dict[str, list[str]]:
    return {
        Q1: ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"],  # 真值 2 筆,rank2 命中 1 筆
        Q2: ["doc_f", "doc_g", "doc_h"],  # 真值 1 筆,rank1 命中
        Q3: ["doc_i", "doc_j"],  # 完全未命中
    }


def make_config(tmp_path: Path, k_values: tuple[int, ...] = (1, 3, 5)) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="synthetic-baseline",
        eval_set=tmp_path / "eval.json",
        retriever=RetrieverConfig(type="scripted", params={}),
        k_values=k_values,
        results_dir=tmp_path / "results",
    )


def write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


VALID_YAML = """
experiment_id: synthetic-baseline
eval_set: data/eval/eval_v0_synthetic.json
retriever:
  type: scripted
  params: {}
k_values: [1, 3, 5]
results_dir: results
"""


# --------------------------------------------------------------------------
# 設定檔載入
# --------------------------------------------------------------------------


def test_load_valid_config(tmp_path: Path) -> None:
    config = load_config(write_yaml(tmp_path / "c.yaml", VALID_YAML))
    assert config.experiment_id == "synthetic-baseline"
    assert config.k_values == (1, 3, 5)
    assert config.retriever.type == "scripted"
    assert config.eval_set == Path("data/eval/eval_v0_synthetic.json")
    assert config.results_dir == Path("results")


def test_load_config_rejects_missing_field(tmp_path: Path) -> None:
    body = VALID_YAML.replace("k_values: [1, 3, 5]\n", "")
    with pytest.raises(ValueError, match="missing field"):
        load_config(write_yaml(tmp_path / "c.yaml", body))


def test_load_config_rejects_empty_k_values(tmp_path: Path) -> None:
    body = VALID_YAML.replace("k_values: [1, 3, 5]", "k_values: []")
    with pytest.raises(ValueError, match="k_values is empty"):
        load_config(write_yaml(tmp_path / "c.yaml", body))


@pytest.mark.parametrize("bad_k", ["[0, 3]", "[1, -2]"])
def test_load_config_rejects_non_positive_k(tmp_path: Path, bad_k: str) -> None:
    body = VALID_YAML.replace("k_values: [1, 3, 5]", f"k_values: {bad_k}")
    with pytest.raises(ValueError, match="k must be >= 1"):
        load_config(write_yaml(tmp_path / "c.yaml", body))


def test_load_config_rejects_blank_experiment_id(tmp_path: Path) -> None:
    body = VALID_YAML.replace("experiment_id: synthetic-baseline", 'experiment_id: "  "')
    with pytest.raises(ValueError, match="experiment_id is empty"):
        load_config(write_yaml(tmp_path / "c.yaml", body))


def test_load_config_rejects_missing_retriever_type(tmp_path: Path) -> None:
    body = VALID_YAML.replace("  type: scripted\n", "")
    with pytest.raises(ValueError, match="missing field"):
        load_config(write_yaml(tmp_path / "c.yaml", body))


# --------------------------------------------------------------------------
# ScriptedRetriever
# --------------------------------------------------------------------------


def test_scripted_retriever_returns_script() -> None:
    retriever = ScriptedRetriever(make_script())
    assert list(retriever.retrieve(Q2, top_k=3)) == ["doc_f", "doc_g", "doc_h"]


def test_scripted_retriever_truncates_to_top_k() -> None:
    retriever = ScriptedRetriever(make_script())
    assert list(retriever.retrieve(Q1, top_k=2)) == ["doc_a", "doc_b"]


def test_scripted_retriever_rejects_unknown_question() -> None:
    retriever = ScriptedRetriever(make_script())
    with pytest.raises(ValueError, match="no scripted result"):
        retriever.retrieve("<未在腳本中的問題>", top_k=3)


# --------------------------------------------------------------------------
# 執行實驗
# --------------------------------------------------------------------------


def test_run_experiment_computes_expected_metrics(tmp_path: Path) -> None:
    """手算真值:
    q001 真值{doc_b,doc_x}: recall@1=0, recall@3=1/2, recall@5=1/2, RR=1/2
    q002 真值{doc_f}:       recall@1=1, recall@3=1,   recall@5=1,   RR=1
    q003 真值{doc_z}:       全 0,                                   RR=0
    平均: recall@1=1/3, recall@3=1/2, recall@5=1/2, MRR=1/2
    """
    result = run_experiment(
        make_config(tmp_path), make_eval_set(), ScriptedRetriever(make_script()), run_id="r1"
    )
    assert result.metrics["recall@1"] == pytest.approx(1 / 3)
    assert result.metrics["recall@3"] == pytest.approx(1 / 2)
    assert result.metrics["recall@5"] == pytest.approx(1 / 2)
    assert result.metrics["mrr"] == pytest.approx(1 / 2)


def test_run_experiment_per_query_matches_eval_set_order(tmp_path: Path) -> None:
    result = run_experiment(
        make_config(tmp_path), make_eval_set(), ScriptedRetriever(make_script()), run_id="r1"
    )
    assert [q.item_id for q in result.per_query] == ["q001", "q002", "q003"]
    assert result.per_query[0].recall["recall@3"] == pytest.approx(0.5)
    assert result.per_query[1].reciprocal_rank == 1.0
    assert result.per_query[2].reciprocal_rank == 0.0


def test_run_experiment_records_eval_set_identity(tmp_path: Path) -> None:
    """結果必須記錄用哪一版 eval set 跑的 —— 失效模式 2。"""
    eval_set = make_eval_set()
    result = run_experiment(
        make_config(tmp_path), eval_set, ScriptedRetriever(make_script()), run_id="r1"
    )
    assert result.eval_set_version == "v0-synthetic"
    assert result.eval_set_hash == compute_content_hash(eval_set.items)


def test_run_experiment_records_config_verbatim(tmp_path: Path) -> None:
    """結果記錄的設定必須與傳入的完全相同 —— 失效模式 3。"""
    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    assert result.config == config


def test_run_experiment_propagates_retrieval_failure(tmp_path: Path) -> None:
    """某題失敗即中止,不產出部分結果 —— 失效模式 4。"""
    partial = {k: v for k, v in make_script().items() if k != Q2}
    with pytest.raises(ValueError, match="no scripted result"):
        run_experiment(
            make_config(tmp_path), make_eval_set(), ScriptedRetriever(partial), run_id="r1"
        )


def test_run_experiment_honours_all_k_values(tmp_path: Path) -> None:
    """指標名稱與 k 值必須對應 —— 失效模式 5。"""
    config = make_config(tmp_path, k_values=(1, 2, 10))
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    assert set(result.metrics) == {
        "recall@1",
        "recall@2",
        "recall@10",
        "hit@1",
        "hit@2",
        "hit@10",  # 2026-09-01:hit@k 併入聚合
        "mrr",
    }


# --------------------------------------------------------------------------
# 結果寫檔與覆蓋保護
# --------------------------------------------------------------------------


def test_write_result_round_trip(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    path = write_result(result, config.results_dir)
    assert path.exists()
    assert read_result(path) == result


def test_write_result_refuses_to_overwrite(tmp_path: Path) -> None:
    """覆蓋即永久損失 —— 失效模式 1。"""
    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    path = write_result(result, config.results_dir)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_result(result, config.results_dir)
    assert path.read_text(encoding="utf-8") == before, "原檔案不得被更動"


def test_write_result_path_includes_experiment_and_run_id(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    path = write_result(result, config.results_dir)
    assert path.parent.name == "synthetic-baseline"
    assert path.name == "r1.json"


def test_written_result_is_valid_json_with_chinese(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    payload = json.loads(write_result(result, config.results_dir).read_text(encoding="utf-8"))
    assert payload["eval_set_version"] == "v0-synthetic"
    assert payload["config"]["experiment_id"] == "synthetic-baseline"


# --------------------------------------------------------------------------
# 合成 eval set 資產
# --------------------------------------------------------------------------


def test_synthetic_eval_set_is_loadable_and_neutral() -> None:
    """入版控的合成資料必須可載入,且題目為中性占位符。"""
    from selectrag.evaluation.eval_set import load_eval_set

    eval_set = load_eval_set(Path("data/eval/eval_v0_synthetic.json"))
    assert eval_set.version == "v0-synthetic"
    assert len(eval_set.items) == 10
    for item in eval_set.items:
        assert item.question.startswith("<") and item.question.endswith(">")


# --------------------------------------------------------------------------
# Property-based(Tier 3)
# --------------------------------------------------------------------------

k_lists = st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4, unique=True)


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(k_values=k_lists)
def test_property_per_query_count_matches_eval_set(
    k_values: list[int], tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp = tmp_path_factory.mktemp("prop")
    config = make_config(tmp, k_values=tuple(k_values))
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    assert len(result.per_query) == len(make_eval_set().items)


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(k_values=k_lists)
def test_property_recall_in_unit_interval_and_monotonic(
    k_values: list[int], tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp = tmp_path_factory.mktemp("prop")
    config = make_config(tmp, k_values=tuple(sorted(k_values)))
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    values = [result.metrics[f"recall@{k}"] for k in sorted(k_values)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values), "k 遞增時 recall 不得下降"


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(k_values=k_lists, run_id=st.text(alphabet="abc123", min_size=1, max_size=5))
def test_property_result_round_trip(
    k_values: list[int], run_id: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp = tmp_path_factory.mktemp("prop")
    config = make_config(tmp, k_values=tuple(k_values))
    result = run_experiment(
        config, make_eval_set(), ScriptedRetriever(make_script()), run_id=run_id
    )
    path = write_result(result, config.results_dir)
    assert read_result(path) == result


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(run_id=st.text(alphabet="abc123", min_size=1, max_size=5))
def test_property_second_write_always_refuses(
    run_id: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """對任意 run_id,第二次寫入必被拒且原檔不變 —— 失效模式 1 的核心不變量。"""
    tmp = tmp_path_factory.mktemp("prop")
    config = make_config(tmp)
    result = run_experiment(
        config, make_eval_set(), ScriptedRetriever(make_script()), run_id=run_id
    )
    path = write_result(result, config.results_dir)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_result(result, config.results_dir)
    assert path.read_text(encoding="utf-8") == before


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(k_values=k_lists)
def test_property_records_correct_eval_set_hash(
    k_values: list[int], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """結果記錄的 hash 必須等於獨立計算值,且不受 k_values 影響。

    補洞:round-trip 不變量在「hash 記成空字串」的實作下依然成立
    (property-only 變異測試證實),必須明寫 hash 的正確性。
    """
    eval_set = make_eval_set()
    config = make_config(tmp_path_factory.mktemp("hash"), k_values=tuple(k_values))
    result = run_experiment(config, eval_set, ScriptedRetriever(make_script()), run_id="r1")
    assert result.eval_set_hash == compute_content_hash(eval_set.items)
    assert result.eval_set_version == eval_set.version


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(k_values=k_lists)
def test_property_retrieval_depth_matches_max_k(
    k_values: list[int], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """檢索深度必須是 max(k_values),否則大 k 的指標根本沒看到那麼多結果。

    補洞:單調性不變量抓不到 depth=min(k) 的錯誤——那會讓所有 recall@k 相等,
    而「相等」依然滿足「不下降」(property-only 變異測試證實)。
    """
    script = make_script()
    config = make_config(tmp_path_factory.mktemp("depth"), k_values=tuple(k_values))
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(script), run_id="r1")
    depth = max(k_values)
    for query_result, question in zip(result.per_query, [Q1, Q2, Q3], strict=True):
        assert len(query_result.retrieved) == min(depth, len(script[question]))


# --------------------------------------------------------------------------
# CLI(一行指令跑完)
# --------------------------------------------------------------------------


def _prepare_cli_workspace(tmp_path: Path) -> Path:
    """在 tmp_path 建出 eval set、檢索腳本與設定檔,回傳設定檔路徑。"""
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            freeze(list(make_eval_set().items), version="v0-synthetic", frozen_at="2026-08-27"),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(make_script(), ensure_ascii=False), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment_id: cli-test
eval_set: {eval_path}
retriever:
  type: scripted
  params:
    script: {script_path}
k_values: [1, 3, 5]
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from selectrag.runner.__main__ import main

    config_path = _prepare_cli_workspace(tmp_path)
    assert main(["prog", str(config_path), "cli-run"]) == 0

    out = capsys.readouterr().out
    assert "recall@1" in out and "mrr" in out
    assert "v0-synthetic" in out
    assert (tmp_path / "results" / "cli-test" / "cli-run.json").exists()


def test_cli_without_arguments_returns_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    from selectrag.runner.__main__ import main

    assert main(["prog"]) == 2
    assert "用法" in capsys.readouterr().err


def test_cli_generates_run_id_when_omitted(tmp_path: Path) -> None:
    """未指定 run_id 時自動以 UTC 時間戳命名。"""
    from selectrag.runner.__main__ import main

    config_path = _prepare_cli_workspace(tmp_path)
    assert main(["prog", str(config_path)]) == 0
    written = list((tmp_path / "results" / "cli-test").glob("*.json"))
    assert len(written) == 1
    assert written[0].stem.endswith("Z")


def test_build_retriever_rejects_unknown_type() -> None:
    from selectrag.runner.__main__ import build_retriever

    with pytest.raises(ValueError, match="unknown retriever type"):
        build_retriever(RetrieverConfig(type="<不存在的型別>", params={}))


def test_build_retriever_requires_script_param() -> None:
    from selectrag.runner.__main__ import build_retriever

    with pytest.raises(ValueError, match="requires params.script"):
        build_retriever(RetrieverConfig(type="scripted", params={}))


def test_format_report_includes_identity_and_metrics(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import format_report

    config = make_config(tmp_path)
    result = run_experiment(config, make_eval_set(), ScriptedRetriever(make_script()), run_id="r1")
    report = format_report(result, tmp_path / "r1.json")
    assert "synthetic-baseline" in report
    assert "v0-synthetic" in report
    assert "recall@3" in report


def test_build_scored_scripted_retriever(tmp_path: Path) -> None:
    from selectrag.retrieval.scripted import ScoredScriptedRetriever
    from selectrag.runner.__main__ import build_retriever
    from selectrag.runner.config import RetrieverConfig

    script = tmp_path / "s.json"
    script.write_text(json.dumps({"<問>": [["c1", 0.9], ["c2", 0.1]]}), encoding="utf-8")
    r = build_retriever(RetrieverConfig("scripted", {"script": str(script), "scored": True}))
    assert isinstance(r, ScoredScriptedRetriever)
    assert list(r.retrieve_scored("<問>", 1)) == [("c1", 0.9)]
    assert list(r.retrieve("<問>", 2)) == ["c1", "c2"]
    with pytest.raises(ValueError):
        r.retrieve_scored("<無>", 1)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"<問>": [["c1"]]}), encoding="utf-8")
    with pytest.raises(ValueError):
        build_retriever(RetrieverConfig("scripted", {"script": str(bad), "scored": True}))
