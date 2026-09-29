"""demo 純邏輯層測試——場景為 2026-09-18 定案的驗收條件。文字皆為中性占位符。"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.demo import core
from selectrag.demo.core import (
    CiteStats,
    Hit,
    LLMConfig,
    Mark,
    PromptConfig,
    RetrieverPool,
    abstain_text,
    cite_stats,
    cited_indices,
    load_demo_config,
    locator_text,
    low_coverage,
    make_messages,
    mark_low_coverage,
    mark_truth,
    pick_eval_item,
    sentence_support,
    should_abstain,
    stream_answer,
    summarize_cites,
)
from selectrag.evaluation.eval_set import EvalItem, EvalSet, freeze
from selectrag.evaluation.faithfulness import coverage3
from selectrag.retrieval.scripted import ScriptedRetriever

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fake = _load_tool("fake_openai_server")


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    srv = fake.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _write_cfg(
    tmp_path: Path,
    presets: dict[str, str] | None = None,
    drop: str | None = None,
    abstain: dict[str, float] | None = None,
) -> Path:
    doc = {
        "presets": presets
        if presets is not None
        else {"left": "configs/a.yaml", "right": "configs/b.yaml"},
        "chunks": "results/chunks.jsonl",
        "eval_set": "data/eval/x.json",
        "prompt": {
            "top_k": 3,
            "system": "<s>",
            "template": "問題:{question}\n\n參考段落:\n{context}",
        },
        "llm": {
            "url": "http://127.0.0.1:1",
            "model": "<m>",
            "max_tokens": 64,
            "seed": 42,
            "timeout": 5,
        },
    }
    if abstain is not None:
        doc["abstain"] = abstain
    if drop:
        doc.pop(drop)
    p = tmp_path / "demo.yaml"
    import yaml

    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return p


# 7 不變量:core 不 import gradio
def test_core_does_not_import_gradio() -> None:
    assert "gradio" not in sys.modules
    import re

    src = Path(core.__file__).read_text(encoding="utf-8")  # 只看 import 陳述,不看註解/docstring
    assert not re.search(r"^\s*(import gradio|from gradio)", src, flags=re.MULTILINE)


# 1 設定
def test_load_demo_config_ok_and_errors(tmp_path: Path) -> None:
    cfg = load_demo_config(_write_cfg(tmp_path))
    assert set(cfg.presets) == {"left", "right"} and cfg.prompt.top_k == 3
    assert cfg.llm.model == "<m>" and cfg.llm.timeout == 5.0
    with pytest.raises(ValueError):
        load_demo_config(_write_cfg(tmp_path, presets={"only": "configs/a.yaml"}))
    for missing in ("chunks", "eval_set", "prompt", "llm"):
        with pytest.raises(ValueError):
            load_demo_config(_write_cfg(tmp_path, drop=missing))
    # 子區段缺鍵
    import yaml

    for section, key in (("prompt", "template"), ("llm", "model")):
        doc = yaml.safe_load(_write_cfg(tmp_path).read_text(encoding="utf-8"))
        doc[section].pop(key)
        bad = tmp_path / f"bad_{section}.yaml"
        bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError):
            load_demo_config(bad)


# 2 檢索快取
def test_retriever_pool_caches_per_preset(tmp_path: Path) -> None:
    rows = {
        "c1": {"chunk_id": "c1", "text": "<甲>", "doc_title": "<文件A>", "heading_path": "<節1>"},
        "c2": {"chunk_id": "c2", "text": "<乙>", "doc_title": "<文件B>"},
    }
    calls: list[str] = []

    def builder(preset: str) -> ScriptedRetriever:
        calls.append(preset)
        return ScriptedRetriever({"<問>": ["c2", "c1"]})

    cfg = load_demo_config(_write_cfg(tmp_path))
    pool = RetrieverPool(cfg, rows, builder)
    h1 = pool.hits("left", "<問>", 2)
    h2 = pool.hits("left", "<問>", 1)
    assert calls == ["left"]  # 同 preset 只建一次
    assert h1 == [Hit("c2", 1, "<文件B>", "", "<乙>"), Hit("c1", 2, "<文件A>", "<節1>", "<甲>")]
    assert h2 == [h1[0]]
    pool.hits("right", "<問>", 1)
    assert calls == ["left", "right"]
    with pytest.raises(KeyError):
        pool.hits("nope", "<問>", 1)


# 3 真值標色
def _hits(ids: list[str]) -> list[Hit]:
    return [Hit(c, i + 1, "", "", "") for i, c in enumerate(ids)]


def test_mark_truth_hand_computed() -> None:
    left, right = mark_truth(_hits(["a", "b", "c"]), _hits(["b", "d"]), ["b", "d"])
    assert left == [Mark("a", 1, False, True), Mark("b", 2, True, False), Mark("c", 3, False, True)]
    assert right == [Mark("b", 1, True, False), Mark("d", 2, True, True)]
    l2, r2 = mark_truth(_hits(["a"]), _hits(["a"]), [])
    assert l2 == [Mark("a", 1, False, False)] and r2 == [Mark("a", 1, False, False)]


@given(
    st.lists(st.text(alphabet="abcde", min_size=1, max_size=1), max_size=5, unique=True),
    st.lists(st.text(alphabet="abcde", min_size=1, max_size=1), max_size=5, unique=True),
    st.frozensets(st.text(alphabet="abcde", min_size=1, max_size=1), max_size=5),
)
@settings(max_examples=150)
def test_mark_truth_property(la: list[str], ra: list[str], truth: frozenset[str]) -> None:
    left, right = mark_truth(_hits(la), _hits(ra), truth)
    assert [m.chunk_id for m in left] == la and [m.chunk_id for m in right] == ra
    # in_truth 只看 truth,與另一欄無關;only_this_side 對稱:同時在兩欄的 chunk 兩邊皆 False
    for m in left + right:
        assert m.in_truth == (m.chunk_id in truth)
    both = set(la) & set(ra)
    for m in left + right:
        assert m.only_this_side == (m.chunk_id not in both)


# 4 引用檢查
def test_cited_indices_hand_computed() -> None:
    assert cited_indices("依 [1] 與 [3],[2] 無。", 3) == ({1, 2, 3}, set())
    assert cited_indices("依 [4] 與 [0]。", 3) == (set(), {0, 4})
    assert cited_indices("沒有引用。", 3) == (set(), set())
    assert cited_indices("[2][2]", 2) == ({2}, set())


@given(st.text(alphabet="[]0123456789甲 ", max_size=40), st.integers(1, 9))
@settings(max_examples=200)
def test_cited_indices_property(answer: str, n: int) -> None:
    valid, invalid = cited_indices(answer, n)
    assert valid <= set(range(1, n + 1))
    assert not (invalid & set(range(1, n + 1)))


# 5 prompt 與串流
def test_make_messages_uses_hit_texts() -> None:
    p = PromptConfig(3, "<s>", "問題:{question}\n\n參考段落:\n{context}")
    msgs = make_messages(
        p, "<問>", _hits(["a", "b"]) and [Hit("a", 1, "", "", "<甲>"), Hit("b", 2, "", "", "<乙>")]
    )
    assert msgs[0] == {"role": "system", "content": "<s>"}
    assert "[1] <甲>\n[2] <乙>" in msgs[1]["content"] and "<問>" in msgs[1]["content"]


def test_stream_answer_accumulates(server: str) -> None:
    llm = LLMConfig(server, "<m>", 64, 42, 5.0)
    parts = list(stream_answer(llm, [{"role": "user", "content": "<問>"}]))
    assert parts and parts[-1].startswith("<回覆:") and parts[-1].endswith(">")
    assert all(parts[i + 1].startswith(parts[i]) for i in range(len(parts) - 1))  # 累積
    assert len(parts) == len(parts[-1])  # 逐字


def test_stream_answer_reports_error_instead_of_raising() -> None:
    llm = LLMConfig("http://127.0.0.1:1", "<m>", 64, 42, 1.0)
    parts = list(stream_answer(llm, [{"role": "user", "content": "<問>"}]))
    assert len(parts) == 1 and parts[0].startswith("[錯誤]")


# 6 eval 題目
def test_pick_eval_item() -> None:
    es = EvalSet("v", "t", (EvalItem("q1", "<問1>", frozenset({"c1", "c2"})),))
    assert pick_eval_item(es, "q1") == ("<問1>", frozenset({"c1", "c2"}))
    with pytest.raises(KeyError):
        pick_eval_item(es, "q9")


# 7 引用統計(2026-09-18)
def _marks(truth_ranks: set[int], n: int = 3) -> list[Mark]:
    return [Mark(f"c{i}", i, i in truth_ranks, False) for i in range(1, n + 1)]


def test_cite_stats_hand_computed() -> None:
    st_ = cite_stats("A[1]。B[1][3]。C。", _marks({1}))
    assert st_ == CiteStats(frozenset({1, 3}), frozenset(), True, True, 3, 1)


def test_cite_stats_abstention_not_uncited() -> None:
    st_ = cite_stats("參考資料未提及。", _marks({1}))
    assert st_.sentences == 1 and st_.uncited_sentences == 0 and st_.valid == frozenset()


def test_cite_stats_out_of_range() -> None:
    st_ = cite_stats("依 [4]。", _marks({1}))
    assert st_.invalid == frozenset({4}) and st_.cites_truth is False


def test_cite_stats_truth_requires_valid() -> None:
    assert cite_stats("依 [2]。", _marks({1})).cites_truth is False
    assert cite_stats("依 [2]。", _marks({2})).cites_truth is True
    free = cite_stats("依 [2]。", _marks(set()))
    assert free.has_truth is False and free.cites_truth is False


def test_cite_stats_cite_after_terminator_belongs_to_prior_sentence() -> None:
    # 模型常把 [n] 放在句號之後:「…症狀。[3]」是一句已引用,不是「一句未引用 + 一段孤立標記」
    st_ = cite_stats("甲。[3]", _marks({3}))
    assert st_.sentences == 1 and st_.uncited_sentences == 0 and st_.cites_truth is True
    st2 = cite_stats("甲。[1] 乙。[2][3]", _marks({1}))
    assert st2.sentences == 2 and st2.uncited_sentences == 0


def test_cite_stats_splits_on_all_terminators() -> None:
    st_ = cite_stats("甲[1]!乙?丙[2]。", _marks(set()))
    assert st_.sentences == 3 and st_.uncited_sentences == 1


def test_summarize_cites_hand_computed() -> None:
    stats = [
        CiteStats(frozenset({1}), frozenset(), True, True, 2, 0),  # 有效、含真值
        CiteStats(frozenset({2}), frozenset({5}), True, False, 2, 1),  # 有效但越界、不含真值
        CiteStats(frozenset({3}), frozenset(), False, False, 1, 0),  # 有效但該題無真值(自由題)
        CiteStats(frozenset(), frozenset(), True, False, 1, 1),  # 無引用
    ]
    got = summarize_cites(stats)
    assert got["n"] == 4
    assert got["cite_rate"] == pytest.approx(3 / 4)
    assert got["invalid_rate"] == pytest.approx(1 / 4)
    assert got["truth_cite_n"] == 2 and got["truth_cite_rate"] == pytest.approx(1 / 2)
    assert got["uncited_sentence_rate"] == pytest.approx(2 / 6)


def test_summarize_cites_empty_and_zero_denominator() -> None:
    got = summarize_cites([])
    assert got["n"] == 0 and got["truth_cite_n"] == 0
    rates = ("cite_rate", "invalid_rate", "truth_cite_rate", "uncited_sentence_rate")
    assert all(got[k] == 0.0 for k in rates)
    only_invalid = summarize_cites([CiteStats(frozenset(), frozenset({9}), True, False, 0, 0)])
    assert only_invalid["truth_cite_n"] == 0 and only_invalid["uncited_sentence_rate"] == 0.0


_stat = st.builds(
    lambda v, inv, h, t, s, u: CiteStats(
        frozenset(v), frozenset(inv), h, bool(v) and h and t, s + u, u
    ),
    st.sets(st.integers(1, 3), max_size=3),
    st.sets(st.integers(4, 9), max_size=2),
    st.booleans(),
    st.booleans(),
    st.integers(0, 4),
    st.integers(0, 4),
)


@given(st.lists(_stat, max_size=12))
@settings(max_examples=200)
def test_summarize_cites_property(stats: list[CiteStats]) -> None:
    got = summarize_cites(stats)
    for k in ("cite_rate", "invalid_rate", "truth_cite_rate", "uncited_sentence_rate"):
        assert 0.0 <= got[k] <= 1.0
    # 有真值引用者必有 valid:truth_cite_n 不超過有效引用回答數
    n_valid = sum(1 for s_ in stats if s_.valid)
    assert got["truth_cite_n"] == sum(1 for s_ in stats if s_.valid and s_.has_truth) <= n_valid
    assert got["cite_rate"] == pytest.approx(n_valid / len(stats) if stats else 0.0)
    # 獨立重算:比例 × 分母 == 引用到真值的回答數(分母是「有 valid 且有真值」,不是 n)
    truth_hits = sum(1 for s_ in stats if s_.valid and s_.has_truth and s_.cites_truth)
    assert got["truth_cite_rate"] * got["truth_cite_n"] == pytest.approx(truth_hits)


@given(st.text(alphabet="甲乙。!?,", max_size=30))
@settings(max_examples=200)
def test_cite_stats_sentence_split_property(answer: str) -> None:
    st_ = cite_stats(answer, _marks({1}))
    assert st_.sentences >= st_.uncited_sentences >= 0
    assert st_.uncited_sentences == st_.sentences  # 無 [n] 亦無「參考資料未提及」
    # 獨立重算:三種終止符各自都是句界
    expected = 0
    for ch in "。!?":
        answer = answer.replace(ch, "\x00")
    expected = sum(1 for seg in answer.split("\x00") if seg.strip())
    assert st_.sentences == expected
    assert st_.valid == frozenset() and st_.invalid == frozenset()


def test_answer_footer_and_cited_tag() -> None:
    app = _load_tool("demo_app")
    marks = _marks({1})
    with_truth = app.answer_footer("A[1]。B。", marks)
    assert "引用 [1](含真值)" in with_truth and "未標引用句 1/2" in with_truth
    no_truth = app.answer_footer("A[2]。", marks)
    assert "含真值" not in no_truth and "引用 [2]" in no_truth
    hits = [Hit(f"c{i}", i, f"<文件{i}>", "", "<內容>") for i in range(1, 4)]
    html_ = app.render_hits(hits, marks, frozenset({2}))
    assert html_.count("已引用") == 1 and "真值" in html_


def _cite_rate_setup(tmp_path: Path, url: str) -> tuple[Any, Any, Any]:
    app = _load_tool("demo_app")
    rows = {"c1": {"text": "<甲>"}, "c2": {"text": "<乙>"}}
    es = EvalSet(
        "v", "t", (EvalItem("q1", "<問1>", frozenset({"c1"})), EvalItem("q2", "<問2>", frozenset()))
    )
    cfg = core.DemoConfig(
        presets={"l": Path("l.yaml"), "r": Path("r.yaml")},
        chunks=tmp_path / "c.jsonl",
        eval_set=tmp_path / "e.json",
        prompt=PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}"),
        llm=LLMConfig(url, "<m>", 32, 42, 2.0),
    )
    pool = RetrieverPool(
        cfg, rows, lambda _n: ScriptedRetriever({"<問1>": ["c1", "c2"], "<問2>": ["c2"]})
    )
    return app, cfg, pool, es


def test_cite_rate_mode_writes_json_without_text(server: str, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    app, cfg, pool, es = _cite_rate_setup(tmp_path, server)
    out = tmp_path / "cite.json"
    assert app.cite_rate(cfg, pool, es, 0, out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert set(doc["per_preset"]) == {"l", "r"} and len(doc["per_item"]) == 4
    assert doc["per_preset"]["l"]["n"] == 2 and doc["per_preset"]["l"]["cite_rate"] == 0.0
    assert "answer" not in out.read_text(encoding="utf-8") and "<甲>" not in out.read_text(
        encoding="utf-8"
    )
    assert (
        "警告" in capsys.readouterr().out
    )  # 假伺服器回覆無 [n] → cite_rate 0 警告(pre-mortem 緩解)
    assert app.cite_rate(cfg, pool, es, 1, out) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["per_preset"]["l"]["n"] == 1


def test_cite_rate_mode_returns_1_on_llm_error(tmp_path: Path) -> None:
    app, cfg, pool, es = _cite_rate_setup(tmp_path, "http://127.0.0.1:1")
    out = tmp_path / "cite.json"
    assert app.cite_rate(cfg, pool, es, 0, out) == 1
    assert not out.exists()


# 8 拒答門檻(2026-09-19)
class _ScoredScripted:
    def __init__(self, table: dict[str, list[tuple[str, float]]]) -> None:
        self._t = table

    def retrieve(self, question: str, top_k: int) -> list[str]:
        return [c for c, _ in self._t[question][:top_k]]

    def retrieve_scored(self, question: str, top_k: int) -> list[tuple[str, float]]:
        return self._t[question][:top_k]


def test_pool_fills_score_when_retriever_exposes_it(tmp_path: Path) -> None:
    rows = {"c1": {"text": "<甲>"}, "c2": {"text": "<乙>"}}
    cfg = load_demo_config(_write_cfg(tmp_path))
    pool = RetrieverPool(
        cfg, rows, lambda _n: _ScoredScripted({"<問>": [("c2", 0.9), ("c1", 0.4)]})
    )
    hits = pool.hits("left", "<問>", 2)
    assert [h.score for h in hits] == [0.9, 0.4] and [h.chunk_id for h in hits] == ["c2", "c1"]
    plain = RetrieverPool(cfg, rows, lambda _n: ScriptedRetriever({"<問>": ["c1"]}))
    assert plain.hits("left", "<問>", 1)[0].score is None


def test_should_abstain_semantics() -> None:
    hi = [Hit("c1", 1, "", "", "", 0.80), Hit("c2", 2, "", "", "", 0.10)]
    assert should_abstain(hi, 0.50) is False
    assert should_abstain(hi, 0.80) is False  # 恰等於門檻不拒
    assert should_abstain(hi, 0.81) is True
    assert should_abstain([], 0.0) is True  # 空結果必拒
    with pytest.raises(ValueError):
        should_abstain([Hit("c1", 1, "", "", "")], 0.5)  # 沒分數卻設門檻 = 設定錯,不靜默


def test_config_abstain_section(tmp_path: Path) -> None:
    cfg = load_demo_config(_write_cfg(tmp_path, abstain={"right": 0.35}))
    assert cfg.abstain == {"right": 0.35}
    assert load_demo_config(_write_cfg(tmp_path)).abstain == {}
    with pytest.raises(ValueError):
        load_demo_config(_write_cfg(tmp_path, abstain={"nope": 0.1}))  # 未知 preset


def test_abstain_text_prefix_and_detail() -> None:
    t = abstain_text([Hit("c1", 1, "", "", "", 0.12)], 0.35)
    assert t.startswith(core.ABSTAIN_PREFIX) and "0.120" in t and "0.350" in t
    assert "無檢索結果" in abstain_text([], 0.35)


def test_answer_stream_abstains_without_calling_llm(tmp_path: Path) -> None:
    app = _load_tool("demo_app")
    cfg = load_demo_config(_write_cfg(tmp_path, abstain={"right": 0.5}))  # llm url 指向死端口
    low = [Hit("c1", 1, "", "", "", 0.2)]
    out = list(app.answer_stream(cfg, "right", "<問>", low))
    assert len(out) == 1 and out[0].startswith(core.ABSTAIN_PREFIX)  # 沒打 LLM(否則會是 [錯誤])
    assert list(app.answer_stream(cfg, "left", "<問>", low))[0].startswith(
        "[錯誤]"
    )  # 左欄無門檻 → 打 LLM
    assert app.answer_footer(out[0], _marks(set())).startswith("\n\n(拒答")


def test_cite_rate_counts_abstained_answers(server: str, tmp_path: Path) -> None:
    app, cfg, pool, es = _cite_rate_setup(tmp_path, server)
    cfg = core.DemoConfig(
        cfg.presets, cfg.chunks, cfg.eval_set, cfg.prompt, cfg.llm, abstain={"l": 0.5}
    )
    scored = _ScoredScripted({"<問1>": [("c1", 0.9), ("c2", 0.1)], "<問2>": [("c2", 0.1)]})
    pool = RetrieverPool(cfg, {"c1": {"text": "<甲>"}, "c2": {"text": "<乙>"}}, lambda _n: scored)
    out = tmp_path / "cite.json"
    assert app.cite_rate(cfg, pool, es, 0, out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["per_preset"]["l"]["abstain_n"] == 1 and doc["per_preset"]["l"]["n"] == 1
    assert doc["per_preset"]["r"]["abstain_n"] == 0 and doc["per_preset"]["r"]["n"] == 2


def test_abstain_grid_collect_and_sweep(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    grid = _load_tool("abstain_grid")
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps({"chunk_id": c, "text": "<x>"}) for c in ("c1", "c2")) + "\n",
        encoding="utf-8",
    )
    es_path = tmp_path / "e.json"
    items = [EvalItem("q1", "<問1>", frozenset({"c1"})), EvalItem("q2", "<問2>", frozenset({"c1"}))]
    es_path.write_text(json.dumps(freeze(items, "v", "t")), encoding="utf-8")
    off = tmp_path / "off.json"
    off.write_text(json.dumps({"items": [{"id": "o1", "question": "<外1>"}]}), encoding="utf-8")
    script = {"<問1>": [["c1", 0.9]], "<問2>": [["c2", 0.3]], "<外1>": [["c2", 0.05]]}
    for name in ("l", "r"):
        (tmp_path / f"{name}.json").write_text(json.dumps(script), encoding="utf-8")
        (tmp_path / f"{name}.yaml").write_text(
            f"experiment_id: {name}\neval_set: {es_path}\nretriever:\n  type: scripted\n"
            f"  params:\n    script: {tmp_path}/{name}.json\n    scored: true\n"
            f"k_values: [1]\nresults_dir: {tmp_path}\n",
            encoding="utf-8",
        )
    cfg = _write_cfg(
        tmp_path, presets={"l": str(tmp_path / "l.yaml"), "r": str(tmp_path / "r.yaml")}
    )
    import yaml

    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["chunks"], doc["eval_set"] = str(tmp_path / "chunks.jsonl"), str(es_path)
    cfg.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "scores.json"
    assert grid.main(["collect", str(cfg), str(off), str(out), "--k", "1"]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 6 and {r["kind"] for r in rows} == {"eval", "offcorpus"}
    assert (
        grid.main(["sweep", str(out), "--grid", "3", "--max-false", "0.0", "--min-true", "1.0"])
        == 0
    )
    text = capsys.readouterr().out
    assert "高原" in text and "== l" in text and "== r" in text


@given(st.floats(0, 1, allow_nan=False), st.floats(0, 1, allow_nan=False))
@settings(max_examples=200)
def test_should_abstain_property(top: float, thr: float) -> None:
    # 獨立重算:恰等於門檻不拒,嚴格小於才拒
    hits = [Hit("c1", 1, "", "", "", top)]
    assert should_abstain(hits, thr) == (top < thr)


# 9 逐字覆蓋(爭議點 1:引用有效 ≠ 引用正確)
def _hits_text(texts: list[str]) -> list[Hit]:
    return [Hit(f"c{i}", i, "", "", t) for i, t in enumerate(texts, start=1)]


def test_sentence_support_hand_computed() -> None:
    hits = _hits_text(["甲乙丙丁戊己庚", "壬癸子丑寅卯辰"])
    sup = sentence_support("甲乙丙丁戊己庚[1]。壬癸子丑寅卯辰[2]。無引用句。", hits)
    assert [x.cites for x in sup] == [(1,), (2,), ()]
    assert sup[0].coverage == pytest.approx(1.0) and sup[1].coverage == pytest.approx(1.0)
    assert sup[2].coverage is None
    assert sup[0].sentence == "甲乙丙丁戊己庚"  # 標記已剝除


def test_sentence_support_uses_only_cited_chunks() -> None:
    hits = _hits_text(["甲乙丙丁戊己庚", "壬癸子丑寅卯辰"])
    sup = sentence_support("甲乙丙丁戊己庚[2]。", hits)  # 引錯段落 → 覆蓋率為 0
    assert sup[0].cites == (2,) and sup[0].coverage == pytest.approx(0.0)
    multi = sentence_support("甲乙丙丁壬癸子[1][2]。", hits)  # 多段合併計算
    assert multi[0].cites == (1, 2) and multi[0].coverage > 0.5


def test_sentence_support_ignores_out_of_range_and_abstain() -> None:
    hits = _hits_text(["甲乙丙丁戊己庚"])
    sup = sentence_support("甲乙丙丁戊己庚[7]。參考資料未提及。", hits)
    assert sup[0].cites == () and sup[0].coverage is None  # 越界不算有效引用
    assert len(sup) == 1  # 棄答句不列
    nested = sentence_support("[[1]1]甲乙丙。", hits)  # 剝一次會再拼出 [1],須剝到穩定
    assert nested[0].sentence == "甲乙丙" and nested[0].cites == (1,)


def test_low_coverage_counts() -> None:
    sup = [
        core.SentenceSupport("a", (1,), 0.9),
        core.SentenceSupport("b", (1,), 0.3),
        core.SentenceSupport("c", (), None),
        core.SentenceSupport("d", (2,), 0.5),
    ]
    assert low_coverage(sup, 0.5) == (3, 1)  # 恰等於門檻不算低
    assert low_coverage([], 0.5) == (0, 0)
    with pytest.raises(ValueError):
        low_coverage(sup, 1.5)


_PARTS = [
    "甲乙丙丁",
    "壬癸子丑",
    "乙丙丁壬",
    "[1]",
    "[2]",
    "[3]",
    "。",
    "!",
    "[",
]  # 乙丙丁壬:對段 1 覆蓋恰 0.5


@given(st.lists(st.sampled_from(_PARTS), max_size=10).map("".join))
@settings(max_examples=300)
def test_sentence_support_property(answer: str) -> None:
    hits = _hits_text(["甲乙丙丁", "壬癸子丑"])  # 兩段字元不重疊:引錯段落覆蓋率必為 0
    sup = sentence_support(answer, hits)
    for x in sup:
        assert all(1 <= c <= 2 for c in x.cites)
        assert (x.coverage is None) == (x.cites == ())
        if x.coverage is not None:
            assert 0.0 <= x.coverage <= 1.0
        assert not re.search(r"\[\d+\]", x.sentence)  # 引用標記已剝除(孤立的 [ 是合法文字)
    n_cited, n_low = low_coverage(sup, 0.5)
    assert 0 <= n_low <= n_cited <= len(sup)
    # 獨立重算:覆蓋率只對「所引段落」算(引錯段落不能靠別段撐高);低覆蓋句數嚴格小於門檻才計
    text = {h.rank: h.text for h in hits}
    for x in sup:
        if x.cites:
            assert x.coverage == pytest.approx(
                coverage3(x.sentence, "\n".join(text[c] for c in x.cites))
            )
    assert n_low == sum(1 for x in sup if x.coverage is not None and x.coverage < 0.5)


# 10 檢索耗時(爭議點 4)
def test_pool_hits_timed_reports_ms(tmp_path: Path) -> None:
    rows = {"c1": {"text": "<甲>"}}
    cfg = load_demo_config(_write_cfg(tmp_path))
    pool = RetrieverPool(cfg, rows, lambda _n: ScriptedRetriever({"<問>": ["c1"]}))
    hits, ms = pool.hits_timed("left", "<問>", 1)
    assert hits == pool.hits("left", "<問>", 1) and ms >= 0.0
    with pytest.raises(KeyError):
        pool.hits_timed("nope", "<問>", 1)


def test_answer_footer_support_line_and_latency_header() -> None:
    app = _load_tool("demo_app")
    hits = _hits_text(["甲乙丙丁戊己庚", "壬癸子丑寅卯辰"])
    marks = _marks({1})
    foot = app.answer_footer("甲乙丙丁戊己庚[1]。亂寫的句子[2]。", marks, hits)
    assert "逐字覆蓋 min 0.00" in foot and "的引用句 1/2" in foot and "非品質分數" in foot
    assert "逐字覆蓋:無有效引用句" in app.answer_footer("沒有引用。", marks, hits)
    assert "逐字覆蓋" not in app.answer_footer("沒有引用。", marks)  # 不給 hits 維持舊格式
    html_ = app.render_hits(hits, _marks(set(), 2), ms=12.3)
    assert "檢索 12 ms" in html_ and "檢索" not in app.render_hits(hits, _marks(set(), 2))


def test_cite_rate_reports_support_and_latency(server: str, tmp_path: Path) -> None:
    app, cfg, pool, es = _cite_rate_setup(tmp_path, server)
    out = tmp_path / "cite.json"
    assert app.cite_rate(cfg, pool, es, 0, out) == 0
    row = json.loads(out.read_text(encoding="utf-8"))["per_preset"]["l"]
    assert row["low_coverage_rate"] == 0.0 and row["retrieval_ms_mean"] >= 0.0


def test_abstain_grid_cv_subcommand(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    grid = _load_tool("abstain_grid")
    rows = [
        {
            "preset": "p",
            "item_id": f"e{i}",
            "kind": "eval",
            "scores": [0.6 + 0.01 * i],
            "truth_in_topk": True,
        }
        for i in range(10)
    ] + [
        {
            "preset": "p",
            "item_id": f"o{i}",
            "kind": "offcorpus",
            "scores": [0.01 * i],
            "truth_in_topk": False,
        }
        for i in range(6)
    ]
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"k": 1, "rows": rows}), encoding="utf-8")
    assert (
        grid.main(
            [
                "cv",
                str(f),
                "--k",
                "3",
                "--seed",
                "0",
                "--grid",
                "10",
                "--max-false",
                "0.0",
                "--min-true",
                "1.0",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "== p(k=3" in text and "held-out 平均誤拒 0.000" in text and "無高原折數 0" in text


# 11 出處定位(爭議 4:引用只到 chunk,沒到頁)
def _hit(**kw: Any) -> Hit:
    base = {"chunk_id": "c1", "rank": 1, "doc_title": "<文件>", "heading": "", "text": "<內容>"}
    return Hit(**{**base, **kw})  # type: ignore[arg-type]


def test_locator_pdf_shows_page() -> None:
    h = _hit(source_type="pdf", source_path="data/corpus/nhi/<檔名>.pdf", page=12, span=(0, 497))
    assert locator_text(h) == "<檔名>.pdf 頁 12"


def test_locator_csv_shows_row_not_char() -> None:
    # csv 的 span 是列索引不是字元位移(chunk_csv_per_row),標成字元即為事實錯誤
    one = _hit(source_type="csv", source_path="a/<表>.csv", span=(0, 1))
    assert locator_text(one) == "<表>.csv 第 0 列(0 起算)"
    many = _hit(source_type="csv", source_path="a/<表>.csv", span=(3, 6))
    assert locator_text(many) == "<表>.csv 第 3–5 列(0 起算)"


def test_locator_html_and_pdf_without_page_show_char_offset() -> None:
    html = _hit(source_type="html", source_path="a/<頁>.html", span=(0, 762))
    assert locator_text(html) == "<頁>.html 字元 0–762"
    pdf_no_page = _hit(source_type="pdf", source_path="a/<檔>.pdf", page=None, span=(5, 9))
    assert locator_text(pdf_no_page) == "<檔>.pdf 字元 5–9"


def test_locator_degrades_without_metadata() -> None:
    assert locator_text(_hit()) == "(未知來源)"
    assert locator_text(_hit(source_type="csv", source_path="a/<表>.csv")) == "<表>.csv"


def test_pool_fills_locator_fields(tmp_path: Path) -> None:
    rows = {
        "c1": {
            "text": "<甲>",
            "source_type": "pdf",
            "source_path": "a/<檔>.pdf",
            "page": 7,
            "char_span": [10, 20],
        },
        "c2": {"text": "<乙>"},
    }
    cfg = load_demo_config(_write_cfg(tmp_path))
    pool = RetrieverPool(cfg, rows, lambda _n: ScriptedRetriever({"<問>": ["c1", "c2"]}))
    a, b = pool.hits("left", "<問>", 2)
    assert (a.source_type, a.page, a.span) == ("pdf", 7, (10, 20))
    assert locator_text(a) == "<檔>.pdf 頁 7"
    assert (b.source_type, b.page, b.span) == ("", None, None)  # 缺欄位不炸
    assert locator_text(b) == "(未知來源)"


def test_render_hits_shows_locator_and_truncation() -> None:
    app = _load_tool("demo_app")
    hits = [
        Hit("c1", 1, "<文件>", "", "短文", None, "pdf", "a/<檔>.pdf", 7, (0, 4)),
        Hit("c2", 2, "<文件>", "", "長" * 400, None, "csv", "a/<表>.csv", None, (3, 4)),
    ]
    out = app.render_hits(hits, _marks(set(), 2))
    assert "出處:&lt;檔&gt;.pdf 頁 7" in out
    assert "第 3 列(0 起算)" in out
    assert out.count("截斷,原文更長") == 1  # 只有超過 300 字的那張卡標截斷


# 12 低覆蓋句在 UI 指名(2026-09-20)
_MARK_HITS = ["甲乙丙丁", "壬癸子丑"]


def test_mark_low_coverage_hand_computed() -> None:
    out = mark_low_coverage("甲乙丙丁[1]。亂寫的東西[2]。", _hits_text(_MARK_HITS), 0.5)
    assert out == "甲乙丙丁[1]。⚠亂寫的東西[2]。"


def test_mark_low_coverage_keeps_abstain_and_uncited_plain() -> None:
    plain = "參考資料未提及。沒有引用的句子。"
    assert mark_low_coverage(plain, _hits_text(_MARK_HITS), 0.5) == plain


def test_mark_low_coverage_no_cited_returns_unchanged() -> None:
    assert mark_low_coverage("", _hits_text(_MARK_HITS), 0.5) == ""
    assert mark_low_coverage("純文字無引用", _hits_text(_MARK_HITS), 0.5) == "純文字無引用"


def test_mark_low_coverage_boundary_matches_low_coverage() -> None:
    # 「乙丙丁壬」對段 1 覆蓋恰 0.5(3-gram 乙丙丁、丙丁壬 中 1 個命中)→ 恰等於門檻不標
    answer = "乙丙丁壬[1]。"
    sup = sentence_support(answer, _hits_text(_MARK_HITS))
    assert sup[0].coverage == pytest.approx(0.5)
    assert mark_low_coverage(answer, _hits_text(_MARK_HITS), 0.5) == answer
    assert mark_low_coverage(answer, _hits_text(_MARK_HITS), 0.51) == "⚠乙丙丁壬[1]。"


def test_mark_low_coverage_does_not_disturb_cite_stats() -> None:
    hits = _hits_text(_MARK_HITS)
    answer = "甲乙丙丁[1]。亂寫的東西[2]。"
    marked = mark_low_coverage(answer, hits, 0.5)
    assert cite_stats(marked, _marks({1}, 2)) == cite_stats(answer, _marks({1}, 2))


@given(st.lists(st.sampled_from(_PARTS), max_size=10).map("".join))
@settings(max_examples=300)
def test_mark_low_coverage_property(answer: str) -> None:
    hits = _hits_text(_MARK_HITS)
    marked = mark_low_coverage(answer, hits, 0.5)
    _, n_low = low_coverage(sentence_support(answer, hits), 0.5)
    assert marked.count(core.LOW_MARK) == n_low
    # 只加前綴不改內容:移除標記後非空白字元序列不變
    assert marked.replace(core.LOW_MARK, "").split() == answer.split()


def test_footer_explains_mark_only_when_marking() -> None:
    app = _load_tool("demo_app")
    hits = _hits_text(_MARK_HITS)
    marks = _marks({1}, 2)
    with_low = app.answer_footer("甲乙丙丁[1]。亂寫的東西[2]。", marks, hits)
    assert "無標記不代表已驗證" in with_low
    assert "無標記不代表已驗證" not in app.answer_footer("甲乙丙丁[1]。", marks, hits)


def test_prompt_config_carries_prompt_id() -> None:
    """UI 版本標頭需要 prompt_id;yaml 缺該鍵時回空字串(gauntlet 的合成 yaml 沒有它)。"""
    p = PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}", prompt_id="<v1>")
    assert p.prompt_id == "<v1>"
    assert PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}").prompt_id == ""


def test_version_line_names_prompt_eval_and_model(tmp_path: Path) -> None:
    """展示時的可稽核性:標頭必須讓觀眾一眼看到 prompt 版本、eval 版本、模型。"""
    app = _load_tool("demo_app")
    es = EvalSet("<v1.x>", "t", (EvalItem("q1", "<問1>", frozenset({"c1"})),))
    cfg = core.DemoConfig(
        presets={"l": Path("l.yaml"), "r": Path("r.yaml")},
        chunks=tmp_path / "c.jsonl",
        eval_set=tmp_path / "e.json",
        prompt=PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}", prompt_id="<pid>"),
        llm=LLMConfig("http://x", "<模型名>", 32, 42, 2.0),
    )
    line = app.version_line(cfg, es)
    assert "<pid>" in line and "<v1.x>" in line and "<模型名>" in line
    # 缺 prompt_id 時不留空槽(顯示占位而非空白)
    cfg2 = core.DemoConfig(
        presets=cfg.presets,
        chunks=cfg.chunks,
        eval_set=cfg.eval_set,
        prompt=PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}"),
        llm=cfg.llm,
    )
    assert "?" in app.version_line(cfg2, es)
