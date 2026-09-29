"""真值重複事實掃描工具測試 —— 場景為 2026-09-15 定案的驗收條件。文字皆為中性占位符。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.evaluation.eval_set import EvalItem, freeze

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


td = _load_tool("truth_dupes")


def _rows(**texts: str) -> dict[str, dict[str, object]]:
    return {k: {"chunk_id": k, "doc_id": f"D-{k}", "text": v} for k, v in texts.items()}


# 1 char_ngrams
def test_char_ngrams_basic_short_and_whitespace() -> None:
    assert td.char_ngrams("abcd", 3) == frozenset({"abc", "bcd"})
    assert td.char_ngrams("ab", 3) == frozenset()
    assert td.char_ngrams("a b", 3) == frozenset()
    assert td.char_ngrams("a bc", 3) == frozenset({"abc"})


# 2 jaccard
def test_jaccard_hand_computed() -> None:
    a, b = frozenset({"x", "y", "z"}), frozenset({"y", "z", "w"})
    assert td.jaccard(a, b) == pytest.approx(2 / 4)
    assert td.jaccard(frozenset(), frozenset()) == 0.0
    assert td.jaccard(a, a) == 1.0
    assert td.jaccard(a, b) == td.jaccard(b, a)


# 3 scan 排序、門檻、top、排除自身
def test_scan_orders_thresholds_and_excludes_self() -> None:
    rows = _rows(
        t="甲乙丙丁戊己庚辛",  # 真值
        a="甲乙丙丁戊己庚辛",  # 完全相同 → 1.0
        b="甲乙丙丁戊己庚壬",  # 高度相似
        c="子丑寅卯辰巳午未",  # 完全不同 → 0
        d="甲乙丙丁戊己庚壬",  # 與 b 同分 → 依 id 排在 b 之後
    )
    out = td.scan(["t"], rows, n=3, top=5, threshold=0.5)
    ids = [c for c, _ in out["t"]]
    assert ids == ["a", "b", "d"]
    assert out["t"][0][1] == 1.0
    assert out["t"][1][1] == out["t"][2][1]
    assert all(s >= 0.5 for _, s in out["t"])


def test_scan_respects_top() -> None:
    rows = _rows(t="甲乙丙丁戊己", a="甲乙丙丁戊己", b="甲乙丙丁戊己", c="甲乙丙丁戊己")
    assert len(td.scan(["t"], rows, top=2)["t"]) == 2


# 5/6 錯誤案例
def test_scan_unknown_truth_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        td.scan(["nope"], _rows(a="甲乙丙"))


@pytest.mark.parametrize("kw", [{"threshold": 1.5}, {"threshold": -0.1}, {"top": 0}, {"n": 0}])
def test_scan_bad_params_raise(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        td.scan(["a"], _rows(a="甲乙丙"), **kw)  # type: ignore[arg-type]


# property 不變量
_text = st.text(alphabet="甲乙丙丁戊", min_size=0, max_size=12)


@settings(max_examples=60, deadline=None)
@given(a=_text, b=_text)
def test_property_jaccard_symmetric_and_bounded(a: str, b: str) -> None:
    ga, gb = td.char_ngrams(a, 2), td.char_ngrams(b, 2)
    s = td.jaccard(ga, gb)
    assert 0.0 <= s <= 1.0
    assert s == td.jaccard(gb, ga)


@settings(max_examples=60, deadline=None)
@given(texts=st.lists(_text, min_size=1, max_size=6), top=st.integers(1, 6), thr=st.floats(0, 1))
def test_property_scan_invariants(texts: list[str], top: int, thr: float) -> None:
    rows = _rows(**{f"c{i}": t for i, t in enumerate(texts)})
    out = td.scan(["c0"], rows, n=2, top=top, threshold=thr)["c0"]
    ids = [c for c, _ in out]
    scores = [s for _, s in out]
    assert "c0" not in ids
    assert len(out) <= top
    assert scores == sorted(scores, reverse=True)
    assert all(thr <= s <= 1.0 for s in scores)
    # 獨立重算:每個列出的分數等於直接算的 Jaccard
    for cid, s in out:
        assert s == td.jaccard(td.char_ngrams(texts[0], 2), td.char_ngrams(rows[cid]["text"], 2))


# 4/7 CLI
def test_cli_writes_report_marks_already_truth_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _rows(
        t1="甲乙丙丁戊己庚辛", t2="甲乙丙丁戊己庚辛", x="甲乙丙丁戊己庚壬", z="子丑寅卯辰巳午未"
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()) + "\n", encoding="utf-8"
    )
    ev = tmp_path / "eval.json"
    ev.write_text(
        json.dumps(freeze([EvalItem("q1", "<問題>", frozenset({"t1", "t2"}))], "t", "t")),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    assert td.main([str(ev), str(chunks), str(out), "--threshold", "0.5"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["params"] == {"n": 3, "top": 5, "threshold": 0.5}
    assert rep["summary"] == {
        "truths": 2,
        "with_candidates": 2,
        "candidates_total": 4,
        "already_truth": 2,
    }
    item = next(i for i in rep["items"] if i["truth_id"] == "t1")
    assert item["question_id"] == "q1"
    cands = {c["chunk_id"]: c for c in item["candidates"]}
    assert cands["t2"]["already_truth"] is True and cands["t2"]["score"] == 1.0
    assert cands["x"]["already_truth"] is False and cands["x"]["doc_id"] == "D-x"
    assert "z" not in cands
    printed = capsys.readouterr().out
    assert "truths=2" in printed and "t1" in printed
    assert td.main([str(ev), str(chunks), str(out)]) == 3


def test_scan_keeps_candidate_exactly_at_threshold() -> None:
    # 2-gram:t={"ab"},a={"ab","bc"} → Jaccard 恰為 1/2,門檻 0.5 必須保留(>= 而非 >)
    rows = _rows(t="ab", a="abc")
    assert td.scan(["t"], rows, n=2, top=5, threshold=0.5)["t"] == [("a", 0.5)]


@settings(max_examples=60, deadline=None)
@given(a=_text, b=_text)
def test_property_jaccard_matches_independent_set_arithmetic(a: str, b: str) -> None:
    # 不透過 td.jaccard 重算:交集/聯集以集合運算獨立求得;兩者皆空 → 0
    ga, gb = td.char_ngrams(a, 2), td.char_ngrams(b, 2)
    union = len(ga | gb)
    expected = len(ga & gb) / union if union else 0.0
    assert td.jaccard(ga, gb) == expected
    if ga and gb and not (ga & gb):
        assert td.jaccard(ga, gb) == 0.0
