"""tools/support_sheet.py:低覆蓋句抽樣表。文字皆為中性占位符。"""

from __future__ import annotations

import csv
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from selectrag.demo.core import DemoConfig, LLMConfig, PromptConfig, RetrieverPool
from selectrag.evaluation.eval_set import EvalItem, EvalSet
from selectrag.retrieval.scripted import ScriptedRetriever

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fake = _load("fake_openai_server")
sheet = _load("support_sheet")


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    srv = fake.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _setup(url: str, tmp_path: Path) -> tuple[Any, RetrieverPool, EvalSet]:
    cfg = DemoConfig(
        presets={"l": Path("l.yaml"), "r": Path("r.yaml")},
        chunks=tmp_path / "c.jsonl",
        eval_set=tmp_path / "e.json",
        prompt=PromptConfig(2, "<s>", "問題:{question}\n\n參考段落:\n{context}"),
        llm=LLMConfig(url, "<m>", 32, 42, 5.0),
    )
    rows = {"c1": {"text": "<段落甲>"}, "c2": {"text": "<段落乙>"}}
    es = EvalSet("v", "t", (EvalItem("q1", "<問1>", frozenset({"c1"})),))
    pool = RetrieverPool(cfg, rows, lambda _n: ScriptedRetriever({"<問1>": ["c1", "c2"]}))
    return cfg, pool, es


def test_collect_rows_keeps_only_low_coverage(server: str, tmp_path: Path) -> None:
    # 假伺服器回覆與段落無關且無 [n] → 無有效引用句 → 零「作答」列,
    # 但 2026-09-22 起每個組合仍留一列「無引用回答」(發現 47:過去整組消失)
    cfg, pool, es = _setup(server, tmp_path)
    rows, n_cited, n_low = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert n_cited == 0 and n_low == 0
    assert [r[sheet.KIND_COL] for r in rows] == ["無引用回答", "無引用回答"]


def test_collect_rows_raises_on_llm_error(tmp_path: Path) -> None:
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    with pytest.raises(RuntimeError):
        sheet.collect_rows(cfg, pool, es, 0, 0.5)


def test_write_refuses_overwrite_and_falls_back_to_csv(tmp_path: Path) -> None:
    out = tmp_path / "s.xlsx"
    rows = [["l", "m", "q1", "<問>", "<句>", "1", 0.2, "<段落>", "", ""]]
    path = sheet.write(rows, out)
    assert path.exists() and path.suffix in (".xlsx", ".csv")
    with pytest.raises(FileExistsError):
        sheet.write(rows, path if path.suffix == ".xlsx" else out)


def test_main_writes_and_reports(server: str, tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup(server, tmp_path)
    monkeypatch.setattr(sheet, "load_demo_config", lambda _p: cfg)
    monkeypatch.setattr(sheet, "load_chunk_rows", lambda _p: {})
    monkeypatch.setattr(sheet, "load_eval_set", lambda _p: es)
    monkeypatch.setattr(sheet, "RetrieverPool", lambda *a, **k: pool)
    out = tmp_path / "out.xlsx"
    assert sheet.main([str(tmp_path / "cfg.yaml"), str(out), "--sample", "0"]) == 0
    printed = capsys.readouterr().out
    assert "有引用句 0" in printed and "band=low 抽樣框 0 列" in printed and "寫出 0 列" in printed
    written = out if out.exists() else out.with_suffix(".csv")
    assert written.exists()
    if written.suffix == ".csv":
        with written.open(encoding="utf-8-sig") as fh:
            assert next(csv.reader(fh)) == sheet.HEADER


# 覆蓋率分帶抽樣(爭議 2)+ 格式覆寫(爭議 3),2026-09-20
def _row(cov: float, preset: str = "l", model: str = "m") -> list[object]:
    return [preset, model, "q1", "<問>", "<句>", "1", cov, "<段落>", "", ""]


def test_filter_band_selects_by_coverage() -> None:
    rows = [_row(0.2), _row(0.5), _row(0.9)]
    assert [r[6] for r in sheet.filter_band(rows, "low", 0.5)] == [0.2]
    assert [r[6] for r in sheet.filter_band(rows, "high", 0.5)] == [0.5, 0.9]
    assert len(sheet.filter_band(rows, "all", 0.5)) == 3
    with pytest.raises(ValueError):
        sheet.filter_band(rows, "middle", 0.5)


def test_collect_rows_carries_model_and_all_coverages(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。亂寫[2]。"]))
    rows, n_cited, n_low = sheet.collect_rows(cfg, pool, es, 0, 0.5, model="<格式A>")
    assert n_cited == 4 and n_low == 2  # 兩 preset × (一句有據 + 一句編造)
    assert len(rows) == 4 and {r[1] for r in rows} == {"<格式A>"}  # 全列帶格式標記
    assert sorted({round(float(r[6]), 2) for r in rows}) == [0.0, 1.0]


def test_collect_rows_restricts_to_named_presets(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    rows, n_cited, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, presets=("r",))
    assert n_cited == 1 and {r[0] for r in rows} == {"r"}
    # 守衛的價值在訊息要列出可用 preset(RetrieverPool 只會拋出 KeyError('nope'))
    with pytest.raises(KeyError, match="unknown preset"):
        sheet.collect_rows(cfg, pool, es, 0, 0.5, presets=("nope",))


def test_main_band_and_model_flow(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "load_demo_config", lambda _p: cfg)
    monkeypatch.setattr(sheet, "load_chunk_rows", lambda _p: {})
    monkeypatch.setattr(sheet, "load_eval_set", lambda _p: es)
    monkeypatch.setattr(sheet, "RetrieverPool", lambda *a, **k: pool)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。亂寫[2]。"]))
    out = tmp_path / "high.xlsx"
    args = [str(tmp_path / "c.yaml"), str(out), "--band", "high", "--sample", "0"]
    assert sheet.main([*args, "--preset", "l", "--model", "<格式A>"]) == 0
    printed = capsys.readouterr().out
    assert "model=<格式A>" in printed and "band=high 抽樣框 1 列" in printed
    written = out if out.exists() else out.with_suffix(".csv")
    assert written.exists()


# 必要成分命中,2026-09-20
def _components(tmp_path: Path) -> Path:
    p = tmp_path / "components.json"
    p.write_text(
        '{"version":"t","items":{"q1":['
        '{"name":"<成分甲>","aliases":["<段落甲>"]},'
        '{"name":"<成分乙>","aliases":["<段落乙>"]}]}}',
        encoding="utf-8",
    )
    return p


def test_header_unchanged_without_components() -> None:
    # B7:未給 --components 時表頭逐字不變(既有三份 xlsx 的格式不破)
    assert sheet.header(None) == sheet.HEADER
    assert len(sheet.HEADER) == 14  # 2026-09-22 追加列型/兩個藥名欄/真值名次


def test_header_gains_two_columns_with_components() -> None:
    assert sheet.header({"q1": ()}) == [
        *sheet.HEADER,
        "必要成分命中",
        "缺漏成分",
        "禁止表述命中",
        "題層禁止命中(整題歸零)",
        "命中但未引用",
        "命中但引用不支撐",
    ]


def test_collect_rows_fills_component_columns(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # B6:答案只提到成分甲 → 命中甲、缺漏乙,同題每列填同值
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。亂寫[2]。"]))
    comps = sheet.load_components(_components(tmp_path))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=comps)
    assert len(rows) == 4
    for r in rows:
        assert len(r) == 20
        assert r[14] == "<成分甲>" and r[15] == "<成分乙>"
        assert r[16] == "" and r[17] == ""


def test_collect_rows_leaves_component_columns_blank_for_unlisted_item(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    # B8:題目不在清單中 → 兩欄留空且不 raise
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components={"other-q": ()})
    assert rows and all(all(c == "" for c in r[14:20]) for r in rows)


def test_collect_rows_omits_component_columns_when_not_requested(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert rows and all(len(r) == 14 for r in rows)


def test_main_components_flow(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "load_demo_config", lambda _p: cfg)
    monkeypatch.setattr(sheet, "load_chunk_rows", lambda _p: {})
    monkeypatch.setattr(sheet, "load_eval_set", lambda _p: es)
    monkeypatch.setattr(sheet, "RetrieverPool", lambda *a, **k: pool)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    out = tmp_path / "comp.xlsx"
    rc = sheet.main(
        [
            str(tmp_path / "cfg.yaml"),
            str(out),
            "--band",
            "all",
            "--sample",
            "0",
            "--components",
            str(_components(tmp_path)),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    # 成分統計只印診斷,不參與任何分數(N2)
    assert "成分清單 1 題" in printed
    written = out if out.exists() else out.with_suffix(".csv")
    if written.suffix == ".csv":
        with written.open(encoding="utf-8-sig") as fh:
            head = next(csv.reader(fh))
            assert head[-6:] == [
                "必要成分命中",
                "缺漏成分",
                "禁止表述命中",
                "題層禁止命中(整題歸零)",
                "命中但未引用",
                "命中但引用不支撐",
            ]


def test_collect_rows_reports_forbidden_hit(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # S-C4:命中禁止表述時第三欄填該成分,且不進「命中」欄(答錯 ≠ 答對)
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    comp = tmp_path / "forbid.json"
    comp.write_text(
        '{"version":"t","items":{"q1":['
        '{"name":"<成分甲>","aliases":["<絕不出現>"],"forbidden":["<段落甲>"]}]}}',
        encoding="utf-8",
    )
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=sheet.load_components(comp))
    assert rows
    for r in rows:
        assert r[14] == "" and r[15] == "" and r[16] == "<成分甲>" and r[17] == ""
        assert r[18] == "" and r[19] == ""


def test_collect_rows_reports_item_level_forbidden(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # S-C8:題層禁止表述命中 → 第四欄填,成分層三欄照實填(兩個機制分開統計)
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    comp = tmp_path / "item_forbid.json"
    comp.write_text(
        '{"version":"t","items":{"q1":{"components":['
        '{"name":"<成分甲>","aliases":["<絕不出現>"]}],'
        '"forbidden":["<段落甲>"]}}}',
        encoding="utf-8",
    )
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=sheet.load_components(comp))
    assert rows
    for r in rows:
        assert r[14] == "" and r[15] == "<成分甲>"
        assert r[16] == "" and r[17] == "<段落甲>" and r[18] == "" and r[19] == ""


def test_collect_rows_flags_unsupported_component(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # B2:成分在完整回答裡命中,但只有另一句有引用 → 第五欄標出該成分
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(
        sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。<絕不出現>。"])
    )
    comp = tmp_path / "unsup.json"
    comp.write_text(
        '{"version":"t","items":{"q1":['
        '{"name":"<有引用>","aliases":["<段落甲>"]},'
        '{"name":"<無引用>","aliases":["<絕不出現>"]}]}}',
        encoding="utf-8",
    )
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=sheet.load_components(comp))
    assert rows
    for r in rows:
        assert r[14] == "<有引用>,<無引用>"  # 完整回答兩者皆中
        assert r[18] == "<無引用>"  # 但只有前者有引用支撐


def test_collect_rows_unsupported_uses_all_cited_sentences(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """引用支撐必須看該題**全部**引用句,不是當前列那一句。

    2026-09-21 錨點殺傷力驗證抓到的洞:先前測試只有一句有引用,`cited_sentences`
    與 `[sup.sentence]` 恰好相同,所以「只看當前句」那個變異存活。
    """
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(
        sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。<段落乙>[2]。"])
    )
    comp = tmp_path / "multi.json"
    comp.write_text(
        '{"version":"t","items":{"q1":['
        '{"name":"<甲>","aliases":["<段落甲>"]},'
        '{"name":"<乙>","aliases":["<段落乙>"]}]}}',
        encoding="utf-8",
    )
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=sheet.load_components(comp))
    assert len(rows) >= 4, len(rows)  # 兩 preset × 兩句
    for r in rows:
        assert r[14] == "<甲>,<乙>", r[14]
        assert r[18] == "", f"兩句都有引用,不該有無支撐成分:{r[18]}"


def test_collect_rows_splits_uncited_from_citation_mismatch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """兩種缺陷必須分欄:①成分只在未引用句 ②句子有引用但所引段落不含該成分。

    2026-09-21 第二輪跑分暴露:原本的單一欄位只抓得到 ①,`b3-q12` 與 `b2-q04`
    那種「引用錯置」(句子有引用、所引段落卻不含該成分)全部從欄位底下溜過去。
    """
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    # 句 1 引用 [1](=<段落甲>),句中寫「<在段落裡>」但段落只有 <段落甲> → 引用不支撐
    # 句 2 無引用,含「<只在未引用句>」→ 未引用
    monkeypatch.setattr(
        sheet,
        "stream_answer",
        lambda *_a, **_k: iter(["<段落甲><在段落裡>[1]。<只在未引用句>。"]),
    )
    comp = tmp_path / "split.json"
    comp.write_text(
        '{"version":"t","items":{"q1":['
        '{"name":"<有支撐>","aliases":["<段落甲>"]},'
        '{"name":"<引用不支撐>","aliases":["<在段落裡>"]},'
        '{"name":"<未引用>","aliases":["<只在未引用句>"]}]}}',
        encoding="utf-8",
    )
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, components=sheet.load_components(comp))
    assert rows
    for r in rows:
        assert len(r) == 20, len(r)
        assert r[14] == "<有支撐>,<引用不支撐>,<未引用>"  # 完整回答三者皆中
        assert r[18] == "<未引用>", r[18]  # 只在未引用句
        assert r[19] == "<引用不支撐>", r[19]  # 在引用句但段落不含 → 兩欄互斥


def test_dump_keeps_items_that_produce_no_rows(server: str, tmp_path: Path) -> None:
    """dump 存在的理由:假伺服器回覆無 [n] → 零列,但該題仍必須留下原始回答。

    這是 xlsx 的結構性盲點(2026-09-22 v14 的 b2-q11 / b4-q15):整題消失時,
    「被擠掉」與「真的棄答」在表上無法分辨。dump 是唯一的回溯路徑。
    """
    cfg, pool, es = _setup(server, tmp_path)
    dump: list[dict[str, Any]] = []
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5, dump_to=dump)
    assert all(r[sheet.KIND_COL] != "作答" for r in rows)  # 無作答列
    assert len(dump) == 2  # 兩個 preset × 一題
    assert {d["preset"] for d in dump} == {"l", "r"}
    for d in dump:
        assert d["item_id"] == "q1"
        assert d["answer"]  # 有內容,不是空字串
        assert [h["rank"] for h in d["hits"]] == [1, 2]


def test_dump_none_collects_nothing(server: str, tmp_path: Path) -> None:
    """不給 dump_to 時行為與舊版逐字相同(預設路徑不受影響)。"""
    cfg, pool, es = _setup(server, tmp_path)
    before = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    after = sheet.collect_rows(cfg, pool, es, 0, 0.5, dump_to=None)
    assert before == after


def test_abstain_row_appears_when_no_cited_sentence(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """發現 47:棄答句被 sentence_support 剔除,整個 (preset, 題) 組合從表上消失。

    首輪 32 / v14 25 / v15 21 個組合不在任何分母裡,而 v15 的 21 筆棄答中
    至少 11 筆真值 chunk 就在取回結果內(錯誤棄答)。棄答必須是一種列型。
    """
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["參考資料未提及。"]))
    rows, n_cited, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert n_cited == 0  # 沒有任何有引用句
    assert len(rows) == 2  # 但兩個 preset 各留一列
    kinds = {r[sheet.KIND_COL] for r in rows}
    assert kinds == {"棄答"}
    # 真值 c1 在 scripted 的第一名 → 這是「該取到卻棄答」的訊號
    assert {r[sheet.TRUTH_RANK_COL] for r in rows} == {1}


def test_unanswered_row_is_distinguished_from_abstain(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """回答了但一句都沒標引用(v14 的 b2-q11 hybrid 型)必須與棄答分開。"""
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<無引用標記的回答>"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert {r[sheet.KIND_COL] for r in rows} == {"無引用回答"}


def test_filter_band_drops_rows_without_coverage(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """棄答列沒有覆蓋率,不屬於 low / high 任一帶(否則 float("") 直接炸)。"""
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["參考資料未提及。"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert sheet.filter_band(rows, "all", 0.5) == rows
    assert sheet.filter_band(rows, "low", 0.5) == []
    assert sheet.filter_band(rows, "high", 0.5) == []


def test_answered_row_carries_term_columns(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """作答列帶兩個藥名對照欄;段落有而句中沒有的拉丁詞要被標出(發現 42/46)。"""
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    pool = RetrieverPool(
        cfg,
        {"c1": {"text": "<段落> methotrexate <說明>"}, "c2": {"text": "<乙>"}},
        lambda _n: ScriptedRetriever({"<問1>": ["c1"]}),
    )
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["使用甲基喋吟治療[1]。"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert rows[0][sheet.KIND_COL] == "作答"
    assert rows[0][sheet.MISSING_COL] == "methotrexate"
    assert rows[0][sheet.TRUTH_RANK_COL] == ""  # 只有棄答列填名次


def test_write_does_not_pad_header_without_components(tmp_path: Path) -> None:
    """未給 --components 時表頭必須恰為 14 欄,不得被欄寬設定補出 None 欄。

    openpyxl 的 ws.cell() 會建立儲存格,所以欄寬迴圈跑過表頭長度就會污染表頭
    (2026-09-22 由 gauntlet 5l 的新斷言抓到的既有缺陷)。
    """
    out = tmp_path / "h.xlsx"
    row = ["l", "m", "q1", "<問>", "<句>", "1", 0.2, "<段落>", "", "", "作答", "", "", ""]
    path = sheet.write([row], out)
    if path.suffix == ".csv":
        pytest.skip("openpyxl 不在此環境")
    from openpyxl import load_workbook

    ws = load_workbook(path).active
    assert [c.value for c in next(ws.iter_rows(max_row=1))] == sheet.HEADER


def test_abstain_row_carries_cites_from_whole_answer(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """棄答列必須填整段回答的有效引用編號(第六輪 jsonl 實證:22 筆棄答有 21 筆帶 [n])。

    這一欄過去寫死空字串,丟掉最強的訊號——模型**引了段落卻說沒提到**
    (實例形態:「參考資料未提及 [3]」)。那是發現 48「宣稱缺席」的直接實證,
    不是推論;句級的 sentence_support 拿不到它,因為它整句跳過棄答句。
    """
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["參考資料未提及 [2]"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert {r[sheet.KIND_COL] for r in rows} == {"棄答"}
    assert {r[5] for r in rows} == {"2"}  # 引用編號欄


def test_abstain_row_omits_out_of_range_cites(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """越界編號不算引用,與 sentence_support 的有效性判準一致(top_k=2 時 [9] 無效)。"""
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["參考資料未提及 [9]"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert {r[5] for r in rows} == {""}


def test_unanswered_row_never_has_valid_cites(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """「無引用回答」列的引用欄必為空——有有效 [n] 就會成為作答列,兩者互斥。

    原本我寫的是「無引用回答列也要填整段的有效編號」,但那個狀態**邏輯上不存在**:
    一個有效引用會讓該句算出覆蓋率而成為作答列。這裡改成鎖住列型與引用欄的一致性,
    這樣若日後有人把列型判斷改成別的條件(例如改看句數),這條會擋下來。
    """
    cfg, pool, es = _setup("http://127.0.0.1:1", tmp_path)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<無引用標記的回答>"]))
    rows, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert {r[sheet.KIND_COL] for r in rows} == {"無引用回答"}
    assert {r[5] for r in rows} == {""}
    # 反向:同一個回答加上有效引用,就該變成作答列(而非無引用回答列)
    monkeypatch.setattr(sheet, "stream_answer", lambda *_a, **_k: iter(["<段落甲>[1]。"]))
    rows2, _, _ = sheet.collect_rows(cfg, pool, es, 0, 0.5)
    assert {r[sheet.KIND_COL] for r in rows2} == {"作答"}
