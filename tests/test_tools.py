"""tools/ 一次性工具的最小煙霧測試(改善項目 12)。測試資料一律中性占位符。"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_make_synthetic_eval_is_deterministic() -> None:
    mod = _load("make_synthetic_eval")
    items_a, script_a = mod.build()
    items_b, script_b = mod.build()
    assert len(items_a) == mod.ITEM_COUNT
    assert items_a == items_b and script_a == script_b, "同程式碼必須產生同樣的檔案"
    assert set(script_a) == {item.question for item in items_a}, "每題都要有對應的假檢索結果"


def _write_dataset(path: Path) -> Path:
    header = "許可證字號,註銷狀態,許可證種類,主成分略述,適應症,中文品名,申請商名稱\n"
    rows = [
        "<許可證A>,,製　劑,INGREDIENT-X,<適應症甲>,<品名一>,<廠商一>",
        "<許可證B>,,製　劑,INGREDIENT-X,<適應症甲>,<品名二>,<廠商二>",
        "<許可證C>,已註銷,製　劑,INGREDIENT-X,<適應症甲>,<品名三>,<廠商三>",
        "<許可證D>,,製　劑,INGREDIENT-X;;OTHER,<適應症乙>,<品名四>,<廠商四>",
        "<許可證E>,,製　劑,INGREDIENT-Y,<適應症丙>,<品名五>,<廠商五>",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8-sig")
    return path


def test_pick_leaflets_filters_and_picks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load("pick_leaflets")
    rows = mod.load_rows(_write_dataset(tmp_path / "ds.csv"))
    active = mod.active_single_ingredient(rows)
    # 已註銷者與複方(;;)被排除
    assert {r["許可證字號"] for r in active} == {"<許可證A>", "<許可證B>", "<許可證E>"}
    mod.pick(rows, "ingredient-x", count=10, same_indication=True)
    out = capsys.readouterr().out
    assert "<許可證A>" in out and "<許可證B>" in out and "<許可證E>" not in out


def test_pick_leaflets_rejects_missing_columns(tmp_path: Path) -> None:
    mod = _load("pick_leaflets")
    bad = tmp_path / "bad.csv"
    bad.write_text("欄位甲,欄位乙\n1,2\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="缺少必要欄位"):
        mod.load_rows(bad)


def test_check_eval_corpus_reports_missing(tmp_path: Path) -> None:
    """真值 chunk 缺席時回報題目與 id;全數存在時 missing 為空。"""
    import json

    from selectrag.evaluation.eval_set import EvalItem, freeze

    mod = _load("check_eval_corpus")
    eval_path = tmp_path / "e.json"
    items = [EvalItem(id="q1", question="<q>", relevant_doc_ids=frozenset({"c1", "c2"}))]
    eval_path.write_text(json.dumps(freeze(items, "t", "t")), encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "c1", "text": "x"}) + "\n", encoding="utf-8")
    total, missing = mod.missing_ids(eval_path, chunks)
    assert (total, missing) == (2, [("q1", "c2")])
    chunks.write_text(
        "\n".join(json.dumps({"chunk_id": c, "text": "x"}) for c in ("c1", "c2")) + "\n",
        encoding="utf-8",
    )
    assert mod.missing_ids(eval_path, chunks) == (2, [])


def test_audit_dense_hits_lists_non_truth_candidates(tmp_path: Path) -> None:
    """只列 dense top-k 中不在真值集的 chunk,並標記是否也出現在 BM25 top-k。"""
    import json

    from selectrag.evaluation.eval_set import EvalItem, freeze

    mod = _load("audit_dense_hits")
    eval_path = tmp_path / "e.json"
    items = [
        EvalItem(id="q1", question="<q1>", relevant_doc_ids=frozenset({"c1"})),
        EvalItem(id="q2", question="<q2>", relevant_doc_ids=frozenset({"c9"})),
    ]
    eval_path.write_text(json.dumps(freeze(items, "t", "t")), encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps({"chunk_id": c, "text": f"<{c}>"}) for c in ("c1", "c2", "c3")) + "\n",
        encoding="utf-8",
    )

    def result(name: str, retrieved: dict[str, list[str]]) -> Path:
        p = tmp_path / f"{name}.json"
        p.write_text(
            json.dumps(
                {"per_query": [{"item_id": k, "retrieved": v} for k, v in retrieved.items()]}
            ),
            encoding="utf-8",
        )
        return p

    dense = result("dense", {"q1": ["c1", "c2"], "q2": ["c3", "c2"]})
    bm25 = result("bm25", {"q1": ["c1", "c3"], "q2": ["c2"]})
    rows = mod.build_candidates(eval_path, chunks, dense, bm25, top_k=2)
    assert [r["item_id"] for r in rows] == ["q1", "q2"]
    assert [c["chunk_id"] for c in rows[0]["candidates"]] == ["c2"]  # c1 是真值,不列
    assert rows[0]["candidates"][0]["also_in_bm25_topk"] is False
    assert rows[1]["dense_hit_in_topk"] is False
    assert [(c["chunk_id"], c["also_in_bm25_topk"]) for c in rows[1]["candidates"]] == [
        ("c3", False),
        ("c2", True),
    ]
    assert rows[1]["candidates"][0]["text"] == "<c3>"
