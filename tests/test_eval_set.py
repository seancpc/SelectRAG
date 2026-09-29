"""eval set schema 測試 —— 場景為 2026-08-27 定案的驗收條件。

所有 fixture 內容皆為合成資料,非真實醫療文件。
"""

import json
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from selectrag.evaluation.eval_set import (
    EvalItem,
    EvalSetIntegrityError,
    compute_content_hash,
    freeze,
    load_eval_set,
)
from selectrag.evaluation.retrieval_metrics import recall_at_k

# --------------------------------------------------------------------------
# 測試資料(合成)
# --------------------------------------------------------------------------


def make_items() -> list[EvalItem]:
    return [
        EvalItem(
            id="q001",
            question="<問題文字 A>",
            relevant_doc_ids=frozenset({"doc_a", "doc_b"}),
            reference_answer="<參考答案 A>",
            notes="<標註理由 A>",
        ),
        EvalItem(id="q002", question="<問題文字 B>", relevant_doc_ids=frozenset({"doc_c"})),
        EvalItem(
            id="q003",
            question="臨床指引查詢的中文題目",
            relevant_doc_ids=frozenset({"doc_d", "doc_e"}),
            notes="全形標點:、。「」",
        ),
    ]


def write_frozen(path: Path, items: list[EvalItem], version: str = "v1") -> Path:
    payload = freeze(items, version=version, frozen_at="2026-08-27")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 載入與凍結完整性
# --------------------------------------------------------------------------


def test_load_valid_frozen_file(tmp_path: Path) -> None:
    path = write_frozen(tmp_path / "eval.json", make_items())
    result = load_eval_set(path)
    assert result.version == "v1"
    assert result.frozen_at == "2026-08-27"
    assert len(result.items) == 3
    assert result.items[0].relevant_doc_ids == frozenset({"doc_a", "doc_b"})
    assert result.items[1].reference_answer is None


def test_load_detects_tampered_content(tmp_path: Path) -> None:
    """改一個字就必須被抓到 —— 失效模式 1。"""
    path = write_frozen(tmp_path / "eval.json", make_items())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["question"] = "<問題文字 A 被改過>"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError, match="content hash mismatch"):
        load_eval_set(path)


def test_load_detects_tampered_relevant_ids(tmp_path: Path) -> None:
    """真值被改動同樣要被抓到。"""
    path = write_frozen(tmp_path / "eval.json", make_items())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][1]["relevant_doc_ids"] = ["doc_c", "doc_z"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError, match="content hash mismatch"):
        load_eval_set(path)


def test_load_detects_item_count_mismatch(tmp_path: Path) -> None:
    path = write_frozen(tmp_path / "eval.json", make_items())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["item_count"] = 99
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError, match="item_count mismatch"):
        load_eval_set(path)


def test_hash_is_insensitive_to_item_order() -> None:
    """題目順序不是語意的一部分 —— 重排不該觸發假警報。"""
    items = make_items()
    assert compute_content_hash(items) == compute_content_hash(list(reversed(items)))


def test_hash_is_sensitive_to_content() -> None:
    """任一字元改動必須改變 hash —— 失效模式 1。"""
    items = make_items()
    changed = [
        *items[:1],
        EvalItem(id="q002", question="<改過>", relevant_doc_ids=frozenset({"doc_c"})),
        *items[2:],
    ]
    assert compute_content_hash(items) != compute_content_hash(changed)


def test_hash_is_insensitive_to_json_key_order(tmp_path: Path) -> None:
    """鍵順序不同但內容相同的檔案,hash 必須一致 —— 失效模式 2。"""
    path_a = write_frozen(tmp_path / "a.json", make_items())
    payload = json.loads(path_a.read_text(encoding="utf-8"))
    reordered = {
        "items": [dict(reversed(list(item.items()))) for item in payload["items"]],
        "manifest": payload["manifest"],
    }
    path_b = tmp_path / "b.json"
    path_b.write_text(json.dumps(reordered, ensure_ascii=False), encoding="utf-8")
    assert load_eval_set(path_b).items == load_eval_set(path_a).items


# --------------------------------------------------------------------------
# 結構驗證 —— 壞資料一律 raise
# --------------------------------------------------------------------------


def test_rejects_empty_eval_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval set is empty"):
        write_frozen(tmp_path / "eval.json", [])


def test_rejects_duplicate_item_id(tmp_path: Path) -> None:
    items = make_items()
    items[1] = EvalItem(id="q001", question="<重複 id>", relevant_doc_ids=frozenset({"doc_x"}))
    with pytest.raises(ValueError, match="duplicate item id"):
        write_frozen(tmp_path / "eval.json", items)


def test_rejects_empty_relevant_doc_ids(tmp_path: Path) -> None:
    items = make_items()
    items[1] = EvalItem(id="q009", question="<無真值>", relevant_doc_ids=frozenset())
    with pytest.raises(ValueError, match="empty relevant_doc_ids"):
        write_frozen(tmp_path / "eval.json", items)


def test_rejects_blank_question(tmp_path: Path) -> None:
    items = make_items()
    items[1] = EvalItem(id="q009", question="   ", relevant_doc_ids=frozenset({"doc_x"}))
    with pytest.raises(ValueError, match="question is empty"):
        write_frozen(tmp_path / "eval.json", items)


def test_rejects_blank_item_id(tmp_path: Path) -> None:
    items = make_items()
    items[1] = EvalItem(id="", question="<空 id>", relevant_doc_ids=frozenset({"doc_x"}))
    with pytest.raises(ValueError, match="item id is empty"):
        write_frozen(tmp_path / "eval.json", items)


def test_load_rejects_missing_field(tmp_path: Path) -> None:
    path = write_frozen(tmp_path / "eval.json", make_items())
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["items"][0]["question"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_eval_set(path)


def test_load_rejects_missing_manifest(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"items": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_eval_set(path)


def test_load_rejects_duplicate_ids_in_relevant_doc_ids(tmp_path: Path) -> None:
    """JSON 的 list 可能含重複,轉 set 前必須抓到 —— 失效模式 4。"""
    path = write_frozen(tmp_path / "eval.json", make_items())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][1]["relevant_doc_ids"] = ["doc_c", "doc_c"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(path)


# --------------------------------------------------------------------------
# 中文編碼與 round-trip
# --------------------------------------------------------------------------


def test_chinese_text_survives_round_trip(tmp_path: Path) -> None:
    """繁中與全形標點必須原樣保存 —— 失效模式 5。"""
    path = write_frozen(tmp_path / "eval.json", make_items())
    raw = path.read_text(encoding="utf-8")
    assert "臨床指引查詢的中文題目" in raw, "不應被 \\uXXXX 轉義"
    loaded = load_eval_set(path)
    assert loaded.items[2].question == "臨床指引查詢的中文題目"
    assert loaded.items[2].notes == "全形標點:、。「」"


def test_round_trip_preserves_all_items(tmp_path: Path) -> None:
    original = make_items()
    path = write_frozen(tmp_path / "eval.json", original)
    loaded = load_eval_set(path)
    assert list(loaded.items) == original


# --------------------------------------------------------------------------
# 與指標模組的契約相容性
# --------------------------------------------------------------------------


def test_relevant_doc_ids_feeds_directly_into_metrics(tmp_path: Path) -> None:
    """真值型別必須能直接餵給 recall_at_k,指標層不做型別轉換。"""
    path = write_frozen(tmp_path / "eval.json", make_items())
    item = load_eval_set(path).items[0]
    assert recall_at_k(["doc_a", "doc_z"], item.relevant_doc_ids, k=2) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Property-based(Tier 3)
# --------------------------------------------------------------------------

ids = st.text(alphabet="abcdef0123456789", min_size=1, max_size=4)
texts = st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != "")


@st.composite
def eval_items(draw: st.DrawFn) -> list[EvalItem]:
    chosen_ids = draw(st.lists(ids, min_size=1, max_size=6, unique=True))
    return [
        EvalItem(
            id=item_id,
            question=draw(texts),
            relevant_doc_ids=frozenset(draw(st.sets(ids, min_size=1, max_size=4))),
            reference_answer=draw(st.one_of(st.none(), texts)),
            notes=draw(st.one_of(st.none(), texts)),
        )
        for item_id in chosen_ids
    ]


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(items=eval_items())
def test_property_round_trip(
    items: list[EvalItem], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """freeze → load 後內容完全相同。"""
    path = tmp_path_factory.mktemp("prop") / "eval.json"
    write_frozen(path, items)
    assert list(load_eval_set(path).items) == items


@given(items=eval_items())
def test_property_hash_stable_across_reordering(items: list[EvalItem]) -> None:
    assert compute_content_hash(items) == compute_content_hash(list(reversed(items)))


@given(items=eval_items(), suffix=st.text(min_size=1, max_size=3))
def test_property_hash_changes_when_question_changes(items: list[EvalItem], suffix: str) -> None:
    """任何題目文字的改動都必須改變 hash。"""
    before = compute_content_hash(items)
    mutated = [
        EvalItem(
            id=items[0].id,
            question=items[0].question + suffix,
            relevant_doc_ids=items[0].relevant_doc_ids,
            reference_answer=items[0].reference_answer,
            notes=items[0].notes,
        ),
        *items[1:],
    ]
    assert compute_content_hash(mutated) != before


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(items=eval_items())
def test_property_item_count_preserved(
    items: list[EvalItem], tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("prop") / "eval.json"
    write_frozen(path, items)
    assert len(load_eval_set(path).items) == len(items)


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(items=eval_items(), suffix=st.text(min_size=1, max_size=3))
def test_property_content_tampering_is_detected(
    items: list[EvalItem], suffix: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """凍結後任何內容改動都必須被偵測 —— 失效模式 1 的核心不變量。

    補洞:round-trip 與 hash 穩定性不變量在「hash 比對失效」的實作下依然成立,
    唯有明寫「竄改必被抓到」的不變量能抓到它(property-only 變異測試證實)。
    """
    path = tmp_path_factory.mktemp("tamper") / "eval.json"
    write_frozen(path, items)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["question"] += suffix
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError):
        load_eval_set(path)


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(items=eval_items())
def test_property_item_removal_is_detected(
    items: list[EvalItem], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """刪題必須被 item_count 或 hash 抓到。"""
    assume(len(items) >= 2)
    path = tmp_path_factory.mktemp("removal") / "eval.json"
    write_frozen(path, items)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].pop()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError):
        load_eval_set(path)


@settings(deadline=None)  # 含檔案 I/O,deadline 會偶發誤判
@given(items=eval_items(), delta=st.integers(min_value=1, max_value=5))
def test_property_manifest_count_tampering_is_detected(
    items: list[EvalItem], delta: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """manifest 被改但 items 沒改,必須被 item_count 比對抓到。

    補洞:刪題會先觸發 hash 不符,所以「刪題被偵測」的不變量無法考驗 item_count
    這條防線;唯有只動 manifest 才會逼它出手(property-only 變異測試證實)。
    """
    path = tmp_path_factory.mktemp("manifest") / "eval.json"
    write_frozen(path, items)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["item_count"] += delta
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvalSetIntegrityError, match="item_count mismatch"):
        load_eval_set(path)
