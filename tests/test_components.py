"""必要成分命中判定(多成分題的答案完整性人工輔助欄)。

成分內容為領域真值,此處全部使用合成的中性成分與答案文字。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.evaluation.components import (
    ComponentSpec,
    ItemComponents,
    completeness,
    component_hits,
    item_completeness,
    item_forbidden_hits,
    load_components,
    normalize_glyphs,
    unsupported_hits,
)


def _write(tmp_path: Path, payload: object) -> Path:
    p = tmp_path / "components.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _ok_payload() -> dict[str, object]:
    return {
        "version": "test-v1",
        "items": {
            "x-q01": [
                {"name": "<成分甲>", "aliases": ["<表述甲一>", "<表述甲二>"]},
                {"name": "<成分乙>", "aliases": ["<表述乙>"]},
            ]
        },
    }


# --- B1 load_components -------------------------------------------------


def test_load_returns_specs_by_item(tmp_path: Path) -> None:
    got = load_components(_write(tmp_path, _ok_payload()))
    assert set(got) == {"x-q01"}
    assert got["x-q01"].components == (
        ComponentSpec("<成分甲>", ("<表述甲一>", "<表述甲二>")),
        ComponentSpec("<成分乙>", ("<表述乙>",)),
    )
    assert got["x-q01"].forbidden == ()


def test_load_rejects_missing_name(tmp_path: Path) -> None:
    bad = {"version": "v", "items": {"x-q01": [{"aliases": ["<表述>"]}]}}
    with pytest.raises(ValueError, match="missing field 'name'"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_missing_aliases(tmp_path: Path) -> None:
    bad = {"version": "v", "items": {"x-q01": [{"name": "<成分甲>"}]}}
    with pytest.raises(ValueError, match="missing field 'aliases'"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_empty_aliases(tmp_path: Path) -> None:
    bad = {"version": "v", "items": {"x-q01": [{"name": "<成分甲>", "aliases": []}]}}
    with pytest.raises(ValueError, match="empty aliases"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_duplicate_name_in_same_item(tmp_path: Path) -> None:
    bad = {
        "version": "v",
        "items": {
            "x-q01": [
                {"name": "<成分甲>", "aliases": ["<表述一>"]},
                {"name": "<成分甲>", "aliases": ["<表述二>"]},
            ]
        },
    }
    with pytest.raises(ValueError, match="duplicate component name"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_item_with_no_components(tmp_path: Path) -> None:
    bad = {"version": "v", "items": {"x-q01": []}}
    with pytest.raises(ValueError, match="no components"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_missing_items_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing field 'items'"):
        load_components(_write(tmp_path, {"version": "v"}))


def test_load_rejects_blank_alias(tmp_path: Path) -> None:
    bad = {"version": "v", "items": {"x-q01": [{"name": "<成分甲>", "aliases": ["  "]}]}}
    with pytest.raises(ValueError, match="blank alias"):
        load_components(_write(tmp_path, bad))


# --- B2 normalize_glyphs ------------------------------------------------


def test_normalize_folds_fullwidth_and_strips_space() -> None:
    assert normalize_glyphs("（一條血管）　1０800 點") == "(一條血管)10800點"


def test_normalize_keeps_chinese_numerals() -> None:
    # 語意正規化是禁區(N4):中文數字不得被改寫成阿拉伯數字
    assert normalize_glyphs("四十五歲") == "四十五歲"


def test_normalize_folds_fullwidth_comma_and_colon() -> None:
    assert normalize_glyphs("甲,乙:丙") == "甲,乙:丙"


# --- B3/B4 component_hits -----------------------------------------------

SPECS = (
    ComponentSpec("<成分甲>", ("<表述甲一>", "<表述甲二>")),
    ComponentSpec("<成分乙>", ("<表述乙>",)),
)


def test_hits_partial() -> None:
    hit, miss, blocked = component_hits("前綴<表述甲一>後綴", SPECS)
    assert hit == ("<成分甲>",)
    assert miss == ("<成分乙>",)
    assert blocked == ()


def test_hits_any_alias_counts() -> None:
    # B4:第二個等價表述命中即算(「每二年進行一次」型的措辭變體)
    hit, miss, _ = component_hits("前綴<表述甲二>後綴", SPECS)
    assert hit == ("<成分甲>",)
    assert miss == ("<成分乙>",)


def test_hits_all() -> None:
    hit, miss, _ = component_hits("<表述甲一>與<表述乙>", SPECS)
    assert hit == ("<成分甲>", "<成分乙>")
    assert miss == ()


def test_hits_none() -> None:
    hit, miss, _ = component_hits("完全無關的答案", SPECS)
    assert hit == ()
    assert miss == ("<成分甲>", "<成分乙>")


def test_hits_normalizes_both_sides() -> None:
    # 答案帶全形與空白、alias 為半形,仍應命中(B2 的界線只到字形)
    specs = (ComponentSpec("<成分>", ("(一條血管)10800",)),)
    hit, _, _ = component_hits("支付點數為（一條血管）　1０800 點", specs)
    assert hit == ("<成分>",)


def test_hits_preserves_spec_order() -> None:
    specs = (
        ComponentSpec("<丙>", ("<表述丙>",)),
        ComponentSpec("<甲>", ("<表述甲>",)),
    )
    hit, _, _ = component_hits("<表述甲><表述丙>", specs)
    assert hit == ("<丙>", "<甲>")


def test_hits_empty_specs_raises() -> None:
    with pytest.raises(ValueError, match="empty component specs"):
        component_hits("任意答案", ())


# --- B5 completeness ----------------------------------------------------


def test_completeness_hand_computed() -> None:
    specs = SPECS + (ComponentSpec("<成分丙>", ("<表述丙>",)),)
    assert completeness("<表述甲一>與<表述乙>", specs) == pytest.approx(2 / 3)


def test_completeness_full_and_zero() -> None:
    assert completeness("<表述甲一><表述乙>", SPECS) == pytest.approx(1.0)
    assert completeness("無關", SPECS) == pytest.approx(0.0)


def test_completeness_empty_specs_raises() -> None:
    with pytest.raises(ValueError, match="empty component specs"):
        completeness("任意答案", ())


# --- property -----------------------------------------------------------


# 註:以下策略的別名一律以 `<…>` 包裹,邊界字元永遠非數字,因此 property 的
# 「獨立重算」可以用樸素的 `in` —— 那正是獨立性的來源(不拿被測的 `_contains`
# 去驗證 `_contains`)。數字邊界規則由手算測試與專屬 property 覆蓋。


@st.composite
def _specs_and_answer(draw: st.DrawFn) -> tuple[tuple[ComponentSpec, ...], str]:
    names = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    specs = tuple(ComponentSpec(n, (f"<{n}>",)) for n in names)
    present = draw(st.lists(st.sampled_from(names), max_size=5, unique=True))
    answer = "".join(f"<{n}>" for n in present)
    return specs, answer


@settings(max_examples=200, deadline=None)
@given(_specs_and_answer())
def test_property_hit_and_miss_partition_specs(
    case: tuple[tuple[ComponentSpec, ...], str],
) -> None:
    specs, answer = case
    hit, miss, blocked = component_hits(answer, specs)
    names = tuple(s.name for s in specs)
    # 不變量一:命中 / 缺漏 / 禁止 三者恰好分割成分清單,且各自保序
    assert set(hit) | set(miss) | set(blocked) == set(names)
    assert len(hit) + len(miss) + len(blocked) == len(names)
    for part in (hit, miss, blocked):
        assert part == tuple(n for n in names if n in set(part))


@settings(max_examples=200, deadline=None)
@given(_specs_and_answer())
def test_property_completeness_equals_hit_fraction(
    case: tuple[tuple[ComponentSpec, ...], str],
) -> None:
    specs, answer = case
    hit, _, _ = component_hits(answer, specs)
    # 不變量二:獨立重算(不呼叫 component_hits 的結果當唯一來源)
    expected = sum(
        1 for s in specs if any(normalize_glyphs(a) in normalize_glyphs(answer) for a in s.aliases)
    ) / len(specs)
    assert completeness(answer, specs) == pytest.approx(expected)
    assert completeness(answer, specs) == pytest.approx(len(hit) / len(specs))


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=40))
def test_property_normalize_is_idempotent(s: str) -> None:
    once = normalize_glyphs(s)
    assert normalize_glyphs(once) == once


# --- S-C2 禁止表述(2026-09-20 核准,來自核准表 r20 的四倍劑量案例)-------


def test_forbidden_blocks_the_component() -> None:
    # 「每日2粒」是「2粒」的超集:子字串比對無法表達否定,故以禁止表述封死
    specs = (ComponentSpec("<單次劑量>", ("<每次2粒>", "<2粒>"), forbidden=("<每日2粒>",)),)
    hit, miss, blocked = component_hits("用法為<每日2粒>", specs)
    assert hit == ()
    assert miss == ()
    assert blocked == ("<單次劑量>",)


def test_forbidden_absent_keeps_the_hit() -> None:
    specs = (ComponentSpec("<單次劑量>", ("<每次2粒>",), forbidden=("<每日2粒>",)),)
    hit, miss, blocked = component_hits("用法為<每次2粒>", specs)
    assert hit == ("<單次劑量>",)
    assert (miss, blocked) == ((), ())


def test_forbidden_wins_over_alias_hit() -> None:
    # 同句同時含正解與禁止表述時以禁止為準:答錯優先於答對
    specs = (ComponentSpec("<劑量>", ("<每次2粒>",), forbidden=("<每日2粒>",)),)
    hit, _, blocked = component_hits("<每次2粒>但又寫<每日2粒>", specs)
    assert hit == ()
    assert blocked == ("<劑量>",)


def test_blocked_not_counted_in_completeness() -> None:
    specs = (
        ComponentSpec("<甲>", ("<表述甲>",), forbidden=("<禁甲>",)),
        ComponentSpec("<乙>", ("<表述乙>",)),
    )
    assert completeness("<禁甲><表述乙>", specs) == pytest.approx(0.5)


def test_load_parses_forbidden_and_defaults_empty(tmp_path: Path) -> None:
    payload = {
        "version": "v",
        "items": {
            "x-q01": [
                {"name": "<甲>", "aliases": ["<表述甲>"], "forbidden": ["<禁甲>"]},
                {"name": "<乙>", "aliases": ["<表述乙>"]},
            ]
        },
    }
    got = load_components(_write(tmp_path, payload))
    assert got["x-q01"].components[0].forbidden == ("<禁甲>",)
    assert got["x-q01"].components[1].forbidden == ()


def test_load_rejects_blank_forbidden(tmp_path: Path) -> None:
    bad = {
        "version": "v",
        "items": {"x-q01": [{"name": "<甲>", "aliases": ["<表述甲>"], "forbidden": ["  "]}]},
    }
    with pytest.raises(ValueError, match="blank forbidden"):
        load_components(_write(tmp_path, bad))


def test_load_rejects_forbidden_equal_to_alias(tmp_path: Path) -> None:
    # 同一表述同時列為正解與禁止 → 該成分永遠不可能命中,是清單筆誤
    bad = {
        "version": "v",
        "items": {"x-q01": [{"name": "<甲>", "aliases": ["<表述甲>"], "forbidden": ["<表述甲>"]}]},
    }
    with pytest.raises(ValueError, match="forbidden overlaps aliases"):
        load_components(_write(tmp_path, bad))


# --- S-C3 正規化擴充(2026-09-20,來自核准表 r9 / r14 / r23)------------


def test_normalize_folds_cjk_radicals() -> None:
    # 全語料實測只有這三種部首字元(⻑ 24 次、⺠ 6 次、⻄ 4 次);NFKC 對此區無映射
    assert normalize_glyphs("延⻑") == normalize_glyphs("延長")
    assert normalize_glyphs("⺠眾") == normalize_glyphs("民眾")
    assert normalize_glyphs("⻄藥") == normalize_glyphs("西藥")


def test_normalize_folds_case() -> None:
    assert normalize_glyphs("IU/Kg") == normalize_glyphs("iu/kg")


def test_normalize_strips_thousands_separator_between_digits() -> None:
    assert normalize_glyphs("10,800點") == normalize_glyphs("10800點")


def test_normalize_keeps_comma_not_between_digits() -> None:
    # 只脫數字之間的逗號:句中的逗號是內容,不可移除
    assert normalize_glyphs("甲,乙") == "甲,乙"
    assert normalize_glyphs("2粒,每6小時") == "2粒,每6小時"


def test_normalize_folds_nfkc_compat_forms() -> None:
    assert normalize_glyphs("30㎏") == normalize_glyphs("30kg")


def test_normalize_still_keeps_chinese_numerals_after_extension() -> None:
    # N4 界線未因擴充而失守
    assert normalize_glyphs("三次") != normalize_glyphs("3次")


def test_normalize_idempotent_on_known_counterexample() -> None:
    """property 測試找到過的反例,固化成手算回歸測試。

    2026-09-20 gauntlet:「省掉末尾的 NFKC」這個變異**存活**——冪等性只有 property
    測試守著,而 st.text 要隨機湊出「字母 + U+00AF」這種序列靠運氣,不保證每輪重現。
    機制:NFKC 把 ¯ 展成「空白 + combining macron」,去空白後 macron 貼上前一字母,
    下一次呼叫才組合成 ū。少了末尾那次 NFKC,同一段文字兩次比對結果不同。
    """
    once = normalize_glyphs("U¯")
    assert normalize_glyphs(once) == once
    assert once == "ū"  # ū:已組合完成,不是 u + combining macron


# --- S-C8 題層禁止表述(2026-09-20 核准)-------------------------------
# 與成分層 forbidden 是不同機制,不可混用同一欄位:
#   保護型(成分層):會假性滿足某個成分 → 掛在該成分上把它歸零
#   懲罰型(題層):不命中任何別名,證明模型整題答到別的標的去了 → item 層級失敗


def _item_payload(item_forbidden: list[str] | None = None) -> dict[str, object]:
    item: dict[str, object] = {
        "components": [
            {"name": "<甲>", "aliases": ["<表述甲>"]},
            {"name": "<乙>", "aliases": ["<表述乙>"]},
        ]
    }
    if item_forbidden is not None:
        item["forbidden"] = item_forbidden
    return {"version": "v", "items": {"x-q01": item}}


def test_load_accepts_dict_form_with_item_forbidden(tmp_path: Path) -> None:
    got = load_components(_write(tmp_path, _item_payload(["<誤歸條文>"])))
    assert got["x-q01"].forbidden == ("<誤歸條文>",)
    assert tuple(c.name for c in got["x-q01"].components) == ("<甲>", "<乙>")


def test_load_accepts_legacy_list_form(tmp_path: Path) -> None:
    # 既有清單的 items 值是純 list,必須繼續可讀(題層 forbidden 視為空)
    legacy = {"version": "v", "items": {"x-q01": [{"name": "<甲>", "aliases": ["<表述甲>"]}]}}
    got = load_components(_write(tmp_path, legacy))
    assert got["x-q01"].forbidden == ()
    assert len(got["x-q01"].components) == 1


def test_load_rejects_blank_item_forbidden(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blank item forbidden"):
        load_components(_write(tmp_path, _item_payload(["  "])))


def test_load_rejects_item_forbidden_overlapping_any_alias(tmp_path: Path) -> None:
    # 題層禁止表述若等於任一成分的正解表述,該題永遠歸零,是清單筆誤
    with pytest.raises(ValueError, match="item forbidden overlaps"):
        load_components(_write(tmp_path, _item_payload(["<表述甲>"])))


def test_item_forbidden_hits_reports_matches(tmp_path: Path) -> None:
    item = load_components(_write(tmp_path, _item_payload(["<誤歸條文>", "<另一條文>"])))["x-q01"]
    assert item_forbidden_hits("答案提到<誤歸條文>", item) == ("<誤歸條文>",)
    assert item_forbidden_hits("乾淨的答案", item) == ()


def test_item_completeness_zeroed_by_item_forbidden(tmp_path: Path) -> None:
    item = load_components(_write(tmp_path, _item_payload(["<誤歸條文>"])))["x-q01"]
    # 兩個成分都答到,但整題命中題層禁止表述 → 歸零
    assert item_completeness("<表述甲><表述乙><誤歸條文>", item) == pytest.approx(0.0)
    # 無題層禁止表述命中時等於成分層比例
    assert item_completeness("<表述甲><表述乙>", item) == pytest.approx(1.0)
    assert item_completeness("<表述甲>", item) == pytest.approx(0.5)


def test_item_completeness_matches_component_layer_without_item_forbidden(
    tmp_path: Path,
) -> None:
    item = load_components(_write(tmp_path, _item_payload()))["x-q01"]
    assert item_completeness("<表述甲>", item) == pytest.approx(
        completeness("<表述甲>", item.components)
    )


def test_item_forbidden_normalized_like_aliases(tmp_path: Path) -> None:
    item = load_components(_write(tmp_path, _item_payload(["(誤歸)10800"])))["x-q01"]
    assert item_forbidden_hits("寫成（誤歸）　1０,800", item) == ("(誤歸)10800",)


# --- property:核心語意的不變量(2026-09-21 補)-------------------------
# 2026-09-20 gauntlet 4b(property 子集)判讀:73/144,存活者裡有三個不是
# 「策略刻意不產生的輸入」而是**新模組的核心語意**——多 alias、成分層 forbidden、
# 題層 forbidden 三者當時只有手算測試,完全沒有不變量守著
# (核心語意存活即真洞,必須補不變量而非手算案例)。


@st.composite
def _specs_with_multi_alias(draw: st.DrawFn) -> tuple[tuple[ComponentSpec, ...], str, str]:
    """回 (specs, answer, 未出現在答案中的字串)。每個成分帶 2~3 個等價表述。"""
    names = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    specs = []
    for n in names:
        n_alias = draw(st.integers(min_value=2, max_value=3))
        specs.append(ComponentSpec(n, tuple(f"<{n}#{i}>" for i in range(n_alias))))
    # 每個成分隨機挑「哪一個 alias(或都不)」出現在答案裡
    parts = []
    for spec in specs:
        idx = draw(st.integers(min_value=-1, max_value=len(spec.aliases) - 1))
        if idx >= 0:
            parts.append(spec.aliases[idx])
    return tuple(specs), "".join(parts), "<絕不出現的表述>"


@settings(max_examples=200, deadline=None)
@given(_specs_with_multi_alias())
def test_property_any_alias_suffices(case: tuple[tuple[ComponentSpec, ...], str, str]) -> None:
    """不變量:成分命中 ⟺ 它**任一** alias 出現在答案裡(不限第一個)。

    殺的變異:「只認第一個等價表述」——那會讓措辭變體被誤判為漏答。
    """
    specs, answer, _ = case
    hit, miss, blocked = component_hits(answer, specs)
    assert blocked == ()
    for spec in specs:
        expected = any(normalize_glyphs(a) in normalize_glyphs(answer) for a in spec.aliases)
        assert (spec.name in hit) is expected
        assert (spec.name in miss) is not expected


@settings(max_examples=200, deadline=None)
@given(_specs_with_multi_alias())
def test_property_adding_unmatched_alias_never_changes_result(
    case: tuple[tuple[ComponentSpec, ...], str, str],
) -> None:
    """不變量(單調性):加一個不在答案裡的 alias,判定結果完全不變。

    與上一條配對封死兩個方向:上一條防「漏看後面的 alias」,
    這一條防「多一個 alias 就誤判命中」。
    """
    specs, answer, absent = case
    before = component_hits(answer, specs)
    widened = tuple(ComponentSpec(s.name, (*s.aliases, absent), s.forbidden) for s in specs)
    assert component_hits(answer, widened) == before


@st.composite
def _specs_with_forbidden(draw: st.DrawFn) -> tuple[tuple[ComponentSpec, ...], str]:
    """回 (specs, answer)。部分成分帶禁止表述,且該表述可能出現在答案裡。"""
    names = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    specs = []
    parts = []
    for n in names:
        has_forbidden = draw(st.booleans())
        forbidden = (f"<禁{n}>",) if has_forbidden else ()
        specs.append(ComponentSpec(n, (f"<{n}>",), forbidden))
        if draw(st.booleans()):
            parts.append(f"<{n}>")
        if has_forbidden and draw(st.booleans()):
            parts.append(f"<禁{n}>")
    return tuple(specs), "".join(parts)


@settings(max_examples=300, deadline=None)
@given(_specs_with_forbidden())
def test_property_forbidden_always_blocks(case: tuple[tuple[ComponentSpec, ...], str]) -> None:
    """不變量:答案含任一禁止表述的成分,**必定**落在 blocked,且不在 hit 也不在 miss。

    殺的變異:「禁止表述被忽略」(四倍劑量錯誤拿到滿分)、
    「禁止表述命中後仍繼續比對 alias」(答錯與答對同時成立)。
    """
    specs, answer = case
    hit, miss, blocked = component_hits(answer, specs)
    norm = normalize_glyphs(answer)
    for spec in specs:
        tripped = any(normalize_glyphs(b) in norm for b in spec.forbidden)
        if tripped:
            assert spec.name in blocked
            assert spec.name not in hit and spec.name not in miss
        else:
            assert spec.name not in blocked
    # blocked 的成分不得計入 completeness 的分子
    assert completeness(answer, specs) == pytest.approx(len(hit) / len(specs))


@st.composite
def _item_with_forbidden(draw: st.DrawFn) -> tuple[ItemComponents, str]:
    """回 (item, answer)。題層禁止表述可能出現在答案裡。"""
    names = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    specs = tuple(ComponentSpec(n, (f"<{n}>",)) for n in names)
    item_forbidden = tuple(f"<題禁{i}>" for i in range(draw(st.integers(min_value=0, max_value=2))))
    parts = [f"<{n}>" for n in names if draw(st.booleans())]
    parts += [b for b in item_forbidden if draw(st.booleans())]
    return ItemComponents(specs, item_forbidden), "".join(parts)


@settings(max_examples=300, deadline=None)
@given(_item_with_forbidden())
def test_property_item_forbidden_zeroes_completeness(case: tuple[ItemComponents, str]) -> None:
    """不變量:答案含任一**題層**禁止表述 ⟹ item_completeness 恰為 0.0;否則等於成分層比例。

    殺的變異:「題層禁止表述被忽略」(整題答到別的標的仍拿滿分)。
    這條與成分層的不變量刻意分開——兩者是不同機制(懲罰型 vs 保護型),
    合成一條就測不出「兩個機制被寫進同一個欄位」這種錯。
    """
    item, answer = case
    norm = normalize_glyphs(answer)
    tripped = [b for b in item.forbidden if normalize_glyphs(b) in norm]
    assert item_forbidden_hits(answer, item) == tuple(tripped)
    if tripped:
        assert item_completeness(answer, item) == pytest.approx(0.0)
    else:
        assert item_completeness(answer, item) == pytest.approx(
            completeness(answer, item.components)
        )


# --- B1 數字邊界(2026-09-21 核准;首輪跑分暴露的假漏判)-----------------
# 模型寫「支付點數為1500,」,別名 1500 不能靠綁單位解決(「點數為1500」會誤中
# 「點數為15000」)。規則:別名邊界是數字時,答案中相鄰字元不得是數字。


def test_digit_boundary_blocks_longer_number() -> None:
    specs = (ComponentSpec("<點數>", ("1500",)),)
    hit, _, _ = component_hits("支付點數為1500,於病患回診當次申報", specs)
    assert hit == ("<點數>",)  # 後接逗號 → 命中
    hit2, miss2, _ = component_hits("支付點數為15000點", specs)
    assert hit2 == () and miss2 == ("<點數>",)  # 後接 0 → 不命中


def test_digit_boundary_checks_both_sides() -> None:
    specs = (ComponentSpec("<發現費>", ("650",)),)
    for ans, want in [
        ("每一個案支付發現費650點", True),
        ("支付6500點", False),  # 後接數字
        ("共1650點", False),  # 前接數字
        ("650", True),  # 整串即別名
    ]:
        hit, _, _ = component_hits(ans, specs)
        assert (hit == ("<發現費>",)) is want, (ans, hit)


def test_digit_boundary_scans_all_occurrences() -> None:
    # 第一次出現不合邊界,後面出現合格 → 仍應命中(不可只看第一個位置)
    specs = (ComponentSpec("<點數>", ("1500",)),)
    hit, _, _ = component_hits("先寫15000,後面才寫1500點", specs)
    assert hit == ("<點數>",)


def test_digit_boundary_not_applied_to_non_digit_edges() -> None:
    # 別名邊界非數字時行為不變(純子字串)
    specs = (ComponentSpec("<甲>", ("每二年一次",)),)
    hit, _, _ = component_hits("補助對象每二年一次檢查", specs)
    assert hit == ("<甲>",)


def test_digit_boundary_applies_to_forbidden_too() -> None:
    # 禁止表述走同一條比對路徑,邊界規則必須一致
    specs = (ComponentSpec("<劑量>", ("每次2粒",), forbidden=("每日2粒",)),)
    _, _, blocked = component_hits("用法為每日2粒", specs)
    assert blocked == ("<劑量>",)
    hit, _, blocked2 = component_hits("用法為每日22粒", specs)
    assert blocked2 == ()  # 「每日2粒」的尾是數字?否——尾是「粒」,故仍為純子字串比對
    assert hit == ()


@settings(max_examples=300, deadline=None)
@given(st.integers(min_value=1, max_value=9999), st.integers(min_value=0, max_value=9))
def test_property_digit_boundary_never_matches_longer_number(n: int, suffix: int) -> None:
    """不變量:純數字別名絕不命中「以它為前綴的更長數字」。

    殺的變異:邊界檢查被拿掉(1500 誤中 15000、650 誤中 6500)。
    """
    specs = (ComponentSpec("<n>", (str(n),)),)
    hit, _, _ = component_hits(f"數值為{n}{suffix}", specs)
    assert hit == ()
    # 但後面接非數字時必須命中
    hit2, _, _ = component_hits(f"數值為{n}點", specs)
    assert hit2 == ("<n>",)


@settings(max_examples=300, deadline=None)
@given(st.integers(min_value=1, max_value=999), st.integers(min_value=0, max_value=9))
def test_property_boundary_scans_all_positions(n: int, suffix: int) -> None:
    """不變量:前面有不合邊界的出現時,後面合格的出現仍必須命中。

    殺的變異:邊界比對只看第一個出現位置(`start = i + 1` 改成 `return False`)。
    2026-09-21 property 子集判讀:這是 `_contains` 的核心語意,原本只有手算測試守著。
    """
    specs = (ComponentSpec("<n>", (str(n),)),)
    hit, _, _ = component_hits(f"先寫{n}{suffix}然後才寫{n}點", specs)
    assert hit == ("<n>",)


# --- B2 引用支撐(2026-09-21 核准)---------------------------------------
# b1-q19 首輪:completeness=1 但唯一被引用的句子只有「補助對象每二年進行一次檢查」
# ——成分對完整回答計算,support 只收有引用的句子,兩者母體不同。
# 決議(核准):不改計算母體,改為把矛盾**顯示出來**,兩個訊號都保留。


def test_unsupported_hits_lists_components_absent_from_cited_text() -> None:
    specs = (
        ComponentSpec("<甲>", ("<表述甲>",)),
        ComponentSpec("<乙>", ("<表述乙>",)),
    )
    # 完整回答含兩個成分,但只有甲那句有引用
    hit, _, _ = component_hits("<表述甲>。<表述乙>。", specs)
    assert hit == ("<甲>", "<乙>")
    assert unsupported_hits(hit, ["<表述甲>。"], specs) == ("<乙>",)


def test_unsupported_hits_empty_when_all_supported() -> None:
    specs = (ComponentSpec("<甲>", ("<表述甲>",)),)
    assert unsupported_hits(("<甲>",), ["前綴<表述甲>後綴"], specs) == ()


def test_unsupported_hits_all_when_no_cited_sentences() -> None:
    specs = (ComponentSpec("<甲>", ("<表述甲>",)),)
    assert unsupported_hits(("<甲>",), [], specs) == ("<甲>",)


def test_unsupported_hits_uses_digit_boundary() -> None:
    # 引用句寫 15000、完整回答寫 1500 → 該成分無引用支撐
    specs = (ComponentSpec("<點數>", ("1500",)),)
    assert unsupported_hits(("<點數>",), ["支付15000點"], specs) == ("<點數>",)
    assert unsupported_hits(("<點數>",), ["支付1500點"], specs) == ()


def test_unsupported_hits_ignores_names_not_in_hit() -> None:
    specs = (ComponentSpec("<甲>", ("<表述甲>",)), ComponentSpec("<乙>", ("<表述乙>",)))
    # 只傳甲進 hit,乙不該出現在結果裡(即使引用句也沒有它)
    assert unsupported_hits(("<甲>",), ["無關內容"], specs) == ("<甲>",)


@st.composite
def _hit_and_cited(draw: st.DrawFn) -> tuple[tuple[ComponentSpec, ...], tuple[str, ...], list[str]]:
    """回 (specs, hit, cited_sentences)。部分命中成分的別名刻意不出現在引用句裡。"""
    names = draw(
        st.lists(
            st.text(alphabet="abcdef", min_size=1, max_size=3),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    specs = tuple(ComponentSpec(n, (f"<{n}>",)) for n in names)
    hit = tuple(draw(st.lists(st.sampled_from(names), max_size=4, unique=True)))
    # 引用句只含 hit 的一個子集
    supported = draw(st.lists(st.sampled_from(hit), max_size=4, unique=True)) if hit else []
    cited = [f"<{n}>" for n in supported]
    return specs, hit, cited


@settings(max_examples=300, deadline=None)
@given(_hit_and_cited())
def test_property_unsupported_is_exactly_hit_without_alias_in_cited(
    case: tuple[tuple[ComponentSpec, ...], tuple[str, ...], list[str]],
) -> None:
    """不變量:unsupported_hits 恰為「hit 中、別名未出現在引用句裡」的那些,且保序。

    殺的變異:引用支撐恆為空(completeness 與 support 的矛盾被掩蓋)。
    2026-09-21 的 4b 判讀:此函式的核心語意原本只有手算測試守著。
    """
    specs, hit, cited = case
    unsup = unsupported_hits(hit, cited, specs)
    by_name = {s.name: s for s in specs}
    joined = normalize_glyphs("".join(cited))
    expected = tuple(
        n
        for n in hit
        if n in by_name and not any(normalize_glyphs(a) in joined for a in by_name[n].aliases)
    )
    assert unsup == expected
    # 結構不變量:一定是 hit 的子序列,且與「有支撐的那些」互斥
    assert set(unsup) <= set(hit)
    assert unsup == tuple(n for n in hit if n in set(unsup))
