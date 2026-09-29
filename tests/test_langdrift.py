"""簡體字元佔比(語言漂移)——場景為 2026-09-17 定案的驗收條件(手工字表方案)。"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.perf.langdrift import SIMPLIFIED_ONLY, drift_summary, simplified_ratio

# 簡繁共用字(合法繁體字):字表必須排除,否則繁體輸出會被誤計
SHARED = "后云里干面几丑谷斗周系缺没床强户温黄决余骨呼叠翻禁咨表症"
TRAD = "參考資料未提及藥物醫療診斷劑量說明這個時間會對應"


# 1 手算
def test_ratio_hand_computed() -> None:
    # 参资药 在字表;考料未提及物 不在;共 9 個 CJK 字
    assert simplified_ratio("参考资料未提及药物") == pytest.approx(3 / 9)
    assert simplified_ratio("参考资料 abc 123") == pytest.approx(2 / 4)


def test_ratio_zero_cases() -> None:
    assert simplified_ratio(TRAD) == 0.0
    assert simplified_ratio("abc 123") == 0.0
    assert simplified_ratio("") == 0.0
    assert simplified_ratio("后來云云") == 0.0  # 共用字不計


# 2 摘要
def test_drift_summary_hand_computed() -> None:
    s = drift_summary({"b": "参考", "a": TRAD, "c": "药药药"})
    assert s == {
        "n": 3,
        "texts_with_simplified": 2,
        "ratio_mean": pytest.approx((0.5 + 0.0 + 1.0) / 3),
        "ratio_max": 1.0,
        "ids": ["b", "c"],
    }
    assert drift_summary({}) == {
        "n": 0,
        "texts_with_simplified": 0,
        "ratio_mean": 0.0,
        "ratio_max": 0.0,
        "ids": [],
    }


# 3 字表健康
def test_table_health() -> None:
    assert not (SIMPLIFIED_ONLY & set(SHARED))
    assert not (SIMPLIFIED_ONLY & set(TRAD))
    assert all(0x4E00 <= ord(c) <= 0x9FFF for c in SIMPLIFIED_ONLY)
    assert len(SIMPLIFIED_ONLY) >= 300
    assert {"参", "资", "药", "体", "医", "这", "们"} <= SIMPLIFIED_ONLY


# 4 property
@given(st.text(max_size=80))
@settings(max_examples=200)
def test_ratio_property_range(text: str) -> None:
    assert 0.0 <= simplified_ratio(text) <= 1.0


@given(st.text(alphabet=TRAD + "abc ,。", max_size=60))
@settings(max_examples=100)
def test_ratio_property_traditional_is_zero(text: str) -> None:
    assert simplified_ratio(text) == 0.0


@given(st.lists(st.sampled_from(sorted(SIMPLIFIED_ONLY)), min_size=1, max_size=30))
@settings(max_examples=100)
def test_ratio_property_all_simplified_is_one(chars: list[str]) -> None:
    assert simplified_ratio("".join(chars) + " ab") == 1.0


_MIXED = st.text(alphabet=sorted(SIMPLIFIED_ONLY)[:40] + list(TRAD + " a1"), max_size=30)


@given(st.dictionaries(st.text(min_size=1, max_size=4), _MIXED, max_size=8))
@settings(max_examples=100)
def test_summary_property_invariants(texts: dict[str, str]) -> None:
    s = drift_summary(texts)
    assert s["n"] == len(texts)
    assert s["ratio_max"] >= s["ratio_mean"] >= 0.0
    assert s["texts_with_simplified"] == len(s["ids"])
    assert s["ids"] == sorted(s["ids"])
    assert all(simplified_ratio(texts[i]) > 0 for i in s["ids"])
    if texts:  # max 與 mean 各自獨立重算(單邊 max ≥ mean 擋不住「max 回傳平均」)
        ratios = [simplified_ratio(v) for v in texts.values()]
        assert s["ratio_max"] == max(ratios)
        assert s["ratio_mean"] == pytest.approx(sum(ratios) / len(ratios))
