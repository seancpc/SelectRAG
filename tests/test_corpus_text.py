"""正規化、行合併、標題解析的測試 —— 場景為 2026-08-30 定案的驗收條件。

所有測試資料為合成文字,取材自實際盤點出的格式,但不含真實文件內容。
"""

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.corpus.headings import HeadingStyle, detect_style, parse_heading
from selectrag.corpus.normalize import merge_lines, normalize_text

# --------------------------------------------------------------------------
# 正規化(失效模式 3)
# --------------------------------------------------------------------------


def test_normalize_removes_kangxi_radicals() -> None:
    """PDF 抽出的康熙部首必須還原成一般漢字,否則檢索完全比對不到。"""
    assert normalize_text("⽤法⽤量") == "用法用量"
    assert normalize_text("⼀般⾎液") == "一般血液"


def test_normalize_converts_fullwidth_punctuation() -> None:
    assert normalize_text("甲，乙：丙（丁）") == "甲,乙:丙(丁)"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("甲   乙\t\t丙") == "甲 乙 丙"
    assert normalize_text("  前後空白  ") == "前後空白"


def test_normalize_is_idempotent() -> None:
    once = normalize_text("⽤法⽤量，每⽇⼀次")
    assert normalize_text(once) == once


# --------------------------------------------------------------------------
# 行合併(失效模式 1)
# --------------------------------------------------------------------------


def never_heading(_: str) -> bool:
    return False


def official_heading(line: str) -> bool:
    return parse_heading(line, HeadingStyle.OFFICIAL) is not None


def test_merge_stops_at_sentence_ending_punctuation() -> None:
    lines = ["本方案自 115 年 8 月 1 日生效。", "一、收案條件如下"]
    assert merge_lines(lines, official_heading) == [
        "本方案自 115 年 8 月 1 日生效。",
        "一、收案條件如下",
    ]


def test_merge_stops_before_heading_even_without_punctuation() -> None:
    """無標點但下一行是標題 → 不合併(這是 58% 無標點行的主要救援)。"""
    lines = ["得於門診施行", "(一) 應有專任醫師"]
    assert merge_lines(lines, official_heading) == ["得於門診施行", "(一) 應有專任醫師"]


def test_merge_joins_when_no_punctuation_and_next_is_not_heading() -> None:
    lines = ["本保險醫事服務機構申報之藥品", "以收載於本標準者為限。"]
    assert merge_lines(lines, never_heading) == [
        "本保險醫事服務機構申報之藥品以收載於本標準者為限。"
    ]


def test_merge_treats_short_line_as_paragraph_end() -> None:
    """未滿行者結束段落 —— 滿行(句子未完)與其後的短行合併,短行之後斷開。

    手算:三行長度 24/2/17,平均 14.33,短行門檻 8.6。
    行1 滿行且無標點 → 與行2 合併;行2 未滿行 → 段落結束;行3 獨立成段。
    """
    long_a = "這是一個相當長的行用來拉高平均行長作為基準值使用"
    long_b = "另一個相當長的行同樣用來維持平均值"
    result = merge_lines([long_a, "短行", long_b], never_heading)
    assert len(result) == 2, "短行應結束段落,而非自己獨立成段"
    assert result[0] == long_a + "短行"
    assert result[1] == long_b


def test_merge_preserves_all_characters() -> None:
    """合併不得吃掉任何字元(失效模式 4 的第一道防線)。"""
    lines = ["甲乙丙", "丁戊己", "庚辛壬。"]
    merged = merge_lines(lines, never_heading)
    assert "".join(merged).replace(" ", "") == "甲乙丙丁戊己庚辛壬。"


def test_merge_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no lines"):
        merge_lines([], never_heading)


# --------------------------------------------------------------------------
# 標題解析:公文體系(失效模式 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("壹、前言", 1),
        ("一、收案條件", 2),
        ("(一) 適用對象", 3),
        ("1、應檢附文件", 4),
        ("(1) 診斷證明", 5),
        ("A. 基本日常生活功能", 6),
    ],
)
def test_official_heading_levels(line: str, level: int) -> None:
    result = parse_heading(line, HeadingStyle.OFFICIAL)
    assert result is not None
    assert result[0] == level
    assert result[1] == line


def test_official_style_rejects_leaflet_pattern() -> None:
    """仿單式標題不應被公文規則解析 —— 兩套體系必須分開。"""
    assert parse_heading("1.1 有效成分及含量", HeadingStyle.OFFICIAL) is None


# --------------------------------------------------------------------------
# 標題解析:仿單體系
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("1 成分", 1),
        ("1.1 有效成分及含量", 2),
        ("1.2 其他成分", 2),
        ("2.3.1 特殊族群", 3),
    ],
)
def test_leaflet_heading_levels(line: str, level: int) -> None:
    result = parse_heading(line, HeadingStyle.LEAFLET)
    assert result is not None
    assert result[0] == level
    assert result[1] == line


def test_leaflet_style_rejects_official_pattern() -> None:
    assert parse_heading("壹、前言", HeadingStyle.LEAFLET) is None


# --------------------------------------------------------------------------
# 日期陷阱(實測誤判率 1%,但必須擋掉)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["115.08.01 第七版修訂", "102.01.01 修訂第三版", "99/01/01 實施"],
)
@pytest.mark.parametrize("style", [HeadingStyle.OFFICIAL, HeadingStyle.LEAFLET])
def test_dates_are_never_headings(line: str, style: HeadingStyle) -> None:
    assert parse_heading(line, style) is None


def test_non_heading_lines_return_none() -> None:
    for line in ["這是一般內文", "", "   ", "沒有編號的段落文字"]:
        assert parse_heading(line, HeadingStyle.OFFICIAL) is None


# --------------------------------------------------------------------------
# 體系自動判別
# --------------------------------------------------------------------------


def test_detect_official_style() -> None:
    lines = ["壹、前言", "一、目的", "(一) 適用對象", "內文若干"]
    assert detect_style(lines) is HeadingStyle.OFFICIAL


def test_detect_leaflet_style() -> None:
    lines = ["1 成分", "1.1 有效成分及含量", "1.2 其他成分", "內文若干"]
    assert detect_style(lines) is HeadingStyle.LEAFLET


def test_detect_style_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no lines"):
        detect_style([])


# --------------------------------------------------------------------------
# Property-based(Tier 3)
# --------------------------------------------------------------------------

texts = st.text(min_size=0, max_size=50)


@given(text=texts)
def test_property_normalize_output_is_nfkc_stable(text: str) -> None:
    """正規化後的文字再做 NFKC 不會改變 —— 失效模式 3 的核心不變量。"""
    out = normalize_text(text)
    assert unicodedata.normalize("NFKC", out) == out


@given(text=texts)
def test_property_normalize_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


line_strategy = st.text(alphabet="甲乙丙丁戊己庚辛。", min_size=1, max_size=12)


@given(lines=st.lists(line_strategy, min_size=1, max_size=8))
@settings(max_examples=200)
def test_property_merge_preserves_content(lines: list[str]) -> None:
    """合併前後的字元序列(去空白)必須完全相同 —— 失效模式 4。"""
    merged = merge_lines(lines, never_heading)
    assert "".join(merged).replace(" ", "") == "".join(lines).replace(" ", "")


@given(lines=st.lists(line_strategy, min_size=1, max_size=8))
def test_property_merge_never_increases_line_count(lines: list[str]) -> None:
    assert len(merge_lines(lines, never_heading)) <= len(lines)


# --------------------------------------------------------------------------
# 補洞:property-only 變異測試揭露的盲點(kill 歸屬最先失敗的測試,全套件分數會掩蓋不變量自身的盲點)
# --------------------------------------------------------------------------


@given(
    before=st.lists(st.text(alphabet="甲乙丙", min_size=3, max_size=8), min_size=1, max_size=4),
    after=st.lists(st.text(alphabet="丁戊己", min_size=3, max_size=8), min_size=1, max_size=4),
)
@settings(max_examples=200)
def test_property_merge_never_crosses_heading(before: list[str], after: list[str]) -> None:
    """標題兩側的內容永不合併 —— 失效模式 1 的核心不變量。

    補洞:既有的合併 property 全用 never_heading,從未考驗標題邊界
    (property-only 變異測試證實)。
    """
    heading = "一、章節標題"
    merged = merge_lines([*before, heading, *after], official_heading)
    # 標題必須獨立成段:若與前文黏合,標題就不在行首,parse_heading 再也認不出它,
    # 整份文件的階層結構會全毀
    assert heading in merged, f"標題應獨立成段,實際:{merged}"
    for segment in merged:
        assert not (
            any(b in segment for b in before) and any(a in segment for a in after)
        ), "標題前後的內容被合併到同一段"


@given(
    long_len=st.integers(min_value=30, max_value=60),
    short_len=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=150)
def test_property_short_line_ends_paragraph(long_len: int, short_len: int) -> None:
    """未滿行必定結束段落 —— 短行之後的內容不得被併入同一段。

    補洞:無任何不變量表達「短行結束段落」,該規則的變異可存活。
    """
    long_a, short, long_b = "甲" * long_len, "乙" * short_len, "丙" * long_len
    merged = merge_lines([long_a, short, long_b], never_heading)
    assert len(merged) >= 2
    assert not any("乙" in seg and "丙" in seg for seg in merged), "短行之後應斷開"


@given(
    n_official=st.integers(min_value=3, max_value=8),
    n_leaflet=st.integers(min_value=0, max_value=2),
)
def test_property_detect_style_follows_majority(n_official: int, n_leaflet: int) -> None:
    """體系判別必須依命中分布,而非固定回傳某一種。

    補洞:30 份仿單若被判成公文式會全部切壞,但無 property 守著這件事。
    """
    official = [f"{ch}、標題" for ch in "一二三四五六七八九"[:n_official]]
    leaflet = [f"{i}.1 小節" for i in range(1, n_leaflet + 1)]
    assert detect_style([*official, *leaflet]) is HeadingStyle.OFFICIAL
    flipped = [f"{i}.{j} 小節" for i in range(1, n_official + 1) for j in (1, 2)]
    assert detect_style([*flipped, *official[:1]]) is HeadingStyle.LEAFLET
