"""src/selectrag/evaluation/terms.py:文字皆為中性占位符或公開藥名(非病歷資料)。"""

from __future__ import annotations

import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from selectrag.evaluation.terms import foreign_terms, latin_terms, missing_verbatim


def test_latin_terms_needs_three_letters() -> None:
    # mg / IU / 單字母編號是單位與標號,不是專有名詞
    assert latin_terms("劑量 500mg,IU 單位,A 款,見 ALK 檢測") == ("ALK",)


def test_latin_terms_dedups_and_keeps_order() -> None:
    assert latin_terms("aspirin 與 Aspirin 與 tablet") == ("aspirin", "tablet")


def test_foreign_terms_treats_question_as_legitimate_source() -> None:
    # v14 掃描實證:句中有、段落沒有的英文詞全部來自題目;不把題目列為來源會全列誤報
    got = foreign_terms("每次給予 Metronidazole 500mg", ["<段落無此詞>"], "Metronidazole 的用法?")
    assert got == ()


def test_foreign_terms_flags_term_in_neither_source() -> None:
    got = foreign_terms("使用 Foobarin 治療", ["<段落無此詞>"], "<問題無此詞>")
    assert got == ("Foobarin",)


def test_foreign_terms_is_case_insensitive() -> None:
    assert foreign_terms("使用 methotrexate", ["…Methotrexate 25mg…"], "") == ()


def test_foreign_terms_respects_letter_boundary() -> None:
    # ALK 不得被段落裡的 ALKBH5 當成有據
    assert foreign_terms("ALK 陽性", ["…ALKBH5…"], "") == ("ALK",)
    assert foreign_terms("ALK 陽性", ["…ALK 陽性…"], "") == ()


def test_foreign_terms_empty_on_pure_chinese() -> None:
    assert foreign_terms("<純中文句>", ["<純中文段落>"], "") == ()


def test_missing_verbatim_flags_translated_drug_name() -> None:
    # r69 的形態:段落有 methotrexate,句子寫成不存在的中文藥名
    assert missing_verbatim("使用甲基喋吟治療", ["…methotrexate 25mg…"]) == ("methotrexate",)


def test_missing_verbatim_empty_when_copied() -> None:
    assert missing_verbatim("使用 methotrexate", ["…methotrexate…"]) == ()


def test_missing_verbatim_empty_without_segments() -> None:
    assert missing_verbatim("<任意句>", []) == ()


def test_missing_verbatim_known_false_positive_shape() -> None:
    # 已知限制:句子只答段落一部分時必然誤報,由人工過濾;不得當指標
    assert missing_verbatim("每次 2 粒", ["…aspirin…tablet…"]) == ("aspirin", "tablet")


@given(st.text())
def test_property_latin_terms_all_results_are_substrings(s: str) -> None:
    # 基準是 NFKC 後的輸入:latin_terms 對 NFKC 後的文字抽詞,所以相容字元
    # (如 U+FB01 ﬁ → fi)展開後的詞不會是原字串的子字串。仍鎖住「回傳的詞
    # 必須真的出現在輸入裡」,只是把比較基準寫對(2026-09-22 規格澄清)。
    folded = unicodedata.normalize("NFKC", s)
    for t in latin_terms(s):
        assert t in folded
        assert len(t) >= 3


@given(st.text(), st.text())
def test_property_foreign_terms_is_subset_of_sentence_terms(sentence: str, seg: str) -> None:
    got = foreign_terms(sentence, [seg], "")
    assert set(t.casefold() for t in got) <= set(t.casefold() for t in latin_terms(sentence))


@given(st.text())
def test_property_foreign_terms_empty_when_sentence_is_its_own_segment(s: str) -> None:
    # 句子本身就是所引段落 → 沒有任何詞是憑空出現的
    assert foreign_terms(s, [s], "") == ()


@given(st.text())
def test_property_missing_verbatim_empty_when_sentence_is_its_own_segment(s: str) -> None:
    assert missing_verbatim(s, [s]) == ()


def test_missing_verbatim_survives_whitespace_like_control_chars() -> None:
    """property 反例固化為手算測試:`AAA\\x85A` 曾被判成「段落沒照抄 AAA」。

    機制:`"\\x85".isspace()` 為 True,所以去空白的正規化(`components.normalize_glyphs`)
    會把 `AAA\\x85A` 黏成 `AAAA`,字母邊界於是在每個出現位置都看到相鄰字母而回 False。
    抽詞端保留分隔、比對端去掉分隔,兩端不一致就會產生這種誤報。
    property 負責發現,但要靠隨機湊出這個序列,換 seed 就漏——必須有手算測試守著。
    """
    assert missing_verbatim("AAA\x85A", ["AAA\x85A"]) == ()
    # 同型:NEL 之外的其他空白類控制字元
    assert missing_verbatim("BBB\x1cB", ["BBB\x1cB"]) == ()
    # 反向:段落真的沒有該詞時仍要標出(不是把守衛做成恆回空)
    assert missing_verbatim("<純中文>", ["CCC\x85C"]) == ("CCC",)


def test_terms_survive_casefold_expanding_characters() -> None:
    """property 反例固化為手算測試:`AAAß` 曾被判成「憑空出現的 AAA」。

    機制:`"ß".casefold()` 展開成 `"ss"`,所以 casefold 後的段落是 `aaass`——
    原本在 `AAA` 之後的非字母邊界(ß)變成了字母 `s`,字母邊界判斷因此失效。
    這類字元(ß、土耳其 İ)不被 NFKC 轉換,只在 casefold 時才擴張,
    所以「先抽詞、再到 casefold 空間做子字串搜尋」這個架構本身對齊不了兩個空間。
    """
    assert foreign_terms("AAAß", ["AAAß"], "") == ()
    assert missing_verbatim("AAAß", ["AAAß"]) == ()
    # 反向:真的不同的詞仍要標出(守衛不得做成恆回空)
    assert foreign_terms("AAAß", ["<段落無此詞>"], "") == ("AAA",)


@given(st.text())
def test_property_foreign_terms_empty_when_question_is_the_sentence(s: str) -> None:
    """題目是合法來源:句子逐字等於題目時,沒有任何詞是憑空出現的。

    補這條的理由(2026-09-22 第 4b 層):既有 property 的 question 一律傳 "",
    所以「題目算來源」這個核心語意完全沒有不變量守著——把 question 從來源移除
    的變異在 property 子集存活。這一欄若丟掉題目來源會整欄誤報(答句重述題目
    藥名即被標),是最該被鎖住的行為。
    """
    assert foreign_terms(s, [], s) == ()
