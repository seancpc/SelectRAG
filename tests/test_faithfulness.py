"""faithfulness judge 純邏輯測試 —— 場景為 2026-09-16 定案的驗收條件。文字皆為中性占位符。"""

from __future__ import annotations

import json
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.evaluation.faithfulness import (
    aggregate,
    cohen_kappa,
    is_abstention,
    judge_prompt,
    parse_verdict,
    perturb_number,
    split_claims,
)


# 1 切句
def test_split_claims_punctuation_and_cleanup() -> None:
    assert split_claims("甲。乙!丙?丁;戊") == ["甲", "乙", "丙", "丁", "戊"]
    assert split_claims("甲。。 。\n乙。") == ["甲", "乙"]
    assert split_claims("") == []
    assert split_claims("甲,乙。") == ["甲,乙"]  # 逗號不切


def test_split_claims_drops_abstention_sentences() -> None:
    assert split_claims("甲。參考資料未提及。") == ["甲"]
    assert split_claims("參考資料未提及。") == []


# 2 拒答
def test_is_abstention_only_when_all_sentences_abstain() -> None:
    assert is_abstention("參考資料未提及。") is True
    assert is_abstention("無法回答") is True
    assert is_abstention("甲。參考資料未提及。") is False
    assert is_abstention("") is False


# 3 verdict 解析
def test_parse_verdict_accepts_valid_and_rejects_invalid() -> None:
    assert parse_verdict('{"verdict": "supported", "evidence_chunk": 2}') == {
        "verdict": "supported",
        "evidence_chunk": 2,
    }
    assert parse_verdict('{"verdict": "unsupported", "evidence_chunk": null}') == {
        "verdict": "unsupported",
        "evidence_chunk": None,
    }
    for bad in (
        '{"verdict": "maybe", "evidence_chunk": 1}',
        '{"verdict": "supported", "evidence_chunk": null}',  # supported 必須有證據
        '{"verdict": "supported", "evidence_chunk": "1"}',
        '{"verdict": "supported"}',
        "not json",
        "[]",
    ):
        with pytest.raises(ValueError):
            parse_verdict(bad)


# 4 彙總
def test_aggregate_hand_computed_excludes_abstained_from_denominator() -> None:
    items = [
        {"verdict": "supported"},
        {"verdict": "supported"},
        {"verdict": "unsupported"},
        {"verdict": "contradicted"},
    ]
    a = aggregate(items, n_abstained_answers=3)
    assert a["n_claims"] == 4 and a["n_abstained_answers"] == 3
    assert a["faithfulness"] == pytest.approx(0.5)
    assert a["unsupported_rate"] == pytest.approx(0.25)
    assert a["contradiction_rate"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        aggregate([], n_abstained_answers=5)
    with pytest.raises(ValueError):
        aggregate([{"verdict": "maybe"}])


# 5 κ
def test_cohen_kappa_hand_computed() -> None:
    assert cohen_kappa(["s", "s", "u"], ["s", "s", "u"]) == 1.0
    assert cohen_kappa(["s", "s", "u", "u"], ["s", "u", "s", "u"]) == pytest.approx(0.0)
    # 手算:a=[s,s,s,u,u,u,c,c] b=[s,s,u,u,u,c,c,c] → 一致 6/8;pe=(3*2+3*3+2*3)/64=21/64
    a = ["s", "s", "s", "u", "u", "u", "c", "c"]
    b = ["s", "s", "u", "u", "u", "c", "c", "c"]
    assert cohen_kappa(a, b) == pytest.approx((6 / 8 - 21 / 64) / (1 - 21 / 64))
    assert cohen_kappa(["s", "s"], ["s", "s"]) == 1.0  # 單一標籤:pe=1,定義為完全一致
    with pytest.raises(ValueError):
        cohen_kappa(["s"], ["s", "u"])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# 6 合成負例
def test_perturb_number_changes_first_number_only() -> None:
    rng = random.Random(0)
    out = perturb_number("每日 250 點,每年 3 次", rng)
    assert out is not None and out != "每日 250 點,每年 3 次"
    assert out.endswith(",每年 3 次") and out.startswith("每日 ")
    assert perturb_number("沒有數字", rng) is None


# 7 prompt 組裝
def test_judge_prompt_structure() -> None:
    msgs = judge_prompt("<問題>", "[1] <段落>", "<主張>")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "supported" in msgs[0]["content"] and "contradicted" in msgs[0]["content"]
    assert "<問題>" in msgs[1]["content"] and "[1] <段落>" in msgs[1]["content"]
    assert "<主張>" in msgs[1]["content"]
    with pytest.raises(ValueError):
        judge_prompt("<問題>", "[1] <段落>", "")


# property 不變量
@settings(max_examples=60, deadline=None)
@given(s=st.text(alphabet="甲乙。!?; \n", max_size=20))
def test_property_split_claims_no_empty_and_order(s: str) -> None:
    out = split_claims(s)
    assert all(c and c == c.strip() for c in out)
    assert all(not any(p in c for p in "。!?;") for c in out)
    # 順序保留:每個主張在原文中的位置遞增
    pos, last = 0, -1
    for c in out:
        i = s.find(c, pos)
        assert i > last
        last, pos = i, i + len(c)


_v = st.sampled_from(["supported", "unsupported", "contradicted"])


@settings(max_examples=60, deadline=None)
@given(vs=st.lists(_v, min_size=1, max_size=30), n_abs=st.integers(0, 5))
def test_property_aggregate_rates_sum_to_one(vs: list[str], n_abs: int) -> None:
    a = aggregate([{"verdict": v} for v in vs], n_abs)
    assert a["faithfulness"] + a["unsupported_rate"] + a["contradiction_rate"] == pytest.approx(1.0)
    assert a["n_claims"] == len(vs)
    assert a["faithfulness"] == pytest.approx(vs.count("supported") / len(vs))


@settings(max_examples=60, deadline=None)
@given(a=st.lists(st.sampled_from("suc"), min_size=1, max_size=20), data=st.data())
def test_property_kappa_bounded_symmetric_and_identity(a: list[str], data: st.DataObject) -> None:
    b = data.draw(st.lists(st.sampled_from("suc"), min_size=len(a), max_size=len(a)))
    k = cohen_kappa(a, b)
    assert -1.0 - 1e-9 <= k <= 1.0 + 1e-9
    assert k == pytest.approx(cohen_kappa(b, a))
    assert cohen_kappa(a, a) == 1.0


@settings(max_examples=60, deadline=None)
@given(
    pre=st.text(alphabet="甲乙 ", max_size=5),
    n=st.integers(0, 9999),
    post=st.text(alphabet="甲乙 ", max_size=5),
    seed=st.integers(0, 1000),
)
def test_property_perturb_changes_only_number(pre: str, n: int, post: str, seed: int) -> None:
    text = f"{pre}{n}{post}"
    out = perturb_number(text, random.Random(seed))
    assert out is not None and out != text
    assert out.startswith(pre) and out.endswith(post)
    mid = out[len(pre) : len(out) - len(post)]
    assert mid.isdigit() and int(mid) != n


def test_parse_verdict_roundtrip_with_json_dumps() -> None:
    d = {"verdict": "contradicted", "evidence_chunk": 3}
    assert parse_verdict(json.dumps(d)) == d


@settings(max_examples=60, deadline=None)
@given(
    s=st.text(alphabet="甲乙。!?; ", max_size=20),
    marker=st.sampled_from(["參考資料未提及", "無法回答"]),
)
def test_property_abstention_sentence_never_becomes_claim(s: str, marker: str) -> None:
    assert split_claims(s + "。" + marker + "。") == split_claims(s)
    assert split_claims(marker + "。" + s) == split_claims(s)


@settings(max_examples=60, deadline=None)
@given(
    v=st.sampled_from(["supported", "unsupported", "contradicted"]),
    ev=st.one_of(st.none(), st.integers(0, 9)),
)
def test_property_parse_verdict_acceptance_set(v: str, ev: int | None) -> None:
    raw = json.dumps({"verdict": v, "evidence_chunk": ev})
    if v == "supported" and ev is None:
        with pytest.raises(ValueError):
            parse_verdict(raw)
    else:
        assert parse_verdict(raw) == {"verdict": v, "evidence_chunk": ev}


@settings(max_examples=80, deadline=None)
@given(a=st.lists(st.sampled_from("suc"), min_size=2, max_size=20), data=st.data())
def test_property_kappa_strictly_below_raw_agreement(a: list[str], data: st.DataObject) -> None:
    b = data.draw(st.lists(st.sampled_from("suc"), min_size=len(a), max_size=len(a)))
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)
    pe = sum((a.count(k) / len(a)) * (b.count(k) / len(a)) for k in set(a) | set(b))
    k = cohen_kappa(a, b)
    if pe == 1.0:
        assert k == 1.0
    elif pe > 0 and po < 1.0:
        assert k < po  # 期望一致項必須把分數往下修
    else:
        assert k == pytest.approx(po if pe == 0 else 1.0)


# 8 逐字命中前置層(2026-09-17 核准)
from selectrag.evaluation.faithfulness import coverage3, split_chunks, verbatim_hit  # noqa: E402


def test_coverage3_hand_computed() -> None:
    # 「甲乙丙丁」3-gram = {甲乙丙, 乙丙丁};context 只含「甲乙丙」→ 1/2
    assert coverage3("甲乙丙丁", "甲乙丙戊") == 0.5
    assert coverage3("甲乙丙丁", "甲乙丙丁") == 1.0
    assert coverage3("甲乙丙丁", "戊己庚辛") == 0.0


def test_coverage3_ignores_whitespace_and_punctuation() -> None:
    assert coverage3("甲乙,丙丁。", "x 甲乙丙丁 y") == 1.0
    assert coverage3("甲乙丙丁", "甲乙、丙(丁)") == 1.0


def test_coverage3_short_or_empty() -> None:
    assert coverage3("甲乙", "甲乙丙丁") == 0.0  # 不足 3 字
    assert coverage3("", "甲乙丙丁") == 0.0
    assert coverage3("甲乙丙丁", "") == 0.0


@given(st.text(min_size=3, max_size=30), st.text(max_size=60))
@settings(max_examples=200)
def test_coverage3_property_range_and_substring(claim: str, context: str) -> None:
    v = coverage3(claim, context)
    assert 0.0 <= v <= 1.0
    if coverage3(claim, claim) == 1.0:  # 正規化後至少 3 字
        assert coverage3(claim, context + claim) == 1.0  # 子字串必為 1.0


def test_split_chunks_formats() -> None:
    assert split_chunks("[1] 甲\n[2] 乙\n\n[3] 丙") == [(1, "甲"), (2, "乙"), (3, "丙")]
    assert split_chunks("[2] 甲\n[5] 乙") == [(2, "甲"), (5, "乙")]  # 不重編
    assert split_chunks("無標記全文") == [(1, "無標記全文")]
    assert split_chunks("[1] 甲\n續行\n[2] 乙") == [(1, "甲\n續行"), (2, "乙")]


def test_verbatim_hit_basic_and_guards() -> None:
    chunks = [(1, "戊己庚辛壬癸"), (2, "規定為每日三次共 250 點甲乙丙")]
    assert verbatim_hit("規定為每日三次共 250 點", chunks, min_coverage=0.9) == 2
    # 數字守門:覆蓋率夠但數字不同 → None
    assert verbatim_hit("規定為每日三次共 251 點", chunks, min_coverage=0.5) is None
    # 短句守門:正規化後不足 6 字 → None
    assert verbatim_hit("庚辛壬", chunks, min_coverage=0.5) is None
    assert verbatim_hit("規定為每日三次共 250 點", [], min_coverage=0.9) is None
    assert verbatim_hit("規定為每日三次共 250 點", chunks, min_coverage=0.99) == 2
    assert verbatim_hit("規定為每日三次共 250 點壬癸壬癸", chunks, min_coverage=0.9) is None


def test_verbatim_hit_threshold_is_inclusive() -> None:
    # 「甲乙丙丁戊己庚辛」6 個 3-gram;chunk「甲乙丙丁戊」含其中 3 個 → 覆蓋率恰為 0.5
    chunk = [(1, "甲乙丙丁戊")]
    assert coverage3("甲乙丙丁戊己庚辛", "甲乙丙丁戊") == 0.5
    assert verbatim_hit("甲乙丙丁戊己庚辛", chunk, min_coverage=0.5) == 1  # 等於門檻要命中
    assert verbatim_hit("甲乙丙丁戊己庚辛", chunk, min_coverage=0.51) is None


def test_verbatim_hit_picks_highest_coverage_then_lowest_index() -> None:
    chunks = [(3, "甲乙丙丁戊己庚辛"), (1, "甲乙丙丁戊己庚辛"), (2, "甲乙丙丁戊己")]
    assert verbatim_hit("甲乙丙丁戊己庚辛", chunks, min_coverage=0.5) == 1
    assert (
        verbatim_hit(
            "甲乙丙丁戊己庚辛", [(7, "甲乙丙丁戊己"), (9, "甲乙丙丁戊己庚辛")], min_coverage=0.5
        )
        == 9
    )


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01])
def test_verbatim_hit_rejects_bad_threshold(bad: float) -> None:
    with pytest.raises(ValueError):
        verbatim_hit("甲乙丙丁戊己庚", [(1, "甲乙丙丁戊己庚")], min_coverage=bad)


@given(
    st.lists(
        st.tuples(st.integers(1, 9), st.text(min_size=6, max_size=30)),
        min_size=1,
        max_size=5,
        unique_by=lambda t: t[0],
    )
)
@settings(max_examples=100)
def test_verbatim_hit_property_order_invariant(chunks: list[tuple[int, str]]) -> None:
    claim = chunks[0][1]
    a = verbatim_hit(claim, chunks, min_coverage=0.5)
    b = verbatim_hit(claim, list(reversed(chunks)), min_coverage=0.5)
    assert a == b


_CJK = st.text(
    alphabet=st.characters(min_codepoint=0x4E00, max_codepoint=0x4EFF), min_size=6, max_size=20
)


@given(_CJK)
@settings(max_examples=100)
def test_verbatim_hit_property_exact_self_hit_at_threshold_one(claim: str) -> None:
    # 主張本身為 chunk → 覆蓋率恰 1.0,門檻 1.0(含等於)必命中
    assert verbatim_hit(claim, [(4, claim)], min_coverage=1.0) == 4


@given(_CJK, st.integers(0, 999))
@settings(max_examples=100)
def test_verbatim_hit_property_number_guard(prefix: str, n: int) -> None:
    claim = f"{prefix}{n}"
    chunk = [(1, claim)]
    assert verbatim_hit(claim, chunk, min_coverage=0.5) == 1
    # 只改數字(其餘逐字相同)→ 必不命中
    assert verbatim_hit(f"{prefix}{n + 1}", chunk, min_coverage=0.5) is None


@given(st.text(alphabet=st.characters(min_codepoint=0x4E00, max_codepoint=0x4EFF), max_size=5))
@settings(max_examples=100)
def test_verbatim_hit_property_short_claim_never_hits(claim: str) -> None:
    assert verbatim_hit(claim, [(1, claim + "甲乙丙丁戊己")], min_coverage=0.1) is None


# 9 拒答標記簡體正規化(2026-09-17 核准,變更 1)
from selectrag.evaluation.faithfulness import normalize_variants  # noqa: E402


def test_normalize_variants_hand_computed() -> None:
    assert normalize_variants("参考资料未提及") == "參考資料未提及"
    assert normalize_variants("无法回答") == "無法回答"
    assert normalize_variants("參考資料未提及") == "參考資料未提及"  # 已是繁體不變
    assert normalize_variants("abc 甲乙") == "abc 甲乙"  # 非對映字不變


@given(st.text(max_size=60))
@settings(max_examples=200)
def test_normalize_variants_property_length_preserved(text: str) -> None:
    out = normalize_variants(text)
    assert len(out) == len(text)
    assert normalize_variants(out) == out  # 冪等


def test_abstention_accepts_simplified_variants() -> None:
    assert is_abstention("参考资料未提及。") is True
    assert is_abstention("参考資料未提及。") is True
    assert is_abstention("无法回答") is True
    assert split_claims("甲。参考資料未提及其他規定。") == ["甲"]


def test_abstention_keeps_positional_variants_as_claims() -> None:
    # 「段落中」「資料中」變體仍是主張(刻意不擴大標記),錯誤棄答才會被 judge 判到
    assert is_abstention("參考段落中未提及含量") is False
    assert split_claims("參考資料中未提及有關規定。") == ["參考資料中未提及有關規定"]


_ABST_PIECES = [
    "参考资料未提及",
    "参考資料未提及",
    "參考資料未提及",
    "无法回答",
    "無法回答",
    "甲乙",
    "参考丙",
]


@given(st.lists(st.sampled_from(_ABST_PIECES), max_size=5))
@settings(max_examples=200)
def test_abstention_property_invariant_under_variants(pieces: list[str]) -> None:
    # 拒答判定對簡體變體不變:正規化前後結論相同,且切句結果相同
    text = "。".join(pieces)
    assert is_abstention(text) == is_abstention(normalize_variants(text))
    assert [normalize_variants(c) for c in split_claims(text)] == split_claims(
        normalize_variants(text)
    )


# 10 prompt v3(2026-09-18 核准):雙向收斂四條規則
from selectrag.evaluation.faithfulness import PROMPT_VERSION, SYSTEM_RULES  # noqa: E402


def test_prompt_is_v2_after_v3_rollback() -> None:
    # v3 四條規則零槓桿、B1 拆問前置實驗 3/8,2026-09-18 退回 v2(DECISION_MEMO §6.5)
    assert PROMPT_VERSION == "v2"
    for kw in ("同時規定", "對象", "逐一核對"):  # v2 的三條結構規則
        assert kw in SYSTEM_RULES, kw
    for kw in ("措辭", "列舉"):  # v3 規則不得殘留
        assert kw not in SYSTEM_RULES, kw


def test_prompt_rules_have_no_domain_examples() -> None:
    # Prompt 設計規範:規則只含結構,不含領域字串或具體數值
    import re

    assert not re.search(r"\d", SYSTEM_RULES)
    # 類別詞(藥名、代碼)是結構;具體藥名、單位縮寫、制度名才是領域內容
    for banned in ("mg", "ml", "健保", "Letrozole", "Allopurinol", "Tetracycline"):
        assert banned not in SYSTEM_RULES, banned
