"""B 軸效能量測純邏輯測試 —— 場景為 2026-09-15 定案的驗收條件。文字皆為中性占位符。"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.perf.bench import (
    RequestMetrics,
    StreamCollector,
    build_messages,
    compare_outputs,
    delta_content,
    parse_sse_line,
    percentile,
    summarize,
)


def _chunk(content: str | None, usage: dict[str, int] | None = None) -> dict[str, object]:
    delta: dict[str, str] = {} if content is None else {"content": content}
    d: dict[str, object] = {"choices": [{"delta": delta}]}
    if usage is not None:
        d["usage"] = usage
    return d


# 1 SSE 解析
def test_parse_sse_line_variants() -> None:
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert parse_sse_line("data: [DONE]") == "DONE"
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None
    with pytest.raises(ValueError):
        parse_sse_line("data: not json")


def test_delta_content_handles_role_only_and_empty_choices() -> None:
    assert delta_content(_chunk("<甲>")) == "<甲>"
    assert delta_content({"choices": [{"delta": {"role": "assistant"}}]}) == ""
    assert delta_content({"choices": []}) == ""
    assert delta_content({}) == ""


# 2 TTFT / 延遲 / token 數
def test_collector_ttft_at_first_nonempty_content_and_usage_tokens() -> None:
    c = StreamCollector("q1", t_start=10.0)
    c.feed(10.1, {"choices": [{"delta": {"role": "assistant"}}]})  # 首 chunk 只有 role,不算 TTFT
    c.feed(10.3, _chunk("<甲>"))
    c.feed(10.5, _chunk("<乙>"))
    c.feed(10.6, _chunk(None, usage={"completion_tokens": 7}))
    m = c.finish(10.9)
    assert (m.id, m.output_tokens, m.tokens_source, m.text) == ("q1", 7, "usage", "<甲><乙>")
    assert m.ttft_s == pytest.approx(0.3) and m.latency_s == pytest.approx(0.9)


def test_collector_falls_back_to_chunk_count_without_usage() -> None:
    c = StreamCollector("q2", t_start=0.0)
    c.feed(0.2, _chunk("<甲>"))
    c.feed(0.4, _chunk(""))
    c.feed(0.5, _chunk("<乙>"))
    m = c.finish(1.0)
    assert (m.output_tokens, m.tokens_source, m.ttft_s) == (2, "chunks", 0.2)


def test_collector_no_content_raises_on_finish() -> None:
    c = StreamCollector("q3", t_start=0.0)
    with pytest.raises(ValueError, match="q3"):
        c.finish(1.0)


# 3 百分位與摘要
def test_percentile_nearest_rank_and_errors() -> None:
    v = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(v, 50) == 3.0
    assert percentile(v, 95) == 5.0
    assert percentile(v, 0) == 1.0
    assert percentile(v, 100) == 5.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile(v, 101)


def test_summarize_hand_computed() -> None:
    ms = [
        RequestMetrics("a", 0.1, 1.0, 10, "usage", "x"),
        RequestMetrics("b", 0.3, 2.0, 30, "usage", "y"),
    ]
    s = summarize(ms, wall_s=4.0)
    assert s["n"] == 2
    assert s["ttft_p50_s"] == 0.1 and s["ttft_p95_s"] == 0.3
    assert s["latency_p50_s"] == 1.0 and s["latency_p95_s"] == 2.0
    assert s["output_tokens_total"] == 40
    assert s["tok_per_s"] == pytest.approx(10.0)
    assert s["req_per_s"] == pytest.approx(0.5)
    assert s["tokens_source"] == "usage"
    with pytest.raises(ValueError):
        summarize([], 1.0)


# 4 一致性比對
def test_compare_outputs_hand_computed() -> None:
    a = {"q1": "<甲乙丙>", "q2": "<甲乙丙>", "q3": "<甲>"}
    b = {"q1": "<甲乙丙>", "q2": "<甲乙丁>", "q3": "<甲乙>"}
    r = compare_outputs(a, b)
    assert r["n"] == 3 and r["exact_match"] == 1
    assert r["exact_rate"] == pytest.approx(1 / 3)
    assert r["first_divergence_char"] == {"q1": None, "q2": 3, "q3": 2}  # "<甲" 共同,第 2 字元分歧
    assert r["len_diff"] == {"q1": 0, "q2": 0, "q3": 1}
    with pytest.raises(ValueError):
        compare_outputs({"q1": "x"}, {"q2": "x"})


# 5 prompt 組裝
def test_build_messages_fills_slots_and_numbers_chunks() -> None:
    tpl = "問:{question}\n參考:\n{context}"
    msgs = build_messages("<系統>", tpl, "<問題>", ["<段一>", "<段二>"])
    assert msgs[0] == {"role": "system", "content": "<系統>"}
    assert msgs[1]["role"] == "user"
    assert "[1] <段一>" in msgs[1]["content"] and "[2] <段二>" in msgs[1]["content"]
    assert msgs[1]["content"].startswith("問:<問題>")
    with pytest.raises(ValueError):
        build_messages("<系統>", "沒有槽位", "<問題>", ["<段>"])


# property 不變量
_vals = st.lists(st.floats(0, 100, allow_nan=False), min_size=1, max_size=30)


@settings(max_examples=60, deadline=None)
@given(v=_vals, p=st.floats(0, 100))
def test_property_percentile_within_range_and_monotone(v: list[float], p: float) -> None:
    x = percentile(v, p)
    assert min(v) <= x <= max(v)
    assert x in v
    assert percentile(v, 0) <= x <= percentile(v, 100)


_text = st.text(alphabet="甲乙丙", max_size=6)


@settings(max_examples=60, deadline=None)
@given(a=_text, b=_text)
def test_property_compare_divergence_semantics(a: str, b: str) -> None:
    r = compare_outputs({"q": a}, {"q": b})
    d = r["first_divergence_char"]["q"]
    if a == b:
        assert d is None and r["exact_match"] == 1
    else:
        assert d is not None and a[:d] == b[:d] and (a[d : d + 1] != b[d : d + 1])
    assert r["len_diff"]["q"] == abs(len(a) - len(b))


def test_parse_sse_ignores_event_lines_and_rejects_non_object() -> None:
    assert parse_sse_line("event: ping") is None
    with pytest.raises(ValueError):
        parse_sse_line("data: [1, 2]")


def test_summarize_rejects_nonpositive_wall() -> None:
    with pytest.raises(ValueError):
        summarize([RequestMetrics("a", 0.1, 1.0, 1, "usage", "x")], wall_s=0.0)


@settings(max_examples=60, deadline=None)
@given(v=_vals, p=st.floats(0.01, 100))
def test_property_percentile_nearest_rank_definition(v: list[float], p: float) -> None:
    # nearest-rank:x 是最小的樣本值,使「≤ x 的樣本比例」達到 p%;「< x 的比例」則未達
    x = percentile(v, p)
    n = len(v)
    assert sum(1 for y in v if y <= x) / n >= p / 100 - 1e-12
    assert sum(1 for y in v if y < x) / n < p / 100


@settings(max_examples=60, deadline=None)
@given(
    t_role=st.floats(0.001, 5.0),
    gap=st.floats(0.001, 5.0),
    tail=st.floats(0.0, 5.0),
    n=st.integers(1, 6),
)
def test_property_ttft_ignores_role_only_chunk(
    t_role: float, gap: float, tail: float, n: int
) -> None:
    c = StreamCollector("q", t_start=100.0)
    c.feed(100.0 + t_role, {"choices": [{"delta": {"role": "assistant"}}]})
    t_first = 100.0 + t_role + gap
    for i in range(n):
        c.feed(t_first + i * 0.01, _chunk("<x>"))
    m = c.finish(t_first + n * 0.01 + tail)
    assert m.ttft_s == pytest.approx(t_role + gap)
    assert m.latency_s >= m.ttft_s and m.output_tokens == n


@settings(max_examples=60, deadline=None)
@given(
    toks=st.lists(st.integers(1, 500), min_size=1, max_size=10),
    lat=st.lists(st.floats(0.01, 50.0), min_size=10, max_size=10),
    wall=st.floats(0.01, 100.0),
)
def test_property_throughput_uses_wall_clock_not_latency_sum(
    toks: list[int], lat: list[float], wall: float
) -> None:
    ms = [RequestMetrics(f"q{i}", 0.01, lat[i], t, "usage", "x") for i, t in enumerate(toks)]
    s = summarize(ms, wall)
    assert s["tok_per_s"] == pytest.approx(sum(toks) / wall)
    assert s["req_per_s"] == pytest.approx(len(toks) / wall)
