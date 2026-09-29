"""B 軸效能量測的純邏輯(2026-09-15 核准)。

SSE 解析、TTFT / 延遲 / token 計數、百分位、一致性比對、prompt 組裝。

設計決策:
  TTFT  — 首個「非空 content」抵達時刻減請求送出時刻;只帶 role 的首 chunk 不算(vLLM 會先送 role)
  tokens — 優先取串流末尾 usage.completion_tokens;缺 usage 時退回「非空 content chunk 數」並標記來源
  百分位 — nearest-rank(排序後第 ceil(p/100·n) 個),不插值,回傳值必在樣本中
  一致性 — 兩引擎 greedy 輸出逐題比對;分歧位置以字元計(客戶端不載 tokenizer),另記長度差
  壞資料 — 無 content 的請求、空樣本、id 集合不一致、模板缺槽位,一律 raise
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class RequestMetrics:
    id: str
    ttft_s: float
    latency_s: float
    output_tokens: int
    tokens_source: str  # "usage" | "chunks"
    text: str


def parse_sse_line(line: str) -> dict[str, Any] | str | None:
    """`data: {...}` → dict;`data: [DONE]` → "DONE";空行 / 註解行 → None;非 JSON → raise。"""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "DONE"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"bad SSE payload: {payload[:60]!r}") from e
    if not isinstance(parsed, dict):
        raise ValueError("SSE payload is not an object")
    return parsed


def delta_content(chunk: Mapping[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return str(delta.get("content") or "")


class StreamCollector:
    """累積一個串流請求的事件,產出 RequestMetrics。"""

    def __init__(self, request_id: str, t_start: float) -> None:
        self._id = request_id
        self._t0 = t_start
        self._ttft: float | None = None
        self._parts: list[str] = []
        self._chunks = 0
        self._usage_tokens: int | None = None

    def feed(self, t: float, chunk: Mapping[str, Any]) -> None:
        content = delta_content(chunk)
        if content:
            if self._ttft is None:
                self._ttft = t - self._t0
            self._parts.append(content)
            self._chunks += 1
        usage = chunk.get("usage")
        if usage and usage.get("completion_tokens") is not None:
            self._usage_tokens = int(usage["completion_tokens"])

    def finish(self, t_end: float) -> RequestMetrics:
        if self._ttft is None:
            raise ValueError(f"request {self._id!r} produced no content")
        if self._usage_tokens is not None:
            tokens, source = self._usage_tokens, "usage"
        else:
            tokens, source = self._chunks, "chunks"
        return RequestMetrics(
            self._id, self._ttft, t_end - self._t0, tokens, source, "".join(self._parts)
        )


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(s)))
    return s[rank - 1]


def summarize(metrics: Sequence[RequestMetrics], wall_s: float) -> dict[str, Any]:
    if not metrics:
        raise ValueError("no metrics to summarize")
    if wall_s <= 0:
        raise ValueError("wall_s must be > 0")
    ttft = [m.ttft_s for m in metrics]
    lat = [m.latency_s for m in metrics]
    tokens_total = sum(m.output_tokens for m in metrics)
    sources = sorted({m.tokens_source for m in metrics})
    return {
        "n": len(metrics),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "ttft_mean_s": fmean(ttft),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "latency_mean_s": fmean(lat),
        "output_tokens_total": tokens_total,
        "output_tokens_mean": tokens_total / len(metrics),
        "tok_per_s": tokens_total / wall_s,
        "req_per_s": len(metrics) / wall_s,
        "wall_s": wall_s,
        "tokens_source": sources[0] if len(sources) == 1 else "mixed",
    }


def compare_outputs(a: Mapping[str, str], b: Mapping[str, str]) -> dict[str, Any]:
    if set(a) != set(b):
        raise ValueError("output id sets differ")
    divergence: dict[str, int | None] = {}
    len_diff: dict[str, int] = {}
    for k in sorted(a):
        x, y = a[k], b[k]
        if x == y:
            divergence[k] = None
        else:
            i = 0
            while i < min(len(x), len(y)) and x[i] == y[i]:
                i += 1
            divergence[k] = i
        len_diff[k] = abs(len(x) - len(y))
    exact = sum(1 for v in divergence.values() if v is None)
    return {
        "n": len(a),
        "exact_match": exact,
        "exact_rate": exact / len(a) if a else 0.0,
        "first_divergence_char": divergence,
        "len_diff": len_diff,
    }


def build_messages(
    system: str, template: str, question: str, chunks: Sequence[str]
) -> list[dict[str, str]]:
    if "{question}" not in template or "{context}" not in template:
        raise ValueError("template must contain {question} and {context}")
    context = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1))
    user = template.replace("{question}", question).replace("{context}", context)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
