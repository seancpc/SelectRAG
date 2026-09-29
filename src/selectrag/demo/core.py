"""demo 純邏輯:設定載入、檢索快取、真值標色、引用檢查、串流回答。不 import gradio。

設計決策(2026-09-18):
  presets — 名稱 → 既有 runner 設定檔;retriever 依 preset 名快取,同名只建一次
  標色   — in_truth 只看 eval set 真值;only_this_side 看另一欄有沒有同一 chunk
  引用   — 回答中的 [n] 逐一對 1..n_chunks 核對,越界另列
  串流   — 端點錯誤以一則「[錯誤] …」文字回報,不 raise(單題失敗不能讓 demo 崩潰)
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from selectrag.evaluation.eval_set import EvalSet
from selectrag.evaluation.faithfulness import coverage3
from selectrag.perf.bench import build_messages, delta_content, parse_sse_line
from selectrag.retrieval.base import Retriever

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_CITE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class LLMConfig:
    url: str
    model: str
    max_tokens: int
    seed: int
    timeout: float


@dataclass(frozen=True)
class PromptConfig:
    top_k: int
    system: str
    template: str
    prompt_id: str = ""  # UI 版本標頭用;gauntlet 的合成 yaml 沒有此鍵,故預設空


@dataclass(frozen=True)
class DemoConfig:
    presets: Mapping[str, Path]
    chunks: Path
    eval_set: Path
    prompt: PromptConfig
    llm: LLMConfig
    abstain: Mapping[str, float] = field(default_factory=dict)  # preset → min_score


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    rank: int
    doc_title: str
    heading: str
    text: str
    score: float | None = None  # 檢索器有露出分數才有(拒答門檻用)
    source_type: str = ""  # pdf / csv / html
    source_path: str = ""
    page: int | None = None  # 僅 pdf 有
    span: tuple[int, int] | None = None  # pdf/html 為字元位移;csv 為**列索引**(見 locator_text)


@dataclass(frozen=True)
class Mark:
    chunk_id: str
    rank: int
    in_truth: bool
    only_this_side: bool


def load_demo_config(path: Path) -> DemoConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("presets", "chunks", "eval_set", "prompt", "llm"):
        if key not in raw:
            raise ValueError(f"demo config missing '{key}'")
    presets = {str(k): Path(v) for k, v in dict(raw["presets"]).items()}
    if len(presets) < 2:
        raise ValueError("demo config needs at least 2 presets")
    p, llm = raw["prompt"], raw["llm"]
    for key in ("top_k", "system", "template"):
        if key not in p:
            raise ValueError(f"prompt section missing '{key}'")
    for key in ("url", "model", "max_tokens", "seed"):
        if key not in llm:
            raise ValueError(f"llm section missing '{key}'")
    abstain = {str(k): float(v) for k, v in dict(raw.get("abstain") or {}).items()}
    unknown = set(abstain) - set(presets)
    if unknown:
        raise ValueError(f"abstain refers to unknown preset(s): {sorted(unknown)}")
    return DemoConfig(
        presets=presets,
        chunks=Path(raw["chunks"]),
        eval_set=Path(raw["eval_set"]),
        prompt=PromptConfig(
            int(p["top_k"]), str(p["system"]), str(p["template"]), str(p.get("prompt_id", ""))
        ),
        llm=LLMConfig(
            str(llm["url"]),
            str(llm["model"]),
            int(llm["max_tokens"]),
            int(llm["seed"]),
            float(llm.get("timeout", 120)),
        ),
        abstain=abstain,
    )


class RetrieverPool:
    """依 preset 名快取 retriever;builder(preset 名) 只在第一次呼叫時執行。"""

    def __init__(
        self,
        cfg: DemoConfig,
        rows: Mapping[str, Mapping[str, Any]],
        builder: Callable[[str], Retriever],
    ) -> None:
        self._cfg = cfg
        self._rows = rows
        self._builder = builder
        self._cache: dict[str, Retriever] = {}

    def hits_timed(self, preset: str, question: str, top_k: int) -> tuple[list[Hit], float]:
        """hits 加上檢索耗時(毫秒;首次建置先做完,不計入)。未知 preset 由 hits 拋 KeyError。"""
        if preset in self._cfg.presets and preset not in self._cache:
            self._cache[preset] = self._builder(preset)
        t0 = time.perf_counter()
        out = self.hits(preset, question, top_k)
        return out, (time.perf_counter() - t0) * 1000.0

    def hits(self, preset: str, question: str, top_k: int) -> list[Hit]:
        if preset not in self._cfg.presets:
            raise KeyError(preset)
        if preset not in self._cache:
            self._cache[preset] = self._builder(preset)
        retriever = self._cache[preset]
        scored_fn = getattr(retriever, "retrieve_scored", None)
        if scored_fn is not None:
            scored: list[tuple[str, float | None]] = [
                (c, float(s)) for c, s in scored_fn(question, top_k)
            ]
        else:
            scored = [(c, None) for c in retriever.retrieve(question, top_k)]
        out = []
        for rank, (cid, score) in enumerate(scored, start=1):
            row = self._rows[cid]
            raw_span = row.get("char_span")
            span = (int(raw_span[0]), int(raw_span[1])) if raw_span else None
            raw_page = row.get("page")
            out.append(
                Hit(
                    chunk_id=cid,
                    rank=rank,
                    doc_title=str(row.get("doc_title") or ""),
                    heading=str(row.get("heading_path") or ""),
                    text=str(row["text"]),
                    score=score,
                    source_type=str(row.get("source_type") or ""),
                    source_path=str(row.get("source_path") or ""),
                    page=int(raw_page) if raw_page is not None else None,
                    span=span,
                )
            )
        return out


def mark_truth(
    left: Sequence[Hit], right: Sequence[Hit], truth_ids: Iterable[str]
) -> tuple[list[Mark], list[Mark]]:
    truth = set(truth_ids)
    left_ids = {h.chunk_id for h in left}
    right_ids = {h.chunk_id for h in right}

    def marks(hits: Sequence[Hit], other: set[str]) -> list[Mark]:
        return [
            Mark(h.chunk_id, h.rank, h.chunk_id in truth, h.chunk_id not in other) for h in hits
        ]

    return marks(left, right_ids), marks(right, left_ids)


def cited_indices(answer: str, n_chunks: int) -> tuple[set[int], set[int]]:
    found = {int(m) for m in _CITE.findall(answer)}
    valid = {i for i in found if 1 <= i <= n_chunks}
    return valid, found - valid


@dataclass(frozen=True)
class CiteStats:
    valid: frozenset[int]
    invalid: frozenset[int]
    has_truth: bool
    cites_truth: bool
    sentences: int
    uncited_sentences: int


_SENT_END = re.compile(r"[。!?]")
_TRAILING_CITE = re.compile(r"([。!?])((?:\s*\[\d+\])+)")
ABSTAIN = "參考資料未提及"


def _move_trailing_cites(answer: str) -> tuple[str, list[int]]:
    """把「…。[3]」的引用標記搬到終止符前,並回傳每個搬移後字元對應原字串的位移。

    搬移只是重排字元(終止符與其後的引用標記互換),故位移表一對一;
    有了它,句首位置才能映回**原字串**(標記要加在使用者看到的原文上)。
    """
    chars: list[str] = []
    origin: list[int] = []
    i = 0
    while i < len(answer):
        m = _TRAILING_CITE.match(answer, i)
        if m:
            for j, ch in enumerate(m.group(2)):
                chars.append(ch)
                origin.append(m.start(2) + j)
            chars.append(m.group(1))
            origin.append(i)
            i = m.end()
        else:
            chars.append(answer[i])
            origin.append(i)
            i += 1
    return "".join(chars), origin


def _split_sentence_spans(answer: str) -> list[tuple[int, str]]:
    """回 [(句首在原字串的位移, 去空白的句子文字)];全檔分段的唯一來源。"""
    moved, origin = _move_trailing_cites(answer)
    out: list[tuple[int, str]] = []
    pos = 0
    spans = [(m.start(), m.end()) for m in _SENT_END.finditer(moved)]
    for term_start, term_end in [*spans, (len(moved), len(moved))]:
        seg = moved[pos:term_start]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            out.append((origin[pos + lead], seg.strip()))
        pos = term_end
    return out


def _split_sentences(answer: str) -> list[str]:
    """以 。!? 切句,去空白;「…。[3]」句號後的引用標記歸前一句。"""
    return [text for _, text in _split_sentence_spans(answer)]


def _strip_cites(text: str) -> str:
    """剝到穩定:「[[1]1]」剝一次會再拼出 [1]。"""
    while True:
        stripped = _CITE.sub("", text)
        if stripped == text:
            return text.strip()
        text = stripped


def cite_stats(answer: str, marks: Sequence[Mark]) -> CiteStats:
    """單一回答的引用統計:valid/invalid 沿用 cited_indices;句子以 。!? 切;
    「參考資料未提及」句是規則允許的不標引用,不計入 uncited。"""
    valid, invalid = cited_indices(answer, len(marks))
    truth_ranks = {m.rank for m in marks if m.in_truth}
    sentences = _split_sentences(answer)
    uncited = sum(1 for x in sentences if not _CITE.search(x) and ABSTAIN not in x)
    return CiteStats(
        valid=frozenset(valid),
        invalid=frozenset(invalid),
        has_truth=bool(truth_ranks),
        cites_truth=bool(valid & truth_ranks),
        sentences=len(sentences),
        uncited_sentences=uncited,
    )


def summarize_cites(stats: Sequence[CiteStats]) -> dict[str, float | int]:
    """整批回答的引用比例;分母為 0 時該比例回 0.0。"""
    n = len(stats)
    with_valid = [s for s in stats if s.valid]
    truth_pool = [s for s in with_valid if s.has_truth]
    n_sent = sum(s.sentences for s in stats)
    n_uncited = sum(s.uncited_sentences for s in stats)
    truth_hits = sum(1 for s in truth_pool if s.cites_truth)
    return {
        "n": n,
        "cite_rate": len(with_valid) / n if n else 0.0,
        "invalid_rate": sum(1 for s in stats if s.invalid) / n if n else 0.0,
        "truth_cite_n": len(truth_pool),
        "truth_cite_rate": truth_hits / len(truth_pool) if truth_pool else 0.0,
        "uncited_sentence_rate": n_uncited / n_sent if n_sent else 0.0,
    }


def make_messages(prompt: PromptConfig, question: str, hits: Sequence[Hit]) -> list[dict[str, str]]:
    return build_messages(prompt.system, prompt.template, question, [h.text for h in hits])


def stream_answer(llm: LLMConfig, messages: Sequence[Mapping[str, str]]) -> Iterator[str]:
    body = json.dumps(
        {
            "model": llm.model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": llm.max_tokens,
            "temperature": 0.0,
            "seed": llm.seed,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        llm.url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text = ""
    try:
        with _OPENER.open(req, timeout=llm.timeout) as resp:
            for raw in resp:
                ev = parse_sse_line(raw.decode("utf-8"))
                if ev is None:
                    continue
                if ev == "DONE":
                    break
                piece = delta_content(ev)  # type: ignore[arg-type]
                if piece:
                    text += piece
                    yield text
    except (urllib.error.URLError, OSError, ValueError) as e:
        yield f"[錯誤] {e}"


def pick_eval_item(eval_set: EvalSet, item_id: str) -> tuple[str, frozenset[str]]:
    for item in eval_set.items:
        if item.id == item_id:
            return item.question, item.relevant_doc_ids
    raise KeyError(item_id)


def should_abstain(hits: Sequence[Hit], min_score: float) -> bool:
    """top-1 分數低於門檻即拒答(恰等於不拒);無結果必拒;有門檻卻無分數是設定錯,raise。"""
    if not hits:
        return True
    top = hits[0].score
    if top is None:
        raise ValueError("abstain threshold set but retriever exposes no score")
    return top < min_score


ABSTAIN_PREFIX = "[拒答]"


def abstain_text(hits: Sequence[Hit], min_score: float) -> str:
    """拒答時取代 LLM 回答的固定文字(以 ABSTAIN_PREFIX 開頭,供 UI / 統計辨識)。"""
    top = hits[0].score if hits and hits[0].score is not None else None
    detail = f"最高相關分數 {top:.3f} < 門檻 {min_score:.3f}" if top is not None else "無檢索結果"
    return f"{ABSTAIN_PREFIX} 找不到足夠相關的參考資料({detail}),不生成回答。"


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    cites: tuple[int, ...]  # 有效引用編號(去越界),空 = 未標引用
    coverage: float | None  # 句子 3-gram 被所引段落覆蓋的比例;未引用為 None(診斷用,非品質分數)


def sentence_support(answer: str, hits: Sequence[Hit]) -> list[SentenceSupport]:
    """每句的引用編號與「句子 3-gram 被所引段落覆蓋的比例」;越界編號不算引用,棄答句不列。

    引用有效只代表編號存在;覆蓋率才回答「這句是不是真的來自所引段落」(確定性,不打 LLM)。
    """
    text_by_rank = {h.rank: h.text for h in hits}
    out = []
    for _, raw in _split_sentence_spans(answer):
        if ABSTAIN in raw:
            continue
        cites = tuple(sorted({int(c) for c in _CITE.findall(raw) if int(c) in text_by_rank}))
        sentence = _strip_cites(raw)
        cov = coverage3(sentence, "\n".join(text_by_rank[c] for c in cites)) if cites else None
        out.append(SentenceSupport(sentence, cites, cov))
    return out


def low_coverage(supports: Sequence[SentenceSupport], min_coverage: float) -> tuple[int, int]:
    """回 (有引用句數, 其中覆蓋率 < min_coverage 的句數);恰等於門檻不算低。

    ★ 覆蓋率是**診斷旗標,不是品質分數**(2026-09-19 人工複核 23 句實證):
    低覆蓋不等於編造(12/23 是忠實改寫),高覆蓋也不保證正確(0.351 錯藥、0.484 因果倒置)。
    只能當「這句需人工查證」的排序依據,禁止拿來算分或比較配置優劣。
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    cited = [x for x in supports if x.coverage is not None]
    low = sum(1 for x in cited if x.coverage is not None and x.coverage < min_coverage)
    return len(cited), low


def locator_text(hit: Hit) -> str:
    """人可自行核對的出處:檔名 + 頁(pdf)/ 列(csv)/ 字元位移(html)。

    ★ csv 的 `char_span` 實為**列索引**(`chunk_csv_per_row` 每列一個 chunk),
    佔本語料 79%;一律標「字元」會是事實錯誤,故按 source_type 分支。
    """
    name = Path(hit.source_path).name
    if not name:
        return "(未知來源)"
    if hit.span is None:
        return name
    start, end = hit.span
    if hit.source_type == "pdf" and hit.page is not None:
        return f"{name} 頁 {hit.page}"
    if hit.source_type == "csv":
        rows = f"{start}" if end - start <= 1 else f"{start}–{end - 1}"
        return f"{name} 第 {rows} 列(0 起算)"
    return f"{name} 字元 {start}–{end}"


LOW_MARK = "⚠"


def mark_low_coverage(answer: str, hits: Sequence[Hit], min_coverage: float) -> str:
    """把覆蓋率 < min_coverage 的句子在**句首**加上 LOW_MARK,其餘原樣。

    只加前綴、不改內容;⚠ 意為「逐字覆蓋低,需人工查證」,**無標記不代表已驗證**
    (覆蓋率是診斷旗標,見 low_coverage)。句首位置與 sentence_support 共用同一分段。
    """
    text_by_rank = {h.rank: h.text for h in hits}
    marks: list[int] = []
    for start, raw in _split_sentence_spans(answer):
        if ABSTAIN in raw:
            continue
        cites = sorted({int(c) for c in _CITE.findall(raw) if int(c) in text_by_rank})
        if not cites:
            continue
        cov = coverage3(_strip_cites(raw), "\n".join(text_by_rank[c] for c in cites))
        if cov < min_coverage:
            marks.append(start)
    out = answer
    for start in reversed(marks):
        out = out[:start] + LOW_MARK + out[start:]
    return out
