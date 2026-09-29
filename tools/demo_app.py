#!/usr/bin/env python3
"""Gradio 雙欄對照 demo(2026-09-18):同一題,左右各用一個 preset 檢索 + 生成,
真值綠框、獨有黃框。

    python3 tools/demo_app.py configs/demo.yaml --port 7860
    python3 tools/demo_app.py configs/demo.yaml --check
        # --check 不起 UI:載入設定、預熱、跑一題印結果(gauntlet 5l 用)
    python3 tools/demo_app.py configs/demo.yaml --cite-rate 0 --out results/demo/cite_v13.json
        # 引用有效率:前 N 題(0=全部)× 每個 preset,寫 JSON(不存回答與 chunk 文字)

啟動順序:載 chunks / eval set → 對每個 preset 建 retriever 並跑一次預熱查詢
→ 對 LLM 端點打 2 筆暖機(F8)→ 才開 UI;任一步失敗即退出。UI 層只呼叫 selectrag.demo.core,不含邏輯。
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

from selectrag.demo.core import (
    ABSTAIN_PREFIX,
    LOW_MARK,
    DemoConfig,
    Hit,
    Mark,
    RetrieverPool,
    abstain_text,
    cite_stats,
    load_demo_config,
    locator_text,
    low_coverage,
    make_messages,
    mark_low_coverage,
    mark_truth,
    pick_eval_item,
    sentence_support,
    should_abstain,
    stream_answer,
    summarize_cites,
)
from selectrag.evaluation.eval_set import EvalSet, load_eval_set
from selectrag.retrieval.rerank import load_chunk_rows
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import load_config

WARMUP = ("<暖機>", "<暖機二>")


def _drain(chunks: Any) -> str:
    last = ""
    for last in chunks:  # noqa: B007 — 只要最後一筆累積文字
        pass
    return str(last)


def make_pool(cfg: DemoConfig, rows: dict[str, dict[str, Any]]) -> RetrieverPool:
    def builder(preset: str) -> Any:
        t0 = time.perf_counter()
        r = build_retriever(load_config(cfg.presets[preset]).retriever)
        print(f"[demo] preset {preset} 建置 {time.perf_counter() - t0:.1f}s")
        return r

    return RetrieverPool(cfg, rows, builder)


def prewarm(cfg: DemoConfig, pool: RetrieverPool, question: str) -> None:
    for name in cfg.presets:
        pool.hits(name, question, cfg.prompt.top_k)
    for w in WARMUP:
        last = _drain(stream_answer(cfg.llm, [{"role": "user", "content": w}]))
        if last.startswith("[錯誤]"):
            raise SystemExit(f"LLM 端點暖機失敗:{last}")
    print("[demo] 預熱完成")


def render_hits(
    hits: list[Hit],
    marks: list[Mark],
    cited: frozenset[int] = frozenset(),
    ms: float | None = None,
) -> str:
    cards = (
        [] if ms is None else [f"<div style='color:#888;font-size:0.85em'>檢索 {ms:.0f} ms</div>"]
    )
    for h, m in zip(hits, marks, strict=True):
        color = "#2e7d32" if m.in_truth else ("#b58900" if m.only_this_side else "#444")
        tag = "真值" if m.in_truth else ("僅本欄" if m.only_this_side else "")
        if h.rank in cited:
            tag = (tag + " · " if tag else "") + "已引用"
        truncated = "…(截斷,原文更長)" if len(h.text) > 300 else ""
        loc = html.escape(locator_text(h))
        cards.append(
            f'<div style="border:2px solid {color};border-radius:6px;padding:8px;margin:6px 0">'
            f"<b>[{h.rank}]</b> {html.escape(h.doc_title)} <small>{html.escape(h.heading)}</small> "
            f'<span style="float:right;color:{color}">{tag}</span><br>'
            f"<span style='font-size:0.8em;color:#888'>出處:{loc}</span><br>"
            f"<span style='font-size:0.9em'>{html.escape(h.text[:300])}{truncated}</span></div>"
        )
    return "".join(cards) or "<i>無結果</i>"


# 句子 3-gram 被所引段落覆蓋率低於此值即標記為「需查證」。
# ★ 診斷旗標不是分數:2026-09-19 人工複核 23 句 → 改寫 12 / 編造 11;
# 高覆蓋句同樣出過錯藥(0.351)與因果倒置(0.484)。
MIN_COVERAGE = 0.5


def answer_footer(answer: str, marks: list[Mark], hits: list[Hit] | None = None) -> str:
    if answer.startswith(ABSTAIN_PREFIX):
        return "\n\n(拒答:未呼叫 LLM,無引用統計)"
    st_ = cite_stats(answer, marks)
    truth = "(含真值)" if st_.cites_truth else ""
    line = (
        f"\n\n引用 {sorted(st_.valid)}{truth} / 越界 {sorted(st_.invalid)}"
        f" / 未標引用句 {st_.uncited_sentences}/{st_.sentences}"
    )
    if hits is None:
        return line
    sup = sentence_support(answer, hits)
    n_cited, n_low = low_coverage(sup, MIN_COVERAGE)
    covs = [x.coverage for x in sup if x.coverage is not None]
    if not covs:
        return line + "\n逐字覆蓋:無有效引用句"
    flag = f"需查證(覆蓋率 < {MIN_COVERAGE:.1f})的引用句 {n_low}/{n_cited}"
    note = f"\n{LOW_MARK} = 逐字覆蓋低,需人工查證;無標記不代表已驗證" if n_low else ""
    return line + f"\n逐字覆蓋 min {min(covs):.2f} / {flag}——診斷旗標,非品質分數" + note


def run_side(cfg: DemoConfig, pool: RetrieverPool, preset: str, question: str) -> list[Hit]:
    return pool.hits(preset, question, cfg.prompt.top_k)


def run_side_timed(
    cfg: DemoConfig, pool: RetrieverPool, preset: str, question: str
) -> tuple[list[Hit], float]:
    return pool.hits_timed(preset, question, cfg.prompt.top_k)


def answer_stream(cfg: DemoConfig, preset: str, question: str, hits: list[Hit]) -> Any:
    """該 preset 有拒答門檻且 top-1 低於門檻 → 不打 LLM,回單一則拒答文字;否則串流。"""
    thr = cfg.abstain.get(preset)
    if thr is not None and should_abstain(hits, thr):
        return iter([abstain_text(hits, thr)])
    return stream_answer(cfg.llm, make_messages(cfg.prompt, question, hits))


def check_once(cfg: DemoConfig, pool: RetrieverPool, question: str, truth: frozenset[str]) -> int:
    names = list(cfg.presets)
    (left, lms), (right, rms) = (
        run_side_timed(cfg, pool, names[0], question),
        run_side_timed(cfg, pool, names[1], question),
    )
    ml, mr = mark_truth(left, right, truth)
    for name, hits, marks, ms in ((names[0], left, ml, lms), (names[1], right, mr, rms)):
        print(
            f"[{name}] 檢索 {ms:.0f} ms: "
            + ", ".join(
                f"{m.chunk_id}{'✓' if m.in_truth else ''}{'*' if m.only_this_side else ''}"
                for m in marks
            )
        )
        final = _drain(answer_stream(cfg, name, question, hits))
        print(f"  回答:{final[:80]}{answer_footer(final, marks, hits)}")
        if final.startswith("[錯誤]"):
            return 1
    return 0


RATE_KEYS = (
    "n",
    "cite_rate",
    "invalid_rate",
    "truth_cite_n",
    "truth_cite_rate",
    "uncited_sentence_rate",
    "abstain_n",
    "low_coverage_rate",
    "retrieval_ms_mean",
)


def cite_rate(cfg: DemoConfig, pool: RetrieverPool, es: Any, limit: int, out: Path) -> int:
    """前 limit 題(0=全部)× 每個 preset:檢索 → 生成 → cite_stats;任一題 LLM 錯誤即 rc=1。"""
    items = es.items if limit <= 0 else es.items[:limit]
    per_item: list[dict[str, Any]] = []
    per_preset: dict[str, dict[str, float | int]] = {}
    for name in cfg.presets:
        stats = []
        abstain_n = 0
        ms_list: list[float] = []
        n_cited_total = n_low_total = 0
        for item in items:
            hits, ms = run_side_timed(cfg, pool, name, item.question)
            ms_list.append(ms)
            marks, _ = mark_truth(hits, [], item.relevant_doc_ids)
            final = _drain(answer_stream(cfg, name, item.question, hits))
            if final.startswith("[錯誤]"):
                print(f"[cite-rate] {name} {item.id} LLM 錯誤:{final[:80]}")
                return 1
            if final.startswith(ABSTAIN_PREFIX):
                abstain_n += 1
                per_item.append({"preset": name, "item_id": item.id, "abstained": True})
                continue
            st_ = cite_stats(final, marks)
            stats.append(st_)
            n_cited, n_low = low_coverage(sentence_support(final, hits), MIN_COVERAGE)
            n_cited_total += n_cited
            n_low_total += n_low
            per_item.append(
                {
                    "preset": name,
                    "item_id": item.id,
                    "chunk_ids": [h.chunk_id for h in hits],
                    "valid": sorted(st_.valid),
                    "invalid": sorted(st_.invalid),
                    "has_truth": st_.has_truth,
                    "cites_truth": st_.cites_truth,
                    "sentences": st_.sentences,
                    "uncited_sentences": st_.uncited_sentences,
                }
            )
        per_preset[name] = {
            **summarize_cites(stats),
            "abstain_n": abstain_n,
            "low_coverage_rate": n_low_total / n_cited_total if n_cited_total else 0.0,
            "retrieval_ms_mean": sum(ms_list) / len(ms_list) if ms_list else 0.0,
        }
        row = per_preset[name]
        print(
            f"[cite-rate] {name}: "
            + " ".join(
                f"{k}={row[k]:.3f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                for k in RATE_KEYS
            )
        )
        if row["n"] and row["cite_rate"] == 0.0:
            print(f"[cite-rate] 警告:{name} 全部回答無 [n] 引用,檢查模型輸出的引用格式")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"per_preset": per_preset, "per_item": per_item}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return 0


def version_line(cfg: DemoConfig, es: EvalSet) -> str:
    """UI 標頭的可稽核資訊:prompt 版本、eval set 版本、模型。

    展示時觀眾(或面試官)要能一眼確認「這是哪個 prompt、哪份題組、哪個模型」;
    缺 prompt_id 時顯示占位「?」而非留白——空白會被誤讀成「沒有版本管理」。
    """
    pid = cfg.prompt.prompt_id or "?"
    return f"prompt {pid} · eval {es.version} · {cfg.llm.model}"


def launch(
    cfg: DemoConfig, pool: RetrieverPool, eval_items: list[tuple[str, str]], es: Any, port: int
) -> None:
    import gradio as gr  # UI 層才 import;core 永不 import gradio

    names = list(cfg.presets)
    choices = [f"{i}:{q[:30]}" for i, q in eval_items]

    def on_pick(choice: str) -> str:
        return pick_eval_item(es, choice.split(":", 1)[0])[0] if choice else ""

    def on_submit(question: str, choice: str, lp: str, rp: str):  # type: ignore[no-untyped-def]
        truth = pick_eval_item(es, choice.split(":", 1)[0])[1] if choice else frozenset()
        (left, lms), (right, rms) = (
            run_side_timed(cfg, pool, lp, question),
            run_side_timed(cfg, pool, rp, question),
        )
        ml, mr = mark_truth(left, right, truth)
        yield render_hits(left, ml, ms=lms), render_hits(right, mr, ms=rms), "", ""
        gl = answer_stream(cfg, lp, question, left)
        gr_ = answer_stream(cfg, rp, question, right)
        la = ra = ""
        for a in gl:
            la = a
            yield render_hits(left, ml, ms=lms), render_hits(right, mr, ms=rms), la, ra
        for a in gr_:
            ra = a
            yield render_hits(left, ml, ms=lms), render_hits(right, mr, ms=rms), la, ra
        # 串流結束才標記:逐字累積時句界會抖動。⚠ = 該句逐字覆蓋低,需人工查證
        yield (
            render_hits(left, ml, cite_stats(la, ml).valid, lms),
            render_hits(right, mr, cite_stats(ra, mr).valid, rms),
            mark_low_coverage(la, left, MIN_COVERAGE) + answer_footer(la, ml, left),
            mark_low_coverage(ra, right, MIN_COVERAGE) + answer_footer(ra, mr, right),
        )

    with gr.Blocks(title="SelectRAG demo") as ui:
        gr.Markdown(f"<small>{version_line(cfg, es)}</small>")
        gr.Markdown("## SelectRAG:同一題,兩種配置並排(綠框 = eval set 真值;黃框 = 只有本欄撈到)")
        with gr.Row():
            question = gr.Textbox(label="問題", scale=3)
            pick = gr.Dropdown(choices=[""] + choices, label="載入 eval 題目", scale=2)
        with gr.Row():
            lp = gr.Dropdown(choices=names, value=names[0], label="左欄 preset")
            rp = gr.Dropdown(choices=names, value=names[1], label="右欄 preset")
            go = gr.Button("送出", variant="primary")
        with gr.Row():
            with gr.Column():
                lh = gr.HTML(label="左欄檢索")
                la = gr.Textbox(label="左欄回答", lines=6)
            with gr.Column():
                rh = gr.HTML(label="右欄檢索")
                ra = gr.Textbox(label="右欄回答", lines=6)
        pick.change(on_pick, inputs=pick, outputs=question)
        go.click(on_submit, inputs=[question, pick, lp, rp], outputs=[lh, rh, la, ra])
    ui.launch(server_port=port)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--check", action="store_true", help="不起 UI,跑一題後退出")
    ap.add_argument(
        "--cite-rate",
        type=int,
        default=None,
        metavar="N",
        help="引用有效率:前 N 題(0=全部),不起 UI",
    )
    ap.add_argument("--out", type=Path, default=None, help="--cite-rate 的輸出 JSON")
    args = ap.parse_args(argv)
    if args.cite_rate is not None and args.out is None:
        ap.error("--cite-rate 需要 --out")
    cfg = load_demo_config(args.config)
    rows = load_chunk_rows(cfg.chunks)
    es = load_eval_set(cfg.eval_set)
    pool = make_pool(cfg, rows)
    first = es.items[0]
    prewarm(cfg, pool, first.question)
    if args.check:
        return check_once(cfg, pool, first.question, first.relevant_doc_ids)
    if args.cite_rate is not None:
        return cite_rate(cfg, pool, es, args.cite_rate, args.out)
    launch(cfg, pool, [(i.id, i.question) for i in es.items], es, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
