#!/usr/bin/env python3
"""拒答門檻網格(2026-09-19 定案):兩步。

    # 1. 取分數(桌機一次 GPU;eval 100 題 + 語料外題 × 每個 preset,只存 top-k 分數與是否含真值)
    python3 tools/abstain_grid.py collect configs/demo.yaml data/eval/offcorpus_v2.json \
        results/demo/abstain_scores.json
    # 2. 離線掃門檻(等距 grid 格),印每格三種拒答率與高原中點
    python3 tools/abstain_grid.py sweep results/demo/abstain_scores.json \
        --grid 40 --max-false 0.02 --min-true 0.90
    # 3. k 折交叉驗證:每折用其餘折搜門檻,在 held-out 折量誤拒 / 正確拒
    #    (回答「門檻是否只對這 100 題成立」)
    python3 tools/abstain_grid.py cv results/demo/abstain_scores.json --k 5 --seed 42

collect 不打 LLM;preset 無分數即 raise(BM25 有分數但跨題不可比,結果照印供判讀)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from selectrag.demo.abstain import ScoreRow, kfold, plateau, sweep
from selectrag.demo.core import RetrieverPool, load_demo_config
from selectrag.evaluation.eval_set import load_eval_set
from selectrag.retrieval.rerank import load_chunk_rows
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import load_config


def load_offcorpus(path: Path) -> list[tuple[str, str]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc.get("items")
    if not items:
        raise ValueError(f"{path}: no items")
    out = []
    for it in items:
        if "id" not in it or "question" not in it:
            raise ValueError(f"{path}: item needs id and question: {it!r}")
        out.append((str(it["id"]), str(it["question"])))
    return out


def collect(cfg_path: Path, offcorpus: Path, out: Path, k: int) -> int:
    cfg = load_demo_config(cfg_path)
    rows = load_chunk_rows(cfg.chunks)
    es = load_eval_set(cfg.eval_set)
    off = load_offcorpus(offcorpus)
    pool = RetrieverPool(
        cfg, rows, lambda p: build_retriever(load_config(cfg.presets[p]).retriever)
    )
    records: list[dict[str, Any]] = []
    for preset in cfg.presets:
        queries = [(it.id, it.question, "eval", it.relevant_doc_ids) for it in es.items] + [
            (i, q, "offcorpus", frozenset()) for i, q in off
        ]
        for item_id, question, kind, truth in queries:
            hits = pool.hits(preset, question, k)
            if hits and hits[0].score is None:
                raise ValueError(f"preset {preset} exposes no score; cannot collect")
            records.append(
                {
                    "preset": preset,
                    "item_id": item_id,
                    "kind": kind,
                    "scores": [h.score for h in hits],
                    "truth_in_topk": any(h.chunk_id in truth for h in hits),
                }
            )
        mine = [r for r in records if r["preset"] == preset]
        sc = [r["scores"][0] for r in mine if r["scores"]]
        print(f"[collect] {preset}: n={len(mine)} top1 min={min(sc):.4f} max={max(sc):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"k": k, "rows": records}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0


def to_rows(doc: dict[str, Any]) -> list[ScoreRow]:
    return [
        ScoreRow(
            str(r["preset"]),
            str(r["item_id"]),
            str(r["kind"]),
            float(r["scores"][0]) if r["scores"] else None,
            bool(r["truth_in_topk"]),
        )
        for r in doc["rows"]
    ]


def run_sweep(scores: Path, grid: int, max_false: float, min_true: float) -> int:
    rows = to_rows(json.loads(scores.read_text(encoding="utf-8")))
    for preset in sorted({r.preset for r in rows}):
        sub = [r for r in rows if r.preset == preset]
        pts = sweep(sub, grid)
        n_pos = sum(1 for r in sub if r.kind == "eval" and r.truth_in_topk)
        n_bonus = sum(1 for r in sub if r.kind == "eval" and not r.truth_in_topk)
        n_off = sum(1 for r in sub if r.kind == "offcorpus")
        print(f"== {preset}(eval 有真值 {n_pos} / 無真值 {n_bonus} / offcorpus {n_off})")
        print("threshold  誤拒率  正確拒率  順帶拒率")
        for p in pts:
            line = f"{p.threshold:9.4f}  {p.false_abstain:6.3f}  {p.true_abstain:8.3f}"
            print(f"{line}  {p.bonus_abstain:8.3f}")
        pl = plateau(pts, max_false, min_true)
        if pl is None:
            print(f"高原:無(沒有門檻同時滿足 誤拒≤{max_false} 且 正確拒≥{min_true})")
            print("改看上表 trade-off 曲線由人選點")
        else:
            print(f"高原(誤拒≤{max_false}, 正確拒≥{min_true}):{pl[0]:.4f}–{pl[1]:.4f}")
            print(f"中點 {pl[2]:.4f}")
    return 0


def run_cv(scores: Path, k: int, seed: int, grid: int, max_false: float, min_true: float) -> int:
    rows = to_rows(json.loads(scores.read_text(encoding="utf-8")))
    for preset in sorted({r.preset for r in rows}):
        sub = [r for r in rows if r.preset == preset]
        res = kfold(sub, k, seed, grid, max_false, min_true)
        print(f"== {preset}(k={k}, seed={seed})")
        print("fold  threshold  held-out誤拒  held-out正確拒")
        for r in res:
            t = "無高原" if r.threshold is None else f"{r.threshold:.4f}"
            fa, ta = r.heldout_false_abstain, r.heldout_true_abstain
            print(f"{r.fold:4d}  {t:>9}  {fa:11.3f}  {ta:13.3f}")
        ths = [r.threshold for r in res if r.threshold is not None]
        if ths:
            print(f"門檻 min–max {min(ths):.4f}–{max(ths):.4f};")
        mean_fa = sum(r.heldout_false_abstain for r in res) / k
        mean_ta = sum(r.heldout_true_abstain for r in res) / k
        print(
            f"held-out 平均誤拒 {mean_fa:.3f} / 平均正確拒 {mean_ta:.3f};無高原折數 {k - len(ths)}"
        )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("config", type=Path)
    c.add_argument("offcorpus", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--k", type=int, default=3)
    s = sub.add_parser("sweep")
    s.add_argument("scores", type=Path)
    s.add_argument("--grid", type=int, default=40)
    s.add_argument("--max-false", type=float, default=0.02)
    s.add_argument("--min-true", type=float, default=0.90)
    cv = sub.add_parser("cv")
    cv.add_argument("scores", type=Path)
    cv.add_argument("--k", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--grid", type=int, default=40)
    cv.add_argument("--max-false", type=float, default=0.02)
    cv.add_argument("--min-true", type=float, default=0.90)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        return collect(a.config, a.offcorpus, a.out, a.k)
    if a.cmd == "cv":
        return run_cv(a.scores, a.k, a.seed, a.grid, a.max_false, a.min_true)
    return run_sweep(a.scores, a.grid, a.max_false, a.min_true)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
