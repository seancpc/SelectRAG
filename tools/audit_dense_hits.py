#!/usr/bin/env python3
"""採樣偏差抽驗:列出 dense 取回、但不在真值集的 chunk,供人工判定是否為漏標真值。

背景:eval set 真值以 BM25 多輪改寫窮舉建立,dense 找到而 BM25 沒找到的 chunk
可能是漏標——不抽驗,dense 的 recall 有系統性低估。本工具只列候選,不做判定(領域真值屬人類保留區)。

用法:
    python3 tools/audit_dense_hits.py data/eval/eval_v1.1_full100.json results/chunks.jsonl \\
        results/dense-bge-m3-ctx-full100/desktop.json results/bm25-ctx-tuned-full100/desktop.json \\
        results/audit/dense_ctx_candidates.json --top-k 5

輸出 JSON:每題的問題、既有真值、候選 chunk(dense 名次、是否也在 BM25 top-k、全文)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from selectrag.evaluation.eval_set import load_eval_set


def _per_query(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {q["item_id"]: list(q["retrieved"]) for q in data["per_query"]}


def _chunk_texts(path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                texts[row["chunk_id"]] = row["text"]
    return texts


def build_candidates(
    eval_path: Path, chunks_path: Path, dense_path: Path, bm25_path: Path, top_k: int
) -> list[dict[str, Any]]:
    """回傳每題的候選清單;沒有候選的題目不列(已全命中或 dense 取回皆為真值)。"""
    eval_set = load_eval_set(eval_path)
    texts = _chunk_texts(chunks_path)
    dense = _per_query(dense_path)
    bm25 = _per_query(bm25_path)
    out: list[dict[str, Any]] = []
    for item in eval_set.items:
        if item.id not in dense:
            raise ValueError(f"dense result lacks item {item.id}")
        truth = item.relevant_doc_ids
        bm25_top = set(bm25.get(item.id, [])[:top_k])
        candidates = [
            {
                "chunk_id": cid,
                "dense_rank": rank,
                "also_in_bm25_topk": cid in bm25_top,
                "text": texts.get(cid, "<chunk 不在語料中>"),
            }
            for rank, cid in enumerate(dense[item.id][:top_k], start=1)
            if cid not in truth
        ]
        if candidates:
            out.append(
                {
                    "item_id": item.id,
                    "question": item.question,
                    "truth": sorted(truth),
                    "dense_hit_in_topk": any(c in truth for c in dense[item.id][:top_k]),
                    "candidates": candidates,
                }
            )
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("dense_result", type=Path)
    ap.add_argument("bm25_result", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"拒絕覆蓋:{args.out}", file=sys.stderr)
        return 3
    rows = build_candidates(
        args.eval_set, args.chunks, args.dense_result, args.bm25_result, args.top_k
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cand = sum(len(r["candidates"]) for r in rows)
    n_dense_only = sum(1 for r in rows for c in r["candidates"] if not c["also_in_bm25_topk"])
    n_miss = sum(1 for r in rows if not r["dense_hit_in_topk"])
    print(f"題目有候選 : {len(rows)}/{len(load_eval_set(args.eval_set).items)}")
    print(
        f"候選 chunk : {n_cand}(其中 dense 獨有 {n_dense_only},BM25 也取回 {n_cand - n_dense_only})"
    )
    print(f"dense top-{args.top_k} 全未命中真值的題目:{n_miss}")
    print(f"輸出       : {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
