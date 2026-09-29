#!/usr/bin/env python3
"""真值重複事實掃描(2026-09-15 核准)。

對 eval set 每筆真值 chunk,列出語料中與它「逐字高度重複」的其他 chunk(字元 n-gram Jaccard),
候選交人工判定是否為未標的等價真值;本工具不修改 eval set。

用法:
    python3 tools/truth_dupes.py data/eval/eval_v1.1_full100.json results/chunks.jsonl \\
        results/audit/truth_dupes_v11.json [--n 3 --top 5 --threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from selectrag.evaluation.eval_set import load_eval_set
from selectrag.retrieval.rerank import load_chunk_rows


def char_ngrams(text: str, n: int) -> frozenset[str]:
    """去空白後的字元 n-gram 集合;長度不足 n 回空集。"""
    s = "".join(text.split())
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def scan(
    truth_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    n: int = 3,
    top: int = 5,
    threshold: float = 0.5,
) -> dict[str, list[tuple[str, float]]]:
    """每筆真值 → [(chunk_id, score)]:分數遞減、同分依 id、排除自身、>= threshold 取前 top 筆。"""
    if n < 1 or top < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("n >= 1, top >= 1, 0 <= threshold <= 1 required")
    for t in truth_ids:
        if t not in rows:
            raise ValueError(f"truth chunk {t!r} not in chunks table")
    grams = {cid: char_ngrams(str(row["text"]), n) for cid, row in rows.items()}
    out: dict[str, list[tuple[str, float]]] = {}
    for t in truth_ids:
        scored = [(cid, jaccard(grams[t], g)) for cid, g in grams.items() if cid != t]
        kept = [(cid, s) for cid, s in scored if s >= threshold]
        kept.sort(key=lambda x: (-x[1], x[0]))
        out[t] = kept[:top]
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"拒絕覆蓋:{args.out} 已存在", file=sys.stderr)
        return 3

    rows = load_chunk_rows(args.chunks)
    pairs = [
        (item.id, t, item.relevant_doc_ids)
        for item in load_eval_set(args.eval_set).items
        for t in sorted(item.relevant_doc_ids)
    ]
    result = scan([t for _, t, _ in pairs], rows, args.n, args.top, args.threshold)

    items = []
    already = 0
    for qid, t, rel in pairs:
        cands = []
        for cid, score in result[t]:
            flag = cid in rel
            already += flag
            cands.append(
                {
                    "chunk_id": cid,
                    "score": score,
                    "doc_id": rows[cid].get("doc_id"),
                    "already_truth": flag,
                }
            )
        items.append({"question_id": qid, "truth_id": t, "candidates": cands})
    summary = {
        "truths": len(items),
        "with_candidates": sum(1 for i in items if i["candidates"]),
        "candidates_total": sum(len(i["candidates"]) for i in items),
        "already_truth": already,
    }
    report = {
        "params": {"n": args.n, "top": args.top, "threshold": args.threshold},
        "summary": summary,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(" ".join(f"{k}={v}" for k, v in summary.items()))
    print("候選最多的前 10 筆真值:")
    for i in sorted(items, key=lambda i: -len(i["candidates"]))[:10]:
        new = sum(1 for c in i["candidates"] if not c["already_truth"])
        print(f"  {i['question_id']:<8}{len(i['candidates'])} 筆(未標 {new}) {i['truth_id'][:60]}")
    print(f"輸出:{args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
