#!/usr/bin/env python3
"""BM25 真實執行層:全語料建索引,對代表性查詢印 top-5 供肉眼檢查。

用法:python3 tools/bm25_smoke.py <chunks.jsonl>
查詢清單刻意含短查詢(2~4 字藥名):pre-mortem 指出 bigram 對短查詢
區分度最弱,這裡的輸出是日後 A 軸報告被挑戰時的第一手對照證據。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from selectrag.retrieval.bm25 import load_bm25_from_jsonl

QUERIES = [
    "metformin",
    "warfarin 禁忌症",
    "糖尿病用藥",
    "洗腎",
    "抗生素 給付規定",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("用法:python3 tools/bm25_smoke.py <chunks.jsonl>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])

    t0 = time.perf_counter()
    retriever = load_bm25_from_jsonl(path)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for query in QUERIES:
        hits = retriever.retrieve(query, top_k=5)
        print(f"  查詢:{query!r} → {len(hits)} 筆")
        for chunk_id in hits:
            print(f"    {chunk_id}")
    query_s = time.perf_counter() - t0

    print(f"  建索引 {build_s:.1f}s;{len(QUERIES)} 個查詢共 {query_s * 1000:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
