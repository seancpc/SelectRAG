#!/usr/bin/env python3
"""驗證 eval set 的真值 chunk_id 是否全部存在於語料建置產物(移植/重建後必跑)。

chunk_id 由「檔名 + 序號 + 內容 hash」組成:檔名不同、pypdf 版本不同、切塊參數不同
都會讓 id 改變,eval set 真值就對不上——本工具讓這種失配在跑實驗前就暴露。

用法:python3 tools/check_eval_corpus.py data/eval/eval_v1.1_full100.json results/chunks.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from selectrag.evaluation.eval_set import load_eval_set


def missing_ids(eval_path: Path, chunks_path: Path) -> tuple[int, list[tuple[str, str]]]:
    """回傳 (真值 id 總數, [(題目 id, 缺少的 chunk_id), …])。"""
    corpus = {
        json.loads(line)["chunk_id"] for line in chunks_path.open(encoding="utf-8") if line.strip()
    }
    eval_set = load_eval_set(eval_path)
    total = 0
    missing: list[tuple[str, str]] = []
    for item in eval_set.items:
        for cid in sorted(item.relevant_doc_ids):
            total += 1
            if cid not in corpus:
                missing.append((item.id, cid))
    return total, missing


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    total, missing = missing_ids(Path(argv[1]), Path(argv[2]))
    if missing:
        print(f"✗ {len(missing)}/{total} 個真值 chunk_id 不在語料中——eval set 與語料失配:")
        for item_id, cid in missing[:20]:
            print(f"    {item_id}  {cid}")
        print("  常見原因:檔名與 SOURCES.md 不一致 / pypdf 版本非 6.16.2 / 切塊參數改動")
        return 1
    print(f"✓ {total}/{total} 個真值 chunk_id 全數存在於語料,eval set 可用")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
