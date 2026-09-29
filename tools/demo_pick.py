#!/usr/bin/env python3
"""從兩份 runner 結果挑 demo 題:左 preset 在 top-k 沒撈到真值、右 preset 撈到了。

    python3 tools/demo_pick.py <左結果.json> <右結果.json> data/eval/eval_v1.2_full100.json [--k 3]

輸出每題 item_id、左右 hit@k、問題文字,供 demo 錄影腳本選題。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def pick(left: dict[str, Any], right: dict[str, Any], k: int) -> list[str]:
    """回左 hit@k 為 0 且右 hit@k 為 1 的 item_id,依右側 reciprocal_rank 高者優先。"""
    key = f"hit@{k}"
    r_by = {q["item_id"]: q for q in right["per_query"]}
    out = []
    for q in left["per_query"]:
        r = r_by.get(q["item_id"])
        if r is None:
            continue
        if q["hit"][key] == 0.0 and r["hit"][key] == 1.0:
            out.append((-r["reciprocal_rank"], q["item_id"]))
    return [i for _, i in sorted(out)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("eval_set")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args(argv)
    left = json.loads(Path(a.left).read_text(encoding="utf-8"))
    right = json.loads(Path(a.right).read_text(encoding="utf-8"))
    ev = json.loads(Path(a.eval_set).read_text(encoding="utf-8"))
    items = {i["id"]: i["question"] for i in ev["items"]}
    ids = pick(left, right, a.k)
    print(f"左輸右贏(hit@{a.k}):{len(ids)} 題")
    for i in ids:
        print(f"  {i}\t{items.get(i, '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
