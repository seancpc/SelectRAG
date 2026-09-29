#!/usr/bin/env python3
"""逐字命中前置層門檻網格掃(對 calib_sheet.csv 的人工標籤;2026-09-17)。

    python3 tools/verbatim_grid.py results/judge/calib/calib_sheet.csv \\
        --thresholds 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0

每門檻印:命中數、命中中人工判「不支持」數(誤放)、「支持」數;
高原 = 誤放為 0 的最高連續門檻區間,門檻取其中點寫進 judge config 的 verbatim_min_coverage。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from selectrag.evaluation.faithfulness import split_chunks, verbatim_hit

REQUIRED = ("context", "claim", "你的判定")
NEGATIVE = "不支持"


def scan(sheet: Path, thresholds: list[float]) -> list[dict[str, Any]]:
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or any(k not in rows[0] for k in REQUIRED):
        raise ValueError(f"sheet must have columns {REQUIRED}")
    out = []
    for t in thresholds:
        hits = [
            r
            for r in rows
            if verbatim_hit(r["claim"], split_chunks(r["context"]), min_coverage=t) is not None
        ]
        false_hits = sum(1 for r in hits if r["你的判定"].strip() == NEGATIVE)
        out.append(
            {
                "threshold": t,
                "hits": len(hits),
                "false_hits": false_hits,
                "true_hits": len(hits) - false_hits,
            }
        )
    return out


def plateau(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """誤放為 0 的最高連續門檻區間 (低, 高);沒有任何零誤放門檻回 None。"""
    zero = [r["threshold"] for r in rows if r["false_hits"] == 0]
    if not zero:
        return None
    hi = zero[-1]
    lo = hi
    ordered = [r["threshold"] for r in rows]
    for t in reversed(ordered[: ordered.index(hi)]):
        if t in zero:
            lo = t
        else:
            break
    return (lo, hi)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[x / 20 for x in range(10, 21)])
    args = ap.parse_args(argv)
    rows = scan(args.sheet, sorted(args.thresholds))
    print("門檻   命中  誤放(不支持)  支持")
    for r in rows:
        print(f"{r['threshold']:<6.2f} {r['hits']:<5} {r['false_hits']:<13} {r['true_hits']}")
    p = plateau(rows)
    if p is None:
        print("高原:無(所有門檻皆有誤放)")
        return 1
    print(f"高原(零誤放):{p[0]:.2f}–{p[1]:.2f},中點 {(p[0] + p[1]) / 2:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
