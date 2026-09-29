#!/usr/bin/env python3
"""兩引擎 greedy 輸出一致性比對(2026-09-15 核准):只判「引擎是否改變輸出」,不判品質。

用法:python3 tools/bench_compare.py results/bench/vllm_bf16_c1.json \\
        results/bench/trtllm_bf16_c1.json [--out x.json]
印出:完全一致率、首個分歧字元位置分佈(p50/p95)、長度差分佈、兩側簡體字元佔比(語言漂移,
確定性字表,與品質無關獨立計);--out 另存明細。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from selectrag.perf.bench import compare_outputs, percentile
from selectrag.perf.langdrift import drift_summary


def load_texts(path: Path) -> dict[str, str]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["id"]): str(r["text"]) for r in rec["requests"]}


def report(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_texts, b_texts = load_texts(a_path), load_texts(b_path)
    r = compare_outputs(a_texts, b_texts)
    r["simplified_a"], r["simplified_b"] = drift_summary(a_texts), drift_summary(b_texts)
    div = [v for v in r["first_divergence_char"].values() if v is not None]
    lens = list(r["len_diff"].values())
    r["divergence_p50_char"] = percentile(div, 50) if div else None
    r["divergence_p95_char"] = percentile(div, 95) if div else None
    r["len_diff_p50"] = percentile(lens, 50)
    r["len_diff_p95"] = percentile(lens, 95)
    r["a"], r["b"] = str(a_path), str(b_path)
    return r


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    r = report(args.a, args.b)
    print(f"完全一致 {r['exact_match']}/{r['n']} = {r['exact_rate']:.2f}")
    print(f"首個分歧字元 p50/p95:{r['divergence_p50_char']}/{r['divergence_p95_char']}")
    print(f"輸出長度差 p50/p95:{r['len_diff_p50']}/{r['len_diff_p95']}")
    sa, sb = r["simplified_a"], r["simplified_b"]
    print(
        f"簡體字元佔比:A {sa['texts_with_simplified']} 題"
        f"(mean {sa['ratio_mean']:.3f} / max {sa['ratio_max']:.3f});"
        f"B {sb['texts_with_simplified']} 題"
        f"(mean {sb['ratio_mean']:.3f} / max {sb['ratio_max']:.3f})"
    )
    if args.out:
        if args.out.exists():
            print(f"拒絕覆蓋:{args.out}", file=sys.stderr)
            return 3
        args.out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
