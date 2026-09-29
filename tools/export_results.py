#!/usr/bin/env python3
"""把桌機端 results/ 的數字摘要匯出成可入 repo 的精簡 JSON(去掉語料文字與模型輸出)。

    python3 tools/export_results.py results docs/results

規則(每類檔案只保留數字與設定,絕不帶語料 chunk 文字或模型回答):
  runner 結果(有 metrics + eval_set_hash)   → 保留 experiment_id / run_id / eval_set_* /
                                               config / metrics;去 per_query
  bench 結果(有 summary + requests)         → 保留 engine / format / model / concurrency / params /
                                               summary / vram_*;去 requests
  judge 結果(有 summary + answers)          → 保留 summary / meta / parse_failure_rate;去 answers
  其他(compare、rerank grid 等純數字 JSON)  → 原樣複製
輸出路徑鏡射來源相對路徑;已存在的目標檔會被覆蓋(這是摘要的重新生成,不是實驗產物)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TEXT_KEYS = ("per_query", "requests", "answers", "results", "rows", "outputs")


def strip(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """回 (精簡後 dict, 類型標籤)。"""
    if "metrics" in record and "eval_set_hash" in record:
        keep = ("experiment_id", "run_id", "eval_set_version", "eval_set_hash", "config", "metrics")
        return {k: record[k] for k in keep if k in record}, "runner"
    if "summary" in record and "requests" in record:
        keep = (
            "engine",
            "format",
            "model",
            "url",
            "concurrency",
            "started_at",
            "params",
            "summary",
            "vram_baseline_mib",
            "vram_peak_mib",
        )
        return {k: record[k] for k in keep if k in record}, "bench"
    if "summary" in record and "answers" in record:
        keep = ("summary", "meta", "parse_failure_rate", "parse_failures")
        return {k: record[k] for k in keep if k in record}, "judge"
    return {k: v for k, v in record.items() if k not in TEXT_KEYS}, "other"


def export(src: Path, dst: Path) -> dict[str, int]:
    if not src.is_dir():
        raise FileNotFoundError(src)
    counts: dict[str, int] = {}
    for path in sorted(src.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not JSON") from e
        if not isinstance(record, dict):
            continue  # list 型(如 decomp_probe)含模型輸出,一律不匯出
        slim, kind = strip(record)
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args(argv)
    counts = export(args.src, args.dst)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no json found")
    print(f"輸出:{args.dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
