#!/usr/bin/env python3
"""ModelOpt FP8 PTQ 轉換(桌機首日驗證第 4 步 / C 軸 FP8 格)。

2026-09-15 拍板:校正資料 = 語料 chunks 隨機 512 筆(seed 固定,不碰 data/eval)。

用法(TRT venv,.venv-trt;需 accelerate,見 requirements-gpu.txt):
    python3 tools/modelopt_fp8.py --run-id day1     # 預設 Qwen/Qwen2.5-7B-Instruct
    python3 tools/modelopt_fp8.py --run-id day1 --calib-n 512 --seed 42

輸出:models/<model 尾段>-fp8/(HF 格式 checkpoint,供 vLLM 與 TRT-LLM 各自載入)、
      results/modelopt_fp8/<run_id>.json(版本、校正參數與樣本 id 雜湊、VRAM 峰值、秒數)。
驗收:exit 0 且 export 目錄含 config.json / *.safetensors、JSON vram_peak_mib < 24564。
注意:mtq.quantize / export_hf_checkpoint 介面依 ModelOpt 0.37 文件所寫,雲端未驗證。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_smoke import VramMonitor, resolve_model_commit  # noqa: E402  # 共用 VRAM 取樣

DEFAULT_MODEL = (
    "Qwen/Qwen2.5-7B-Instruct"  # bf16 原版;與 AWQ 版同家族,C 軸三格式共基底(2026-09-15 拍板)
)
DEFAULT_REVISION: str | None = "a09a35458c702b33eeacc393d103063234e8bc28"  # 2026-09-15 首跑釘
MAX_LENGTH = 512
BATCH = 4


def load_calib_texts(chunks: Path, n: int, seed: int) -> tuple[list[str], str]:
    """從 chunks.jsonl 以固定 seed 抽 n 筆正文;回傳文字與抽中 chunk_id 排序後的 sha256(可稽核)。"""
    rows = [json.loads(line) for line in chunks.open(encoding="utf-8") if line.strip()]
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    ids = sorted(r["chunk_id"] for r in picked)
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return [str(r["text"]) for r in picked], digest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--chunks", type=Path, default=Path("results/chunks.jsonl"))
    ap.add_argument("--calib-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--export-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    out_path = Path("results/modelopt_fp8") / f"{args.run_id}.json"
    export_dir = args.export_dir or Path("models") / f"{args.model.split('/')[-1]}-fp8"
    if out_path.exists() or export_dir.exists():
        print(f"拒絕覆蓋:{out_path} 或 {export_dir} 已存在", file=sys.stderr)
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts, calib_digest = load_calib_texts(args.chunks, args.calib_n, args.seed)

    import modelopt
    import modelopt.torch.quantization as mtq
    import torch
    import transformers
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    record: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "revision_requested": args.revision,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "modelopt": modelopt.__version__,
        },
        "calibration": {
            "source": str(args.chunks),
            "n": args.calib_n,
            "seed": args.seed,
            "max_length": MAX_LENGTH,
            "batch": BATCH,
            "chunk_ids_sha256": calib_digest,
        },
        "quant_config": "FP8_DEFAULT_CFG",
    }

    with VramMonitor() as vm:
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        load_s = time.perf_counter() - t0

        def forward_loop(m: Any) -> None:
            for i in range(0, len(texts), BATCH):
                enc = tok(
                    texts[i : i + BATCH],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to("cuda")
                m(**enc)

        t1 = time.perf_counter()
        with torch.no_grad():
            model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
        quant_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        export_dir.mkdir(parents=True, exist_ok=True)
        export_hf_checkpoint(model, export_dir=str(export_dir))
        tok.save_pretrained(str(export_dir))
        export_s = time.perf_counter() - t2

    files = sorted(p.name for p in export_dir.iterdir())
    record.update(
        {
            "revision_resolved": resolve_model_commit(args.model, args.revision),
            "load_seconds": round(load_s, 1),
            "quantize_seconds": round(quant_s, 1),
            "export_seconds": round(export_s, 1),
            "export_dir": str(export_dir),
            "export_files": files,
            "vram_baseline_mib": vm.baseline_mib,
            "vram_peak_mib": vm.peak_mib,
        }
    )
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型      : {args.model} @ {record['revision_resolved'] or '(未釘,見 JSON)'}")
    print(f"校正      : {args.calib_n} 筆 chunks,seed {args.seed},id 雜湊 {calib_digest[:12]}")
    print(f"耗時      : 載入 {load_s:.0f}s / 量化 {quant_s:.0f}s / 匯出 {export_s:.0f}s")
    print(f"VRAM      : 基線 {vm.baseline_mib} MiB → 峰值 {vm.peak_mib} MiB")
    print(f"匯出      : {export_dir}({len(files)} 檔)")
    print(f"結果檔    : {out_path}")
    ok = "config.json" in files and any(f.endswith(".safetensors") for f in files)
    if not ok:
        print("✗ 匯出目錄缺 config.json 或 safetensors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
