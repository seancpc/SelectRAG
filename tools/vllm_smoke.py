#!/usr/bin/env python3
"""vLLM 首日驗證(桌機首日驗證第 2 步):載入 AWQ 模型、跑固定 prompt、記錄 VRAM 峰值與吞吐。

用法(主 venv,.venv):
    python3 tools/vllm_smoke.py                      # 預設模型、run_id 取時間戳
    python3 tools/vllm_smoke.py --run-id day1        # 指定 run_id(已存在即拒絕覆蓋)
    python3 tools/vllm_smoke.py --revision <commit>  # 覆寫已釘的 HF revision

輸出:results/vllm_smoke/<run_id>.json,含套件版本、模型 commit、VRAM 基線/峰值、
      載入秒數、生成秒數、輸出 token 數與吞吐。TTFT 留給 B 軸的線上服務量測(offline API 不可靠)。
驗收:exit 0 且 JSON 內 cuda_available=true、outputs 三則皆非空、vram_peak_mib < 24564。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 2026-09-13 拍板(繁中語料,7B 與 embedding 可共存)
DEFAULT_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"  # 2026-09-13 首跑解析出的 commit,釘死
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.60  # 刻意留 40% 給日後 embedding 模型共存實驗
SEED = 42

# 中性 smoke prompt:只驗證「模型會回、繁中會出、格式指令會遵守」,不含任何領域資料
PROMPTS = [
    "請用繁體中文以一句話自我介紹。",
    "請列出三個整數,每行一個,不要加任何說明。",
    "以下是一段文字:「<占位文字>」。請用繁體中文用不超過二十字摘要它。",
]


class VramMonitor:
    """背景執行緒以 nvidia-ml-py 每 0.2 秒取樣 GPU 0 已用記憶體,回報基線與峰值(MiB)。"""

    def __init__(self) -> None:
        import pynvml  # nvidia-ml-py 提供的模組名

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline_mib = self._used()
        self.peak_mib = self.baseline_mib
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._h).used // (1024 * 1024))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._used())
            time.sleep(0.2)

    def __enter__(self) -> VramMonitor:
        self._t.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._t.join(timeout=2)
        self.peak_mib = max(self.peak_mib, self._used())
        self._nvml.nvmlShutdown()


def resolve_model_commit(model: str, revision: str | None) -> str | None:
    """從 HF 快取取出實際載入的 commit hash,供首跑後回填 --revision 釘選。"""
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, revision=revision, local_files_only=True))
        return path.name if len(path.name) == 40 else None
    except Exception:  # 純資訊性,取不到就記 null
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="HF commit hash(預設已釘)")
    ap.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args(argv)

    out_path = Path("results/vllm_smoke") / f"{args.run_id}.json"
    if out_path.exists():
        print(f"拒絕覆蓋既有結果:{out_path}(換 --run-id)", file=sys.stderr)
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    record: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "revision_requested": args.revision,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params": {
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "seed": SEED,
            "max_tokens": args.max_tokens,
        },
    }

    with VramMonitor() as vm:
        t0 = time.perf_counter()
        llm = LLM(
            model=args.model,
            revision=args.revision,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            seed=SEED,
        )
        load_s = time.perf_counter() - t0

        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=SEED)
        t1 = time.perf_counter()
        # Instruct 模型必須走 chat template;raw generate 會被當成續寫(2026-09-13 day1 實證)
        results = llm.chat([[{"role": "user", "content": p}] for p in PROMPTS], sp)
        gen_s = time.perf_counter() - t1

    outputs = []
    total_out_tokens = 0
    for prompt, r in zip(PROMPTS, results, strict=True):
        text = r.outputs[0].text
        n = len(r.outputs[0].token_ids)
        total_out_tokens += n
        outputs.append({"prompt": prompt, "text": text, "output_tokens": n})

    record.update(
        {
            "revision_resolved": resolve_model_commit(args.model, args.revision),
            "load_seconds": round(load_s, 2),
            "generate_seconds": round(gen_s, 3),
            "output_tokens_total": total_out_tokens,
            "throughput_tok_per_s": round(total_out_tokens / gen_s, 1) if gen_s > 0 else None,
            "vram_baseline_mib": vm.baseline_mib,
            "vram_peak_mib": vm.peak_mib,
            "outputs": outputs,
        }
    )
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型      : {args.model} @ {record['revision_resolved'] or '(未釘,見 JSON)'}")
    print(
        f"載入      : {load_s:.1f}s;生成 {total_out_tokens} tokens / {gen_s:.2f}s "
        f"= {record['throughput_tok_per_s']} tok/s"
    )
    print(f"VRAM      : 基線 {vm.baseline_mib} MiB → 峰值 {vm.peak_mib} MiB")
    for o in outputs:
        print(f"--- {o['prompt']}\n{o['text'].strip()}")
    print(f"結果檔    : {out_path}")
    empty = [o for o in outputs if not o["text"].strip()]
    if empty:
        print(f"✗ {len(empty)} 則輸出為空", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
