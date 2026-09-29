#!/usr/bin/env python3
"""NV-Embed-v2 首日驗證(桌機首日驗證第 6 步):載入、編碼中性句、記錄 VRAM 峰值與維度。

用法(主 venv,.venv;需 `huggingface-cli login` 且已在 HF 接受 nvidia/NV-Embed-v2 授權):
    python3 tools/nvembed_smoke.py                    # run_id 取時間戳
    python3 tools/nvembed_smoke.py --run-id day1
    python3 tools/nvembed_smoke.py --revision <commit>  # 首跑印出 commit 後釘回 DEFAULT_REVISION

輸出:results/nvembed_smoke/<run_id>.json,含套件版本、模型 commit、dtype、embedding 維度、
      VRAM 基線/峰值、載入秒數、編碼秒數、兩組中性句的 cosine(同義對應高於無關對)。
驗收:exit 0 且 dim == 4096、cosine(同義) > cosine(無關)、vram_peak_mib < 24564。
注意:載入走 transformers AutoModel + trust_remote_code(模型卡作法);此程式碼對 transformers 4.57
      的相容性為首跑驗證項目,失敗即記誠實區。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_smoke import VramMonitor, resolve_model_commit  # noqa: E402  # 共用 VRAM 取樣

DEFAULT_MODEL = "nvidia/NV-Embed-v2"
DEFAULT_REVISION: str | None = (
    "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"  # 2026-09-15 首跑 remote code commit
)
MAX_LENGTH = 512  # 與 bge-m3 索引端上限對齊;正式建索引時另議
QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)

# 中性句:只驗證「同義對 > 無關對」,不含任何領域資料
QUERY = "<問題占位:某項規定的適用條件>"
PASSAGES = [
    "<段落甲:與問題同義的敘述>",
    "<段落乙:與問題同義的另一種敘述>",
    "<段落丙:完全無關的敘述,例如天氣>",
]


def _cosine(a: Any, b: Any) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args(argv)

    out_path = Path("results/nvembed_smoke") / f"{args.run_id}.json"
    if out_path.exists():
        print(f"拒絕覆蓋既有結果:{out_path}(換 --run-id)", file=sys.stderr)
        return 3
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoModel

    # 相容 shim(2026-09-15):NV-Embed-v2 remote code 呼叫 DynamicCache.get_usable_length,
    # transformers ≥ 4.5x 已改名 get_seq_length;只補回舊名,不動模型檔、不降版 transformers
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (  # type: ignore[attr-defined]
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )
        record_shim = "DynamicCache.get_usable_length → get_seq_length"
    else:
        record_shim = None

    record: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "revision_requested": args.revision,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params": {"max_length": MAX_LENGTH, "dtype": "float16"},
        "compat_shim": record_shim,
    }

    with VramMonitor() as vm:
        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            dtype=torch.float16,
        ).to("cuda")
        model.eval()
        load_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            q = model.encode([QUERY], instruction=QUERY_INSTRUCTION, max_length=MAX_LENGTH)
            p = model.encode(PASSAGES, instruction="", max_length=MAX_LENGTH)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        p = torch.nn.functional.normalize(p, p=2, dim=1)
        enc_s = time.perf_counter() - t1

    sims = [_cosine(q[0], p[i]) for i in range(len(PASSAGES))]
    dim = int(p.shape[1])
    record.update(
        {
            "revision_resolved": resolve_model_commit(args.model, args.revision),
            "load_seconds": round(load_s, 2),
            "encode_seconds": round(enc_s, 3),
            "embedding_dim": dim,
            "cosine": dict(
                zip(["同義甲", "同義乙", "無關丙"], [round(s, 4) for s in sims], strict=True)
            ),
            "vram_baseline_mib": vm.baseline_mib,
            "vram_peak_mib": vm.peak_mib,
        }
    )
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型      : {args.model} @ {record['revision_resolved'] or '(未釘,見 JSON)'}")
    print(f"載入      : {load_s:.1f}s;編碼 {len(PASSAGES) + 1} 句 {enc_s:.2f}s;維度 {dim}")
    print(f"VRAM      : 基線 {vm.baseline_mib} MiB → 峰值 {vm.peak_mib} MiB")
    print(f"cosine    : {record['cosine']}")
    print(f"結果檔    : {out_path}")
    ok = dim == 4096 and min(sims[0], sims[1]) > sims[2]
    if not ok:
        print("✗ 驗收未過:維度非 4096 或同義對未高於無關對", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
