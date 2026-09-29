#!/usr/bin/env python3
"""B 軸效能量測客戶端(2026-09-15 核准)。

對 OpenAI-compatible 串流端點量 TTFT / 延遲 / 吞吐 / VRAM。

用法(任一 venv;伺服端需另起):
    python3 tools/bench_serve.py results/prompts_v12.jsonl results/bench \\
        --url http://localhost:8000 --model <served model name> --engine vllm --fmt bf16 \\
        --concurrency 1 4 8 --max-tokens 256 --seed 42 --warmup 5
輸出:<out>/<engine>_<fmt>_c<N>.json(拒絕覆蓋):每題 ttft / latency / tokens / text
      + summary(p50/p95/吞吐)+ VRAM。
純標準庫;不走 proxy(伺服端在本機)。VRAM 取樣需 nvidia-ml-py,無則記 null。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from selectrag.perf.bench import RequestMetrics, StreamCollector, parse_sse_line, summarize

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    for r in rows:
        if "id" not in r or "messages" not in r:
            raise ValueError("prompt row needs id and messages")
    return rows


def stream_one(
    url: str, model: str, row: dict[str, Any], max_tokens: int, seed: int, timeout: float
) -> RequestMetrics:
    body = json.dumps(
        {
            "model": model,
            "messages": row["messages"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    col = StreamCollector(str(row["id"]), t0)
    with _OPENER.open(req, timeout=timeout) as resp:
        for raw in resp:
            ev = parse_sse_line(raw.decode("utf-8"))
            if ev is None:
                continue
            if ev == "DONE":
                break
            col.feed(time.perf_counter(), ev)  # type: ignore[arg-type]
    return col.finish(time.perf_counter())


def run_level(
    url: str,
    model: str,
    rows: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> tuple[list[RequestMetrics], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda r: stream_one(url, model, r, max_tokens, seed, timeout), rows))
    return results, time.perf_counter() - t0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-vram", action="store_true")
    args = ap.parse_args(argv)
    if any(c < 1 for c in args.concurrency) or args.warmup < 0:
        print("concurrency >= 1, warmup >= 0", file=sys.stderr)
        return 2
    rows = load_prompts(args.prompts)
    targets = {c: args.out / f"{args.engine}_{args.fmt}_c{c}.json" for c in args.concurrency}
    clash = [p for p in targets.values() if p.exists()]
    if clash:
        print(f"拒絕覆蓋:{clash[0]}", file=sys.stderr)
        return 3
    args.out.mkdir(parents=True, exist_ok=True)

    for r in rows[: args.warmup]:  # 暖機不計
        stream_one(args.url, args.model, r, args.max_tokens, args.seed, args.timeout)

    for c, path in targets.items():
        vm: Any = None
        if not args.no_vram:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from vllm_smoke import VramMonitor

                vm = VramMonitor()
                vm.__enter__()
            except Exception:  # nvidia-ml-py 不在或無 GPU:記 null
                vm = None
        metrics, wall = run_level(
            args.url, args.model, rows, c, args.max_tokens, args.seed, args.timeout
        )
        if vm is not None:
            vm.__exit__(None, None, None)
        s = summarize(metrics, wall)
        record = {
            "engine": args.engine,
            "format": args.fmt,
            "model": args.model,
            "url": args.url,
            "concurrency": c,
            "started_at": datetime.now(UTC).isoformat(),
            "params": {"max_tokens": args.max_tokens, "seed": args.seed, "warmup": args.warmup},
            "summary": s,
            "vram_baseline_mib": vm.baseline_mib if vm else None,
            "vram_peak_mib": vm.peak_mib if vm else None,
            "requests": [m.__dict__ for m in metrics],
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{args.engine:<8}{args.fmt:<10}c={c:<3} n={s['n']:<4}"
            f"TTFT p50/p95 {s['ttft_p50_s']:.3f}/{s['ttft_p95_s']:.3f}s  "
            f"lat p50/p95 {s['latency_p50_s']:.2f}/{s['latency_p95_s']:.2f}s  "
            f"{s['tok_per_s']:.0f} tok/s  {s['req_per_s']:.2f} req/s  "
            f"VRAM peak {record['vram_peak_mib']}"
        )
    print(f"輸出:{args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
