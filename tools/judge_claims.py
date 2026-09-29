#!/usr/bin/env python3
"""faithfulness judge 客戶端(2026-09-16 核准;第一階段:短回答)。

用法(任一 venv;judge 伺服端需另起):
    # 由 bench 結果自動切句
    python3 tools/judge_claims.py results/judge/vllm_bf16.json \\
        --bench results/bench/vllm_bf16_c1.json --prompts results/prompts_v12.jsonl \\
        --config configs/judge_v1.yaml
    # 由校準用 claims jsonl(calib_sheet build 產出,含合成負例)
    python3 tools/judge_claims.py results/judge/calib.json \\
        --claims results/judge/calib/calib_claims.jsonl --config configs/judge_v1.yaml
輸出:JSON(拒絕覆蓋):meta、summary(aggregate)、answers[](每主張 verdict / evidence / raw)、
      parse_failures;解析失敗率 > 5% 回 exit 1(動工前設定的高風險假設門檻)。
非串流、response_format json_schema、temperature 0、seed。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from selectrag.evaluation.faithfulness import (
    PROMPT_VERSION,
    aggregate,
    is_abstention,
    judge_prompt,
    parse_verdict,
    split_chunks,
    split_claims,
    verbatim_hit,
)

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
        "evidence_chunk": {"type": ["integer", "null"]},
    },
    "required": ["verdict", "evidence_chunk"],
}
MAX_FAILURE_RATE = 0.05


def load_judge_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for k in ("model", "url", "seed", "max_tokens"):
        if k not in raw:
            raise ValueError(f"judge config missing '{k}'")
    return dict(raw)


def context_of(user_content: str) -> tuple[str, str]:
    """prompts jsonl 的 user 內容 → (問題, 參考段落);找不到分隔就整段當段落。"""
    if "參考段落:" in user_content:
        q, ctx = user_content.split("參考段落:", 1)
        return q.replace("問題:", "", 1).strip(), ctx.strip()
    return "", user_content.strip()


def claims_from_bench(bench: Path, prompts: Path) -> list[dict[str, Any]]:
    """每則回答 → 若拒答記一筆 abstained,否則每句一筆 claim。"""
    ctx_by_id: dict[str, tuple[str, str]] = {}
    for line in prompts.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            ctx_by_id[str(r["id"])] = context_of(str(r["messages"][1]["content"]))
    rows = []
    for req in json.loads(bench.read_text(encoding="utf-8"))["requests"]:
        rid = str(req["id"])
        if rid not in ctx_by_id:
            raise ValueError(f"no prompt for answer id {rid!r}")
        q, ctx = ctx_by_id[rid]
        text = str(req["text"])
        if is_abstention(text):
            rows.append({"id": rid, "abstained": True, "claims": []})
            continue
        claims = split_claims(text)
        rows.append(
            {
                "id": rid,
                "abstained": False,
                "question": q,
                "context": ctx,
                "claims": [{"claim_id": f"{rid}#{i}", "claim": c} for i, c in enumerate(claims)],
            }
        )
    return rows


def claims_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for k in ("claim_id", "claim", "question", "context"):
            if k not in r:
                raise ValueError(f"claims row missing '{k}'")
        rows.append(
            {
                "id": str(r["claim_id"]),
                "abstained": False,
                "question": str(r["question"]),
                "context": str(r["context"]),
                "claims": [{"claim_id": str(r["claim_id"]), "claim": str(r["claim"])}],
            }
        )
    return rows


def judge_one(cfg: dict[str, Any], question: str, context: str, claim: str) -> str:
    body = json.dumps(
        {
            "model": cfg["model"],
            "messages": judge_prompt(question, context, claim),
            "temperature": 0.0,
            "seed": int(cfg["seed"]),
            "max_tokens": int(cfg["max_tokens"]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": SCHEMA},
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        str(cfg["url"]).rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(req, timeout=float(cfg.get("timeout", 120))) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def run(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    n_abst = 0
    n_verbatim = 0
    min_cov = cfg.get("verbatim_min_coverage")  # 缺鍵即停用前置層
    for row in rows:
        if row["abstained"]:
            n_abst += 1
            continue
        chunks = split_chunks(row["context"]) if min_cov is not None else []
        for c in row["claims"]:
            hit = (
                verbatim_hit(c["claim"], chunks, min_coverage=float(min_cov))
                if min_cov is not None
                else None
            )
            if hit is not None:
                v = {"verdict": "supported", "evidence_chunk": hit}
                c.update(v)
                c["source"] = "verbatim"
                verdicts.append(v)
                n_verbatim += 1
                continue
            c["source"] = "llm"
            raw = judge_one(cfg, row["question"], row["context"], c["claim"])
            c["raw"] = raw
            try:
                v = parse_verdict(raw)
            except ValueError as e:
                failures.append({"claim_id": c["claim_id"], "error": str(e)})
                c["verdict"] = None
                continue
            c.update(v)
            verdicts.append(v)
    n_total = len(verdicts) + len(failures)
    summary = aggregate(verdicts, n_abst) if verdicts else None
    if summary is not None:
        summary["verbatim_hits"] = n_verbatim
    return {
        "summary": summary,
        "parse_failures": failures,
        "parse_failure_rate": len(failures) / n_total if n_total else 0.0,
        "answers": rows,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--claims", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    if bool(args.claims) == bool(args.bench):
        print("需要 --claims 或 --bench(二選一)", file=sys.stderr)
        return 2
    if args.bench and not args.prompts:
        print("--bench 需搭配 --prompts", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"拒絕覆蓋:{args.out}", file=sys.stderr)
        return 3
    cfg = load_judge_config(args.config)
    rows = (
        claims_from_jsonl(args.claims)
        if args.claims
        else claims_from_bench(args.bench, args.prompts)
    )
    result = run(cfg, rows)
    result["meta"] = {
        "started_at": datetime.now(UTC).isoformat(),
        "judge_model": cfg["model"],
        "judge_revision": cfg.get("revision"),
        "prompt_version": PROMPT_VERSION,
        "verbatim_min_coverage": cfg.get("verbatim_min_coverage"),
        "seed": cfg["seed"],
        "source": str(args.claims or args.bench),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s = result["summary"]
    if s:
        print(
            f"claims={s['n_claims']} abstained={s['n_abstained_answers']} "
            f"faithfulness={s['faithfulness']:.3f} unsupported={s['unsupported_rate']:.3f} "
            f"contradicted={s['contradiction_rate']:.3f}"
        )
    print(f"parse_failures={len(result['parse_failures'])} rate={result['parse_failure_rate']:.3f}")
    print(f"輸出:{args.out}")
    if result["parse_failure_rate"] > MAX_FAILURE_RATE:
        print(f"✗ 解析失敗率超過 {MAX_FAILURE_RATE:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
