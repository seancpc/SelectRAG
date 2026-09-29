#!/usr/bin/env python3
"""方案 B1 前置實驗(2026-09-18,留證):三子問 + 確定性合併,對分層校準集的指定列試跑。

    python3 tools/decomp_probe.py results/judge/calib_v3 results/judge/calib_v3.json \\
        --config configs/judge_v1.yaml \\
        --out results/judge/decomp_probe.json

列選擇:v2(judge_json)與人工不一致的列 + 全部合成負例。
每列三次呼叫(Q1 對象歸屬 / Q2 數值核對 / Q3 列舉核對),
verdict 由 merge() 合併;印出人工 / v2 / 拆問三欄對照與翻對數。prompt 只含結構,無領域範例。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from selectrag.evaluation.faithfulness import normalize_variants

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LABEL = {"支持": "supported", "不支持": "not_supported"}

Q1_SYS = (
    "你是事實查核員。只依據參考段落作答,不得使用外部知識。\n"
    "任務:找出主張所指的對象(藥品、代碼、項目或規定的名稱),再判斷哪一個段落含有「針對該對象」的主張所述那項資訊。\n"
    "段落只對其他對象有該項資訊,不算該對象有;找不到就填 null。\n"
    '只輸出 JSON:{"target": "<對象名稱>", "chunk_with_info": <段落編號或 null>}'
)
Q2_SYS = (
    "你是事實查核員。只依據參考段落作答,不得使用外部知識。\n"
    "任務:列出主張中的每一個數值(含單位與計量基準,如每次、每日、總量、期間),"
    "逐一在段落中找對應的值。數字相同但單位或計量基準不同,same 填 false。主張沒有數值就回空陣列。\n"
    '只輸出 JSON:{"values": [{"claim_value": "<主張中的值>", '
    '"context_value": "<段落中的值或 null>", "same": <true|false>}]}'
)
Q3_SYS = (
    "你是事實查核員。只依據參考段落作答,不得使用外部知識。\n"
    "任務:主張若列舉多個項目(文件、條件、步驟、病症),逐一確認每個項目是否在段落中存在;"
    "不存在的項目 present 填 false。主張沒有列舉就回空陣列。\n"
    '只輸出 JSON:{"items": [{"item": "<項目>", "present": <true|false>}]}'
)
Q1_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string"}, "chunk_with_info": {"type": ["integer", "null"]}},
    "required": ["target", "chunk_with_info"],
}
Q2_SCHEMA = {
    "type": "object",
    "properties": {
        "values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_value": {"type": "string"},
                    "context_value": {"type": ["string", "null"]},
                    "same": {"type": "boolean"},
                },
                "required": ["claim_value", "context_value", "same"],
            },
        }
    },
    "required": ["values"],
}
Q3_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "string"}, "present": {"type": "boolean"}},
                "required": ["item", "present"],
            },
        }
    },
    "required": ["items"],
}


def ask(
    cfg: dict[str, Any],
    system: str,
    question: str,
    context: str,
    claim: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    user = f"問題:{question}\n\n參考段落:\n{context}\n\n主張:{claim}"
    body = json.dumps(
        {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "seed": int(cfg["seed"]),
            "max_tokens": 256,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "q", "schema": schema},
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
    return dict(json.loads(payload["choices"][0]["message"]["content"]))


def is_abstention_claim(claim: str) -> bool:
    return "未提及" in normalize_variants(claim)


def merge(claim: str, q1: dict[str, Any], q2: dict[str, Any], q3: dict[str, Any]) -> str:
    """確定性合併:contradicted 只能來自被指名的差異點。"""
    if is_abstention_claim(claim):
        return "contradicted" if q1.get("chunk_with_info") is not None else "supported"
    if q1.get("chunk_with_info") is None:
        return "unsupported"
    if any(not v.get("same", True) for v in q2.get("values", [])):
        return "contradicted"
    if any(not it.get("present", True) for it in q3.get("items", [])):
        return "contradicted"
    return "supported"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument("judge_json", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"拒絕覆蓋:{args.out}", file=sys.stderr)
        return 3
    cfg = dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    rows = [
        json.loads(line)
        for line in (args.calib_dir / "calib_claims.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    human: dict[str, str] = {}
    with (args.calib_dir / "calib_sheet.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            v = LABEL.get((r.get("你的判定") or "").strip())
            if v:
                human[r["row_id"]] = v
    v2 = {
        c["claim_id"]: c.get("verdict")
        for a in json.loads(args.judge_json.read_text(encoding="utf-8"))["answers"]
        for c in a["claims"]
    }

    def collapse(v: str | None) -> str:
        return "supported" if v == "supported" else "not_supported"

    picked = [
        r
        for r in rows
        if r["row_id"] in human
        and (r.get("synthetic") or collapse(v2.get(r["claim_id"])) != human[r["row_id"]])
    ]
    results = []
    flipped = 0
    neg_ok = 0
    print(f"{'row':<11}{'層':<11}{'人工':<14}{'v2':<14}{'拆問':<14}Q1/Q2/Q3")
    for r in picked:
        q1 = ask(cfg, Q1_SYS, r["question"], r["context"], r["claim"], Q1_SCHEMA)
        q2 = ask(cfg, Q2_SYS, r["question"], r["context"], r["claim"], Q2_SCHEMA)
        q3 = ask(cfg, Q3_SYS, r["question"], r["context"], r["claim"], Q3_SCHEMA)
        d = merge(r["claim"], q1, q2, q3)
        h, old = human[r["row_id"]], collapse(v2.get(r["claim_id"]))
        ok = collapse(d) == h
        if r.get("synthetic"):
            neg_ok += ok
        elif ok and old != h:
            flipped += 1
        results.append(
            {
                **r,
                "human": h,
                "v2": v2.get(r["claim_id"]),
                "decomp": d,
                "q1": q1,
                "q2": q2,
                "q3": q3,
            }
        )
        mark = "✓" if ok else "✗"
        vals, items = q2.get("values", []), q3.get("items", [])
        n_diff = sum(1 for v in vals if not v.get("same"))
        n_miss = sum(1 for i in items if not i.get("present"))
        print(
            f"{r['row_id']:<11}{r.get('stratum', ''):<11}{h:<14}{old:<14}{d + mark:<14}"
            f"{q1.get('chunk_with_info')} / {len(vals)}值{n_diff}異 / "
            f"{len(items)}項{n_miss}缺 | {r['claim'][:28]}"
        )
    n_wrong = sum(1 for r in picked if not r.get("synthetic"))
    n_neg = sum(1 for r in picked if r.get("synthetic"))
    print(f"\nv2 錯的 {n_wrong} 筆翻對 {flipped};負例 {neg_ok}/{n_neg}(門檻:翻對 ≥ 6、負例全對)")
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"輸出:{args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
