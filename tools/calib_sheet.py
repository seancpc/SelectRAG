#!/usr/bin/env python3
"""judge 校準表(2026-09-16 核准):抽題給人工判定,再與 judge 結果算 κ。

    build:python3 tools/calib_sheet.py build results/judge/vllm_bf16.json \\
              results/judge/calib \\
              --n 30 --negatives 10 --seed 42
          → calib_claims.jsonl(供 judge_claims --claims)、
            calib_sheet.csv(UTF-8 BOM,Excel 可開,人工填「你的判定」)
    score:python3 tools/calib_sheet.py score results/judge/calib/calib_sheet.csv \\
              results/judge/calib.json
          → κ(整體)、各類 recall、混淆矩陣、合成負例抓到數
合成負例:從抽中的主張裡挑含數字者改一個數字(perturb_number),
row_id 對人工不可辨;jsonl 記 synthetic 旗標。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from selectrag.evaluation.faithfulness import (
    VERDICTS,
    cohen_kappa,
    coverage3,
    normalize_variants,
    perturb_number,
    split_chunks,
)

HEADER = ["row_id", "question", "context", "claim", "你的判定", "備註"]
# 人工可用二分法填(支持/不支持);二分時 judge 的 unsupported / contradicted 一併塌成 not_supported
LABEL_ALIASES = {"支持": "supported", "不支持": "not_supported"}
COLLAPSE = {
    "supported": "supported",
    "unsupported": "not_supported",
    "contradicted": "not_supported",
}


def _row_id(claim_id: str, synthetic: bool) -> str:
    return hashlib.sha256(f"{claim_id}|{synthetic}".encode()).hexdigest()[:10]


def build(judge_json: Path, out_dir: Path, n: int, negatives: int, seed: int) -> dict[str, int]:
    if n < 1 or negatives < 0:
        raise ValueError("n >= 1, negatives >= 0")
    answers = [
        a
        for a in json.loads(judge_json.read_text(encoding="utf-8"))["answers"]
        if not a["abstained"]
    ]
    if len(answers) < n:
        raise ValueError(f"only {len(answers)} non-abstained answers, need {n}")
    rng = random.Random(seed)
    picked = rng.sample(answers, n)
    rows: list[dict[str, Any]] = []
    for a in picked:
        for c in a["claims"]:
            rows.append(
                {
                    "row_id": _row_id(c["claim_id"], False),
                    "claim_id": c["claim_id"],
                    "question": a["question"],
                    "context": a["context"],
                    "claim": c["claim"],
                    "synthetic": False,
                }
            )
    candidates = [r for r in rows if perturb_number(r["claim"], random.Random(0)) is not None]
    if len(candidates) < negatives:
        raise ValueError(
            f"only {len(candidates)} claims contain a number, need {negatives} negatives"
        )
    for r in rng.sample(candidates, negatives):
        rows.append(
            {
                **r,
                "row_id": _row_id(r["claim_id"], True),
                "claim_id": r["claim_id"] + "-neg",
                "claim": perturb_number(r["claim"], rng),
                "synthetic": True,
            }
        )
    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl, cs = out_dir / "calib_claims.jsonl", out_dir / "calib_sheet.csv"
    if jl.exists() or cs.exists():
        raise FileExistsError(f"拒絕覆蓋:{out_dir}")
    with jl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with cs.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r["row_id"], r["question"], r["context"], r["claim"], "", ""])
    return {
        "answers": n,
        "claims": len(rows) - negatives,
        "negatives": negatives,
        "rows": len(rows),
    }


STRATA = ("verbatim", "paraphrase", "low", "abstention")
QUOTAS = {"verbatim": 4, "paraphrase": 12, "low": 12, "abstention": None}  # None = 全收


def stratum_of(claim: str, context: str, min_coverage: float) -> str:
    """abstention(含「未提及」,正規化後)優先;其餘依對各 chunk 的最大 coverage3 分三層。"""
    if "未提及" in normalize_variants(claim):
        return "abstention"
    cov = max((coverage3(claim, text) for _, text in split_chunks(context)), default=0.0)
    if cov >= min_coverage:
        return "verbatim"
    if cov >= 0.5:
        return "paraphrase"
    return "low"


def _pool(judge_jsons: list[Path], min_coverage: float) -> list[dict[str, Any]]:
    """所有非拒答主張,依(題 id, 正規化主張)去重,附 stratum。"""
    seen: set[tuple[str, str]] = set()
    pool: list[dict[str, Any]] = []
    for path in judge_jsons:
        for a in json.loads(path.read_text(encoding="utf-8"))["answers"]:
            if a["abstained"]:
                continue
            for c in a["claims"]:
                key = (str(a["id"]), normalize_variants(c["claim"]))
                if key in seen:
                    continue
                seen.add(key)
                pool.append(
                    {
                        # 同題同序號在不同格式可能文字不同:claim_id 綴來源檔名,row_id 才不撞號
                        "claim_id": f"{c['claim_id']}@{path.stem}",
                        "question": a["question"],
                        "context": a["context"],
                        "claim": c["claim"],
                        "stratum": stratum_of(c["claim"], a["context"], min_coverage),
                    }
                )
    return pool


def build_stratified(
    judge_jsons: list[Path], out_dir: Path, negatives: int, seed: int, min_coverage: float
) -> dict[str, int]:
    if not judge_jsons:
        raise ValueError("need at least one judge json")
    if negatives < 0:
        raise ValueError("negatives >= 0")
    pool = _pool(judge_jsons, min_coverage)
    quota_sum = sum(q for q in QUOTAS.values() if q is not None)
    if len(pool) < quota_sum / 2:
        raise ValueError(f"pool too small after dedup: {len(pool)} < {quota_sum / 2}")
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    info: dict[str, int] = {}
    for name in STRATA:
        members = [r for r in pool if r["stratum"] == name]
        q = QUOTAS[name]
        chosen = members if q is None or len(members) <= q else rng.sample(members, q)
        info[name] = len(chosen)
        picked.extend(chosen)
    rows = [{"row_id": _row_id(r["claim_id"], False), "synthetic": False, **r} for r in picked]
    candidates = [r for r in rows if perturb_number(r["claim"], random.Random(0)) is not None]
    if len(candidates) < negatives:
        raise ValueError(
            f"only {len(candidates)} claims contain a number, need {negatives} negatives"
        )
    for r in rng.sample(candidates, negatives):
        rows.append(
            {
                **r,
                "row_id": _row_id(r["claim_id"], True),
                "claim_id": r["claim_id"] + "-neg",
                "claim": perturb_number(r["claim"], rng),
                "synthetic": True,
                "stratum": "negative",
            }
        )
    rng.shuffle(rows)
    _write(out_dir, rows)
    info["negatives"] = negatives
    info["rows"] = len(rows)
    return info


def _write(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jl, cs = out_dir / "calib_claims.jsonl", out_dir / "calib_sheet.csv"
    if jl.exists() or cs.exists():
        raise FileExistsError(f"拒絕覆蓋:{out_dir}")
    with jl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with cs.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([r["row_id"], r["question"], r["context"], r["claim"], "", ""])


def score(sheet: Path, judge_json: Path) -> dict[str, Any]:
    human: dict[str, str] = {}
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            v = LABEL_ALIASES.get(
                (r.get("你的判定") or "").strip(), (r.get("你的判定") or "").strip()
            )
            if v:
                if v not in VERDICTS and v != "not_supported":
                    raise ValueError(f"row {r['row_id']}: bad label {v!r}")
                human[r["row_id"]] = v
    if not human:
        raise ValueError("sheet has no labels")
    binary = any(v == "not_supported" for v in human.values())
    claim_text = {
        r["row_id"]: r["claim"]
        for r in csv.DictReader(sheet.open(encoding="utf-8-sig", newline=""))
    }
    judged: dict[str, dict[str, Any]] = {}
    for a in json.loads(judge_json.read_text(encoding="utf-8"))["answers"]:
        for c in a["claims"]:
            judged[str(a["id"])] = c  # --claims 模式:每列一主張,id 即 claim_id
    # row_id ↔ claim_id 對照:jsonl 與 sheet 同目錄
    id_map = {}
    strata: dict[str, str] = {}
    for line in (sheet.parent / "calib_claims.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            id_map[r["row_id"]] = (r["claim_id"], bool(r["synthetic"]))
            if "stratum" in r:
                strata[r["row_id"]] = str(r["stratum"])
    pairs = []
    for row_id, hv in human.items():
        if row_id not in id_map:
            raise ValueError(f"unknown row_id {row_id}")
        cid, syn = id_map[row_id]
        jv_raw = judged.get(cid, {}).get("verdict")
        if jv_raw is None:
            continue  # judge 解析失敗的列不計
        hv2 = COLLAPSE[hv] if binary and hv in COLLAPSE else hv
        jv = COLLAPSE[jv_raw] if binary else jv_raw
        pairs.append((hv2, jv, syn, row_id, jv_raw))
    if not pairs:
        raise ValueError("no overlapping labeled rows")
    labels = ("supported", "not_supported") if binary else VERDICTS
    h = [p[0] for p in pairs]
    j = [p[1] for p in pairs]
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for hv, jv, *_ in pairs:
        confusion[hv][jv] += 1
    per_class_recall = {
        v: (confusion[v][v] / sum(confusion[v].values()) if sum(confusion[v].values()) else None)
        for v in labels
    }
    syn = [p for p in pairs if p[2]]
    neg = "not_supported" if binary else "contradicted"
    disagreements = [
        {
            "row_id": rid,
            "human": hv,
            "judge": jv_raw,
            "synthetic": s,
            "claim": claim_text.get(rid, "")[:60],
        }
        for hv, jv, s, rid, jv_raw in pairs
        if hv != jv
    ]
    result: dict[str, Any] = {}
    if strata:
        per: dict[str, dict[str, Any]] = {}
        for name in sorted({strata[p[3]] for p in pairs if p[3] in strata}):
            sub = [p for p in pairs if strata.get(p[3]) == name]
            hs, js = [p[0] for p in sub], [p[1] for p in sub]
            single = len(set(hs)) < 2 or len(set(js)) < 2
            per[name] = {"n": len(sub), "kappa": None if single else cohen_kappa(hs, js)}
        result["per_stratum"] = per
    return {
        **result,
        "n": len(pairs),
        "mode": "binary" if binary else "3-class",
        "kappa": cohen_kappa(h, j),
        "agreement": sum(1 for a, b in zip(h, j, strict=True) if a == b) / len(pairs),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
        "synthetic_total": len(syn),
        "synthetic_caught_by_judge": sum(1 for p in syn if p[1] == neg),
        "synthetic_caught_by_human": sum(1 for p in syn if p[0] == neg),
        "disagreements": disagreements,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("paths", type=Path, nargs="+", help="judge json… out_dir(最後一個為輸出目錄)")
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--negatives", type=int, default=10)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--stratify", action="store_true", help="方案 B 四層抽樣;可給多個 judge json")
    b.add_argument("--min-coverage", type=float, default=0.925)
    s = sub.add_parser("score")
    s.add_argument("sheet", type=Path)
    s.add_argument("judge_json", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        *judge_jsons, out_dir = args.paths
        if not judge_jsons or (not args.stratify and len(judge_jsons) != 1):
            print(
                "build 需要 judge_json 與 out_dir;多個 judge_json 只在 --stratify 下允許",
                file=sys.stderr,
            )
            return 2
        try:
            info = (
                build_stratified(judge_jsons, out_dir, args.negatives, args.seed, args.min_coverage)
                if args.stratify
                else build(judge_jsons[0], out_dir, args.n, args.negatives, args.seed)
            )
        except FileExistsError as e:
            print(e, file=sys.stderr)
            return 3
        print(" ".join(f"{k}={v}" for k, v in info.items()))
        print(f"輸出:{out_dir}")
        return 0
    r = score(args.sheet, args.judge_json)
    print(f"n={r['n']} mode={r['mode']} kappa={r['kappa']:.3f} agreement={r['agreement']:.3f}")
    print(f"per_class_recall={r['per_class_recall']}")
    print(
        f"synthetic: judge {r['synthetic_caught_by_judge']}/{r['synthetic_total']}, "
        f"human {r['synthetic_caught_by_human']}/{r['synthetic_total']}"
    )
    print(f"confusion={json.dumps(r['confusion'], ensure_ascii=False)}")
    for name, ps in r.get("per_stratum", {}).items():
        k = "n/a(單類)" if ps["kappa"] is None else f"{ps['kappa']:.3f}"
        print(f"  分層 {name:<11} n={ps['n']:<3} kappa={k}")
    for d in r["disagreements"]:
        tag = "合成" if d["synthetic"] else "真實"
        print(f"  不一致 {d['row_id']} [{tag}] 人={d['human']} judge={d['judge']} | {d['claim']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
