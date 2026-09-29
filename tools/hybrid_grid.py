#!/usr/bin/env python3
"""hybrid RRF 網格 + 聯集上限(A 軸步驟 1;2026-09-14 核准)。

用法(桌機,主 venv;需 dense 索引與 bge-m3):
    python3 tools/hybrid_grid.py configs/bm25_ctx_tuned_full100.yaml \\
        configs/dense_bge_m3_ctx_full100.yaml results/hybrid-grid/v11 \\
        --pools 10 20 40 --rrf-ks 10 60 100

兩份 base config 必須指向同一份 eval set。輸出:
  <out>/pool{P}_k{K}.json — 每格一份 runner 結果檔(既有即拒絕覆蓋)
  <out>/union_bound.json  — 各深度 N 的 A / B / 聯集 recall@N、hit@N
                            (pre-mortem 要的「單邊放大池」對照)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selectrag.evaluation.eval_set import load_eval_set
from selectrag.evaluation.retrieval_metrics import hit_at_k, recall_at_k
from selectrag.retrieval.hybrid import HybridRRF
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import ExperimentConfig, RetrieverConfig, load_config
from selectrag.runner.experiment import run_experiment, write_result

DEFAULT_POOLS = (10, 20, 40)
DEFAULT_RRF_KS = (10, 60, 100)


def _per_query(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {q["item_id"]: list(q["retrieved"]) for q in data["per_query"]}


def union_bound(
    truth: dict[str, set[str]], result_a: Path, result_b: Path, depths: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    """各深度 N:A、B 各自的 recall@N / hit@N,與兩者 top-N 聯集的 recall / hit(理論上限)。"""
    a, b = _per_query(result_a), _per_query(result_b)
    out: dict[int, dict[str, float]] = {}
    for n in depths:
        acc = {
            k: 0.0 for k in ("a_recall", "b_recall", "union_recall", "a_hit", "b_hit", "union_hit")
        }
        for qid, rel in truth.items():
            ra, rb = a[qid][:n], b[qid][:n]
            union = list(dict.fromkeys([*ra, *rb]))
            acc["a_recall"] += recall_at_k(ra, rel, n)
            acc["b_recall"] += recall_at_k(rb, rel, n)
            acc["union_recall"] += recall_at_k(union, rel, len(union) or 1)
            acc["a_hit"] += hit_at_k(ra, rel, n)
            acc["b_hit"] += hit_at_k(rb, rel, n)
            acc["union_hit"] += hit_at_k(union, rel, len(union) or 1)
        out[n] = {k: round(v / len(truth), 4) for k, v in acc.items()}
    return out


def run_grid(
    cfg_a: ExperimentConfig,
    cfg_b: ExperimentConfig,
    out: Path,
    pools: tuple[int, ...],
    rrf_ks: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    eval_set = load_eval_set(cfg_a.eval_set)
    base_a = build_retriever(cfg_a.retriever)
    base_b = build_retriever(cfg_b.retriever)
    table: dict[str, dict[str, float]] = {}
    for pool in pools:
        for rrf_k in rrf_ks:
            name = f"pool{pool}_k{rrf_k:g}"
            cfg = ExperimentConfig(
                experiment_id=f"hybrid-{cfg_a.experiment_id}+{cfg_b.experiment_id}",
                eval_set=cfg_a.eval_set,
                retriever=RetrieverConfig(
                    "hybrid",
                    {
                        "pool": pool,
                        "rrf_k": rrf_k,
                        "retrievers": [
                            {"type": cfg_a.retriever.type, "params": dict(cfg_a.retriever.params)},
                            {"type": cfg_b.retriever.type, "params": dict(cfg_b.retriever.params)},
                        ],
                    },
                ),
                k_values=cfg_a.k_values,
                results_dir=out,
            )
            result = run_experiment(
                cfg, eval_set, HybridRRF([base_a, base_b], pool, rrf_k), run_id=name
            )
            path = out / name
            path = write_result(
                result, out.parent
            )  # 先寫到 <out.parent>/<experiment_id>/<name>.json
            final = out / f"{name}.json"
            path.replace(final)
            table[name] = dict(result.metrics)
    return table


def write_union_bound(
    cfg_a: ExperimentConfig, cfg_b: ExperimentConfig, out: Path, depths: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    """兩個 base 各以最深 depth 跑一次,算各深度的單邊與聯集 recall / hit,寫 union_bound.json。"""
    eval_set = load_eval_set(cfg_a.eval_set)
    depth = max(depths)
    paths = []
    for cfg in (cfg_a, cfg_b):
        deep = ExperimentConfig(
            experiment_id=f"base-{cfg.experiment_id}",
            eval_set=cfg.eval_set,
            retriever=cfg.retriever,
            k_values=tuple(sorted({*cfg.k_values, depth})),
            results_dir=out.parent,
        )
        result = run_experiment(
            deep, eval_set, build_retriever(cfg.retriever), run_id=f"depth{depth}"
        )
        paths.append(write_result(result, out.parent))
    truth = {item.id: set(item.relevant_doc_ids) for item in eval_set.items}
    ub = union_bound(truth, paths[0], paths[1], depths)
    (out / "union_bound.json").write_text(
        json.dumps({str(n): m for n, m in ub.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ub


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_a", type=Path)
    ap.add_argument("config_b", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pools", type=int, nargs="+", default=list(DEFAULT_POOLS))
    ap.add_argument("--rrf-ks", type=float, nargs="+", default=list(DEFAULT_RRF_KS))
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"拒絕覆蓋:{args.out} 已有內容", file=sys.stderr)
        return 3
    cfg_a, cfg_b = load_config(args.config_a), load_config(args.config_b)
    if cfg_a.eval_set != cfg_b.eval_set:
        raise ValueError("two base configs must share the same eval_set")
    args.out.mkdir(parents=True, exist_ok=True)
    table = run_grid(cfg_a, cfg_b, args.out, tuple(args.pools), tuple(args.rrf_ks))
    ub = write_union_bound(cfg_a, cfg_b, args.out, tuple(args.pools))
    print(f"{'深度 N':<8}{'A recall':>10}{'B recall':>10}{'聯集 recall':>12}{'聯集 hit':>10}")
    for n, m in ub.items():
        print(
            f"{n:<8}{m['a_recall']:>10.4f}{m['b_recall']:>10.4f}{m['union_recall']:>12.4f}{m['union_hit']:>10.2f}"
        )
    print(f"{'格點':<14}{'recall@3':>10}{'recall@10':>11}{'hit@10':>8}{'mrr':>8}")
    for name, m in table.items():
        print(
            f"{name:<14}{m.get('recall@3', float('nan')):>10.4f}"
            f"{m.get('recall@10', float('nan')):>11.4f}"
            f"{m.get('hit@10', float('nan')):>8.2f}{m['mrr']:>8.4f}"
        )
    print(f"輸出:{args.out}/pool*_k*.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
