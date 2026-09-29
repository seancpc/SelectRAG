#!/usr/bin/env python3
"""reranker 網格 + hard negative 檢核(A 軸步驟 2;2026-09-14 核准)。

用法(桌機,主 venv):
    python3 tools/rerank_grid.py configs/hybrid_rrf_full100.yaml results/chunks.jsonl \\
        results/rerank-grid/v11 --candidates 10 20 30 --reranker bge \\
        --hard-negatives data/eval/hard_negatives.json
    --reranker fake:雜湊替身(無 GPU 的 smoke,gauntlet 5g)
    --reranker nim --nim-url http://localhost:8000 --nim-model <模型名>:reranking NIM(時間盒)

輸出:<out>/cand{N}_{plain|ctx}.json(拒絕覆蓋)、<out>/hard_negative_check.json、
      每格印 recall@3 / recall@10 / hit@10 / MRR 與每題平均重排耗時(ms)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from selectrag.evaluation.eval_set import load_eval_set
from selectrag.retrieval.bm25 import index_text
from selectrag.retrieval.rerank import NIMReranker, Reranker, RerankRetriever, load_chunk_rows
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import ExperimentConfig, RetrieverConfig, load_config
from selectrag.runner.experiment import run_experiment, write_result

BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_REVISION: str | None = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # 桌機首跑 2026-09-14
CTX_VARIANTS: dict[str, tuple[str, ...]] = {"plain": (), "ctx": ("doc_title",)}


class FakeHashReranker:
    """決定性替身:分數 = sha256(question|text) 前 4 bytes;僅供無 GPU 的 smoke。"""

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        return [
            int.from_bytes(hashlib.sha256(f"{question}|{t}".encode()).digest()[:4], "big") / 2**32
            for t in texts
        ]


def make_reranker(
    name: str, revision: str | None, nim_url: str = "", nim_model: str = ""
) -> tuple[Reranker, str, str | None]:
    if name == "fake":
        return FakeHashReranker(), "fake", None
    if name == "nim":
        if not nim_url or not nim_model:
            raise ValueError("--reranker nim 需要 --nim-url 與 --nim-model")
        return NIMReranker(nim_url, nim_model), f"nim:{nim_model}", revision
    if name == "bge":
        from selectrag.retrieval.rerank import CrossEncoderReranker

        return (
            CrossEncoderReranker(BGE_RERANKER, revision),
            BGE_RERANKER,
            revision or _resolve_commit(),
        )
    raise ValueError(f"unknown reranker {name!r}; use bge, nim or fake")


def _resolve_commit() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        name = Path(snapshot_download(BGE_RERANKER, local_files_only=True)).name
        return name if len(name) == 40 else None
    except Exception:  # 純資訊性
        return None


def hard_negative_check(
    eval_path: Path,
    hard_negatives: Path,
    rows: Mapping[str, Mapping[str, Any]],
    reranker: Reranker,
    context_fields: tuple[str, ...],
) -> dict[str, Any]:
    """每筆干擾項:該題真值中的最高 reranker 分數是否 > 干擾項分數;回傳比例與明細。"""
    truth = {
        item.id: (item.question, sorted(item.relevant_doc_ids))
        for item in load_eval_set(eval_path).items
    }
    items = json.loads(hard_negatives.read_text(encoding="utf-8"))["items"]
    details = []
    for hn in items:
        qid = hn["question_id"]
        if qid not in truth:
            raise ValueError(f"hard negative refers to unknown question {qid!r}")
        question, rel = truth[qid]
        texts = [index_text(dict(rows[c]), context_fields, c) for c in [*rel, hn["chunk_id"]]]
        scores = list(reranker.score(question, texts))
        best_truth, neg = max(scores[:-1]), scores[-1]
        details.append(
            {
                "question_id": qid,
                "chunk_id": hn["chunk_id"],
                "type": hn.get("type"),
                "best_truth_score": best_truth,
                "negative_score": neg,
                "truth_wins": best_truth > neg,
            }
        )
    wins = sum(1 for d in details if d["truth_wins"])
    by_type: dict[str, dict[str, Any]] = {}
    for d in details:
        t = str(d["type"])
        agg = by_type.setdefault(t, {"total": 0, "truth_wins": 0})
        agg["total"] += 1
        agg["truth_wins"] += int(d["truth_wins"])
    for agg in by_type.values():
        agg["ratio"] = agg["truth_wins"] / agg["total"]
    return {
        "total": len(details),
        "truth_wins": wins,
        "ratio": wins / len(details) if details else 0.0,
        "by_type": dict(sorted(by_type.items())),
        "details": details,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--candidates", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--reranker", choices=("bge", "nim", "fake"), default="bge")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--nim-url", default="", help="NIM base URL,例 http://localhost:8000")
    ap.add_argument(
        "--nim-model", default="", help="NIM 模型名(釘 image tag/digest 記在 --revision)"
    )
    ap.add_argument("--hard-negatives", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"拒絕覆蓋:{args.out} 已有內容", file=sys.stderr)
        return 3
    args.out.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base_config)
    eval_set = load_eval_set(base_cfg.eval_set)
    rows = load_chunk_rows(args.chunks)
    base = build_retriever(base_cfg.retriever)
    reranker, model, revision = make_reranker(
        args.reranker, args.revision, args.nim_url, args.nim_model
    )
    print(f"reranker : {model} @ {revision or '(未釘)'}")

    print(f"{'格點':<14}{'recall@3':>10}{'recall@10':>11}{'hit@10':>8}{'mrr':>8}{'ms/題':>8}")
    for cand in args.candidates:
        for tag, fields in CTX_VARIANTS.items():
            name = f"cand{cand}_{tag}"
            cfg = ExperimentConfig(
                experiment_id=f"rerank-{base_cfg.experiment_id}",
                eval_set=base_cfg.eval_set,
                retriever=RetrieverConfig(
                    "rerank",
                    {
                        "candidates": cand,
                        "model": model,
                        "revision": revision,
                        "chunks": str(args.chunks),
                        "context_fields": list(fields),
                        "retriever": {
                            "type": base_cfg.retriever.type,
                            "params": dict(base_cfg.retriever.params),
                        },
                    },
                ),
                k_values=base_cfg.k_values,
                results_dir=args.out.parent,
            )
            retriever = RerankRetriever(base, reranker, rows, cand, fields)
            t0 = time.perf_counter()
            result = run_experiment(cfg, eval_set, retriever, run_id=name)
            ms = (time.perf_counter() - t0) * 1000 / len(eval_set.items)
            write_result(result, args.out.parent).replace(args.out / f"{name}.json")
            m = result.metrics
            print(
                f"{name:<14}{m.get('recall@3', float('nan')):>10.4f}"
                f"{m.get('recall@10', float('nan')):>11.4f}"
                f"{m.get('hit@10', float('nan')):>8.2f}{m['mrr']:>8.4f}{ms:>8.0f}"
            )

    if args.hard_negatives:
        for tag, fields in CTX_VARIANTS.items():
            report = hard_negative_check(
                base_cfg.eval_set, args.hard_negatives, rows, reranker, fields
            )
            (args.out / f"hard_negative_check_{tag}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"hard negatives({tag}):真值勝 {report['truth_wins']}/{report['total']}"
                f" = {report['ratio']:.2f}"
            )
            for t, agg in report["by_type"].items():
                print(f"    {t:<32}{agg['truth_wins']:>3}/{agg['total']:<3} = {agg['ratio']:.2f}")
    print(f"輸出:{args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
