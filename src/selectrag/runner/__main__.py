"""實驗 runner 的命令列入口。

用法:python -m selectrag.runner <設定檔> [run_id]
這就是「一行指令可重跑」的入口。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from selectrag.evaluation.eval_set import load_eval_set
from selectrag.retrieval.base import Retriever
from selectrag.retrieval.bm25 import load_bm25_from_jsonl
from selectrag.retrieval.dense import QueryEmbedder, load_dense_from_index
from selectrag.retrieval.hybrid import HybridRRF
from selectrag.retrieval.neighbor import NeighborExpand, load_chunk_order
from selectrag.retrieval.rerank import Reranker, RerankRetriever, load_chunk_rows
from selectrag.retrieval.scripted import ScoredScriptedRetriever, ScriptedRetriever
from selectrag.runner.config import ExperimentConfig, RetrieverConfig, load_config
from selectrag.runner.experiment import ExperimentResult, run_experiment, write_result

EmbedderFactory = Callable[[str, str | None], QueryEmbedder]
RerankerFactory = Callable[[str, str | None], Reranker]


def _default_embedder(model: str, revision: str | None) -> QueryEmbedder:  # pragma: no cover
    from selectrag.retrieval.dense import SentenceTransformerEmbedder  # 僅桌機有此套件

    return SentenceTransformerEmbedder(model, revision)


def _default_reranker(model: str, revision: str | None) -> Reranker:  # pragma: no cover
    from selectrag.retrieval.rerank import CrossEncoderReranker  # 僅桌機有此套件

    return CrossEncoderReranker(model, revision)


def build_retriever(
    config: RetrieverConfig,
    embedder_factory: EmbedderFactory = _default_embedder,
    reranker_factory: RerankerFactory = _default_reranker,
) -> Retriever:
    """依設定建出檢索器;未知型別即 raise,不預設回退到任何實作。

    embedder_factory 只在 type=dense 時使用,測試以替身注入,避免載入真模型。
    params.neighbor_window(選填,預設 0)對 bm25 / dense 皆可用;>0 時以 NeighborExpand 包裝,
    順序表來自 params.chunks(dense 也必須提供)。
    """
    window = int(config.params.get("neighbor_window", 0))
    chunks = config.params.get("chunks")
    if window > 0 and not chunks:  # 先驗參數,再建 base——缺參數不該等到索引載入後才發現
        raise ValueError("neighbor_window > 0 requires params.chunks (path to chunks.jsonl)")
    base = _build_base(config, embedder_factory, reranker_factory)
    if window == 0:
        return base
    return NeighborExpand(base, load_chunk_order(Path(str(chunks))), window)


def _build_base(
    config: RetrieverConfig, embedder_factory: EmbedderFactory, reranker_factory: RerankerFactory
) -> Retriever:
    if config.type == "scripted":
        script_path = config.params.get("script")
        if not script_path:
            raise ValueError("scripted retriever requires params.script")
        script = json.loads(Path(script_path).read_text(encoding="utf-8"))
        if config.params.get("scored"):
            return ScoredScriptedRetriever(script)
        return ScriptedRetriever(script)
    if config.type == "bm25":
        chunks = config.params.get("chunks")
        if not chunks:
            raise ValueError("bm25 retriever requires params.chunks (path to chunks.jsonl)")
        context_fields = tuple(config.params.get("context_fields") or ())
        return load_bm25_from_jsonl(
            Path(chunks),
            context_fields=context_fields,
            k1=float(config.params.get("k1", 1.5)),
            b=float(config.params.get("b", 0.75)),
        )
    if config.type == "dense":
        index = config.params.get("index")
        if not index:
            raise ValueError("dense retriever requires params.index (path to .npz)")
        model = config.params.get("model")
        if not model:
            raise ValueError("dense retriever requires params.model (must match index metadata)")
        revision = config.params.get("revision")
        return load_dense_from_index(
            Path(index), embedder_factory(str(model), revision), str(model), revision
        )
    if config.type == "hybrid":
        nested = config.params.get("retrievers")
        if not nested or not isinstance(nested, list):
            raise ValueError(
                "hybrid retriever requires params.retrievers (list of retriever configs)"
            )
        if "pool" not in config.params:
            raise ValueError("hybrid retriever requires params.pool")
        if "rrf_k" not in config.params:
            raise ValueError("hybrid retriever requires params.rrf_k")
        bases = [
            build_retriever(
                RetrieverConfig(type=str(sub["type"]), params=dict(sub.get("params") or {})),
                embedder_factory,
                reranker_factory,
            )
            for sub in nested
        ]
        return HybridRRF(bases, int(config.params["pool"]), float(config.params["rrf_k"]))
    if config.type == "rerank":
        for key in ("retriever", "candidates", "model", "chunks"):
            if key not in config.params:
                raise ValueError(f"rerank retriever requires params.{key}")
        sub = config.params["retriever"]
        base = build_retriever(
            RetrieverConfig(type=str(sub["type"]), params=dict(sub.get("params") or {})),
            embedder_factory,
            reranker_factory,
        )
        model = str(config.params["model"])
        revision = config.params.get("revision")
        return RerankRetriever(
            base,
            reranker_factory(model, revision),
            load_chunk_rows(Path(str(config.params["chunks"]))),
            int(config.params["candidates"]),
            tuple(config.params.get("context_fields") or ()),
        )
    raise ValueError(f"unknown retriever type: {config.type!r}")


def format_report(result: ExperimentResult, path: Path) -> str:
    """把結果整理成人可讀的摘要。"""
    lines = [
        f"實驗 {result.experiment_id} / run {result.run_id}",
        f"  eval set : {result.eval_set_version}(hash {result.eval_set_hash[:12]}…)",
        f"  題數     : {len(result.per_query)}",
        f"  檢索器   : {result.config.retriever.type}",
        "  指標:",
    ]
    for name, value in result.metrics.items():
        lines.append(f"    {name:<12} = {value:.4f}")
    lines.append(f"  結果檔   : {path}")
    return "\n".join(lines)


def run(config: ExperimentConfig, run_id: str) -> tuple[ExperimentResult, Path]:
    eval_set = load_eval_set(config.eval_set)
    retriever = build_retriever(config.retriever)
    result = run_experiment(config, eval_set, retriever, run_id=run_id)
    return result, write_result(result, config.results_dir)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法:python -m selectrag.runner <設定檔> [run_id]", file=sys.stderr)
        return 2
    config = load_config(Path(argv[1]))
    run_id = argv[2] if len(argv) > 2 else datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result, path = run(config, run_id)
    print(format_report(result, path))
    return 0


if __name__ == "__main__":  # pragma: no cover — 模組守衛,測試經 main() 直接呼叫
    raise SystemExit(main(sys.argv))
