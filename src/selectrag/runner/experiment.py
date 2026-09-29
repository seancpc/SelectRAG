"""實驗執行與結果持久化。

結果檔是實驗的唯一紀錄:必須記錄 eval set 身分,且絕不覆蓋既有檔案。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from selectrag.evaluation.eval_set import EvalSet, compute_content_hash
from selectrag.evaluation.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank
from selectrag.retrieval.base import Retriever
from selectrag.runner.config import ExperimentConfig, RetrieverConfig


@dataclass(frozen=True)
class QueryResult:
    """單題的檢索結果與分數。"""

    item_id: str
    retrieved: tuple[str, ...]
    recall: Mapping[str, float]
    hit: Mapping[str, float]
    reciprocal_rank: float


@dataclass(frozen=True)
class ExperimentResult:
    """一次實驗的完整結果;eval_set_version 與 hash 是可比性的憑據。"""

    experiment_id: str
    run_id: str
    eval_set_version: str
    eval_set_hash: str
    config: ExperimentConfig
    metrics: Mapping[str, float]
    per_query: tuple[QueryResult, ...]


def run_experiment(
    config: ExperimentConfig, eval_set: EvalSet, retriever: Retriever, run_id: str
) -> ExperimentResult:
    """逐題檢索並計分;任一題失敗即中止,不產出部分結果。"""
    depth = max(config.k_values)
    per_query = []
    for item in eval_set.items:
        retrieved = tuple(retriever.retrieve(item.question, top_k=depth))
        per_query.append(
            QueryResult(
                item_id=item.id,
                retrieved=retrieved,
                recall={
                    f"recall@{k}": recall_at_k(retrieved, item.relevant_doc_ids, k)
                    for k in config.k_values
                },
                hit={
                    f"hit@{k}": hit_at_k(retrieved, item.relevant_doc_ids, k)
                    for k in config.k_values
                },
                reciprocal_rank=reciprocal_rank(retrieved, item.relevant_doc_ids),
            )
        )

    metrics: dict[str, float] = {
        f"recall@{k}": fmean(q.recall[f"recall@{k}"] for q in per_query) for k in config.k_values
    }
    for k in config.k_values:
        metrics[f"hit@{k}"] = fmean(q.hit[f"hit@{k}"] for q in per_query)
    metrics["mrr"] = fmean(q.reciprocal_rank for q in per_query)

    return ExperimentResult(
        experiment_id=config.experiment_id,
        run_id=run_id,
        eval_set_version=eval_set.version,
        eval_set_hash=compute_content_hash(eval_set.items),
        config=config,
        metrics=metrics,
        per_query=tuple(per_query),
    )


def _to_dict(result: ExperimentResult) -> dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "run_id": result.run_id,
        "eval_set_version": result.eval_set_version,
        "eval_set_hash": result.eval_set_hash,
        "config": {
            "experiment_id": result.config.experiment_id,
            "eval_set": str(result.config.eval_set),
            "retriever": {
                "type": result.config.retriever.type,
                "params": dict(result.config.retriever.params),
            },
            "k_values": list(result.config.k_values),
            "results_dir": str(result.config.results_dir),
        },
        "metrics": dict(result.metrics),
        "per_query": [
            {
                "item_id": q.item_id,
                "retrieved": list(q.retrieved),
                "recall": dict(q.recall),
                "hit": dict(q.hit),
                "reciprocal_rank": q.reciprocal_rank,
            }
            for q in result.per_query
        ],
    }


def write_result(result: ExperimentResult, results_dir: Path) -> Path:
    """寫出結果檔;檔案已存在即 raise,絕不覆蓋。"""
    directory = results_dir / result.experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.run_id}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.write_text(json.dumps(_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_result(path: Path) -> ExperimentResult:
    """讀回結果檔(供驗證與報告使用)。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    config_raw = raw["config"]
    return ExperimentResult(
        experiment_id=raw["experiment_id"],
        run_id=raw["run_id"],
        eval_set_version=raw["eval_set_version"],
        eval_set_hash=raw["eval_set_hash"],
        config=ExperimentConfig(
            experiment_id=config_raw["experiment_id"],
            eval_set=Path(config_raw["eval_set"]),
            retriever=RetrieverConfig(
                type=config_raw["retriever"]["type"], params=config_raw["retriever"]["params"]
            ),
            k_values=tuple(config_raw["k_values"]),
            results_dir=Path(config_raw["results_dir"]),
        ),
        metrics=raw["metrics"],
        per_query=tuple(
            QueryResult(
                item_id=q["item_id"],
                retrieved=tuple(q["retrieved"]),
                recall=q["recall"],
                hit=q["hit"],
                reciprocal_rank=q["reciprocal_rank"],
            )
            for q in raw["per_query"]
        ),
    )
