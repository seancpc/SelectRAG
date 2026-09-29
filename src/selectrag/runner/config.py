"""實驗設定檔的載入與驗證。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP_LEVEL = ("experiment_id", "eval_set", "retriever", "k_values", "results_dir")
_REQUIRED_RETRIEVER = ("type", "params")


@dataclass(frozen=True)
class RetrieverConfig:
    """檢索器的類型與參數;type 決定 runner 要建哪個實作。"""

    type: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    """一次實驗的完整設定——結果檔會原樣記錄它,確保設定與數字對得起來。"""

    experiment_id: str
    eval_set: Path
    retriever: RetrieverConfig
    k_values: tuple[int, ...]
    results_dir: Path


def _require(mapping: Mapping[str, Any], keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing field '{key}' in {where}")


def load_config(path: Path) -> ExperimentConfig:
    """讀 YAML 設定檔並驗證;缺欄位或參數不合法即 raise。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(raw, _REQUIRED_TOP_LEVEL, "config")
    _require(raw["retriever"], _REQUIRED_RETRIEVER, "retriever config")

    experiment_id = str(raw["experiment_id"])
    if not experiment_id.strip():
        raise ValueError("experiment_id is empty")

    k_values = tuple(int(k) for k in raw["k_values"])
    if not k_values:
        raise ValueError("k_values is empty")
    if any(k < 1 for k in k_values):
        raise ValueError("k must be >= 1")

    return ExperimentConfig(
        experiment_id=experiment_id,
        eval_set=Path(raw["eval_set"]),
        retriever=RetrieverConfig(
            type=str(raw["retriever"]["type"]), params=dict(raw["retriever"]["params"] or {})
        ),
        k_values=k_values,
        results_dir=Path(raw["results_dir"]),
    )
