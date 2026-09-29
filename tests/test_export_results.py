"""export_results:摘要入庫時絕不帶語料文字或模型輸出。文字皆為中性占位符。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path("tools") / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load_tool("export_results")


def test_strip_each_kind_drops_text_fields() -> None:
    runner = {
        "experiment_id": "e",
        "run_id": "r",
        "eval_set_hash": "h",
        "metrics": {"recall@10": 0.9},
        "config": {"k": 1},
        "per_query": [{"id": "q1", "retrieved": ["c1"]}],
    }
    bench = {
        "engine": "vllm",
        "format": "bf16",
        "summary": {"tok_per_s": 1.0},
        "requests": [{"text": "<回答>"}],
    }
    judge = {
        "summary": {"faithfulness": 0.9},
        "meta": {"prompt_version": "v2"},
        "answers": [{"context": "<段落>", "claims": []}],
    }
    other = {"exact_rate": 0.5, "first_divergence_char": {"q1": 3}}
    assert ex.strip(runner) == (
        {
            "experiment_id": "e",
            "run_id": "r",
            "eval_set_hash": "h",
            "config": {"k": 1},
            "metrics": {"recall@10": 0.9},
        },
        "runner",
    )
    assert ex.strip(bench) == (
        {"engine": "vllm", "format": "bf16", "summary": {"tok_per_s": 1.0}},
        "bench",
    )
    assert ex.strip(judge) == (
        {"summary": {"faithfulness": 0.9}, "meta": {"prompt_version": "v2"}},
        "judge",
    )
    assert ex.strip(other) == (other, "other")
    for rec in (runner, bench, judge):
        slim, _ = ex.strip(rec)
        assert not any(k in slim for k in ex.TEXT_KEYS)


def test_export_mirrors_tree_and_skips_lists(tmp_path: Path) -> None:
    src = tmp_path / "results"
    (src / "bench").mkdir(parents=True)
    (src / "bench" / "vllm_bf16_c1.json").write_text(
        json.dumps(
            {
                "engine": "vllm",
                "format": "bf16",
                "summary": {"n": 1},
                "requests": [{"text": "<回答>"}],
            }
        ),
        encoding="utf-8",
    )
    (src / "bench" / "probe.json").write_text(json.dumps([{"raw": "<模型輸出>"}]), encoding="utf-8")
    (src / "run").mkdir()
    (src / "run" / "v1.json").write_text(
        json.dumps(
            {"experiment_id": "e", "eval_set_hash": "h", "metrics": {"m": 1}, "per_query": [1]}
        ),
        encoding="utf-8",
    )
    dst = tmp_path / "out"
    counts = ex.export(src, dst)
    assert counts == {"bench": 1, "runner": 1}
    assert (dst / "bench" / "vllm_bf16_c1.json").exists() and not (
        dst / "bench" / "probe.json"
    ).exists()
    dumped = (dst / "bench" / "vllm_bf16_c1.json").read_text(encoding="utf-8")
    assert "<回答>" not in dumped and "requests" not in dumped
    assert ex.main([str(src), str(dst)]) == 0
    with pytest.raises(FileNotFoundError):
        ex.export(tmp_path / "nope", dst)
    (src / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        ex.export(src, dst)
