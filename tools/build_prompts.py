#!/usr/bin/env python3
"""B 軸 prompt 集建構(2026-09-15 核准)。

100 題經 A 軸推薦配置檢索 top-k,組成 chat messages 寫 jsonl。

用法(主 venv,桌機;需 dense 索引與 reranker 快取):
    python3 tools/build_prompts.py configs/prompt_v12.yaml results/prompts_v12.jsonl
設定檔:experiment_id / eval_set / retriever(與 runner 同格式)/ k_values,
外加 prompt.top_k / system / template / chunks
(template 必含 {question} 與 {context};模板只含結構與占位槽,chunk 正文於執行期填入)。
輸出每列:id / question / chunk_ids / messages。兩引擎共用同一份檔,檢索不在計時範圍。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from selectrag.evaluation.eval_set import load_eval_set
from selectrag.perf.bench import build_messages
from selectrag.retrieval.rerank import load_chunk_rows
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import load_config


def load_prompt_section(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    p = raw.get("prompt")
    if not p:
        raise ValueError("config needs a prompt section")
    for k in ("top_k", "system", "template", "chunks"):
        if k not in p:
            raise ValueError(f"prompt section missing '{k}'")
    if int(p["top_k"]) < 1:
        raise ValueError("prompt.top_k must be >= 1")
    return dict(p)


def build_rows(config_path: Path, retriever_builder: Any = build_retriever) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    p = load_prompt_section(config_path)
    rows_by_id = load_chunk_rows(Path(p["chunks"]))
    retriever = retriever_builder(cfg.retriever)
    out = []
    for item in load_eval_set(cfg.eval_set).items:
        ids = list(retriever.retrieve(item.question, int(p["top_k"])))
        texts = [str(rows_by_id[c]["text"]) for c in ids]
        out.append(
            {
                "id": item.id,
                "question": item.question,
                "chunk_ids": ids,
                "messages": build_messages(p["system"], p["template"], item.question, texts),
            }
        )
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"拒絕覆蓋:{args.out}", file=sys.stderr)
        return 3
    rows = build_rows(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["messages"][1]["content"]) for r in rows]
    print(
        f"prompts   : {len(rows)} 題;user 內容字元數 min/mean/max "
        f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}"
    )
    print(f"輸出      : {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
