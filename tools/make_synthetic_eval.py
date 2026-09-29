#!/usr/bin/env python3
"""產生合成 eval set 與對應的假檢索結果。

用途:讓 runner pipeline 端到端跑通、讓 eval set 格式定案。
**這不是評測資料**——題目一律中性占位符,不含任何醫療內容,
分數沒有任何領域意義,僅供驗證流程正確。

產出決定性:同樣的程式碼永遠產生同樣的檔案(無隨機、無時間戳)。
用法:python3 tools/make_synthetic_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from selectrag.evaluation.eval_set import EvalItem, freeze

OUT_DIR = Path("data/eval")
EVAL_PATH = OUT_DIR / "eval_v0_synthetic.json"
SCRIPT_PATH = OUT_DIR / "eval_v0_synthetic_retrievals.json"
VERSION = "v0-synthetic"
FROZEN_AT = "2026-08-27"
ITEM_COUNT = 10


def build() -> tuple[list[EvalItem], dict[str, list[str]]]:
    """依固定規則產生題目與檢索腳本,讓各題的 recall / RR 各不相同。"""
    items: list[EvalItem] = []
    script: dict[str, list[str]] = {}

    for i in range(1, ITEM_COUNT + 1):
        question = f"<問題文字 {i:02d}>"
        relevant = [f"doc_{i:03d}_{j}" for j in range((i % 3) + 1)]

        # 前段雜訊 → 部分真值(有時刻意漏抓)→ 尾段雜訊
        noise = [f"doc_noise_{i:03d}_{j}" for j in range(i % 4)]
        hits = relevant[: max(0, len(relevant) - (i % 2))]
        tail = [f"doc_tail_{i:03d}_{j}" for j in range(3)]

        items.append(
            EvalItem(
                id=f"q{i:03d}",
                question=question,
                relevant_doc_ids=frozenset(relevant),
                reference_answer=f"<參考答案 {i:02d}>",
                notes="合成資料,非真實內容",
            )
        )
        script[question] = noise + hits + tail

    return items, script


def main() -> int:
    items, script = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = freeze(items, version=VERSION, frozen_at=FROZEN_AT)
    EVAL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SCRIPT_PATH.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已產生 {EVAL_PATH}({len(items)} 題,{VERSION})")
    print(f"已產生 {SCRIPT_PATH}({len(script)} 筆檢索腳本)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
