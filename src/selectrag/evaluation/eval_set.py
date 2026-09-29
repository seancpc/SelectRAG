"""eval set 的 schema、載入與凍結完整性檢查。

真值形式為 binary relevance(一組 doc id),與 retrieval_metrics 的介面相容。
壞資料一律 raise,延續指標模組的策略。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUIRED_TOP_LEVEL = ("manifest", "items")
_REQUIRED_MANIFEST = ("version", "frozen_at", "item_count", "content_hash")
_REQUIRED_ITEM = ("id", "question", "relevant_doc_ids")


class EvalSetIntegrityError(Exception):
    """凍結資訊(hash / 題數)與實際內容不符——代表檔案被改過。"""


@dataclass(frozen=True)
class EvalItem:
    """一道評測題目。relevant_doc_ids 即人工判定的真值。"""

    id: str
    question: str
    relevant_doc_ids: frozenset[str]
    reference_answer: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class EvalSet:
    """一份凍結的評測題組。"""

    version: str
    frozen_at: str
    items: tuple[EvalItem, ...]


def _to_dict(item: EvalItem) -> dict[str, Any]:
    """轉成可序列化字典;doc id 排序以確保表述唯一。"""
    return {
        "id": item.id,
        "question": item.question,
        "relevant_doc_ids": sorted(item.relevant_doc_ids),
        "reference_answer": item.reference_answer,
        "notes": item.notes,
    }


def compute_content_hash(items: Sequence[EvalItem]) -> str:
    """對正規化後的題目內容計算 sha256;與題目順序、JSON 鍵順序無關。"""
    canonical = json.dumps(
        [_to_dict(item) for item in sorted(items, key=lambda i: i.id)],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate(items: Sequence[EvalItem]) -> None:
    """結構驗證:壞題目一律拒絕,不靜默略過。"""
    if not items:
        raise ValueError("eval set is empty")
    seen: set[str] = set()
    for item in items:
        if not item.id.strip():
            raise ValueError("item id is empty")
        if item.id in seen:
            raise ValueError(f"duplicate item id: {item.id}")
        seen.add(item.id)
        if not item.question.strip():
            raise ValueError(f"question is empty for item {item.id}")
        if not item.relevant_doc_ids:
            raise ValueError(f"empty relevant_doc_ids for item {item.id}")


def freeze(items: Sequence[EvalItem], version: str, frozen_at: str) -> dict[str, Any]:
    """產生含 manifest 的可序列化結構。"""
    _validate(items)
    return {
        "manifest": {
            "version": version,
            "frozen_at": frozen_at,
            "item_count": len(items),
            "content_hash": compute_content_hash(items),
        },
        "items": [_to_dict(item) for item in items],
    }


def _require(mapping: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing field '{key}' in {where}")


def _parse_item(raw: Mapping[str, Any]) -> EvalItem:
    _require(raw, _REQUIRED_ITEM, "item")
    doc_ids = list(raw["relevant_doc_ids"])
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"duplicate doc id in relevant_doc_ids for item {raw['id']}")
    return EvalItem(
        id=raw["id"],
        question=raw["question"],
        relevant_doc_ids=frozenset(doc_ids),
        reference_answer=raw.get("reference_answer"),
        notes=raw.get("notes"),
    )


def load_eval_set(path: Path) -> EvalSet:
    """讀檔 → 結構驗證 → 凍結完整性驗證。任一關失敗即 raise。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload, _REQUIRED_TOP_LEVEL, "eval set file")
    manifest = payload["manifest"]
    _require(manifest, _REQUIRED_MANIFEST, "manifest")

    items = tuple(_parse_item(raw) for raw in payload["items"])
    _validate(items)

    if manifest["item_count"] != len(items):
        raise EvalSetIntegrityError(
            f"item_count mismatch: manifest says {manifest['item_count']}, file has {len(items)}"
        )
    actual_hash = compute_content_hash(items)
    if actual_hash != manifest["content_hash"]:
        raise EvalSetIntegrityError(
            f"content hash mismatch: manifest says {manifest['content_hash']}, "
            f"computed {actual_hash}"
        )
    return EvalSet(version=manifest["version"], frozen_at=manifest["frozen_at"], items=items)
