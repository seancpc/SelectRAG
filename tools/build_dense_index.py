#!/usr/bin/env python3
"""桌機離線建 dense 索引(A 軸對照組;2026-09-13 核准)。

用法(主 venv):
    python3 tools/build_dense_index.py --chunks results/chunks.jsonl \\
        --out results/dense_index/bge_m3_plain.npz --embedder bge-m3
    python3 tools/build_dense_index.py --chunks results/chunks.jsonl \\
        --out results/dense_index/bge_m3_ctx.npz --embedder bge-m3 \\
        --context-fields doc_title heading_path
    python3 tools/build_dense_index.py ... --embedder fake      # 無 GPU 的 smoke(gauntlet 5e)

索引文本組裝與 BM25 共用 selectrag.retrieval.bm25.index_text,兩組對照的脈絡語意完全相同。
首跑未帶 --revision 時,metadata 記 revision_pinned=false 並印出解析到的 commit,
請釘回 DEFAULT_REVISION。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from selectrag.retrieval.bm25 import index_text
from selectrag.retrieval.dense import IndexMetadata, QueryEmbedder, save_index

BGE_M3 = "BAAI/bge-m3"
# 2026-09-14 桌機首跑解析,釘死
DEFAULT_REVISION: str | None = "5617a9f61b028005a4858fdac845db406aefb181"
FAKE_DIM = 16
BATCH = 64


class FakeHashEmbedder:
    """決定性替身:sha256(文字)展開成 FAKE_DIM 維並 L2 normalize;僅供無 GPU 的 smoke。"""

    def embed(self, texts: Sequence[str]) -> Any:
        rows = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            raw = np.frombuffer(digest[:FAKE_DIM], dtype=np.uint8).astype(np.float32) - 127.5
            rows.append(raw / np.linalg.norm(raw))
        return np.asarray(rows, dtype=np.float32)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path, context_fields: tuple[str, ...]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("chunk_id", "text"):
                if field not in row:
                    raise ValueError(f"{path.name} line {line_no}: missing field '{field}'")
            ids.append(row["chunk_id"])
            texts.append(index_text(row, context_fields, f"{path.name} line {line_no}"))
    return ids, texts


def make_embedder(name: str, revision: str | None) -> tuple[QueryEmbedder, str, str | None]:
    """回傳 (embedder, model 名, 解析到的 revision);fake 不涉及任何模型。"""
    if name == "fake":
        return FakeHashEmbedder(), "fake", None
    if name == "bge-m3":
        from selectrag.retrieval.dense import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(BGE_M3, revision)
        return emb, BGE_M3, revision or _resolve_commit(BGE_M3)
    raise ValueError(f"unknown embedder {name!r}; use bge-m3 or fake")


def _resolve_commit(model: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        name = Path(snapshot_download(model, local_files_only=True)).name
        return name if len(name) == 40 else None
    except Exception:  # 純資訊性
        return None


def embed_all(embedder: QueryEmbedder, texts: list[str]) -> Any:
    parts = [np.asarray(embedder.embed(texts[i : i + BATCH])) for i in range(0, len(texts), BATCH)]
    return np.concatenate(parts, axis=0).astype(np.float32)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedder", choices=("bge-m3", "fake"), required=True)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--context-fields", nargs="*", default=[])
    args = ap.parse_args(argv)

    if args.out.exists():
        print(f"拒絕覆蓋既有索引:{args.out}", file=sys.stderr)
        return 3
    fields = tuple(args.context_fields)
    ids, texts = read_rows(args.chunks, fields)
    if not ids:
        print(f"無 chunk 可索引:{args.chunks}", file=sys.stderr)
        return 2

    embedder, model, revision = make_embedder(args.embedder, args.revision)
    t0 = time.perf_counter()
    vectors = embed_all(embedder, texts)
    elapsed = time.perf_counter() - t0
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    meta = IndexMetadata(
        model=model,
        revision=revision,
        dim=int(vectors.shape[1]),
        context_fields=fields,
        chunks_sha256=sha256_of(args.chunks),
        normalized=True,
        revision_pinned=args.embedder != "fake" and args.revision is not None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_index(args.out, ids, vectors, meta)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"索引     : {args.out}({size_mb:.1f} MB)")
    print(f"模型     : {model} @ {revision or '(未釘,請釘回 DEFAULT_REVISION)'}")
    print(f"chunks   : {len(ids)} 筆,dim={meta.dim},context_fields={list(fields)}")
    print(f"embedding: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
