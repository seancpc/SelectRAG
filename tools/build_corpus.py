#!/usr/bin/env python3
"""把語料目錄轉成 chunk JSONL。

語料檔本身留在本機 data/corpus/(gitignore),不進 repo;
本腳本與產出的 chunk 檔都是可重現的中間產物。

用法:
    python3 tools/build_corpus.py data/corpus results/chunks.jsonl
    python3 tools/build_corpus.py data/corpus out.jsonl --csv-mode grouped --min-chars 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from selectrag.corpus.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP,
    chunk_csv_grouped,
    chunk_csv_per_row,
    chunk_document,
)
from selectrag.corpus.document import Chunk
from selectrag.corpus.loaders import load_csv_rows, load_html, load_pdf

SUFFIX_LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html}


def build_chunks(
    corpus_dir: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
    csv_mode: str = "per-row",
    group_size: int = 10,
) -> tuple[list[Chunk], list[tuple[str, str]]]:
    """走訪語料目錄產出 chunks;回傳 (chunks, 失敗清單)。

    單一檔案失敗不中止整批——但失敗會被收集並回報,不靜默略過。
    """
    if not corpus_dir.is_dir():
        raise ValueError(f"not a directory: {corpus_dir}")
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"no documents found in {corpus_dir}")

    chunks: list[Chunk] = []
    failures: list[tuple[str, str]] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in SUFFIX_LOADERS:
                document = SUFFIX_LOADERS[suffix](path)
                chunks.extend(chunk_document(document, max_chars, overlap, min_chars))
            elif suffix == ".csv":
                rows = load_csv_rows(path)
                if csv_mode == "grouped":
                    chunks.extend(chunk_csv_grouped(rows, path.stem, str(path), group_size))
                else:
                    chunks.extend(chunk_csv_per_row(rows, path.stem, str(path)))
        except Exception as exc:  # noqa: BLE001 — 逐檔隔離,失敗如實回報
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
    return chunks, failures


def write_jsonl(chunks: Sequence[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def summarise(chunks: Sequence[Chunk]) -> str:
    if not chunks:
        return "  (無 chunk)"
    sizes = sorted(len(c.text) for c in chunks)
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.source_type] = by_type.get(chunk.source_type, 0) + 1
    tiny = sum(1 for s in sizes if s < 50)
    lines = [
        f"  總計 {len(chunks)} chunks:" + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
        f"  長度 P10={sizes[len(sizes) // 10]} 中位={sizes[len(sizes) // 2]} "
        f"P90={sizes[len(sizes) * 9 // 10]} 最大={sizes[-1]}",
        f"  過短(<50 字元):{tiny} 個 ({tiny * 100 // len(sizes)}%)",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="語料目錄 → chunk JSONL")
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--csv-mode", choices=("per-row", "grouped"), default="per-row")
    parser.add_argument("--group-size", type=int, default=10)
    args = parser.parse_args(argv[1:])

    chunks, failures = build_chunks(
        args.corpus_dir,
        args.max_chars,
        args.overlap,
        args.min_chars,
        args.csv_mode,
        args.group_size,
    )
    write_jsonl(chunks, args.output)
    print(f"語料目錄:{args.corpus_dir}")
    print(summarise(chunks))
    print(f"  輸出:{args.output}")
    if failures:
        print(f"  ⚠ {len(failures)} 個檔案失敗:", file=sys.stderr)
        for name, reason in failures:
            print(f"    {name}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
