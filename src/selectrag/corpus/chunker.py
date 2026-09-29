"""切塊:結構優先 + 長度上限(方案 C)。

先依標題階層切出語意完整的單元,超過上限者再以固定長度切分並保留
heading_path,兼顧 citation 精確度(§5.3)與 chunk 大小可控。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from selectrag.corpus.document import Chunk, Document
from selectrag.corpus.headings import detect_style, parse_heading
from selectrag.corpus.normalize import normalize_text

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 100
# 下限:實測顯示純結構切塊會產生 47~62% 短於 50 字元的碎片,
# 那種長度的 chunk 語意不完整,會讓 dense 檢索失去鑑別度
DEFAULT_MIN_CHARS = 200


def _make_id(doc_id: str, index: int, text: str) -> str:
    """id 綁定內容,同一輸入重跑必得相同結果。"""
    digest = hashlib.sha256(f"{doc_id}|{index}|{text}".encode()).hexdigest()
    return f"{doc_id}#{index:04d}#{digest[:8]}"


def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
    """把超長文字切成有重疊的片段。"""
    step = max_chars - overlap
    return [text[i : i + max_chars] for i in range(0, len(text), step) if text[i : i + max_chars]]


def _page_of(offsets: Sequence[int], position: int) -> int | None:
    """由字元位置反查頁碼(1 起算),供 citation 使用。"""
    if not offsets:
        return None
    page = 1
    for index, start in enumerate(offsets, start=1):
        if position >= start:
            page = index
    return page


def _common_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """兩條章節路徑的共同前綴——合併後的 chunk 只能宣稱屬於共同祖先。"""
    shared: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        shared.append(a)
    return tuple(shared)


def _validate(document: Document, max_chars: int, overlap: int, min_chars: int) -> None:
    if not document.text.strip():
        raise ValueError("empty document: nothing to chunk")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    if not 0 <= min_chars < max_chars:
        raise ValueError("min_chars must be >= 0 and < max_chars")


def chunk_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """依標題階層切塊;短單元合併至下限,超長單元再切分。"""
    _validate(document, max_chars, overlap, min_chars)
    units = _collect_units(document)
    merged = _merge_short_units(units, max_chars, min_chars)
    return _emit_chunks(document, merged, max_chars, overlap)


def _collect_units(document: Document) -> list[tuple[tuple[str, ...], str, int]]:
    """依標題切出 (heading_path, 內容, 起始位置) 的原始單元。"""
    lines = [ln for ln in document.text.split("\n") if ln.strip()]
    style = detect_style(lines)

    units: list[tuple[tuple[str, ...], str, int]] = []
    path: list[str] = []
    buffer: list[str] = []
    unit_start = 0
    cursor = 0

    for line in lines:
        heading = parse_heading(line, style)
        if heading:
            body = normalize_text("\n".join(buffer))
            if body:
                units.append((tuple(path), body, unit_start))
            buffer.clear()
            level, title = heading
            del path[level - 1 :]
            path.append(title)
            unit_start = cursor
        else:
            buffer.append(line)
        cursor += len(line) + 1

    body = normalize_text("\n".join(buffer))
    if body:
        units.append((tuple(path), body, unit_start))
    return units


def _merge_short_units(
    units: list[tuple[tuple[str, ...], str, int]], max_chars: int, min_chars: int
) -> list[tuple[tuple[str, ...], str, int]]:
    """把短於下限的單元與相鄰單元合併。

    三個保護條件:不得超過 max_chars、不得跨越無共同前綴的邊界、
    路徑取共同前綴(不謊稱屬於某個特定子節)。
    """
    merged: list[tuple[tuple[str, ...], str, int]] = []
    for path, body, start in units:
        if not merged:
            merged.append((path, body, start))
            continue
        prev_path, prev_body, prev_start = merged[-1]
        shared = _common_prefix(prev_path, path)
        too_long = len(prev_body) + len(body) + 1 > max_chars
        if len(prev_body) >= min_chars or not shared or too_long:
            merged.append((path, body, start))
        else:
            merged[-1] = (shared, f"{prev_body}\n{body}", prev_start)
    return merged


def _emit_chunks(
    document: Document,
    units: list[tuple[tuple[str, ...], str, int]],
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    # 文件身分(藥名等)常只出現在首行,chunk 層必須自帶,檢索器才有機會用上
    lines = (ln.strip() for ln in document.text.split("\n"))
    doc_title = next((ln for ln in lines if ln), document.doc_id)
    chunks: list[Chunk] = []
    for path, body, start in units:
        for piece in _split_oversized(body, max_chars, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=_make_id(document.doc_id, index, piece),
                    doc_id=document.doc_id,
                    doc_title=doc_title,
                    text=piece,
                    source_path=document.source_path,
                    source_type=document.source_type,
                    page=_page_of(document.page_offsets, start),
                    heading_path=path,
                    char_span=(start, start + len(body)),
                )
            )
    return chunks


def _row_text(row: dict[str, str]) -> str:
    """把一列轉成自我描述的文字(含欄位名,否則 chunk 脫離表頭就沒有語意)。"""
    return " ".join(f"{key}:{value}" for key, value in row.items() if value.strip())


def _csv_chunk(
    text: str, doc_id: str, source_path: str, index: int, span: tuple[int, int]
) -> Chunk:
    body = normalize_text(text)
    return Chunk(
        chunk_id=_make_id(doc_id, index, body),
        doc_id=doc_id,
        doc_title=doc_id,
        text=body,
        source_path=source_path,
        source_type="csv",
        page=None,
        heading_path=(),
        char_span=span,
    )


def chunk_csv_per_row(rows: Sequence[dict[str, str]], doc_id: str, source_path: str) -> list[Chunk]:
    """逐列一個 chunk(定位精確,但實測中位數僅 129 字元)。"""
    if not rows:
        raise ValueError("no rows to chunk")
    return [
        _csv_chunk(_row_text(row), doc_id, source_path, index, (index, index + 1))
        for index, row in enumerate(rows)
    ]


def chunk_csv_grouped(
    rows: Sequence[dict[str, str]], doc_id: str, source_path: str, group_size: int = 10
) -> list[Chunk]:
    """每 group_size 列合併為一個 chunk(語意較豐富,定位較粗)。"""
    if not rows:
        raise ValueError("no rows to chunk")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(rows), group_size)):
        block = rows[start : start + group_size]
        text = "\n".join(_row_text(row) for row in block)
        chunks.append(_csv_chunk(text, doc_id, source_path, index, (start, start + len(block))))
    return chunks
