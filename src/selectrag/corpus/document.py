"""語料的資料結構。

Chunk 是檢索與 citation 的最小單元:除了文字本身,必須帶足以反查原文的
metadata(來源、頁碼、章節路徑),否則 §5.3 的 citation 正確率無從計算。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """一個檢索單元。text 保證已 NFKC 正規化。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    source_path: str
    source_type: str
    page: int | None
    heading_path: tuple[str, ...]
    char_span: tuple[int, int]


@dataclass(frozen=True)
class Document:
    """一份已載入並正規化的文件。

    page_offsets 記錄每頁在 text 中的起始位置,供 char_span → 頁碼的反查。
    非 PDF 來源為空 tuple。
    """

    doc_id: str
    source_path: str
    source_type: str
    text: str
    page_offsets: tuple[int, ...]
