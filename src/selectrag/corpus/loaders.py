"""三種來源的載入器:PDF / HTML / CSV。

各自的特性(已由 42 份真實語料盤點驗證):
  PDF  — 有文字層,但每行僅 14~21 字元,需行合併
  HTML — 有 <main>/<article> 語意標籤,footer/nav 在其外,可精準抽正文
  CSV  — 結構化,逐列或分組成 chunk(兩版對照,由使用者拍板)
"""

from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from selectrag.corpus.document import Document
from selectrag.corpus.headings import detect_style, parse_heading
from selectrag.corpus.normalize import merge_lines, normalize_text

# 自閉合標籤沒有結束標籤,計數時必須排除,否則區塊深度永遠不會歸零
_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)
_SKIP_TAGS = frozenset({"script", "style"})


class _MainExtractor(HTMLParser):
    """抽出 <main> 內的可見文字;script/style 與區塊外的內容一律排除。"""

    def __init__(self, target: str = "main") -> None:
        super().__init__(convert_charrefs=True)
        self.target = target
        self.depth = 0
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[Any]) -> None:
        if tag in _VOID_TAGS:
            return
        if tag == self.target and self.depth == 0:
            self.depth = 1
            return
        if self.depth:
            self.depth += 1
            if tag in _SKIP_TAGS:
                self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not self.depth:
            return
        if tag in _SKIP_TAGS and self.skip:
            self.skip -= 1
        self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth and not self.skip and data.strip():
            self.parts.append(data.strip())


def load_pdf(path: Path) -> Document:
    """抽取 PDF 文字並正規化;無文字層即 raise(不做 OCR)。"""
    reader = PdfReader(path)
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    if not any(p.strip() for p in raw_pages):
        raise ValueError(f"no text layer in {path.name}; OCR is out of scope")

    all_lines = [normalize_text(ln) for page in raw_pages for ln in page.split("\n") if ln.strip()]
    style = detect_style(all_lines)

    segments: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for raw in raw_pages:
        offsets.append(cursor)
        lines = [normalize_text(ln) for ln in raw.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = merge_lines(lines, lambda ln: parse_heading(ln, style) is not None)
        page_text = "\n".join(merged)
        segments.append(page_text)
        cursor += len(page_text) + 1

    return Document(
        doc_id=path.stem,
        source_path=str(path),
        source_type="pdf",
        text="\n".join(segments),
        page_offsets=tuple(offsets),
    )


def load_html(path: Path) -> Document:
    """抽取 <main> 內的正文;找不到即 raise。"""
    html = path.read_text(encoding="utf-8", errors="replace")
    for tag in ("main", "article"):
        extractor = _MainExtractor(tag)
        extractor.feed(html)
        if extractor.parts:
            # 逐段正規化後以換行接合:normalize_text 會把 \s+ 壓成空白,
            # 若先 join 再正規化,區塊邊界會消失、整份文件變成一行而無法切塊
            pieces = [normalize_text(part) for part in extractor.parts]
            text = "\n".join(piece for piece in pieces if piece)
            return Document(
                doc_id=path.stem,
                source_path=str(path),
                source_type="html",
                text=text,
                page_offsets=(),
            )
    raise ValueError(f"no main content found in {path.name}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """讀入 CSV 為列的序列(切塊策略由 chunker 決定)。"""
    text = path.read_text(encoding="utf-8-sig")
    rows = [dict(r) for r in csv.DictReader(io.StringIO(text))]
    if not rows:
        raise ValueError(f"no rows in {path.name}")
    return rows
