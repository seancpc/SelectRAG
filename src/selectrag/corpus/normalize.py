"""文字正規化與 PDF 行合併。

PDF 抽出的文字每行只有 14~21 字元(視覺換行被當成真換行),不合併會切出
破碎半句;而實測 58% 的行沒有句末標點,單靠標點判斷段落結束會嚴重誤判。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

# 句末標點:出現在行尾即視為段落結束
_SENTENCE_ENDINGS = "。；;：:！!？?"


def normalize_text(text: str) -> str:
    """NFKC 正規化並壓縮空白;消除 PDF 抽出的康熙部首等異體字。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _ends_paragraph(
    line: str, next_line: str | None, is_heading: Callable[[str], bool], short_threshold: float
) -> bool:
    if line and line[-1] in _SENTENCE_ENDINGS:
        return True
    if next_line is None or is_heading(next_line):
        return True
    return len(line) < short_threshold


def merge_lines(
    lines: Sequence[str],
    is_heading: Callable[[str], bool],
    short_line_ratio: float = 0.6,
) -> list[str]:
    """把 PDF 的視覺換行合併回段落。

    判定段落結束的三個條件(任一成立即結束):
      1. 行尾有句末標點
      2. 下一行是標題
      3. 本行明顯短於平均行長(未滿行)
    """
    if not lines:
        raise ValueError("no lines to merge")

    short_threshold = sum(len(x) for x in lines) / len(lines) * short_line_ratio
    merged: list[str] = []
    buffer = ""
    for index, line in enumerate(lines):
        buffer += line
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        if is_heading(line) or _ends_paragraph(line, next_line, is_heading, short_threshold):
            merged.append(buffer)
            buffer = ""
    # 最後一行必觸發 _ends_paragraph(next_line is None),buffer 到此必為空,不需收尾
    return merged
