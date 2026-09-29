"""標題階層解析——公文與仿單用兩套完全不同的體系。

全量盤點(38 份 PDF)證實:
  健保署公文:壹、/ 一、/ (一) / 1、/ (1) / A.     —— 仿單 0 份使用中文數字階層
  藥品仿單:  「1 成分」「1.1 有效成分及含量」     —— 30/30 份使用,公文僅 2~3 份
一套 pattern 打天下會讓 30 份仿單全部切壞。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum

# 日期必須先擋掉:「115.08.01 第七版修訂」會被數字標題規則誤判(實測誤判率 1%)
_DATE = re.compile(r"^\d{2,4}[.\-/]\d{1,2}[.\-/]\d{1,2}")

_OFFICIAL_LEVELS = (
    re.compile(r"^[壹貳參肆伍陸柒捌玖拾]+[、.]"),
    re.compile(r"^[一二三四五六七八九十]+[、.]"),
    re.compile(r"^[(（][一二三四五六七八九十]+[)）]"),
    re.compile(r"^\d+[、.](?!\d)"),
    re.compile(r"^[(（]\d+[)）]"),
    re.compile(r"^[A-Za-z][.、)]\s"),
)

# 仿單:「1 成分」「1.1 有效成分及含量」,層級 = 數字段數
_LEAFLET = re.compile(r"^(\d+(?:\.\d+)*)\s+\S")


class HeadingStyle(Enum):
    """標題體系。"""

    OFFICIAL = "official"
    LEAFLET = "leaflet"


def parse_heading(line: str, style: HeadingStyle) -> tuple[int, str] | None:
    """解析一行是否為標題,回傳 (層級, 標題全文);不是標題則回傳 None。

    層級從 1 起算。日期(如「115.08.01 第七版修訂」)一律不視為標題。
    """
    stripped = line.strip()
    if not stripped or _DATE.match(stripped):
        return None

    if style is HeadingStyle.OFFICIAL:
        for level, pattern in enumerate(_OFFICIAL_LEVELS, start=1):
            if pattern.match(stripped):
                return level, stripped
        return None

    match = _LEAFLET.match(stripped)
    if match:
        return match.group(1).count(".") + 1, stripped
    return None


def detect_style(lines: Sequence[str]) -> HeadingStyle:
    """依標題 pattern 的命中分布判斷文件屬於哪套體系。"""
    if not lines:
        raise ValueError("no lines to inspect")
    official = sum(1 for ln in lines if parse_heading(ln, HeadingStyle.OFFICIAL))
    leaflet = sum(1 for ln in lines if parse_heading(ln, HeadingStyle.LEAFLET))
    return HeadingStyle.LEAFLET if leaflet > official else HeadingStyle.OFFICIAL
