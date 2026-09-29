#!/usr/bin/env python3
"""從食藥署藥品許可證資料集篩出「同成分多廠牌」的仿單清單。

用途:為 A 軸實驗建立干擾集——同成分不同廠牌的仿單內容高度雷同,
是檢驗 reranker 價值的關鍵題材。

資料集來源:食藥署「西藥、醫療器材、化粧品許可證」資料集(UTF-8-BOM CSV)。
**資料集本身請放本機 data/corpus/,不要進 repo(26MB)。**

用法:
    python3 tools/pick_leaflets.py 許可證資料集.csv LOPERAMIDE
    python3 tools/pick_leaflets.py 許可證資料集.csv METFORMIN --count 12 --same-indication
    python3 tools/pick_leaflets.py 許可證資料集.csv --list-top 20
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

DETAIL_URL = "https://mcp.fda.gov.tw/im_detail_1/{}"
# 注意:許可證種類的「製　劑」中間是全形空格,不是半形
LICENCE_KIND = "製　劑"
REQUIRED_COLUMNS = ("許可證字號", "註銷狀態", "許可證種類", "主成分略述", "適應症", "中文品名")


def load_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"資料集是空的:{path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"資料集缺少必要欄位:{missing}(實際欄位:{list(rows[0])[:6]}…)")
    return rows


def active_single_ingredient(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """只留未註銷、製劑、單一成分者——註銷的仿單可能已下架抓不到。"""
    return [
        r
        for r in rows
        if r["註銷狀態"].strip() != "已註銷"
        and r["許可證種類"].strip() == LICENCE_KIND
        and r["主成分略述"]
        and ";;" not in r["主成分略述"]
    ]


def normalise(text: str) -> str:
    """NFKC 正規化並壓縮空白——資料集含全形字與異體字。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def list_top(rows: list[dict[str, str]], limit: int) -> None:
    counter = Counter(r["主成分略述"].strip().upper() for r in active_single_ingredient(rows))
    print(f"# 單一成分且未註銷的製劑,依筆數排序(前 {limit} 名)")
    print(f"# 可用總筆數 = {len(active_single_ingredient(rows))}")
    for name, count in counter.most_common(limit):
        print(f"{count:>5}  {name}")


def pick(rows: list[dict[str, str]], ingredient: str, count: int, same_indication: bool) -> None:
    key = ingredient.strip().upper()
    hits = [r for r in active_single_ingredient(rows) if key in r["主成分略述"].upper()]
    if not hits:
        raise SystemExit(f"找不到成分含 {key!r} 的有效製劑;可先用 --list-top 查看可選成分")

    if same_indication:
        # 適應症正規化後分組,取最大群——同文異署才是最強干擾
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in hits:
            groups[normalise(r["適應症"])].append(r)
        largest = max(groups.values(), key=len)
        print(f"# 適應症分組:{len(groups)} 種寫法,取最大群({len(largest)} 筆)", file=sys.stderr)
        hits = largest

    selected = hits[:count]
    print(f"# 成分 {key}:符合 {len(hits)} 筆,輸出 {len(selected)} 筆", file=sys.stderr)
    print("# 抓取日期:<填入抓取當日>")
    for i, r in enumerate(selected, start=1):
        licence = r["許可證字號"].strip()
        print()
        print(f"{i:02d}. {licence}  {r['中文品名'].strip()}  [{r.get('申請商名稱', '').strip()}]")
        print(f"    適應症:{r['適應症'].strip()[:60]}")
        print(f"    {DETAIL_URL.format(quote(licence))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="許可證資料集 CSV 路徑")
    parser.add_argument("ingredient", nargs="?", help="成分關鍵字(如 LOPERAMIDE)")
    parser.add_argument("--count", type=int, default=10, help="輸出筆數(預設 10)")
    parser.add_argument(
        "--same-indication", action="store_true", help="只取適應症文字完全相同的最大群(干擾最強)"
    )
    parser.add_argument("--list-top", type=int, metavar="N", help="改為列出前 N 名成分")
    args = parser.parse_args(argv)

    rows = load_rows(args.dataset)
    if args.list_top:
        list_top(rows, args.list_top)
    elif args.ingredient:
        pick(rows, args.ingredient, args.count, args.same_indication)
    else:
        parser.error("請指定成分關鍵字,或用 --list-top 查看可選成分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
