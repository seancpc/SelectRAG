"""多成分題的「必要成分」清單與命中判定(2026-09-20)。

背景:`b1-q19`(補助對象分在兩個代碼)、`b1-q22`(點數 + 是否含特材)這類題目要答兩件
獨立的事,但單句 claim 天生答不完,而既有指標都看不到——檢索層真值齊全、support 層
答的那半有據。此模組提供確定性的成分命中判定,寫進 `tools/support_sheet.py` 的人工
輔助欄。

**成分命中不得進入任何分數或配置比較**,理由與 §6.6 的逐字覆蓋率同:
判定用的是字面命中,把它當品質分數會讓分數變成措辭一致度的代理。

正規化的界線只到**字形**——全半形與空白;**語意**差異(中文數字與
阿拉伯數字互換、同義詞)一律由 aliases 明列,程式不猜。清單內容屬領域真值,人工核准。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUIRED_COMPONENT = ("name", "aliases")

# CJK Radicals Supplement 的部首字元:NFKC 對此區**沒有**映射,必須自行折疊。
# 範圍取自全語料實測(2026-09-20):只出現這三種,共 34 次。新增語料時應重掃。
_RADICALS = {"\u2ed1": "長", "\u2ea0": "民", "\u2ec4": "西"}

# 千分位逗號:只脫「數字,數字」之間的,句中的逗號是內容
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


@dataclass(frozen=True)
class ComponentSpec:
    """一個必要成分,與它的等價表述。命中任一 alias 即視為答到該成分。

    forbidden 是**禁止表述**:命中任一即該成分不計分,且與「漏答」分開回報。
    存在理由(2026-09-20 核准表 r20):子字串比對無法表達否定——正解「每次2粒」
    的裸別名「2粒」會同時命中錯誤答案「每日2粒」(日劑量差四倍),而那是本語料
    反覆出現的錯型。單位錯置、因果倒置這類錯誤都只能靠這一欄攔。
    """

    name: str
    aliases: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class ItemComponents:
    """一題的必要成分,與**題層**禁止表述。

    題層 forbidden 與成分層 forbidden 是不同機制,刻意分開(2026-09-20 核准):
      - 成分層(保護型):該表述會**假性滿足**某個成分(如「每日2粒」滿足「2粒」),
        所以掛在那個成分上把它歸零
      - 題層(懲罰型):該表述不命中任何別名,它證明的是模型**整題答到別的標的去了**
        (如 Stiripentol 題答出相鄰條文的「迷走神經刺激術」),屬 item 層級失敗
    混用同一欄位會讓「答不全」與「答錯題」統計成同一個數字。
    """

    components: tuple[ComponentSpec, ...]
    forbidden: tuple[str, ...] = ()


def normalize_glyphs(text: str) -> str:
    """折疊字形差異:NFKC + CJK 部首 + 大小寫 + 千分位逗號 + 空白。

    刻意不動中文數字與同義詞(N4)——那屬語意層,由 aliases 明列。這條界線是
    為了不重踩 §6.6 逐字覆蓋率的坑:一旦程式開始猜「45 歲 = 四十五歲」,判定
    就變成措辭相似度的代理。

    NFKC 負責全形 ASCII、㎏ 這類相容字、圈號數字;CJK Radicals Supplement
    (如「延⻑」的 ⻑)NFKC 沒有映射,由 _RADICALS 補。
    """
    folded = "".join(_RADICALS.get(ch, ch) for ch in text)
    # NFKC → casefold → NFKC 是 Unicode 的 NFKC_Casefold 順序。
    folded = unicodedata.normalize("NFKC", folded).casefold()
    folded = unicodedata.normalize("NFKC", folded)
    folded = _THOUSANDS.sub("", folded)
    folded = "".join(ch for ch in folded if not ch.isspace())
    # 最後這次 NFKC 不可省(property 測試的反例 "U¯"):NFKC 把 ¯ 展成
    # 「空白 + combining macron」,去空白後 macron 貼上前一字母,下次呼叫才會
    # 組合成 ū——少了這步此函式就不冪等,同一段文字兩次比對可能結果不同。
    return unicodedata.normalize("NFKC", folded)


def _contains(haystack: str, needle: str) -> bool:
    """子字串比對,但**數字邊界**:別名邊緣是數字時,答案中相鄰字元不得是數字。

    存在理由(2026-09-21 首輪跑分):模型寫「支付點數為1500,」,而別名綁單位寫成
    `1500點` 就漏判;改寫成 `點數為1500` 又會誤中「點數為15000」。純子字串比對
    在「數字是另一個數字的前綴」時必然出錯——`650` 會命中 `6500` 與 `1650`。

    必須掃過**所有**出現位置:第一次出現不合邊界時,後面的出現仍可能合格
    (「先寫15000,後面才寫1500點」)。

    已知限制:只擋數字相鄰,擋不了小數位擴寫——別名 `2.5` 在答案寫 `2.50mg` 時
    不命中(尾字 `5` 後接 `0`)。那類變體仍需由 aliases 明列。
    """
    if not needle:
        return False
    head_digit = needle[0].isdigit()
    tail_digit = needle[-1].isdigit()
    if not head_digit and not tail_digit:
        return needle in haystack
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return False
        before_ok = not (head_digit and i > 0 and haystack[i - 1].isdigit())
        end = i + len(needle)
        after_ok = not (tail_digit and end < len(haystack) and haystack[end].isdigit())
        if before_ok and after_ok:
            return True
        start = i + 1


def _parse_component(raw: Mapping[str, Any], item_id: str) -> ComponentSpec:
    for key in _REQUIRED_COMPONENT:
        if key not in raw:
            raise ValueError(f"missing field '{key}' in component of item {item_id}")
    aliases = tuple(raw["aliases"])
    if not aliases:
        raise ValueError(f"empty aliases for component {raw['name']!r} of item {item_id}")
    for alias in aliases:
        if not alias.strip():
            raise ValueError(f"blank alias for component {raw['name']!r} of item {item_id}")
    forbidden = tuple(raw.get("forbidden", ()))
    for bad in forbidden:
        if not bad.strip():
            raise ValueError(f"blank forbidden for component {raw['name']!r} of item {item_id}")
    overlap = {normalize_glyphs(a) for a in aliases} & {normalize_glyphs(b) for b in forbidden}
    if overlap:
        raise ValueError(
            f"forbidden overlaps aliases for component {raw['name']!r} of item {item_id}: "
            f"{sorted(overlap)}"
        )
    return ComponentSpec(name=raw["name"], aliases=aliases, forbidden=forbidden)


def _parse_item(raw: Any, item_id: str) -> ItemComponents:
    """題目節點可為 dict(含題層 forbidden)或純 list(既有清單格式,向後相容)。"""
    if isinstance(raw, Mapping):
        raws = raw["components"]
        item_forbidden = tuple(raw.get("forbidden", ()))
    else:
        raws = raw
        item_forbidden = ()
    specs = tuple(_parse_component(r, item_id) for r in raws)
    if not specs:
        raise ValueError(f"item {item_id} has no components")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate component name in item {item_id}")
    for bad in item_forbidden:
        if not bad.strip():
            raise ValueError(f"blank item forbidden in item {item_id}")
    # 題層禁止表述等於任一成分的正解表述 → 該題永遠歸零,是清單筆誤
    all_aliases = {normalize_glyphs(a) for spec in specs for a in spec.aliases}
    overlap = all_aliases & {normalize_glyphs(b) for b in item_forbidden}
    if overlap:
        raise ValueError(f"item forbidden overlaps aliases in item {item_id}: {sorted(overlap)}")
    return ItemComponents(components=specs, forbidden=item_forbidden)


def load_components(path: Path) -> dict[str, ItemComponents]:
    """讀成分清單。壞資料一律 raise,延續 eval_set 與指標模組的策略。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "items" not in payload:
        raise ValueError("missing field 'items' in components file")
    return {item_id: _parse_item(raw, item_id) for item_id, raw in payload["items"].items()}


def component_hits(
    answer: str, specs: Sequence[ComponentSpec]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """回 (命中, 缺漏, 禁止表述命中) 三組成分名,各依 specs 原順序。

    三者恰好分割成分清單。禁止表述與「漏答」分開回報,因為兩者的意義不同:
    漏答是不完整,命中禁止表述是**答錯**;把兩者混成一個數字就看不出錯型。
    禁止表述優先於 alias 命中(同句同時出現時以答錯為準)。

    空的成分清單是壞資料而非「全部命中」,一律 raise。
    """
    if not specs:
        raise ValueError("empty component specs")
    norm_answer = normalize_glyphs(answer)
    hit: list[str] = []
    miss: list[str] = []
    blocked: list[str] = []
    for spec in specs:
        if any(_contains(norm_answer, normalize_glyphs(bad)) for bad in spec.forbidden):
            blocked.append(spec.name)
            continue
        found = any(_contains(norm_answer, normalize_glyphs(alias)) for alias in spec.aliases)
        (hit if found else miss).append(spec.name)
    return tuple(hit), tuple(miss), tuple(blocked)


def completeness(answer: str, specs: Sequence[ComponentSpec]) -> float:
    """命中成分數佔必要成分數的比例;禁止表述命中不計入分子。

    **診斷用,不得進入分數比較**(N2)。
    """
    hit, _, _ = component_hits(answer, specs)
    return len(hit) / len(specs)


def item_forbidden_hits(answer: str, item: ItemComponents) -> tuple[str, ...]:
    """回命中的**題層**禁止表述,依清單原順序。命中任何一個即整題判為答錯。"""
    norm_answer = normalize_glyphs(answer)
    return tuple(bad for bad in item.forbidden if _contains(norm_answer, normalize_glyphs(bad)))


def item_completeness(answer: str, item: ItemComponents) -> float:
    """整題 completeness:題層禁止表述一命中即 0.0,否則等於成分層比例。

    **診斷用,不得進入分數比較**(N2)。歸零與「成分全漏」在數值上相同但意義不同,
    故 `item_forbidden_hits` 要與成分層分開統計,不可只看這個數字。
    """
    if item_forbidden_hits(answer, item):
        return 0.0
    return completeness(answer, item.components)


def unsupported_hits(
    hit: Sequence[str], cited_sentences: Sequence[str], specs: Sequence[ComponentSpec]
) -> tuple[str, ...]:
    """回「命中了但在**有引用的句子**裡找不到依據」的成分名,依 hit 原順序。

    存在理由(2026-09-21 首輪跑分):`component_hits` 對**完整回答**計算,而 support
    只收有引用的句子,兩者母體不同——`b1-q19` 因此出現 completeness = 1 但兩組補助
    對象都沒有任何引用支撐。決議是**不改計算母體**(改了會讓「答對但沒標引用」被
    算成漏答),而是把矛盾顯示出來,讓完整性與可稽核性兩個訊號都保留。

    比對走與 `component_hits` 同一條路徑(含數字邊界),否則兩欄的判準會不一致。
    """
    by_name = {spec.name: spec for spec in specs}
    joined = normalize_glyphs("".join(cited_sentences))
    out = []
    for name in hit:
        spec = by_name.get(name)
        if spec is None:
            continue
        if not any(_contains(joined, normalize_glyphs(a)) for a in spec.aliases):
            out.append(name)
    return tuple(out)
