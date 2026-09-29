"""句中拉丁字母專有名詞與所引段落的對照(2026-09-22,發現 42 的處置)。

背景:v14 想用 prompt 規則要求「藥名照抄原文」,實測無效——模型把 methotrexate 寫成
「甲基喋吟」、azathioprine 寫成「硫脈」,同批還出現與藥名無關的「C 肻口」亂碼。三者同型:
量化模型的罕用字字元層漂移。prompt 管得住「要不要翻」,管不住「字寫錯」;禁止表述清單
更是換一個錯法就漏。本模組改以**確定性後處理**產生人工輔助欄。

兩個互補偵測(單獨任一都有盲點):
  foreign_terms   句中有、所引段落與題目都沒有的拉丁字母詞 → 憑空出現的英文名
  missing_verbatim 所引段落有、句子沒照抄的拉丁字母詞     → 被譯寫(如 methotrexate → 中文)

為何不直接找「句中的中文藥名」:改寫本來就是正常行為(§6.8 的 142 筆),
拿中文 token 比對段落會把絕大多數正常句標成誤報。missing_verbatim 因此刻意
**不猜被譯成哪個詞**,只報「段落有 X、這句沒照抄 X」,由人工看一眼即知。

**兩欄都不進任何分數**(與成分命中同一原則,也是 §6.6 覆蓋率的教訓):missing_verbatim 在
「句子只答段落一部分」時必然誤報,當成指標就會變成措辭一致度的代理。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# 長度 ≥ 3 才算專有名詞候選:mg / IU / ml 是單位,單字母是款項編號,都不是藥名。
_LATIN = re.compile(r"[A-Za-z]{3,}")


def _nfkc(text: str) -> str:
    """NFKC,**保留空白**。抽詞用這個,所以回傳的詞形是原文的大小寫。"""
    return unicodedata.normalize("NFKC", text)


def latin_terms(text: str) -> tuple[str, ...]:
    """回文中的拉丁字母詞(長度 ≥ 3),**依首次出現順序去重**,保留首次出現的原形。

    去重以 casefold 後的形式為準(`aspirin` 與 `Aspirin` 是同一個詞),但回傳原形,
    讓人工欄位看得出模型實際寫的大小寫。抽詞只做 NFKC(全形 Ａ 要折成 A 才抓得到),
    **不做 casefold**:折掉原形,人工欄位就看不出模型寫的大小寫。
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _LATIN.finditer(_nfkc(text)):
        term = m.group(0)
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return tuple(out)


def _keys(texts: Sequence[str]) -> set[str]:
    """一組文字裡所有拉丁字母詞的比對鍵(casefold 後)。

    ★ 比對一律是**詞集對詞集**,不是「詞在文字裡做子字串搜尋」。
    後者需要自行判斷字母邊界,而邊界在 casefold 空間對齊不了:`"ß".casefold()`
    展開成 `"ss"`,`AAAß` 的段落 casefold 後是 `aaass`,原本的非字母邊界變成字母
    (2026-09-22 由 property 抓到,反例 `AAAß`);去空白的正規化則會讓控制字元
    兩側的詞黏合(反例 `AAA\x85A`)。改用詞集後,邊界由 `_LATIN` 的 regex 本身保證:
    它只吃連續的 `[A-Za-z]`,所以 `ALK` 與 `ALKBH5` 天然是兩個不同的鍵。
    """
    return {t.casefold() for text in texts for t in latin_terms(text)}


def foreign_terms(sentence: str, segments: Sequence[str], question: str) -> tuple[str, ...]:
    """回句中出現、但**所引段落與題目都沒有**的拉丁字母詞(= 憑空出現的英文名)。

    question 是合法來源而非寬鬆處置:v14 掃全部 178 句實證,「句中有、段落沒有」的
    英文詞**全部**來自題目(問某藥的用法,答句自然會重述藥名)。不把題目列為來源,
    這一欄會整欄誤報而失去作用。
    """
    known = _keys([*segments, question])
    return tuple(t for t in latin_terms(sentence) if t.casefold() not in known)


def missing_verbatim(sentence: str, segments: Sequence[str]) -> tuple[str, ...]:
    """回所引段落有、但句子**沒照抄**的拉丁字母詞(= 疑似被譯寫)。

    刻意不猜「被譯成哪個中文詞」:段落有 `methotrexate` 而句子沒有,人工看一眼就知道
    它被寫成了什麼(r69 是「甲基喋吟」)。反向去猜句中的中文 token 會把正常改寫
    整批標成誤報。

    **已知限制**:句子只答段落一部分時必然誤報(段落列了多個藥名、句子只答其中一項),
    所以這是人工輔助欄,不得進入任何分數。無所引段落時回空——沒有段落就沒有照抄義務。
    """
    if not segments:
        return ()
    said = _keys([sentence])
    return tuple(t for t in latin_terms("\n".join(segments)) if t.casefold() not in said)
