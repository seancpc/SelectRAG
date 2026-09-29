"""faithfulness judge 純邏輯(2026-09-16 核准;第一階段:短回答)。

設計決策:
  主張  — 短回答以句為單位(。!?;切),拒答句剔除;
          句內任一事實矛盾即 contradicted,全部有據才 supported(判定規則寫進 prompt)
  拒答  — 全部句子皆為拒答標記才算拒答回答;拒答回答不計入 faithfulness 分母,另列數量
  verdict — 三選一 + evidence_chunk;supported 必須附證據編號,否則 raise(說有據卻指不出來源)
  κ     — Cohen's kappa,標準定義;長度不等 / 空序列 raise
  負例  — 改第一個阿拉伯數字串成不同值(+1 或 ×2),供校準集「必須抓到」的合成 contradicted
  prompt — 只含結構與規則,不含任何領域範例(Prompt 設計規範)
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

VERDICTS = ("supported", "unsupported", "contradicted")
ABSTENTION_MARKERS = ("參考資料未提及", "無法回答")
_SPLIT = re.compile(r"[。!?;]")
_NUMBER = re.compile(r"\d+")

# v1 校準 κ 0.789 → v2 加兩條結構規則 0.845(隨機集);分層集 v2 κ 0.488。
# v3 加四條雙向規則零槓桿(42 筆判定一筆未變)、方案 B1 拆問前置實驗 3/8,皆退回;見 DECISION_MEMO §6.5
PROMPT_VERSION = "v2"
SYSTEM_RULES = (
    "你是嚴格的事實查核員。只能依據提供的參考段落判定一條主張,不得使用外部知識。\n"
    "判定規則:\n"
    "- supported:主張中的每一個事實都能在某一段落找到明確依據;evidence_chunk 填該段落編號。\n"
    "- 主張若以「且」「並」「同時」把多個條件合併,段落必須對同一個對象同時規定這些條件;"
    "段落中屬於不同類別、不同項目或互斥選項的條件被合併成一個,判 contradicted。\n"
    "- 主張所指的對象(藥名、代碼、項目)必須與段落中該規定所屬的對象一致;"
    "把段落中相鄰的其他藥品、其他代碼、前一項目的規定歸給主張的對象,判 contradicted。\n"
    "- 主張中的每個數字、單位、期間都要逐一核對,任一不符即 contradicted。\n"
    "- contradicted:主張中任一事實與段落明確衝突(數值、條件、對象不同);"
    "evidence_chunk 填衝突段落編號。\n"
    "- unsupported:段落既未支持也未反駁;evidence_chunk 填 null。\n"
    '只輸出 JSON:{"verdict": <supported|unsupported|contradicted>, "evidence_chunk": <整數或 null>}'
)


# 只對映拒答標記用到的簡體變體字(語言漂移實測:参考資料未提及 / 参考资料未提及),不做全文簡繁轉換
_VARIANTS = str.maketrans({"参": "參", "资": "資", "无": "無", "说": "說", "题": "題"})


def normalize_variants(text: str) -> str:
    """把拒答標記相關的簡體變體字對映回繁體;其餘字元原樣、長度不變。"""
    return text.translate(_VARIANTS)


def _sentences(answer: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(answer) if s.strip()]


def _is_abstention_sentence(s: str) -> bool:
    t = normalize_variants(s)
    return any(m in t for m in ABSTENTION_MARKERS)


def split_claims(answer: str) -> list[str]:
    """切句、去空白;拒答句不算主張。"""
    return [s for s in _sentences(answer) if not _is_abstention_sentence(s)]


def is_abstention(answer: str) -> bool:
    """全部句子皆為拒答標記(且至少一句)才算拒答回答。"""
    sents = _sentences(answer)
    return bool(sents) and all(_is_abstention_sentence(s) for s in sents)


def parse_verdict(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"verdict is not JSON: {raw[:60]!r}") from e
    if not isinstance(obj, dict) or set(obj) != {"verdict", "evidence_chunk"}:
        raise ValueError("verdict object must have exactly verdict and evidence_chunk")
    verdict, ev = obj["verdict"], obj["evidence_chunk"]
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    if ev is not None and (not isinstance(ev, int) or isinstance(ev, bool)):
        raise ValueError("evidence_chunk must be int or null")
    if verdict == "supported" and ev is None:
        raise ValueError("supported verdict must cite an evidence_chunk")
    return {"verdict": verdict, "evidence_chunk": ev}


def aggregate(items: Sequence[Mapping[str, Any]], n_abstained_answers: int = 0) -> dict[str, Any]:
    if not items:
        raise ValueError("no claims to aggregate")
    counts = dict.fromkeys(VERDICTS, 0)
    for it in items:
        v = it["verdict"]
        if v not in counts:
            raise ValueError(f"unknown verdict {v!r}")
        counts[v] += 1
    n = len(items)
    return {
        "n_claims": n,
        "n_abstained_answers": n_abstained_answers,
        "faithfulness": counts["supported"] / n,
        "unsupported_rate": counts["unsupported"] / n,
        "contradiction_rate": counts["contradicted"] / n,
        "counts": counts,
    }


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or len(a) != len(b):
        raise ValueError("kappa needs two non-empty sequences of equal length")
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pe = sum((list(a).count(k) / n) * (list(b).count(k) / n) for k in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def perturb_number(text: str, rng: random.Random) -> str | None:
    """把第一個數字串改成不同值;無數字回 None。"""
    m = _NUMBER.search(text)
    if not m:
        return None
    n = int(m.group())
    new = n + 1 if rng.random() < 0.5 or n == 0 else n * 2
    return text[: m.start()] + str(new) + text[m.end() :]


_STRIP = re.compile(r"[\s,,。、;:;:()()\[\]「」『』.]")
_CHUNK_MARK = re.compile(r"^\[(\d+)\] ?", re.MULTILINE)
_MIN_CLAIM_CHARS = 6


def _norm(text: str) -> str:
    return _STRIP.sub("", text)


def _grams3(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(len(text) - 2)}


def coverage3(claim: str, context: str) -> float:
    """claim 字元 3-gram 被 context 覆蓋的比例(去空白與中英標點);claim 不足 3 字回 0.0。"""
    g = _grams3(_norm(claim))
    if not g:
        return 0.0
    return len(g & _grams3(_norm(context))) / len(g)


def split_chunks(context: str) -> list[tuple[int, str]]:
    """按行首「[n] 」切段,編號取標記數字不重編;無標記 → [(1, 全文)]。"""
    marks = list(_CHUNK_MARK.finditer(context))
    if not marks:
        return [(1, context)]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(context)
        out.append((int(m.group(1)), context[m.end() : end].strip()))
    return out


def verbatim_hit(
    claim: str, chunks: Sequence[tuple[int, str]], *, min_coverage: float
) -> int | None:
    """逐字命中:對單一 chunk 覆蓋率 ≥ 門檻、claim 的每個數字串都在該 chunk、claim ≥ 6 字。

    多個命中取覆蓋率最高者,同分取編號最小;無命中回 None。
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if len(_norm(claim)) < _MIN_CLAIM_CHARS:
        return None
    nums = set(_NUMBER.findall(claim))
    best: tuple[float, int] | None = None
    for idx, text in chunks:
        cov = coverage3(claim, text)
        if cov < min_coverage or not nums <= set(_NUMBER.findall(text)):
            continue
        key = (cov, -idx)
        if best is None or key > best:
            best = key
    return None if best is None else -best[1]


def judge_prompt(question: str, context: str, claim: str) -> list[dict[str, str]]:
    if not claim.strip():
        raise ValueError("claim is empty")
    user = f"問題:{question}\n\n參考段落:\n{context}\n\n待判定主張:{claim}"
    return [{"role": "system", "content": SYSTEM_RULES}, {"role": "user", "content": user}]
