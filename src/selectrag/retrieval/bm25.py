"""Okapi BM25 檢索器(A 軸控制組)— 純標準庫實作。

設計決策(2026-08-30 核准):
  斷詞  — CJK 字元 bigram + ASCII 連續字母數字詞(小寫化),不引入 jieba;
          查詢與文件走同一個 tokenize,正規化不一致即 recall 無聲歸零
  IDF   — Lucene 變體 ln((N-df+0.5)/(df+0.5)+1),恆非負;
          原始 Robertson IDF 在 df > N/2 時為負,中文 bigram 的 df 很容易過半
  平手  — 同分依索引插入順序,保證跨次執行 bit-for-bit 可重現
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from selectrag.corpus.normalize import normalize_text

_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[㐀-鿿]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """NFKC 正規化後切 token:ASCII 詞照切,CJK 連續段切成 bigram。

    單獨一個 CJK 字元(切不出 bigram)保留為單字 token,不得丟失。
    """
    normalized = normalize_text(text).lower()
    tokens: list[str] = []
    cursor = 0
    for match in _CJK.finditer(normalized):
        tokens.extend(_ASCII_WORD.findall(normalized[cursor : match.start()]))
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        cursor = match.end()
    tokens.extend(_ASCII_WORD.findall(normalized[cursor:]))
    return tokens


class BM25Retriever:
    """對 (id, text) 序列建倒排索引;retrieve 回傳分數遞減的 id。"""

    def __init__(
        self,
        entries: Sequence[tuple[str, str]],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if not entries:
            raise ValueError("empty corpus: nothing to index")
        ids = [entry_id for entry_id, _ in entries]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate ids in corpus: {dupes}")

        self._k1 = k1
        self._b = b
        self._ids = ids
        self._tf: list[Counter[str]] = []
        for entry_id, text in entries:
            tokens = tokenize(text)
            if not tokens:
                raise ValueError(f"document {entry_id!r} has no tokens after normalization")
            self._tf.append(Counter(tokens))

        n = len(self._tf)
        self._doc_len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = sum(self._doc_len) / n
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {
            token: math.log((n - count + 0.5) / (count + 0.5) + 1) for token, count in df.items()
        }

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        return [doc_id for doc_id, _ in self.retrieve_scored(question, top_k)]

    def retrieve_scored(self, question: str, top_k: int) -> Sequence[tuple[str, float]]:
        """與 retrieve 同序,附 BM25 分數(跨題不可比,只供同題內比較與拒答門檻實驗)。"""
        query_tokens = tokenize(question)
        if not query_tokens:
            raise ValueError("query has no tokens after normalization")

        scored: list[tuple[float, int]] = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._doc_len[index] / self._avgdl)
            for token in query_tokens:
                count = tf.get(token, 0)
                if count:
                    score += self._idf[token] * count * (self._k1 + 1) / (count + norm)
            if score > 0.0:
                scored.append((score, index))
        # 分數遞減;同分依插入順序(index 遞增)——sort 穩定 + key 只含分數
        scored.sort(key=lambda pair: -pair[0])
        return [(self._ids[index], score) for score, index in scored[:top_k]]


_CONTEXT_FIELDS = ("doc_title", "heading_path")


def _context_text(row: dict[str, object], field: str, where: str) -> str:
    if field not in row:
        raise ValueError(f"{where}: missing field '{field}' required by context_fields")
    value = row[field]
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value)


def index_text(row: dict[str, object], context_fields: tuple[str, ...], where: str) -> str:
    """索引文本的唯一組裝規則:脈絡欄位依序在前、正文在後,以空白串接。

    BM25 與 dense 共用本函式——兩組對照的「文件脈絡」語意必須完全相同,
    否則 A 軸的比較不公平(共用欄位必須來自同一份定義)。
    """
    for field in context_fields:
        if field not in _CONTEXT_FIELDS:
            raise ValueError(f"unknown context field {field!r}; allowed: {_CONTEXT_FIELDS}")
    context = [_context_text(row, field, where) for field in context_fields]
    return " ".join([*context, str(row["text"])])


def load_bm25_from_jsonl(
    path: Path,
    context_fields: tuple[str, ...] = (),
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25Retriever:
    """從項目 6 的 chunks.jsonl 建索引;缺 chunk_id/text 欄位即 raise。

    context_fields 指定要串接進索引文本的文件脈絡欄位(doc_title / heading_path);
    檢索回傳的仍是 chunk_id,retrieve 介面不變。預設空 = 只索引正文,
    即 A 軸控制組行為——對照組設定不得汙染控制組。
    """
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("chunk_id", "text"):
                if field not in row:
                    raise ValueError(f"{path.name} line {line_no}: missing field '{field}'")
            where = f"{path.name} line {line_no}"
            entries.append((row["chunk_id"], index_text(row, context_fields, where)))
    return BM25Retriever(entries, k1=k1, b=b)
