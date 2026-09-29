"""BM25 檢索器測試 —— 場景為 2026-08-30 定案的驗收條件。"""

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.retrieval.bm25 import BM25Retriever, load_bm25_from_jsonl, tokenize

# --------------------------------------------------------------------------
# S2:斷詞(CJK bigram + ASCII 詞)
# --------------------------------------------------------------------------


def test_tokenize_mixed_cjk_and_ascii() -> None:
    """CJK 逐字 bigram,ASCII 連續字母數字為一詞(小寫化)。"""
    assert tokenize("metformin 500mg 糖尿病") == ["metformin", "500mg", "糖尿", "尿病"]


def test_tokenize_single_cjk_char_kept() -> None:
    """單一 CJK 字元(切不出 bigram)保留為單字 token,不得丟失。"""
    assert tokenize("胃") == ["胃"]


def test_tokenize_ascii_lowercased() -> None:
    assert tokenize("Metformin HCl") == ["metformin", "hcl"]


def test_tokenize_nfkc_normalised() -> None:
    """S3 前置:全形/異體字經 NFKC 還原後才切,查詢與文件同套規則。"""
    assert tokenize("ABC") == ["abc"]


# --------------------------------------------------------------------------
# S1:手算正確性(獨立重算公式,鎖死 Lucene 變體 IDF 與長度正規化)
# --------------------------------------------------------------------------

CORPUS = [
    ("d1", "apple banana apple"),
    ("d2", "banana cherry"),
    ("d3", "cherry cherry durian"),
]


def expected_bm25(query_token: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    """測試側獨立重算(與實作不共享程式碼),含截斷前的全部分數。"""
    docs = {doc_id: text.split() for doc_id, text in CORPUS}
    n = len(docs)
    avgdl = sum(len(t) for t in docs.values()) / n
    df = sum(1 for tokens in docs.values() if query_token in tokens)
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    scores = {}
    for doc_id, tokens in docs.items():
        tf = tokens.count(query_token)
        if tf:
            denom = tf + k1 * (1 - b + b * len(tokens) / avgdl)
            scores[doc_id] = idf * tf * (k1 + 1) / denom
    return scores


def test_hand_computed_single_hit() -> None:
    """只有 d1 含 apple → 恰回 d1。"""
    retriever = BM25Retriever(CORPUS)
    assert list(retriever.retrieve("apple", top_k=10)) == ["d1"]


def test_hand_computed_ranking_tf_beats_length_penalty() -> None:
    """cherry:d3(tf=2, 較長)必須贏 d2(tf=1, 較短)——手算 0.6455 > 0.5296。"""
    scores = expected_bm25("cherry")
    assert scores["d3"] > scores["d2"], scores
    retriever = BM25Retriever(CORPUS)
    assert list(retriever.retrieve("cherry", top_k=10)) == ["d3", "d2"]


def test_idf_never_negative() -> None:
    """df 過半(banana 出現 2/3 份)時 Robertson 原始 IDF 為負;
    Lucene 變體必須恆正,含常見詞的文件不得被倒扣出榜。"""
    retriever = BM25Retriever(CORPUS)
    result = list(retriever.retrieve("banana", top_k=10))
    assert set(result) == {"d1", "d2"}


# --------------------------------------------------------------------------
# S3:查詢與文件同套正規化
# --------------------------------------------------------------------------


def test_fullwidth_query_matches_halfwidth_doc() -> None:
    """全形查詢必須命中半形文件。

    全形字以 U+FFxx 逃逸寫死:肉眼看似全形的字面值可能在編輯途中被正規化成
    ASCII,使本測試無聲退化成「半形查半形」(2026-08-30 變異存活即因此)。
    """
    fullwidth_aspirin = "\uff21\uff33\uff30\uff29\uff32\uff29\uff2e"  # ASPIRIN 全形
    assert all(ord(c) > 0xFF00 for c in fullwidth_aspirin)
    retriever = BM25Retriever([("d1", "aspirin 100mg"), ("d2", "warfarin 5mg")])
    assert list(retriever.retrieve(fullwidth_aspirin, top_k=1)) == ["d1"]


def test_cjk_query_hits_cjk_doc() -> None:
    """中文查詢必須經同一套 bigram 斷詞才比對得到中文文件。"""
    retriever = BM25Retriever([("d1", "糖尿病用藥指引"), ("d2", "高血壓治療原則")])
    assert list(retriever.retrieve("糖尿病", top_k=1)) == ["d1"]


# --------------------------------------------------------------------------
# 公式分項鎖定(變異測試揭露:只鎖排序不鎖公式,三個公式變異可同時存活)
# --------------------------------------------------------------------------


def test_idf_ranks_rare_token_doc_above_common_token_doc() -> None:
    """查詢含一罕見詞一常見詞時,含罕見詞的文件必須排最前。

    IDF 若退化為常數,五份文件同分、平手規則讓 d1 搶第一。
    """
    docs = [(f"d{i}", "aa") for i in range(1, 5)] + [("d5", "bb")]
    retriever = BM25Retriever(docs)
    assert list(retriever.retrieve("aa bb", top_k=5))[0] == "d5"


def test_length_norm_prefers_shorter_doc_at_equal_tf() -> None:
    """同 tf 之下短文件必須贏長文件;b 失效則同分、長文件因插入序搶先。"""
    retriever = BM25Retriever([("dlong", "aa cc dd ee ff"), ("dshort", "aa bb")])
    assert list(retriever.retrieve("aa", top_k=2)) == ["dshort", "dlong"]


def test_tf_saturation_caps_high_tf_in_long_doc() -> None:
    """tf 飽和:長文件靠堆詞頻不得壓過短文件的單次精準命中。

    手算(k1=1.5, b=0.75, avgdl=7):dA(tf=2, dl=12)=0.465·idf·2.5,
    dB(tf=1, dl=2)=0.590·idf·2.5 → dB 在前;分母漏 tf 時 dA 反超。
    """
    retriever = BM25Retriever([("dA", "aa aa cc dd ee ff gg hh ii jj kk ll"), ("dB", "aa bb")])
    assert list(retriever.retrieve("aa", top_k=2)) == ["dB", "dA"]


# --------------------------------------------------------------------------
# S4/S5/S9:零重疊、截斷、空查詢
# --------------------------------------------------------------------------


def test_zero_overlap_returns_empty() -> None:
    """無共同 token → 空列表,不得回傳任意文件充數(recall 虛高)。"""
    retriever = BM25Retriever(CORPUS)
    assert list(retriever.retrieve("elderberry", top_k=5)) == []


def test_top_k_truncates_to_best() -> None:
    docs = [(f"d{i}", "target " * (i + 1)) for i in range(5)]
    retriever = BM25Retriever(docs)
    result = list(retriever.retrieve("target", top_k=3))
    assert len(result) == 3
    full = list(retriever.retrieve("target", top_k=10))
    assert result == full[:3]


def test_empty_query_raises() -> None:
    retriever = BM25Retriever(CORPUS)
    with pytest.raises(ValueError, match="no tokens"):
        retriever.retrieve("   ", top_k=5)


# --------------------------------------------------------------------------
# S6:平手決定性
# --------------------------------------------------------------------------


def test_tie_break_is_insertion_order() -> None:
    """完全同文者同分,依索引插入順序排。"""
    retriever = BM25Retriever([("z9", "same text"), ("a1", "same text"), ("m5", "same text")])
    assert list(retriever.retrieve("same", top_k=10)) == ["z9", "a1", "m5"]


# --------------------------------------------------------------------------
# S7/S8:壞資料一律 raise
# --------------------------------------------------------------------------


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        BM25Retriever([])


def test_duplicate_ids_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BM25Retriever([("d1", "text one"), ("d1", "text two")])


def test_doc_with_no_tokens_raises() -> None:
    """索引內容空白的文件是壞資料,不得靜默入索引(它永遠檢索不到)。"""
    with pytest.raises(ValueError, match="no tokens"):
        BM25Retriever([("d1", "ok text"), ("d2", "   ")])


# --------------------------------------------------------------------------
# S10:JSONL 載入
# --------------------------------------------------------------------------


def rows_to_jsonl_with_blank(rows: list[dict[str, str]]) -> str:
    lines = [json.dumps(r) for r in rows]
    return lines[0] + "\n\n" + "\n".join(lines[1:]) + "\n"


def test_load_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "c1", "text": "aspirin indications", "doc_id": "x"},
        {"chunk_id": "c2", "text": "warfarin dosage", "doc_id": "y"},
    ]
    # 中段夾一個空行:JSONL 常見於手工串接,載入器必須跳過而非炸掉
    path.write_text(rows_to_jsonl_with_blank(rows), encoding="utf-8")
    retriever = load_bm25_from_jsonl(path)
    assert list(retriever.retrieve("warfarin", top_k=1)) == ["c2"]


def test_load_from_jsonl_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "no id here"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_id"):
        load_bm25_from_jsonl(path)


# --------------------------------------------------------------------------
# 不變量(property-based)
# --------------------------------------------------------------------------

_WORDS = ["aa", "bb", "cc", "dd", "ee"]
corpus_strategy = st.lists(
    st.lists(st.sampled_from(_WORDS), min_size=1, max_size=6).map(" ".join),
    min_size=1,
    max_size=8,
)
query_strategy = st.lists(st.sampled_from(_WORDS), min_size=1, max_size=3).map(" ".join)


def _build(texts: list[str]) -> BM25Retriever:
    return BM25Retriever([(f"d{i}", t) for i, t in enumerate(texts)])


@settings(deadline=None)
@given(texts=corpus_strategy, query=query_strategy, k=st.integers(1, 10))
def test_property_prefix_consistency(texts: list[str], query: str, k: int) -> None:
    """retrieve(q, k) 必為 retrieve(q, k+n) 的前綴——截斷不得改變順序。"""
    retriever = _build(texts)
    small = list(retriever.retrieve(query, top_k=k))
    big = list(retriever.retrieve(query, top_k=k + 5))
    assert big[:k] == small


@settings(deadline=None)
@given(texts=corpus_strategy, query=query_strategy, k=st.integers(1, 10))
def test_property_output_ids_valid_and_unique(texts: list[str], query: str, k: int) -> None:
    retriever = _build(texts)
    result = list(retriever.retrieve(query, top_k=k))
    assert len(result) <= k
    assert len(set(result)) == len(result), "輸出不得有重複 id"
    valid = {f"d{i}" for i in range(len(texts))}
    assert set(result) <= valid, "輸出必須是索引內的 id"


@settings(deadline=None)
@given(texts=corpus_strategy, query=query_strategy, k=st.integers(1, 10))
def test_property_deterministic(texts: list[str], query: str, k: int) -> None:
    retriever = _build(texts)
    assert list(retriever.retrieve(query, top_k=k)) == list(retriever.retrieve(query, top_k=k))


@settings(deadline=None)
@given(texts=corpus_strategy, query=query_strategy)
def test_property_hits_iff_token_overlap(texts: list[str], query: str) -> None:
    """回傳的每份文件必與查詢有 token 交集;有交集的文件在 k 夠大時必入榜。"""
    retriever = _build(texts)
    result = set(retriever.retrieve(query, top_k=len(texts)))
    q_tokens = set(query.split())
    overlapping = {f"d{i}" for i, t in enumerate(texts) if q_tokens & set(t.split())}
    assert result == overlapping


# --------------------------------------------------------------------------
# S11:runner 整合
# --------------------------------------------------------------------------


def test_build_retriever_bm25(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever
    from selectrag.runner.config import RetrieverConfig

    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({"chunk_id": "c1", "text": "aspirin"}) + "\n", encoding="utf-8")
    retriever = build_retriever(RetrieverConfig(type="bm25", params={"chunks": str(path)}))
    assert list(retriever.retrieve("aspirin", top_k=1)) == ["c1"]


def test_build_retriever_bm25_missing_chunks_param() -> None:
    from selectrag.runner.__main__ import build_retriever
    from selectrag.runner.config import RetrieverConfig

    with pytest.raises(ValueError, match="chunks"):
        build_retriever(RetrieverConfig(type="bm25", params={}))


# --------------------------------------------------------------------------
# 文件脈絡入索引(2026-09-01)
# --------------------------------------------------------------------------


def write_ctx_jsonl(path: Path, rows_data: list[dict[str, object]]) -> Path:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows_data]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


CTX_ROWS: list[dict[str, object]] = [
    {
        "chunk_id": "c1",
        "text": "本藥用於治療某症狀",
        "doc_title": "甲藥錠 ALPHADRUG",
        "heading_path": ["2 適應症"],
    },
    {
        "chunk_id": "c2",
        "text": "本藥用於治療某症狀",
        "doc_title": "乙藥錠 BETADRUG",
        "heading_path": ["1 性狀"],
    },
]


def test_doc_title_field_enables_docname_query(tmp_path: Path) -> None:
    """B1:正文相同、標題不同;查藥名 → 只回標題相符者;預設查不到。"""
    path = write_ctx_jsonl(tmp_path / "c.jsonl", CTX_ROWS)
    plain = load_bm25_from_jsonl(path)
    assert list(plain.retrieve("alphadrug", top_k=2)) == []
    ctx = load_bm25_from_jsonl(path, context_fields=("doc_title",))
    assert list(ctx.retrieve("alphadrug", top_k=2)) == ["c1"]


def test_heading_path_field_enables_heading_query(tmp_path: Path) -> None:
    """B2:詞只在 heading_path → 啟用時命中、預設不命中。"""
    path = write_ctx_jsonl(tmp_path / "c.jsonl", CTX_ROWS)
    assert list(load_bm25_from_jsonl(path).retrieve("適應症", top_k=2)) == []
    ctx = load_bm25_from_jsonl(path, context_fields=("heading_path",))
    assert list(ctx.retrieve("適應症", top_k=2)) == ["c1"]


def test_unknown_context_field_raises(tmp_path: Path) -> None:
    """B3:未知欄位 raise。"""
    path = write_ctx_jsonl(tmp_path / "c.jsonl", CTX_ROWS)
    with pytest.raises(ValueError, match="context field"):
        load_bm25_from_jsonl(path, context_fields=("nonexistent",))


def test_legacy_jsonl_without_doc_title(tmp_path: Path) -> None:
    """B5:舊檔無 doc_title——預設可用;帶 doc_title 即 raise(壞資料不靜默)。"""
    path = write_ctx_jsonl(tmp_path / "old.jsonl", [{"chunk_id": "c1", "text": "some text"}])
    assert list(load_bm25_from_jsonl(path).retrieve("some", top_k=1)) == ["c1"]
    with pytest.raises(ValueError, match="doc_title"):
        load_bm25_from_jsonl(path, context_fields=("doc_title",))


def test_build_retriever_bm25_context_fields(tmp_path: Path) -> None:
    """runner params.context_fields 傳遞至載入器。"""
    from selectrag.runner.__main__ import build_retriever
    from selectrag.runner.config import RetrieverConfig

    path = write_ctx_jsonl(tmp_path / "c.jsonl", CTX_ROWS)
    retriever = build_retriever(
        RetrieverConfig(type="bm25", params={"chunks": str(path), "context_fields": ["doc_title"]})
    )
    assert list(retriever.retrieve("betadrug", top_k=1)) == ["c2"]


def test_build_retriever_bm25_passes_k1_b(tmp_path: Path) -> None:
    """調參值經 config 透傳:b=0 時長度懲罰消失,平手改依插入順序。

    索引順序刻意讓長文件在前:預設 b=0.75 下短文件因長度正規化勝出;
    b=0 時兩者同分,插入序使長文件在前——排序翻轉即證明參數確實生效。
    """
    from selectrag.runner.__main__ import build_retriever
    from selectrag.runner.config import RetrieverConfig

    path = write_ctx_jsonl(
        tmp_path / "c.jsonl",
        [{"chunk_id": "long", "text": "aa cc dd ee ff"}, {"chunk_id": "short", "text": "aa bb"}],
    )
    default = build_retriever(RetrieverConfig(type="bm25", params={"chunks": str(path)}))
    assert list(default.retrieve("aa", top_k=2)) == ["short", "long"]
    flat = build_retriever(
        RetrieverConfig(type="bm25", params={"chunks": str(path), "k1": 1.5, "b": 0.0})
    )
    assert list(flat.retrieve("aa", top_k=2)) == ["long", "short"]


# 分數露出(拒答門檻用):retrieve_scored 與 retrieve 同序、分數遞減、只回有命中者
def test_bm25_retrieve_scored_matches_retrieve() -> None:
    docs = [("d1", "aspirin 100mg"), ("d2", "warfarin 5mg"), ("d3", "aspirin aspirin")]
    retriever = BM25Retriever(docs)
    scored = list(retriever.retrieve_scored("aspirin", 3))
    assert [d for d, _ in scored] == list(retriever.retrieve("aspirin", 3))
    assert all(a >= b for (_, a), (_, b) in zip(scored, scored[1:], strict=False))
    assert all(sc > 0.0 for _, sc in scored) and len(scored) == 2
    assert scored[0][0] == "d3" and scored[0][1] > scored[1][1]  # tf=2 嚴格高於 tf=1,分數不可同值
