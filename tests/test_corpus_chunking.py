"""載入器與切塊器測試 —— 場景為 2026-08-30 定案的驗收條件。

測試資料為合成內容,不依賴 data/corpus/(那在 gitignore 區,CI 取不到)。
"""

import csv
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.corpus.chunker import (
    chunk_csv_grouped,
    chunk_csv_per_row,
    chunk_document,
)
from selectrag.corpus.document import Document
from selectrag.corpus.loaders import load_csv_rows, load_html, load_pdf

# --------------------------------------------------------------------------
# 測試資料
# --------------------------------------------------------------------------

HTML_WITH_MAIN = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>測試</title></head>
<body><nav>導覽列 首頁 關於</nav>
<main><h1>壹、前言</h1><p>本方案自 115 年 8 月 1 日生效。</p>
<script>var noise = 1;</script></main>
<footer>頁尾 隱私權政策 累計瀏覽人次</footer></body></html>"""

HTML_NO_MAIN = "<html><body><div>沒有 main 標籤的內容</div></body></html>"


def make_document(text: str, source_type: str = "pdf") -> Document:
    return Document(
        doc_id="doc001",
        source_path="/synthetic/test.pdf",
        source_type=source_type,
        text=text,
        page_offsets=(0,),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


CSV_ROWS = [
    # 代碼用固定寬度,避免 f"0010{i}" 在 i>=10 時產生歧義字串
    {"代碼": f"{i:05d}B", "名稱": f"<診療項目 {i:02d}>", "點數": str(100 + i)}
    for i in range(1, 26)
]

# --------------------------------------------------------------------------
# HTML 載入器
# --------------------------------------------------------------------------


def test_load_html_extracts_main_only(tmp_path: Path) -> None:
    """只取 <main>,footer 與 nav 必須被排除。"""
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_MAIN, encoding="utf-8")
    doc = load_html(path)
    assert "壹、前言" in doc.text
    assert "本方案自 115 年 8 月 1 日生效。" in doc.text
    assert "導覽列" not in doc.text, "nav 不應進入正文"
    assert "頁尾" not in doc.text, "footer 不應進入正文"
    assert "noise" not in doc.text, "script 內容不應進入正文"
    assert doc.source_type == "html"


def test_load_html_rejects_missing_main(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(HTML_NO_MAIN, encoding="utf-8")
    with pytest.raises(ValueError, match="no main content"):
        load_html(path)


def test_load_html_output_is_normalised(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<html><body><main>⽤法⽤量,每⽇⼀次</main></body></html>", encoding="utf-8")
    assert "用法用量" in load_html(path).text


# --------------------------------------------------------------------------
# CSV 載入器
# --------------------------------------------------------------------------


def test_load_csv_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "data.csv", CSV_ROWS)
    rows = load_csv_rows(path)
    assert len(rows) == 25
    assert rows[0]["代碼"] == "00001B"


def test_load_csv_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_csv_rows(path)


def build_pdf(pages: list[list[str]]) -> bytes:
    """組出最小可用 PDF(Helvetica,每個字串一行)。

    不把二進位 fixture 放進版控:PDF 於測試時即時生成,內容可讀、可 diff。
    文字以 latin-1 編碼寫入,故測試內容一律用 ASCII。
    """
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode()
    objs: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n).encode() + b" >>"),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for pid, cid, lines in zip(page_ids, content_ids, pages, strict=True):
        objs.append(
            (
                pid,
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
                + str(cid).encode()
                + b" 0 R >>",
            )
        )
        body = b"BT /F1 12 Tf 72 720 Td 14 TL\n"
        for line in lines:
            esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += b"(" + esc.encode("latin-1") + b") Tj T*\n"
        body += b"ET"
        objs.append(
            (
                cid,
                b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
            )
        )
    objs.sort()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num, payload in objs:
        offsets[num] = len(out)
        out += str(num).encode() + b" 0 obj\n" + payload + b"\nendobj\n"
    xref = len(out)
    size = max(offsets) + 1
    out += b"xref\n0 " + str(size).encode() + b"\n0000000000 65535 f \n"
    for i in range(1, size):
        out += ("%010d 00000 n \n" % offsets[i]).encode()
    out += (
        b"trailer\n<< /Size "
        + str(size).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    path.write_bytes(build_pdf(pages))
    return path


# --------------------------------------------------------------------------
# PDF 載入器
# --------------------------------------------------------------------------


def test_load_pdf_extracts_text_and_metadata(tmp_path: Path) -> None:
    """PDF 有文字層時抽出正文,並帶上來源中繼資料。"""
    path = write_pdf(
        tmp_path / "guideline.pdf",
        [["1 Indications", "This product is used", "for the stated condition."]],
    )
    doc = load_pdf(path)
    assert doc.doc_id == "guideline"
    assert doc.source_type == "pdf"
    assert doc.source_path == str(path)
    assert "1 Indications" in doc.text
    assert "for the stated condition." in doc.text


def test_load_pdf_merges_wrapped_lines(tmp_path: Path) -> None:
    """未以句末標點結束、且下一行非標題的短行必須合併回同一行。

    合併不插入空白:語料為中文,PDF 斷行處補空白反而會在詞中間製造假分隔。
    """
    path = write_pdf(
        tmp_path / "wrapped.pdf",
        [["1 Indications", "This product is used", "for the stated condition."]],
    )
    lines = load_pdf(path).text.split("\n")
    assert "This product is usedfor the stated condition." in lines, lines


def test_load_pdf_records_page_offsets(tmp_path: Path) -> None:
    """每頁一個位移,首頁為 0 且嚴格遞增,供 chunk 回推頁碼。"""
    path = write_pdf(
        tmp_path / "twopage.pdf",
        [["1 Indications", "Page one body text."], ["2 Dosage", "Page two body text."]],
    )
    doc = load_pdf(path)
    assert len(doc.page_offsets) == 2
    assert doc.page_offsets[0] == 0
    assert doc.page_offsets[1] > 0
    assert "2 Dosage" in doc.text


def test_load_pdf_rejects_missing_text_layer(tmp_path: Path) -> None:
    """掃描件(無文字層)必須 raise,不得靜默回空文件——OCR 不在範圍內。"""
    path = write_pdf(tmp_path / "scanned.pdf", [[]])
    with pytest.raises(ValueError, match="no text layer"):
        load_pdf(path)


# --------------------------------------------------------------------------
# 結構切塊(方案 C)
# --------------------------------------------------------------------------


def test_chunk_keeps_short_unit_intact() -> None:
    """短單元保持完整,不被切開。"""
    text = "壹、前言\n本方案自 115 年 8 月 1 日生效。\n一、目的\n提升照護品質。"
    chunks = chunk_document(make_document(text), min_chars=0)  # 純結構模式
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("壹、前言",)
    assert "本方案自" in chunks[0].text
    assert chunks[1].heading_path == (
        "壹、前言",
        "一、目的",
    ), "level 2 標題在 level 1 之下,路徑累積"


def test_chunk_builds_nested_heading_path() -> None:
    """階層路徑必須累積,這是 citation 的憑據(§5.3)。"""
    text = "壹、前言\n開場說明。\n一、收案條件\n條件說明。\n(一) 適用對象\n對象說明。"
    chunks = chunk_document(make_document(text), min_chars=0)  # 純結構模式
    paths = [c.heading_path for c in chunks]
    assert ("壹、前言",) in paths
    assert ("壹、前言", "一、收案條件") in paths
    assert ("壹、前言", "一、收案條件", "(一) 適用對象") in paths


def test_chunk_splits_oversized_unit_and_keeps_path() -> None:
    """超長單元切分後,每塊仍帶相同的 heading_path。"""
    body = "甲" * 2000
    chunks = chunk_document(make_document(f"壹、前言\n{body}"), max_chars=800, overlap=100)
    assert len(chunks) >= 3
    assert all(c.heading_path == ("壹、前言",) for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)


def test_chunk_skips_empty_units() -> None:
    """標題下無內容者不產生 chunk。"""
    text = "壹、前言\n一、目的\n實際內容在這裡。"
    chunks = chunk_document(make_document(text), min_chars=0)
    assert all(c.text.strip() for c in chunks)
    assert len(chunks) == 1


def test_chunk_ids_are_stable() -> None:
    """同一輸入重跑必須產生相同 id —— 否則結果無法比對。"""
    doc = make_document("壹、前言\n內容甲乙丙。\n一、目的\n內容丁戊己。")
    first = [c.chunk_id for c in chunk_document(doc, min_chars=0)]
    second = [c.chunk_id for c in chunk_document(doc, min_chars=0)]
    assert first == second
    assert len(set(first)) == len(first), "chunk_id 不得重複"


def test_chunk_records_char_span() -> None:
    doc = make_document("壹、前言\n內容甲乙丙。")
    chunk = chunk_document(doc, min_chars=0)[0]
    start, end = chunk.char_span
    assert 0 <= start < end <= len(doc.text)


def test_chunk_rejects_empty_document() -> None:
    with pytest.raises(ValueError, match="empty document"):
        chunk_document(make_document("   "))


def test_chunk_rejects_invalid_parameters() -> None:
    doc = make_document("壹、前言\n內容。")
    with pytest.raises(ValueError, match="max_chars must be"):
        chunk_document(doc, max_chars=0)
    with pytest.raises(ValueError, match="overlap must be"):
        chunk_document(doc, max_chars=100, overlap=100)


# --------------------------------------------------------------------------
# CSV 雙版本(使用者拍板的對照)
# --------------------------------------------------------------------------


def test_csv_per_row_produces_one_chunk_each() -> None:
    chunks = chunk_csv_per_row(CSV_ROWS, doc_id="csv001", source_path="/synthetic/d.csv")
    assert len(chunks) == 25
    assert "00001B" in chunks[0].text
    assert "代碼" in chunks[0].text, "應含欄位名,讓 chunk 自我描述"
    assert chunks[0].source_type == "csv"


def test_csv_grouped_merges_rows() -> None:
    chunks = chunk_csv_grouped(
        CSV_ROWS, doc_id="csv001", source_path="/synthetic/d.csv", group_size=10
    )
    assert len(chunks) == 3
    assert "00001B" in chunks[0].text and "00010B" in chunks[0].text
    assert len(chunks[0].text) > len(
        chunk_csv_per_row(CSV_ROWS, "csv001", "/synthetic/d.csv")[0].text
    )


def test_csv_both_versions_cover_all_rows() -> None:
    """兩種切法都不得遺漏任何一列。"""
    per_row = chunk_csv_per_row(CSV_ROWS, "c", "/p")
    grouped = chunk_csv_grouped(CSV_ROWS, "c", "/p", group_size=10)
    for row in CSV_ROWS:
        assert any(row["代碼"] in c.text for c in per_row)
        assert any(row["代碼"] in c.text for c in grouped)


def test_csv_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="no rows"):
        chunk_csv_per_row([], "c", "/p")


# --------------------------------------------------------------------------
# Property-based(Tier 3)
# --------------------------------------------------------------------------

body_text = st.text(alphabet="甲乙丙丁戊己庚辛。", min_size=1, max_size=200)


@given(body=body_text)
@settings(max_examples=150)
def test_property_chunks_respect_max_chars(body: str) -> None:
    """所有 chunk 不得超過上限 —— 失效模式:切塊失控。"""
    doc = make_document(f"壹、前言\n{body}")
    for chunk in chunk_document(doc, max_chars=100, overlap=20, min_chars=0):
        assert len(chunk.text) <= 100


@given(body=body_text)
@settings(max_examples=150)
def test_property_chunks_are_lossless(body: str) -> None:
    """所有 chunk 串接後必須涵蓋原文全部字元 —— 失效模式 4(內容遺失)。"""
    doc = make_document(f"壹、前言\n{body}")
    combined = "".join(c.text for c in chunk_document(doc, max_chars=100, overlap=20, min_chars=0))
    for char in body:
        assert char in combined


@given(body=body_text)
def test_property_chunk_text_is_nfkc_stable(body: str) -> None:
    import unicodedata

    doc = make_document(f"壹、前言\n{body}")
    for chunk in chunk_document(doc, min_chars=0):
        assert unicodedata.normalize("NFKC", chunk.text) == chunk.text


@given(rows=st.integers(min_value=1, max_value=40), group=st.integers(min_value=1, max_value=8))
def test_property_csv_grouping_covers_every_row(rows: int, group: int) -> None:
    data = [{"代碼": f"X{i:03d}", "名稱": f"<項目 {i}>"} for i in range(rows)]
    chunks = chunk_csv_grouped(data, "c", "/p", group_size=group)
    assert len(chunks) == (rows + group - 1) // group
    for row in data:
        assert any(row["代碼"] in c.text for c in chunks)


# --------------------------------------------------------------------------
# min_chars 合併(修正過度碎片化)
# --------------------------------------------------------------------------


def test_short_units_are_merged_to_reach_minimum() -> None:
    """相鄰的短單元應合併,而非各自成為碎片 chunk。"""
    text = "壹、前言\n甲乙丙。\n一、目的\n丁戊己。\n二、依據\n庚辛壬。"
    chunks = chunk_document(make_document(text), max_chars=800, min_chars=15)
    assert len(chunks) == 1, "三個短單元(各 4 字元)應合併為一塊"
    assert "甲乙丙" in chunks[0].text and "庚辛壬" in chunks[0].text


def test_merged_chunk_takes_common_heading_prefix() -> None:
    """合併後的路徑取共同前綴 —— 不能謊稱屬於某個特定子節。"""
    text = "壹、前言\n一、目的\n甲乙丙。\n二、依據\n丁戊己。"
    chunks = chunk_document(make_document(text), max_chars=800, min_chars=15)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("壹、前言",), "跨兩個 level-2 節,路徑退到共同的 level-1"


def test_merge_does_not_cross_unrelated_top_level() -> None:
    """無共同前綴者不合併 —— 寧可留短 chunk,也不把不相干的內容黏在一起。"""
    text = "壹、前言\n甲乙丙。\n貳、實施方式\n丁戊己。"
    chunks = chunk_document(make_document(text), max_chars=800, min_chars=100)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("壹、前言",)
    assert chunks[1].heading_path == ("貳、實施方式",)


def test_merge_respects_max_chars() -> None:
    """合併不得撐破上限。"""
    text = "壹、前言\n" + "\n".join(f"一、第{i}節\n{'甲' * 90}。" for i in range(1, 6))
    chunks = chunk_document(make_document(text), max_chars=200, min_chars=150)
    assert all(len(c.text) <= 200 for c in chunks)
    assert len(chunks) >= 3


def test_min_chars_must_be_below_max_chars() -> None:
    doc = make_document("壹、前言\n內容甲乙丙。")
    with pytest.raises(ValueError, match="min_chars must be"):
        chunk_document(doc, max_chars=100, overlap=10, min_chars=100)


@given(
    body=st.lists(st.text(alphabet="甲乙丙。", min_size=2, max_size=20), min_size=3, max_size=12)
)
@settings(max_examples=150)
def test_property_no_two_consecutive_short_chunks(body: list[str]) -> None:
    """不得出現連續兩個過短的 chunk —— 那代表合併邏輯失效。

    例外:路徑無共同前綴者不合併,但本測試的資料全在同一頂層之下。
    """
    text = "壹、前言\n" + "\n".join(f"一、第{i}節\n{seg}" for i, seg in enumerate(body))
    chunks = chunk_document(make_document(text), max_chars=800, min_chars=100)
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert not (
            len(first.text) < 100 and len(second.text) < 100
        ), f"連續兩塊都過短:{len(first.text)} / {len(second.text)}"


@given(
    unit_len=st.integers(min_value=250, max_value=900),
    max_chars=st.integers(min_value=50, max_value=200),
)
@settings(max_examples=150)
def test_property_oversized_units_are_actually_split(unit_len: int, max_chars: int) -> None:
    """超長單元必定被切分 —— 且每塊都在上限內。

    補洞:既有的 max_chars property 用 st.text(max_size=200) 產生內容,
    實測 150 個樣本中長度 >100 的有 0 個(中位僅 5 字元),
    從未真正考驗切分邏輯(property-only 變異測試證實)。改用明確長度生成。
    """
    doc = make_document("壹、前言\n" + "甲" * unit_len)
    chunks = chunk_document(doc, max_chars=max_chars, overlap=max_chars // 5, min_chars=0)
    assert len(chunks) > 1, f"{unit_len} 字元的單元在上限 {max_chars} 下必須被切開"
    assert all(len(c.text) <= max_chars for c in chunks)


@given(
    body_a=st.integers(min_value=5, max_value=40),
    body_b=st.integers(min_value=5, max_value=40),
)
@settings(max_examples=150)
def test_property_merge_never_crosses_unrelated_top_level(body_a: int, body_b: int) -> None:
    """無共同前綴的頂層之間永不合併 —— 失效模式 5(citation 錯位)。

    補洞:既有 property 的測試資料都在同一頂層之下,從未考驗這條保護。
    """
    text = f"壹、甲章\n{'甲' * body_a}\n貳、乙章\n{'乙' * body_b}"
    chunks = chunk_document(make_document(text), max_chars=800, min_chars=500)
    assert len(chunks) == 2, "不同頂層不得合併,即使兩者都短於 min_chars"
    for chunk in chunks:
        assert not ("甲" in chunk.text and "乙" in chunk.text)


# --------------------------------------------------------------------------
# doc_title(2026-09-01)
# --------------------------------------------------------------------------


def test_chunk_document_backfills_doc_title(tmp_path: Path) -> None:
    """PDF/HTML 文件的所有 chunk 之 doc_title == 全文第一個非空行。"""
    doc = make_document("藥品名稱錠 DRUGNAME TABLETS\n1 適應症\n本藥用於某症狀之治療。" * 3)
    chunks = chunk_document(doc, max_chars=100, overlap=10, min_chars=20)
    assert chunks, "至少要有一個 chunk"
    for chunk in chunks:
        assert chunk.doc_title == "藥品名稱錠 DRUGNAME TABLETS"


def test_csv_chunks_use_doc_id_as_title(tmp_path: Path) -> None:
    """CSV chunk 的 doc_title == 資料集名(doc_id)。"""
    rows = [{"代碼": "00001B", "名稱": "測試項目"}]
    for chunk in chunk_csv_per_row(rows, "支付標準資料集", "/synthetic/x.csv"):
        assert chunk.doc_title == "支付標準資料集"
    for chunk in chunk_csv_grouped(rows, "支付標準資料集", "/synthetic/x.csv"):
        assert chunk.doc_title == "支付標準資料集"


# --------------------------------------------------------------------------
# 覆蓋率缺口補強(改善項目 11)
# --------------------------------------------------------------------------


def test_chunk_csv_grouped_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="no rows"):
        chunk_csv_grouped([], "ds", "/synthetic/x.csv")
    with pytest.raises(ValueError, match="group_size"):
        chunk_csv_grouped([{"代碼": "00001B"}], "ds", "/synthetic/x.csv", group_size=0)


def test_load_pdf_skips_blank_page_but_keeps_offset(tmp_path: Path) -> None:
    """空白頁不產生段落,但頁位移仍逐頁記錄(否則後頁的 page 會回推錯誤)。"""
    path = write_pdf(tmp_path / "blank2.pdf", [["1 Indications", "Body text here."], []])
    doc = load_pdf(path)
    assert len(doc.page_offsets) == 2
    assert "Body text here." in doc.text
