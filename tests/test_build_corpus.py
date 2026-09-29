"""語料建置 CLI 測試 —— 前次 __main__.py 覆蓋率 0% 的教訓。"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_corpus import build_chunks, main, summarise, write_jsonl  # noqa: E402

HTML_DOC = """<html><body><nav>導覽</nav>
<main><h1>壹、前言</h1><p>本方案自 115 年 8 月 1 日生效,適用於全體特約醫事服務機構。</p>
<h2>一、目的</h2><p>提升照護品質並確保給付的一致性與可稽核性。</p></main>
<footer>頁尾</footer></body></html>"""


def make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)  # 同一測試可能呼叫多次
    (corpus / "doc.html").write_text(HTML_DOC, encoding="utf-8")
    rows = [{"代碼": f"{i:05d}B", "名稱": f"<項目 {i}>"} for i in range(1, 13)]
    with (corpus / "table.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["代碼", "名稱"])
        writer.writeheader()
        writer.writerows(rows)
    (corpus / "readme.txt").write_text("非語料檔,應被略過", encoding="utf-8")
    return corpus


def test_build_chunks_handles_all_supported_types(tmp_path: Path) -> None:
    chunks, failures = build_chunks(make_corpus(tmp_path))
    assert not failures
    types = {c.source_type for c in chunks}
    assert types == {"html", "csv"}, "不支援的副檔名應被略過而非失敗"


def test_build_chunks_csv_mode_grouped(tmp_path: Path) -> None:
    per_row, _ = build_chunks(make_corpus(tmp_path), csv_mode="per-row")
    grouped, _ = build_chunks(make_corpus(tmp_path), csv_mode="grouped", group_size=10)
    csv_per_row = [c for c in per_row if c.source_type == "csv"]
    csv_grouped = [c for c in grouped if c.source_type == "csv"]
    assert len(csv_per_row) == 12
    assert len(csv_grouped) == 2


def test_build_chunks_collects_failures_without_aborting(tmp_path: Path) -> None:
    """單檔失敗不中止整批,但必須被回報 —— 不得靜默略過。"""
    corpus = make_corpus(tmp_path)
    (corpus / "broken.html").write_text("<html><body>沒有 main</body></html>", encoding="utf-8")
    chunks, failures = build_chunks(corpus)
    assert chunks, "其餘檔案仍應產出 chunk"
    assert len(failures) == 1
    assert failures[0][0] == "broken.html"
    assert "no main content" in failures[0][1]


def test_build_chunks_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        build_chunks(tmp_path / "nope")


def test_build_chunks_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no documents found"):
        build_chunks(empty)


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    chunks, _ = build_chunks(make_corpus(tmp_path))
    out = tmp_path / "nested" / "chunks.jsonl"
    write_jsonl(chunks, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(chunks)
    first = json.loads(lines[0])
    assert set(first) == {
        "chunk_id",
        "doc_id",
        "text",
        "source_path",
        "source_type",
        "page",
        "heading_path",
        "char_span",
        "doc_title",  # 2026-09-01:文件標題入 chunk
    }


def test_write_jsonl_keeps_chinese_readable(tmp_path: Path) -> None:
    chunks, _ = build_chunks(make_corpus(tmp_path))
    out = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, out)
    assert "本方案" in out.read_text(encoding="utf-8"), "不應被 \\uXXXX 轉義"


def test_summarise_reports_distribution(tmp_path: Path) -> None:
    chunks, _ = build_chunks(make_corpus(tmp_path))
    text = summarise(chunks)
    assert "總計" in text and "中位=" in text and "過短" in text


def test_summarise_handles_empty() -> None:
    assert "無 chunk" in summarise([])


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["prog", str(corpus), str(out), "--csv-mode", "grouped"]) == 0
    assert out.exists()
    assert "總計" in capsys.readouterr().out


def test_cli_reports_failures_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(tmp_path)
    (corpus / "broken.html").write_text("<html><body>無 main</body></html>", encoding="utf-8")
    assert main(["prog", str(corpus), str(tmp_path / "o.jsonl")]) == 0
    assert "broken.html" in capsys.readouterr().err
