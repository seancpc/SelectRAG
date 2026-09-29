"""Reranker 測試 —— 場景為 2026-09-14 定案的驗收條件。id / 文字皆為中性占位符,不載入真模型。"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectrag.retrieval.rerank import NIMReranker, RerankRetriever, load_chunk_rows
from selectrag.retrieval.scripted import ScriptedRetriever
from selectrag.runner.config import RetrieverConfig

TOOLS = Path(__file__).resolve().parent.parent / "tools"
Q = "<問題>"
ROWS = {
    f"c{i}": {
        "chunk_id": f"c{i}",
        "text": f"<正文 {i}>",
        "doc_title": f"<標題 {i}>",
        "heading_path": [],
    }
    for i in range(8)
}


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TableReranker:
    """替身:依「文本 → 分數」表打分;記錄收到的文本供斷言。"""

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self._table = table
        self._default = default
        self.seen: list[list[str]] = []

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        self.seen.append(list(texts))
        return [self._table.get(t, self._default) for t in texts]


class BrokenReranker:
    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        return [1.0] * (len(texts) + 1)


class RecordingBase:
    def __init__(self, result: Sequence[str]) -> None:
        self.calls: list[int] = []
        self._result = list(result)

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        self.calls.append(top_k)
        return self._result[:top_k]


def _rr(
    base: Sequence[str],
    scores: dict[str, float],
    candidates: int = 10,
    fields: tuple[str, ...] = (),
) -> RerankRetriever:
    table = {ROWS[c]["text"]: s for c, s in scores.items()} | {
        f"{ROWS[c]['doc_title']} {ROWS[c]['text']}": s for c, s in scores.items()
    }
    return RerankRetriever(
        ScriptedRetriever({Q: base}), TableReranker(table), ROWS, candidates, fields
    )


# 1~6 手算案例


def test_rerank_orders_by_score_desc() -> None:
    r = _rr(["c0", "c1", "c2"], {"c0": 0.2, "c1": 0.9, "c2": 0.5})
    assert list(r.retrieve(Q, 3)) == ["c1", "c2", "c0"]


def test_rerank_tie_keeps_base_order() -> None:
    r = _rr(["c0", "c1", "c2"], {"c0": 0.5, "c1": 0.5, "c2": 0.9})
    assert list(r.retrieve(Q, 3)) == ["c2", "c0", "c1"]


def test_rerank_truncates_to_top_k() -> None:
    r = _rr(["c0", "c1", "c2"], {"c0": 0.2, "c1": 0.9, "c2": 0.5})
    assert list(r.retrieve(Q, 2)) == ["c1", "c2"]


def test_rerank_base_called_with_candidates() -> None:
    base = RecordingBase([f"c{i}" for i in range(8)])
    RerankRetriever(base, TableReranker({}), ROWS, candidates=6).retrieve(Q, 2)
    assert base.calls == [6]


def test_rerank_text_uses_context_fields() -> None:
    plain = TableReranker({})
    RerankRetriever(ScriptedRetriever({Q: ["c3"]}), plain, ROWS, 5, ()).retrieve(Q, 1)
    assert plain.seen == [["<正文 3>"]]
    ctx = TableReranker({})
    RerankRetriever(ScriptedRetriever({Q: ["c3"]}), ctx, ROWS, 5, ("doc_title",)).retrieve(Q, 1)
    assert ctx.seen == [["<標題 3> <正文 3>"]]


def test_rerank_unknown_chunk_raises() -> None:
    r = RerankRetriever(ScriptedRetriever({Q: ["zz"]}), TableReranker({}), ROWS, 5)
    with pytest.raises(ValueError, match="unknown chunk"):
        r.retrieve(Q, 1)


def test_rerank_rejects_bad_candidates_and_top_k() -> None:
    with pytest.raises(ValueError, match="candidates"):
        RerankRetriever(ScriptedRetriever({Q: ["c0"]}), TableReranker({}), ROWS, 0)
    with pytest.raises(ValueError, match="top_k"):
        _rr(["c0"], {}).retrieve(Q, 0)


def test_rerank_score_length_mismatch_raises() -> None:
    r = RerankRetriever(ScriptedRetriever({Q: ["c0", "c1"]}), BrokenReranker(), ROWS, 5)
    with pytest.raises(ValueError, match="scores"):
        r.retrieve(Q, 2)


def test_rerank_empty_candidates_returns_empty() -> None:
    rr = TableReranker({})
    out = RerankRetriever(ScriptedRetriever({Q: []}), rr, ROWS, 5).retrieve(Q, 3)
    assert list(out) == [] and rr.seen == []


# 7 property

_ids = st.lists(st.sampled_from(list(ROWS)), min_size=1, max_size=8, unique=True)
_scores = st.dictionaries(
    st.sampled_from(list(ROWS)), st.floats(-5, 5, width=32), min_size=0, max_size=8
)


@settings(max_examples=100)
@given(base=_ids, scores=_scores, k=st.integers(1, 8), cand=st.integers(1, 8))
def test_property_output_subset_and_budget(
    base: list[str], scores: dict[str, float], k: int, cand: int
) -> None:
    out = list(_rr(base, scores, candidates=cand).retrieve(Q, k))
    assert set(out) <= set(base[:cand])
    assert len(out) == len(set(out)) and len(out) <= k


@settings(max_examples=100)
@given(base=_ids, scores=_scores, data=st.data())
def test_property_prefix_consistency(
    base: list[str], scores: dict[str, float], data: st.DataObject
) -> None:
    n = data.draw(st.integers(1, 8))
    m = data.draw(st.integers(1, n))
    assert list(_rr(base, scores).retrieve(Q, m)) == list(_rr(base, scores).retrieve(Q, n))[:m]


@settings(max_examples=100)
@given(base=_ids, scores=_scores)
def test_property_scores_monotone_and_ties_keep_base_order(
    base: list[str], scores: dict[str, float]
) -> None:
    out = list(_rr(base, scores).retrieve(Q, 8))
    s = [scores.get(c, 0.0) for c in out]
    assert all(a >= b for a, b in zip(s, s[1:], strict=False))
    for a, b in zip(out, out[1:], strict=False):
        if scores.get(a, 0.0) == scores.get(b, 0.0):
            assert base.index(a) < base.index(b)


# load_chunk_rows


def test_load_chunk_rows(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in ROWS.values()]
    lines.insert(3, "")  # 空白行必須被跳過,不得成為一筆
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = load_chunk_rows(p)
    assert set(rows) == set(ROWS) and rows["c2"]["text"] == "<正文 2>"
    assert len(rows) == len(ROWS)
    p.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_id"):
        load_chunk_rows(p)


# 8 runner


def _chunks_file(tmp_path: Path) -> Path:
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS.values()) + "\n", encoding="utf-8"
    )
    return p


def test_build_retriever_rerank_nested_hybrid(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    sa, sb = tmp_path / "a.json", tmp_path / "b.json"
    sa.write_text(json.dumps({Q: ["c0", "c1", "c2"]}), encoding="utf-8")
    sb.write_text(json.dumps({Q: ["c1", "c2", "c3"]}), encoding="utf-8")
    cfg = RetrieverConfig(
        "rerank",
        {
            "candidates": 4,
            "model": "<m>",
            "revision": "<r>",
            "chunks": str(_chunks_file(tmp_path)),
            "context_fields": ["doc_title"],
            "retriever": {
                "type": "hybrid",
                "params": {
                    "pool": 3,
                    "rrf_k": 60,
                    "retrievers": [
                        {"type": "scripted", "params": {"script": str(sa)}},
                        {"type": "scripted", "params": {"script": str(sb)}},
                    ],
                },
            },
        },
    )
    table = {"<標題 3> <正文 3>": 0.9, "<標題 0> <正文 0>": 0.8}
    r = build_retriever(cfg, reranker_factory=lambda m, rev: TableReranker(table))
    assert isinstance(r, RerankRetriever)
    # hybrid 候選:c1, c2, c0, c3;重排:c3 0.9、c0 0.8、其餘 0 依 hybrid 名次
    assert list(r.retrieve(Q, 4)) == ["c3", "c0", "c1", "c2"]


def test_rerank_requires_params(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    base = {"type": "scripted", "params": {"script": "x"}}
    full = {"candidates": 5, "model": "<m>", "chunks": "c", "retriever": base}
    for missing in ("retriever", "candidates", "model", "chunks"):
        params = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            build_retriever(RetrieverConfig("rerank", params), reranker_factory=lambda m, r: None)


def test_rerank_uses_injected_factory(tmp_path: Path) -> None:
    from selectrag.runner.__main__ import build_retriever

    sa = tmp_path / "a.json"
    sa.write_text(json.dumps({Q: ["c0", "c1"]}), encoding="utf-8")
    seen: list[tuple[str, str | None]] = []

    def factory(model: str, revision: str | None) -> TableReranker:
        seen.append((model, revision))
        return TableReranker({"<正文 1>": 1.0})

    cfg = RetrieverConfig(
        "rerank",
        {
            "candidates": 2,
            "model": "<m>",
            "revision": "<r>",
            "chunks": str(_chunks_file(tmp_path)),
            "retriever": {"type": "scripted", "params": {"script": str(sa)}},
        },
    )
    r = build_retriever(cfg, reranker_factory=factory)
    assert seen == [("<m>", "<r>")]
    assert list(r.retrieve(Q, 2)) == ["c1", "c0"]


# 9 tools/rerank_grid.py


def test_hard_negative_check_hand_computed(tmp_path: Path) -> None:
    mod = _load_tool("rerank_grid")
    from selectrag.evaluation.eval_set import EvalItem, freeze

    eval_path = tmp_path / "e.json"
    items = [
        EvalItem("q1", "<q1>", frozenset({"c0"})),
        EvalItem("q2", "<q2>", frozenset({"c1", "c2"})),
    ]
    eval_path.write_text(json.dumps(freeze(items, "t", "t")), encoding="utf-8")
    hn = tmp_path / "hn.json"
    hn.write_text(
        json.dumps(
            {
                "description": "t",
                "items": [
                    {"question_id": "q1", "chunk_id": "c5", "type": "ta", "note": ""},
                    {"question_id": "q2", "chunk_id": "c6", "type": "tb", "note": ""},
                    {"question_id": "q2", "chunk_id": "c7", "type": "tb", "note": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    # q1:真值 c0 0.9 > c5 0.1 ✓;q2:真值最高 c2 0.6 > c6 0.7?否 → ✗;c7 0.2 → ✓
    table = {
        "<正文 0>": 0.9,
        "<正文 5>": 0.1,
        "<正文 1>": 0.3,
        "<正文 2>": 0.6,
        "<正文 6>": 0.7,
        "<正文 7>": 0.2,
    }
    report = mod.hard_negative_check(eval_path, hn, ROWS, TableReranker(table), ())
    assert report["total"] == 3 and report["truth_wins"] == 2
    assert report["ratio"] == pytest.approx(2 / 3)
    assert [d["chunk_id"] for d in report["details"] if not d["truth_wins"]] == ["c6"]
    assert report["by_type"] == {
        "ta": {"total": 1, "truth_wins": 1, "ratio": 1.0},
        "tb": {"total": 2, "truth_wins": 1, "ratio": 0.5},
    }
    bad = tmp_path / "hn2.json"
    bad.write_text(
        json.dumps({"items": [{"question_id": "q9", "chunk_id": "c5"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="q9"):
        mod.hard_negative_check(eval_path, bad, ROWS, TableReranker(table), ())


def test_rerank_grid_runs_with_fake_and_refuses_overwrite(tmp_path: Path) -> None:
    mod = _load_tool("rerank_grid")
    from selectrag.evaluation.eval_set import EvalItem, freeze

    eval_path = tmp_path / "e.json"
    items = [EvalItem("q1", "<q1>", frozenset({"c1"})), EvalItem("q2", "<q2>", frozenset({"c3"}))]
    eval_path.write_text(json.dumps(freeze(items, "t", "t")), encoding="utf-8")
    sa = tmp_path / "a.json"
    sa.write_text(json.dumps({"<q1>": ["c0", "c1", "c2"], "<q2>": ["c2", "c3"]}), encoding="utf-8")
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        f"experiment_id: base\neval_set: {eval_path}\nretriever:\n  type: scripted\n"
        f"  params:\n    script: {sa}\nk_values: [1, 2]\nresults_dir: {tmp_path / 'r'}\n",
        encoding="utf-8",
    )
    chunks = _chunks_file(tmp_path)
    out = tmp_path / "grid"
    argv = [str(base_cfg), str(chunks), str(out), "--candidates", "2", "3", "--reranker", "fake"]
    assert mod.main(argv) == 0
    names = sorted(p.name for p in out.glob("cand*.json"))
    assert names == ["cand2_ctx.json", "cand2_plain.json", "cand3_ctx.json", "cand3_plain.json"]
    got = json.loads((out / "cand3_plain.json").read_text(encoding="utf-8"))
    assert "recall@1" in got["metrics"]
    assert mod.main(argv) != 0


# ---- NIM 客戶端(本機假伺服器,不連外)----


def _serve(handler_body: bytes, status: int = 200) -> tuple[HTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        received: list[dict[str, object]] = []

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers["Content-Length"])
            H.received.append(json.loads(self.rfile.read(n)))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(handler_body)

        def log_message(self, *args: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_nim_scores_restored_by_index() -> None:
    body = json.dumps(
        {
            "rankings": [
                {"index": 2, "logit": 3.0},
                {"index": 0, "logit": 1.0},
                {"index": 1, "logit": -2.0},
            ]
        }
    )
    srv, url = _serve(body.encode())
    try:
        got = NIMReranker(url, "<模型>").score(Q, ["<a>", "<b>", "<c>"])
        req = srv.RequestHandlerClass.received[-1]  # type: ignore[attr-defined]
    finally:
        srv.shutdown()
    assert got == [1.0, -2.0, 3.0]
    assert req["model"] == "<模型>"
    assert req["query"] == {"text": Q}
    assert req["passages"] == [{"text": "<a>"}, {"text": "<b>"}, {"text": "<c>"}]
    assert srv.RequestHandlerClass.received  # type: ignore[attr-defined]


def test_nim_empty_texts_skips_request() -> None:
    srv, url = _serve(b"{}")
    try:
        assert NIMReranker(url, "<模型>").score(Q, []) == []
        assert srv.RequestHandlerClass.received == []  # type: ignore[attr-defined]
    finally:
        srv.shutdown()


@pytest.mark.parametrize(
    "rankings",
    [
        [{"index": 0, "logit": 1.0}],  # 缺一筆
        [{"index": 0, "logit": 1.0}, {"index": 0, "logit": 2.0}],  # 重複
        [{"index": 0, "logit": 1.0}, {"index": 5, "logit": 2.0}],  # 越界
    ],
)
def test_nim_bad_index_raises(rankings: list[dict[str, float]]) -> None:
    srv, url = _serve(json.dumps({"rankings": rankings}).encode())
    try:
        with pytest.raises(ValueError):
            NIMReranker(url, "<模型>").score(Q, ["<a>", "<b>"])
    finally:
        srv.shutdown()


def test_nim_http_error_raises() -> None:
    srv, url = _serve(b"boom", status=500)
    try:
        with pytest.raises(urllib.error.HTTPError):
            NIMReranker(url, "<模型>").score(Q, ["<a>"])
    finally:
        srv.shutdown()


def test_make_reranker_nim_requires_url_and_model() -> None:
    grid = _load_tool("rerank_grid")
    with pytest.raises(ValueError):
        grid.make_reranker("nim", None)
    rr, model, rev = grid.make_reranker("nim", "<tag>", "http://127.0.0.1:1", "<模型>")
    assert isinstance(rr, NIMReranker) and model == "nim:<模型>" and rev == "<tag>"
    with pytest.raises(ValueError):
        grid.make_reranker("nope", None)


# 分數露出(拒答門檻用,2026-09-19):retrieve_scored 與 retrieve 同序,分數為 reranker 分數
def test_rerank_retrieve_scored_matches_retrieve_and_scores() -> None:
    r = _rr(["c0", "c1", "c2"], {"c0": 0.2, "c1": 0.9, "c2": 0.5})
    scored = list(r.retrieve_scored(Q, 2))
    assert scored == [("c1", 0.9), ("c2", 0.5)]
    assert [c for c, _ in scored] == list(r.retrieve(Q, 2))


def test_rerank_retrieve_scored_empty_base() -> None:
    r = _rr([], {})
    assert list(r.retrieve_scored(Q, 3)) == []
