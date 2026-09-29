"""B 軸工具測試。

假 OpenAI 串流伺服器 → bench_serve 客戶端 → bench_compare;build_prompts 以 scripted 檢索器。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from selectrag.evaluation.eval_set import EvalItem, freeze
from selectrag.retrieval.scripted import ScriptedRetriever

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fake = _load_tool("fake_openai_server")
serve_tool = _load_tool("bench_serve")
compare_tool = _load_tool("bench_compare")
prompts_tool = _load_tool("build_prompts")


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    srv = fake.serve(0, delay_s=0.0, drift_after=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _prompts(path: Path, n: int = 6) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps({"id": f"q{i}", "messages": [{"role": "user", "content": f"<問 {i}>"}]})
                + "\n"
            )
    return path


def test_bench_serve_end_to_end_and_refuses_overwrite(server: str, tmp_path: Path) -> None:
    p = _prompts(tmp_path / "p.jsonl")
    out = tmp_path / "bench"
    argv = [
        str(p),
        str(out),
        "--url",
        server,
        "--model",
        "<m>",
        "--engine",
        "fake",
        "--fmt",
        "x",
        "--concurrency",
        "1",
        "2",
        "--warmup",
        "1",
        "--no-vram",
    ]
    assert serve_tool.main(argv) == 0
    rec = json.loads((out / "fake_x_c2.json").read_text(encoding="utf-8"))
    assert rec["concurrency"] == 2 and rec["summary"]["n"] == 6
    assert rec["summary"]["tokens_source"] == "usage"
    assert all(r["ttft_s"] > 0 and r["latency_s"] >= r["ttft_s"] for r in rec["requests"])
    assert all(r["text"].startswith("<回覆:") for r in rec["requests"])
    assert rec["vram_peak_mib"] is None
    assert serve_tool.main(argv) == 3  # 拒絕覆蓋
    assert (
        serve_tool.main(
            [
                str(p),
                str(tmp_path / "o2"),
                "--url",
                server,
                "--model",
                "m",
                "--engine",
                "e",
                "--fmt",
                "f",
                "--concurrency",
                "0",
            ]
        )
        == 2
    )


def test_bench_compare_detects_drift(server: str, tmp_path: Path) -> None:
    p = _prompts(tmp_path / "p.jsonl", n=4)
    base = [
        str(p),
        str(tmp_path / "b"),
        "--url",
        server,
        "--model",
        "m",
        "--fmt",
        "x",
        "--concurrency",
        "1",
        "--warmup",
        "0",
        "--no-vram",
    ]
    assert serve_tool.main([*base, "--engine", "a"]) == 0
    drift = fake.serve(0, delay_s=0.0, drift_after=2)  # 第 3 題起多一個字元
    threading.Thread(target=drift.serve_forever, daemon=True).start()
    try:
        base2 = [*base]
        base2[base2.index("--url") + 1] = f"http://127.0.0.1:{drift.server_port}"
        assert serve_tool.main([*base2, "--engine", "b"]) == 0
    finally:
        drift.shutdown()
    r = compare_tool.report(tmp_path / "b" / "a_x_c1.json", tmp_path / "b" / "b_x_c1.json")
    assert r["n"] == 4 and r["exact_match"] == 2
    assert r["len_diff_p95"] == 1 and r["divergence_p50_char"] is not None
    out = tmp_path / "cmp.json"
    assert (
        compare_tool.main(
            [
                str(tmp_path / "b" / "a_x_c1.json"),
                str(tmp_path / "b" / "b_x_c1.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["exact_rate"] == 0.5
    assert (
        compare_tool.main(
            [
                str(tmp_path / "b" / "a_x_c1.json"),
                str(tmp_path / "b" / "b_x_c1.json"),
                "--out",
                str(out),
            ]
        )
        == 3
    )  # 拒絕覆蓋


def test_build_prompts_with_scripted_retriever(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    rows = [{"chunk_id": f"c{i}", "doc_id": "D", "text": f"<正文 {i}>"} for i in range(4)]
    chunks.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    ev = tmp_path / "eval.json"
    ev.write_text(
        json.dumps(freeze([EvalItem("q1", "<問題>", frozenset({"c0"}))], "t", "t")),
        encoding="utf-8",
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""experiment_id: t
eval_set: {ev}
retriever:
  type: scripted
  params:
    script: {tmp_path}/script.json
k_values: [1]
results_dir: {tmp_path}
prompt:
  top_k: 2
  chunks: {chunks}
  system: "<系統>"
  template: "問:{{question}}\\n{{context}}"
""",
        encoding="utf-8",
    )
    (tmp_path / "script.json").write_text(
        json.dumps({"<問題>": ["c2", "c0", "c3"]}), encoding="utf-8"
    )
    out = prompts_tool.build_rows(cfg, lambda rc: ScriptedRetriever({"<問題>": ["c2", "c0", "c3"]}))
    assert out[0]["chunk_ids"] == ["c2", "c0"]
    assert "[1] <正文 2>" in out[0]["messages"][1]["content"]
    assert "[2] <正文 0>" in out[0]["messages"][1]["content"]
    target = tmp_path / "p.jsonl"
    assert prompts_tool.main([str(cfg), str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8").splitlines()[0])["id"] == "q1"
    assert prompts_tool.main([str(cfg), str(target)]) == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        cfg.read_text(encoding="utf-8").replace("top_k: 2", "top_k: 0"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        prompts_tool.load_prompt_section(bad)
    with pytest.raises(ValueError):
        prompts_tool.load_prompt_section(ev)


def test_bench_serve_rejects_bad_prompt_rows(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"id": "q"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        serve_tool.load_prompts(p)


# 簡體字元佔比欄(2026-09-17)
def test_bench_compare_reports_simplified_ratio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def bench(name: str, texts: dict[str, str]) -> Path:
        p = tmp_path / name
        p.write_text(
            json.dumps({"requests": [{"id": k, "text": v} for k, v in texts.items()]}),
            encoding="utf-8",
        )
        return p

    a = bench("a.json", {"q1": "參考資料甲", "q2": "乙丙丁"})
    b = bench("b.json", {"q1": "参考资料甲", "q2": "乙丙丁"})
    r = compare_tool.report(a, b)
    assert r["simplified_a"]["texts_with_simplified"] == 0
    assert r["simplified_b"] == {
        "n": 2,
        "texts_with_simplified": 1,
        "ratio_mean": pytest.approx((2 / 5) / 2),  # 参资 2 / CJK 5 字
        "ratio_max": pytest.approx(2 / 5),
        "ids": ["q1"],
    }
    assert r["exact_match"] == 1 and r["n"] == 2  # 既有鍵不變
    assert compare_tool.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("完全一致 1/2") and out[3].startswith("簡體字元佔比:A 0 題")
    assert "B 1 題(mean 0.200 / max 0.400)" in out[3]
