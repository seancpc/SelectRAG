"""judge 工具測試。

假伺服器 judge 模式 → judge_claims → calib_sheet build/score;文字皆為中性占位符。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import threading
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fake = _load_tool("fake_openai_server")
judge = _load_tool("judge_claims")
calib = _load_tool("calib_sheet")


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    srv = fake.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _cfg(tmp_path: Path, url: str) -> Path:
    p = tmp_path / "judge.yaml"
    p.write_text(
        f"model: <judge>\nrevision: <rev>\nurl: {url}\nseed: 42\nmax_tokens: 64\n", encoding="utf-8"
    )
    return p


def _inputs(tmp_path: Path, texts: dict[str, str]) -> tuple[Path, Path]:
    bench = tmp_path / "bench.json"
    bench.write_text(
        json.dumps(
            {"requests": [{"id": k, "text": v} for k, v in texts.items()]}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    prompts = tmp_path / "p.jsonl"
    with prompts.open("w", encoding="utf-8") as fh:
        for k in texts:
            fh.write(
                json.dumps(
                    {
                        "id": k,
                        "messages": [
                            {"role": "system", "content": "<s>"},
                            {
                                "role": "user",
                                "content": f"問題:<問 {k}>\n\n參考段落:\n[1] <段 {k}>",
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return bench, prompts


def test_context_of_splits_question_and_context() -> None:
    assert judge.context_of("問題:<q>\n\n參考段落:\n[1] <c>") == ("<q>", "[1] <c>")
    assert judge.context_of("<沒有分隔>") == ("", "<沒有分隔>")


def test_judge_claims_end_to_end(server: str, tmp_path: Path) -> None:
    bench, prompts = _inputs(
        tmp_path,
        {
            "q1": "每日 250 點。每年 3 次。",
            "q2": "參考資料未提及。",  # 拒答:不進分母
            "q3": "甲。參考資料未提及。",  # 混合:只留「甲」
        },
    )
    out = tmp_path / "j.json"
    argv = [
        str(out),
        "--bench",
        str(bench),
        "--prompts",
        str(prompts),
        "--config",
        str(_cfg(tmp_path, server)),
    ]
    assert judge.main(argv) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["summary"]["n_claims"] == 3 and r["summary"]["n_abstained_answers"] == 1
    assert r["parse_failures"] == [] and r["meta"]["judge_revision"] == "<rev>"
    assert r["meta"]["prompt_version"] == judge.PROMPT_VERSION
    by_id = {a["id"]: a for a in r["answers"]}
    assert by_id["q2"]["abstained"] is True and by_id["q3"]["claims"][0]["claim"] == "甲"
    assert all("verdict" in c for c in by_id["q1"]["claims"])
    assert judge.main(argv) == 3  # 拒絕覆蓋
    assert judge.main([str(tmp_path / "x.json"), "--config", str(_cfg(tmp_path, server))]) == 2
    assert (
        judge.main(
            [
                str(tmp_path / "x.json"),
                "--bench",
                str(bench),
                "--config",
                str(_cfg(tmp_path, server)),
            ]
        )
        == 2
    )


def test_judge_claims_parse_failure_gate(server: str, tmp_path: Path) -> None:
    bench, prompts = _inputs(tmp_path, {"q1": "<壞>。乙。"})  # 1/2 解析失敗 > 5%
    out = tmp_path / "j.json"
    assert (
        judge.main(
            [
                str(out),
                "--bench",
                str(bench),
                "--prompts",
                str(prompts),
                "--config",
                str(_cfg(tmp_path, server)),
            ]
        )
        == 1
    )
    r = json.loads(out.read_text(encoding="utf-8"))
    assert len(r["parse_failures"]) == 1 and r["parse_failure_rate"] == 0.5


def test_judge_claims_missing_prompt_raises(server: str, tmp_path: Path) -> None:
    bench, prompts = _inputs(tmp_path, {"q1": "甲。"})
    prompts.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="q1"):
        judge.claims_from_bench(bench, prompts)


def test_calib_build_and_score(server: str, tmp_path: Path) -> None:
    texts = {f"q{i}": f"每日 {100 + i} 點。條件為 {i} 項。" for i in range(6)}
    bench, prompts = _inputs(tmp_path, texts)
    cfg = _cfg(tmp_path, server)
    first = tmp_path / "first.json"
    assert (
        judge.main(
            [str(first), "--bench", str(bench), "--prompts", str(prompts), "--config", str(cfg)]
        )
        == 0
    )
    out_dir = tmp_path / "calib"
    info = calib.build(first, out_dir, n=4, negatives=3, seed=1)
    assert info == {"answers": 4, "claims": 8, "negatives": 3, "rows": 11}
    rows = [json.loads(line) for line in (out_dir / "calib_claims.jsonl").open(encoding="utf-8")]
    syn = [r for r in rows if r["synthetic"]]
    assert len(syn) == 3 and all(r["claim_id"].endswith("-neg") for r in syn)
    orig = {r["claim_id"]: r["claim"] for r in rows if not r["synthetic"]}
    assert all(orig[r["claim_id"][:-4]] != r["claim"] for r in syn)  # 負例確實改了數字
    with pytest.raises(FileExistsError):
        calib.build(first, out_dir, n=4, negatives=3, seed=1)
    with pytest.raises(ValueError):
        calib.build(first, tmp_path / "c2", n=99, negatives=0, seed=1)
    # judge 對校準主張打分(--claims 模式)
    cj = tmp_path / "calib.json"
    assert (
        judge.main([str(cj), "--claims", str(out_dir / "calib_claims.jsonl"), "--config", str(cfg)])
        == 0
    )
    # 人工判定:全部抄 judge 的答案(κ 應為 1),合成負例人工標 contradicted
    judged = {
        a["id"]: a["claims"][0]["verdict"]
        for a in json.loads(cj.read_text(encoding="utf-8"))["answers"]
    }
    sheet = out_dir / "calib_sheet.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0].keys()) == calib.HEADER and len(table) == 11
    id_map = {r["row_id"]: r for r in rows}
    for t in table:
        r = id_map[t["row_id"]]
        t["你的判定"] = "contradicted" if r["synthetic"] else judged[r["claim_id"]]
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=calib.HEADER)
        w.writeheader()
        w.writerows(table)
    r = calib.score(sheet, cj)
    assert r["n"] == 11 and r["synthetic_total"] == 3 and r["synthetic_caught_by_human"] == 3
    assert 0 <= r["synthetic_caught_by_judge"] <= 3
    assert -1.0 <= r["kappa"] <= 1.0 and 0 <= r["agreement"] <= 1
    assert sum(sum(v.values()) for v in r["confusion"].values()) == 11
    assert calib.main(["score", str(sheet), str(cj)]) == 0
    assert (
        calib.main(
            ["build", str(first), str(out_dir), "--n", "4", "--negatives", "3", "--seed", "1"]
        )
        == 3
    )
    # 壞標籤 / 空表
    bad = tmp_path / "bad.csv"
    bad.write_text(
        sheet.read_text(encoding="utf-8-sig").replace("contradicted", "maybe", 1),
        encoding="utf-8-sig",
    )
    (tmp_path / "calib_claims.jsonl").write_text(
        (out_dir / "calib_claims.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        calib.score(bad, cj)
    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(calib.HEADER) + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        calib.score(empty, cj)


def test_calib_score_binary_labels(server: str, tmp_path: Path) -> None:
    texts = {f"q{i}": f"每日 {100 + i} 點。" for i in range(4)}
    bench, prompts = _inputs(tmp_path, texts)
    cfg = _cfg(tmp_path, server)
    first = tmp_path / "first.json"
    assert (
        judge.main(
            [str(first), "--bench", str(bench), "--prompts", str(prompts), "--config", str(cfg)]
        )
        == 0
    )
    out_dir = tmp_path / "calib"
    calib.build(first, out_dir, n=4, negatives=2, seed=3)
    cj = tmp_path / "calib.json"
    assert (
        judge.main([str(cj), "--claims", str(out_dir / "calib_claims.jsonl"), "--config", str(cfg)])
        == 0
    )
    rows = [json.loads(line) for line in (out_dir / "calib_claims.jsonl").open(encoding="utf-8")]
    judged = {
        a["id"]: a["claims"][0]["verdict"]
        for a in json.loads(cj.read_text(encoding="utf-8"))["answers"]
    }
    sheet = out_dir / "calib_sheet.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        table = list(csv.DictReader(fh))
    id_map = {r["row_id"]: r for r in rows}
    for t in table:  # 人工二分法:抄 judge 塌成二類的答案 → κ 必為 1
        t["你的判定"] = (
            "支持" if judged[id_map[t["row_id"]]["claim_id"]] == "supported" else "不支持"
        )
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=calib.HEADER)
        w.writeheader()
        w.writerows(table)
    r = calib.score(sheet, cj)
    assert r["mode"] == "binary" and r["n"] == 6 and r["kappa"] == 1.0 and r["disagreements"] == []
    assert set(r["confusion"]) == {"supported", "not_supported"}
    # 翻一列人工標籤 → 出現一筆不一致
    table[0]["你的判定"] = "不支持" if table[0]["你的判定"] == "支持" else "支持"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=calib.HEADER)
        w.writeheader()
        w.writerows(table)
    r2 = calib.score(sheet, cj)
    assert len(r2["disagreements"]) == 1 and r2["disagreements"][0]["row_id"] == table[0]["row_id"]
    assert calib.main(["score", str(sheet), str(cj)]) == 0


def test_calib_build_rejects_bad_params(tmp_path: Path) -> None:
    j = tmp_path / "j.json"
    j.write_text(json.dumps({"answers": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        calib.build(j, tmp_path / "o", n=0, negatives=0, seed=1)
    with pytest.raises(ValueError):
        calib.build(j, tmp_path / "o", n=1, negatives=0, seed=1)


# 逐字命中前置層(2026-09-17)
def test_judge_claims_verbatim_prelayer_skips_llm(tmp_path: Path) -> None:
    """全部主張逐字命中 → 不呼叫 LLM(url 指向死埠仍成功)。"""
    bench, prompts = _inputs(tmp_path, {"q1甲乙丙丁": "<段 q1甲乙丙丁>。"})
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(
        "model: <judge>\nrevision: <rev>\nurl: http://127.0.0.1:1\nseed: 42\nmax_tokens: 64\n"
        "timeout: 1\nverbatim_min_coverage: 0.9\n",
        encoding="utf-8",
    )
    out = tmp_path / "j.json"
    assert (
        judge.main(
            [str(out), "--bench", str(bench), "--prompts", str(prompts), "--config", str(cfg)]
        )
        == 0
    )
    r = json.loads(out.read_text(encoding="utf-8"))
    c = r["answers"][0]["claims"][0]
    assert c["verdict"] == "supported" and c["evidence_chunk"] == 1 and c["source"] == "verbatim"
    assert r["summary"]["verbatim_hits"] == 1 and r["meta"]["verbatim_min_coverage"] == 0.9


def test_judge_claims_verbatim_prelayer_disabled_without_key(tmp_path: Path) -> None:
    bench, prompts = _inputs(tmp_path, {"q1甲乙丙丁": "<段 q1甲乙丙丁>。"})
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(
        "model: <judge>\nrevision: <rev>\nurl: http://127.0.0.1:1\nseed: 42\n"
        "max_tokens: 64\ntimeout: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(urllib.error.URLError):
        judge.main(
            [
                str(tmp_path / "j.json"),
                "--bench",
                str(bench),
                "--prompts",
                str(prompts),
                "--config",
                str(cfg),
            ]
        )


def test_judge_claims_verbatim_miss_goes_to_llm(server: str, tmp_path: Path) -> None:
    bench, prompts = _inputs(
        tmp_path, {"q1甲乙丙丁": "<段 q1甲乙丙丁>。完全不同的另一句話子丑寅卯。"}
    )
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(
        f"model: <judge>\nrevision: <rev>\nurl: {server}\nseed: 42\nmax_tokens: 64\n"
        "verbatim_min_coverage: 0.9\n",
        encoding="utf-8",
    )
    out = tmp_path / "j.json"
    assert (
        judge.main(
            [str(out), "--bench", str(bench), "--prompts", str(prompts), "--config", str(cfg)]
        )
        == 0
    )
    r = json.loads(out.read_text(encoding="utf-8"))
    src = [c["source"] for c in r["answers"][0]["claims"]]
    assert src == ["verbatim", "llm"] and r["summary"]["verbatim_hits"] == 1


grid = _load_tool("verbatim_grid")


def test_verbatim_grid_plateau(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import csv

    p = tmp_path / "sheet.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "question", "context", "claim", "你的判定", "備註"])
        w.writerow(["a", "<q>", "[1] 甲乙丙丁戊己庚辛", "甲乙丙丁戊己庚辛", "支持", ""])  # cov 1.0
        w.writerow(
            ["b", "<q>", "[1] 甲乙丙丁戊己庚辛", "甲乙丙丁戊己壬癸", "不支持", ""]
        )  # cov 4/6
        w.writerow(["c", "<q>", "[1] 甲乙丙丁戊己庚辛", "子丑寅卯辰巳午未", "支持", ""])  # cov 0
    rows = grid.scan(p, [0.5, 0.7, 0.9])
    assert rows == [
        {"threshold": 0.5, "hits": 2, "false_hits": 1, "true_hits": 1},
        {"threshold": 0.7, "hits": 1, "false_hits": 0, "true_hits": 1},
        {"threshold": 0.9, "hits": 1, "false_hits": 0, "true_hits": 1},
    ]
    assert grid.plateau(rows) == (0.7, 0.9)
    assert grid.main([str(p), "--thresholds", "0.5", "0.7", "0.9"]) == 0
    assert "高原" in capsys.readouterr().out


def test_verbatim_grid_rejects_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("row_id,claim\na,b\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        grid.scan(p, [0.5])


# 分層抽樣(2026-09-17 核准,變更 2)
_CTX = "[1] 甲乙丙丁戊己庚辛壬癸 250 點\n[2] 子丑寅卯辰巳午未申酉"


def test_stratum_of_rules() -> None:
    assert calib.stratum_of("甲乙丙丁戊己庚辛壬癸 250 點", _CTX, 0.925) == "verbatim"
    assert calib.stratum_of("甲乙丙丁戊己庚辛壬癸 250 點", _CTX, 1.0) == "verbatim"  # 等於門檻
    assert calib.stratum_of("甲乙丙丁戊己庚天地玄黃", _CTX, 0.925) == "paraphrase"  # 5/9
    assert calib.stratum_of("天地玄黃宇宙洪荒日月盈昃", _CTX, 0.925) == "low"
    assert calib.stratum_of("參考資料中未提及甲乙丙丁", _CTX, 0.925) == "abstention"
    assert calib.stratum_of("参考资料中未提及甲乙丙丁", _CTX, 0.925) == "abstention"  # 簡體同判
    assert (
        calib.stratum_of("甲乙丙丁戊己庚辛壬癸 250 點未提及", _CTX, 0.925) == "abstention"
    )  # 優先於覆蓋率


def _judge_json(path: Path, answers: list[tuple[str, str, list[str]]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "id": qid,
                        "abstained": False,
                        "question": f"<問 {qid}>",
                        "context": ctx,
                        "claims": [
                            {"claim_id": f"{qid}#{i}", "claim": c, "verdict": "supported"}
                            for i, c in enumerate(claims)
                        ],
                    }
                    for qid, ctx, claims in answers
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_build_stratified_dedup_quota_and_files(tmp_path: Path) -> None:
    ctx = _CTX
    verb = [f"甲乙丙丁戊己庚辛壬癸 {250} 點"] * 1
    para = [f"甲乙丙丁戊己庚辛天地第 {i} 項" for i in range(15)]  # 覆蓋率 6/11–6/12;15 > 配額 12
    low = [f"天地玄黃宇宙洪荒第 {i} 條" for i in range(5)]  # 5 < 配額 12 → 全收
    abst = ["參考資料中未提及甲乙丙丁"]
    a = _judge_json(
        tmp_path / "a.json",
        [("q1", ctx, verb + para[:8]), ("q2", ctx, low + abst)],
    )
    b = _judge_json(
        tmp_path / "b.json",
        [("q1", ctx, verb + para[8:]), ("q2", ctx, low)],  # verb 與 low 與 a 重複 → 去重
    )
    out = tmp_path / "calib"
    info = calib.build_stratified([a, b], out, negatives=3, seed=1, min_coverage=0.925)
    assert info["verbatim"] == 1 and info["paraphrase"] == 12
    assert info["low"] == 5 and info["abstention"] == 1 and info["negatives"] == 3
    assert info["rows"] == 1 + 12 + 5 + 1 + 3
    rows = [json.loads(line) for line in (out / "calib_claims.jsonl").open(encoding="utf-8")]
    assert {r["stratum"] for r in rows} == {
        "verbatim",
        "paraphrase",
        "low",
        "abstention",
        "negative",
    }
    assert len({r["claim"] for r in rows if not r["synthetic"]}) == 19  # 去重後無重複主張
    with (out / "calib_sheet.csv").open(encoding="utf-8-sig", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0].keys()) == calib.HEADER and len(table) == 22  # 表頭不含 stratum
    with pytest.raises(FileExistsError):
        calib.build_stratified([a, b], out, negatives=3, seed=1, min_coverage=0.925)
    assert (
        calib.main(
            ["build", str(a), str(b), str(tmp_path / "c2"), "--stratify", "--negatives", "3"]
        )
        == 0
    )


def test_build_stratified_rejects_small_pool(tmp_path: Path) -> None:
    a = _judge_json(tmp_path / "a.json", [("q1", _CTX, ["天地玄黃宇宙洪荒第一條"])])
    with pytest.raises(ValueError):
        calib.build_stratified([a], tmp_path / "c", negatives=0, seed=1, min_coverage=0.925)
    with pytest.raises(ValueError):
        calib.build_stratified([], tmp_path / "c", negatives=0, seed=1, min_coverage=0.925)


def test_score_reports_per_stratum_kappa(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    out.mkdir()
    rows = [
        {"row_id": "r1", "claim_id": "q1#0", "claim": "甲", "synthetic": False, "stratum": "low"},
        {"row_id": "r2", "claim_id": "q1#1", "claim": "乙", "synthetic": False, "stratum": "low"},
        {
            "row_id": "r3",
            "claim_id": "q2#0",
            "claim": "丙",
            "synthetic": False,
            "stratum": "paraphrase",
        },
        {
            "row_id": "r4",
            "claim_id": "q2#1",
            "claim": "丁",
            "synthetic": False,
            "stratum": "paraphrase",
        },
    ]
    with (out / "calib_claims.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps({**r, "question": "<q>", "context": "<c>"}, ensure_ascii=False) + "\n"
            )
    sheet = out / "calib_sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(calib.HEADER)
        # low 層:人工 支持/不支持,judge 反過來 → κ = -1;paraphrase 層:人工全支持 → 單類 n/a
        w.writerow(["r1", "<q>", "<c>", "甲", "支持", ""])
        w.writerow(["r2", "<q>", "<c>", "乙", "不支持", ""])
        w.writerow(["r3", "<q>", "<c>", "丙", "支持", ""])
        w.writerow(["r4", "<q>", "<c>", "丁", "支持", ""])
    judged = {"q1#0": "contradicted", "q1#1": "supported", "q2#0": "supported", "q2#1": "supported"}
    cj = tmp_path / "cj.json"
    cj.write_text(
        json.dumps(
            {
                "answers": [
                    {"id": k, "abstained": False, "claims": [{"claim_id": k, "verdict": v}]}
                    for k, v in judged.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    r = calib.score(sheet, cj)
    assert r["per_stratum"]["low"] == {"n": 2, "kappa": -1.0}
    assert r["per_stratum"]["paraphrase"] == {"n": 2, "kappa": None}  # 單類
    assert set(r["per_stratum"]) == {"low", "paraphrase"}


def test_score_without_stratum_field_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    out.mkdir()
    with (out / "calib_claims.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"row_id": "r1", "claim_id": "q1#0", "claim": "甲", "synthetic": False})
            + "\n"
        )
    sheet = out / "calib_sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(calib.HEADER)
        w.writerow(["r1", "<q>", "<c>", "甲", "supported", ""])
    cj = tmp_path / "cj.json"
    cj.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "id": "q1#0",
                        "abstained": False,
                        "claims": [{"claim_id": "q1#0", "verdict": "supported"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    r = calib.score(sheet, cj)
    assert "per_stratum" not in r and r["n"] == 1


def test_build_stratified_claim_ids_unique_across_sources(tmp_path: Path) -> None:
    """bug 2026-09-18:四格式同題同 claim_id 但文字不同 → claim_id / row_id 撞號,score 掉列。"""
    low_a = [f"天地玄黃宇宙洪荒第 {i} 條" for i in range(8)]
    low_b = [f"日月盈昃辰宿列張第 {i} 條" for i in range(8)]  # 同 claim_id、不同文字
    a = _judge_json(tmp_path / "vllm_awq.json", [("q1", _CTX, low_a)])
    b = _judge_json(tmp_path / "vllm_gptq.json", [("q1", _CTX, low_b)])
    out = tmp_path / "calib"
    calib.build_stratified([a, b], out, negatives=2, seed=1, min_coverage=0.925)
    rows = [json.loads(line) for line in (out / "calib_claims.jsonl").open(encoding="utf-8")]
    assert len(rows) == 14
    assert len({r["claim_id"] for r in rows}) == 14
    assert len({r["row_id"] for r in rows}) == 14
    assert all("@vllm_awq" in r["claim_id"] or "@vllm_gptq" in r["claim_id"] for r in rows)
