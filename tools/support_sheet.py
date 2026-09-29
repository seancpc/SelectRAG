#!/usr/bin/env python3
"""抽樣低逐字覆蓋的引用句,連同所引段落輸出供人工判型(桌機執行,產物不進 repo)。

    # 爭議 1(低覆蓋句判型)
    python3 tools/support_sheet.py configs/demo.yaml results/demo/support_review.xlsx \
        --limit 40 --min-coverage 0.5 --sample 30 --seed 42
    # 爭議 2(高覆蓋句抽驗,估真實無據率而非下界)
    python3 tools/support_sheet.py configs/demo.yaml results/demo/support_high.xlsx \
        --limit 40 --band high --sample 20 --seed 42
    # 爭議 3(跨格式配對;固定題目與 preset,只換 --model)
    python3 tools/support_sheet.py configs/demo.yaml results/demo/support_gptq.xlsx \
        --limit 20 --band all --sample 0 --preset bm25_ctx_tuned --model <格式名稱>

對 eval set 前 limit 題 × 每個 preset:檢索 → 生成 → 逐句算「句子 3-gram 被所引段落覆蓋率」,
取覆蓋率 < min-coverage 的句子隨機抽 sample 筆,寫出「句子 / 所引段落全文 / 覆蓋率」供判定:
改寫(內容對、措辭不同)/ 跨段落綜合(一句綜合多段只標一段)/ 編造(段落沒有此內容)。
覆蓋率只是抽樣排序依據,不是品質分數——2026-09-19 首輪複核 23 句:改寫 12 / 編造 11 / 綜合 0。
有 openpyxl 就寫 xlsx,否則寫同名 .csv(UTF-8 BOM)。拒絕覆蓋既有檔案。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from selectrag.demo.core import (
    ABSTAIN,
    Hit,
    RetrieverPool,
    cited_indices,
    load_demo_config,
    make_messages,
    sentence_support,
    stream_answer,
)
from selectrag.evaluation.abstain_audit import is_abstain, truth_rank
from selectrag.evaluation.components import (
    ItemComponents,
    component_hits,
    item_forbidden_hits,
    load_components,
    unsupported_hits,
)
from selectrag.evaluation.eval_set import load_eval_set
from selectrag.evaluation.terms import foreign_terms, missing_verbatim
from selectrag.retrieval.rerank import load_chunk_rows
from selectrag.runner.__main__ import build_retriever
from selectrag.runner.config import load_config

HEADER = [
    "preset",
    "model",
    "item_id",
    "question",
    "句子(已剝除引用標記)",
    "引用編號",
    "覆蓋率",
    "所引段落全文",
    "判定(改寫/綜合/編造)",
    "備註",
    # 以下四欄為 2026-09-22 新增,表格自第六輪起為新基準
    "列型",
    "外來詞(段落與題目皆無)",
    "段落有但未照抄",
    "棄答時真值最佳名次",
]


COVERAGE_COL = 6  # rows 的覆蓋率欄位索引(HEADER 順序)
KIND_COL = 10  # 列型:作答 / 棄答 / 無引用回答
FOREIGN_COL = 11
MISSING_COL = 12
TRUTH_RANK_COL = 13

# 必要成分欄:只在給了 --components 時追加,否則表頭逐字不變
COMPONENT_HEADER = [
    "必要成分命中",
    "缺漏成分",
    "禁止表述命中",
    "題層禁止命中(整題歸零)",
    "命中但未引用",
    "命中但引用不支撐",
]


def header(components: Mapping[str, ItemComponents] | None) -> list[str]:
    """給了成分清單才追加兩欄;未給時回原表頭(既有產出的格式不破)。"""
    return HEADER if components is None else [*HEADER, *COMPONENT_HEADER]


def filter_band(rows: list[list[Any]], band: str, min_coverage: float) -> list[list[Any]]:
    """按覆蓋率分帶挑列:low = < 門檻(疑似不忠實)、high = ≥ 門檻、all = 全取。

    ★ high 帶存在的理由:覆蓋率**不是** support 的代理(2026-09-19 實證),
    只看 low 帶會讓「無據引用率」變成選擇性樣本的下界;要估真實率必須也抽 high 帶。
    """
    if band == "all":
        return list(rows)
    # 棄答列與無引用回答列沒有覆蓋率,不屬於任何一帶:分帶的語意就是「按覆蓋率挑」。
    # 不跳過的話 float("") 會直接炸(2026-09-22)。
    scored = [r for r in rows if r[COVERAGE_COL] != ""]
    if band == "low":
        return [r for r in scored if float(r[COVERAGE_COL]) < min_coverage]
    if band == "high":
        return [r for r in scored if float(r[COVERAGE_COL]) >= min_coverage]
    raise ValueError(f"unknown band {band!r}; use low, high or all")


def collect_rows(
    cfg: Any,
    pool: RetrieverPool,
    es: Any,
    limit: int,
    min_coverage: float,
    presets: Sequence[str] | None = None,
    model: str = "",
    components: Mapping[str, ItemComponents] | None = None,
    dump_to: list[dict[str, Any]] | None = None,
) -> tuple[list[list[Any]], int, int]:
    """回 (**全部**有引用句的列, 有引用句總數, 低覆蓋句總數);分帶交給 filter_band。

    presets=None 表示 config 的全部 preset;model 只寫進輸出欄位供跨格式比較時辨識,
    實際用哪個模型由 cfg.llm.model 決定(main 以 --model 覆寫)。

    components 給定時每列追加成分三欄與題層禁止一欄(判定對**整段回答**,故同題每列同值);
    題層禁止與成分層禁止刻意分開:前者是「整題答到別的標的」,後者是「假性滿足某成分」;
    「命中但未引用」與「命中但引用不支撐」兩欄互斥,顯示 completeness 與 support 的母體差異
    (B2),不改任一方的計算;後者對應 support 的「引用錯置」錯型;
    清單未涵蓋的題目留空而不 raise——清單只覆蓋多成分題。成分不進任何分數。

    dump_to 給定時,**每個 (preset, item) 都追加一筆完整回答**,與是否產生列無關。
    存在理由(2026-09-22 v14 跑分):本函式只寫入「有引用且 coverage 非 None」的句子,
    整段回答用完即丟——當某題一句都沒被收進表(答案無引用標記、或棄答),它會從 xlsx
    整題消失,而「被擠掉」與「真的棄答」在表上無法分辨(v14 的 b2-q11 / b4-q15 即卡在此)。
    dump 是唯一能回溯的原始輸出,含語料原文故**不進 repo**。
    """
    rows: list[list[Any]] = []
    n_cited = n_low = 0
    items = es.items if limit <= 0 else es.items[:limit]
    names = list(cfg.presets) if presets is None else list(presets)
    for name in names:
        if name not in cfg.presets:
            raise KeyError(f"unknown preset {name!r}; config has {sorted(cfg.presets)}")
        for item in items:
            hits: list[Hit] = pool.hits(name, item.question, cfg.prompt.top_k)
            text_by_rank = {h.rank: h.text for h in hits}
            answer = ""
            for piece in stream_answer(cfg.llm, make_messages(cfg.prompt, item.question, hits)):
                answer = piece  # 串流累積,取最後一筆
            if answer.startswith("[錯誤]"):
                raise RuntimeError(f"{name} {item.id} LLM 錯誤:{answer[:80]}")
            if dump_to is not None:
                dump_to.append(
                    {
                        "preset": name,
                        "model": model,
                        "item_id": item.id,
                        "question": item.question,
                        "answer": answer,
                        "hits": [{"rank": h.rank, "chunk_id": h.chunk_id} for h in hits],
                    }
                )
            # 兩段處理:成分的「引用支撐」要看該題**全部**有引用的句子,
            # 所以必須先收完才能組列(B2,2026-09-21)。
            sups = [s for s in sentence_support(answer, hits) if s.coverage is not None]
            cited_sentences = [s.sentence for s in sups]
            # 該題**所有被引用的段落**文字,供「引用不支撐」那一欄比對
            cited_segments = [text_by_rank[c] for s in sups for c in s.cites]
            for sup in sups:
                assert sup.coverage is not None  # 上面已過濾,narrow 給 mypy
                n_cited += 1
                if sup.coverage < min_coverage:
                    n_low += 1
                own_segments = [text_by_rank[c] for c in sup.cites]
                cited_text = "\n---\n".join(own_segments)
                # 藥名對照用**該句自己的**引用段落,不是全題的:句子只該對它引的那幾塊負責
                row = [
                    name,
                    model,
                    item.id,
                    item.question,
                    sup.sentence,
                    ",".join(str(c) for c in sup.cites),
                    round(sup.coverage, 3),
                    cited_text,
                    "",
                    "",
                    "作答",
                    ",".join(foreign_terms(sup.sentence, own_segments, item.question)),
                    ",".join(missing_verbatim(sup.sentence, own_segments)),
                    "",
                ]
                if components is not None:
                    spec_item = components.get(item.id)
                    if spec_item is not None and spec_item.components:
                        hit, miss, blocked = component_hits(answer, spec_item.components)
                        item_bad = item_forbidden_hits(answer, spec_item)
                        # 兩種缺陷分欄且互斥(2026-09-21 第二輪跑分):
                        #   未引用     = 成分只出現在**沒有引用標記**的句子裡
                        #   引用不支撐 = 成分在引用句裡,但**所引段落**不含它(引用錯置)
                        # 單一欄位只抓得到前者,b3-q12 / b2-q04 那種引用錯置會溜過去。
                        uncited = unsupported_hits(hit, cited_sentences, spec_item.components)
                        in_segments = unsupported_hits(hit, cited_segments, spec_item.components)
                        mismatch = tuple(n for n in in_segments if n not in uncited)
                        row += [
                            ",".join(hit),
                            ",".join(miss),
                            ",".join(blocked),
                            ",".join(item_bad),
                            ",".join(uncited),
                            ",".join(mismatch),
                        ]
                    else:
                        row += ["", "", "", "", "", ""]
                rows.append(row)
            if not sups:
                # 發現 47:沒有任何有引用句的組合過去整個消失(首輪 32 / v14 25 / v15 21),
                # 完整率與 support 因此只在「有作答的題」上計算,錯誤棄答整類看不見。
                # 棄答與「回答了但一句都沒標引用」分開:後者是 v14 的 b2-q11 hybrid 型。
                rows.append(
                    [
                        name,
                        model,
                        item.id,
                        item.question,
                        answer,
                        # 整段回答的有效引用編號:句級的 sentence_support 拿不到它,
                        # 因為它整句跳過棄答句。第六輪 jsonl 實證 22 筆棄答有 21 筆帶 [n],
                        # 而「引了段落卻說沒提到」是發現 48「宣稱缺席」的直接實證。
                        ",".join(str(c) for c in sorted(cited_indices(answer, len(hits))[0])),
                        "",
                        "",
                        "",
                        "",
                        "棄答" if is_abstain(answer, ABSTAIN) else "無引用回答",
                        "",
                        "",
                        truth_rank(item.relevant_doc_ids, [h.chunk_id for h in hits]) or "",
                    ]
                    + (["", "", "", "", "", ""] if components is not None else [])
                )
    return rows, n_cited, n_low


def write(rows: list[list[Any]], out: Path, head: Sequence[str] | None = None) -> Path:
    if out.exists():
        raise FileExistsError(f"拒絕覆蓋:{out}")
    try:
        from openpyxl import Workbook  # type: ignore[import-untyped]
        from openpyxl.styles import Alignment  # type: ignore[import-untyped]
    except ImportError:
        out = out.with_suffix(".csv")
        if out.exists():
            raise FileExistsError(f"拒絕覆蓋:{out}") from None
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(head or HEADER))
            w.writerows(rows)
        return out
    wb = Workbook()
    ws = wb.active
    ws.title = "support"
    ws.append(list(head or HEADER))
    for r in rows:
        ws.append(r)
    widths = [16, 18, 10, 34, 48, 10, 9, 70, 16, 24]
    widths += [12, 26, 26, 14]  # 列型 / 外來詞 / 未照抄 / 真值名次
    widths += [22, 22, 22, 24, 22, 24]  # 成分六欄
    # ★ 只套到實際欄數:openpyxl 存取 ws.cell(row=1, column=i) 會**建立**該儲存格,
    # 套超過表頭長度就會在表頭列補出 None 欄(2026-09-22 由 gauntlet 5l 的新斷言抓到;
    # 未給 --components 的產出一直多 6 個空欄,因為沒有人驗過那個分支的表頭)。
    for i, wd in enumerate(widths[: len(head or HEADER)], 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = wd
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(out)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=40, help="取 eval set 前 N 題(0=全部)")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=30, help="抽樣筆數(0=全取)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--band",
        choices=("low", "high", "all"),
        default="low",
        help="low=覆蓋率 < 門檻、high=≥ 門檻(估真實率)、all=全取(跨格式配對)",
    )
    ap.add_argument("--preset", action="append", default=None, help="只跑指定 preset(可重複)")
    ap.add_argument("--model", default=None, help="覆寫 cfg.llm.model(跨格式比較用)")
    ap.add_argument(
        "--dump-answers",
        type=Path,
        default=None,
        help="把每題完整回答寫成 jsonl(含語料原文,不進 repo);整題無引用句時仍留一筆",
    )
    ap.add_argument(
        "--components",
        type=Path,
        default=None,
        help="必要成分清單(多成分題的答案完整性輔助欄;不進任何分數)",
    )
    a = ap.parse_args(argv)
    cfg = load_demo_config(a.config)
    model = a.model or cfg.llm.model
    if a.model:
        cfg = replace(cfg, llm=replace(cfg.llm, model=a.model))
    rows_tbl = load_chunk_rows(cfg.chunks)
    es = load_eval_set(cfg.eval_set)
    pool = RetrieverPool(
        cfg, rows_tbl, lambda p: build_retriever(load_config(cfg.presets[p]).retriever)
    )
    # 覆蓋檢查放在**開跑前**:15 分鐘的生成跑完才拒絕寫檔等於整輪白跑
    if a.dump_answers is not None and a.dump_answers.exists():
        raise FileExistsError(f"拒絕覆蓋:{a.dump_answers}")
    dump: list[dict[str, Any]] | None = [] if a.dump_answers else None
    components = load_components(a.components) if a.components else None
    if components is not None:
        print(f"[support] 成分清單 {len(components)} 題(人工輔助欄,不進分數)")
    rows, n_cited, n_low = collect_rows(
        cfg,
        pool,
        es,
        a.limit,
        a.min_coverage,
        presets=a.preset,
        model=model,
        components=components,
        dump_to=dump if dump is not None else None,
    )
    if dump is not None:
        assert a.dump_answers is not None
        with a.dump_answers.open("w", encoding="utf-8") as fh:
            for rec in dump:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[support] 原始回答 {len(dump)} 筆 → {a.dump_answers}")
    rate = n_low / n_cited if n_cited else 0.0
    print(
        f"[support] model={model} 有引用句 {n_cited},低於 {a.min_coverage} 的 {n_low}({rate:.3f})"
    )
    rows = filter_band(rows, a.band, a.min_coverage)
    print(f"[support] band={a.band} 抽樣框 {len(rows)} 列")
    if a.sample > 0 and len(rows) > a.sample:
        rows = random.Random(a.seed).sample(rows, a.sample)
    rows.sort(key=lambda r: (r[2], r[0]))  # 同題相鄰,跨格式合併後便於配對判定
    path = write(rows, a.out, header(components))
    print(f"[support] 寫出 {len(rows)} 列 → {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
