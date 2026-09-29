#!/usr/bin/env bash
# SelectRAG gauntlet — 單一入口,依序跑完所有檢測層,任一層失敗即中止。
# 用法:bash tools/gauntlet.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### 0. 清除前次產物(避免讀到舊結果)"
rm -rf .coverage coverage.xml htmlcov .pytest_cache
rm -f results/synthetic-baseline/gauntlet.json results/*/gauntlet-*.json
echo

echo "### 0b. 工具版本(必須與 requirements-dev.txt 釘選值一致)"
python3 -m pytest --version 2>&1 | head -1
python3 -m mypy --version
python3 -m ruff --version
echo

echo "### 1. 測試 + 覆蓋率(固定順序)"
python3 -m pytest tests -p no:randomly \
  --cov=selectrag --cov-branch --cov-report=term-missing
echo

echo "### 2. 靜態型別(mypy strict)"
python3 -m mypy
echo

echo "### 3. Lint + format"
python3 -m ruff check .
python3 -m ruff format --check .
echo

echo "### 4. 變異測試(全套件)"
python3 tools/mutants.py
echo

echo "### 4b. 變異測試(僅 property 子集 — 驗證不變量本身的殺傷力)"
# 只容忍 rc 1(有存活者,屬資訊);殘留(3)、基線不綠(4)、腳本過期(2)必須讓 gauntlet 失敗——
# 過去的 `|| echo` 會把這些全部吞掉,報出全綠
PROP_RC=0
python3 tools/mutants.py property || PROP_RC=$?
if [ "$PROP_RC" -eq 1 ]; then
  echo "    ↑ 存活者若為 property 策略刻意不產生的輸入(k<1、重複 id),屬預期,由手算錯誤案例覆蓋"
elif [ "$PROP_RC" -ne 0 ]; then
  echo "✗ 第 4b 層異常結束(rc=$PROP_RC),結果不可信"; exit "$PROP_RC"
fi
echo

echo "### 5. 真實執行(CLI 一行指令跑完整條 pipeline)"
python3 -m selectrag.runner configs/synthetic_baseline.yaml gauntlet
echo
echo "### 5b. 防護路徑真實觸發(備援路徑必須真實觸發實測,不能只測主路徑)"
echo "  [覆蓋保護] 以相同 run_id 再跑一次,預期被拒:"
if python3 -m selectrag.runner configs/synthetic_baseline.yaml gauntlet 2>/dev/null; then
    echo "  ✗ 覆蓋保護失效 —— 結果檔會被靜默蓋掉"
    exit 1
fi
echo "  ✓ 覆蓋保護攔截成功"
echo "  [竄改偵測] 竄改凍結檔後載入,預期被拒:"
python3 - <<'PY'
import json
import shutil
import tempfile
from pathlib import Path

from selectrag.evaluation.eval_set import EvalSetIntegrityError, load_eval_set

source = Path("data/eval/eval_v0_synthetic.json")
with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / "tampered.json"
    shutil.copy(source, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["items"][0]["question"] = "<被竄改>"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        load_eval_set(target)
    except EvalSetIntegrityError as exc:
        print(f"  ✓ 竄改偵測攔截成功:{str(exc)[:55]}…")
    else:
        raise SystemExit("  ✗ 竄改未被偵測 —— 凍結機制失效")
PY
echo
echo "### 5c. 語料前處理(有語料才跑;語料在 gitignore 區,CI 可能取不到)"
CORPUS_TMP="$(mktemp -d)"
if [ -d data/corpus ] && [ -n "$(find data/corpus -type f -name '*.pdf' -o -name '*.csv' -o -name '*.html' 2>/dev/null | head -1)" ]; then
    python3 tools/build_corpus.py data/corpus "$CORPUS_TMP/chunks.jsonl"
else
    echo "  ⏭ 跳過:data/corpus 無語料檔(正式數字須在有語料的機器上量測)"
fi
echo

echo "### 5d. BM25 真實執行(全語料建索引 + 代表性查詢 top-5)"
if [ -d data/corpus ] && [ -n "$(find data/corpus -type f -name '*.pdf' -o -name '*.csv' -o -name '*.html' 2>/dev/null | head -1)" ]; then
    # 沿用 5c 建好的語料(不重建);runner 設定檔指向 results/chunks.jsonl,同步一份
    cp "$CORPUS_TMP/chunks.jsonl" results/chunks.jsonl
    python3 tools/bm25_smoke.py results/chunks.jsonl
    echo "  [對照] 100 題 v1-full100:純正文 / 純正文調參 / +文件脈絡 / +文件脈絡調參 / 鄰塊 w1 / w2"
    RUN_ID="gauntlet-$(date +%s)"
    for CFG in bm25_full100 bm25_tuned_full100 bm25_ctx_full100 bm25_ctx_tuned_full100 bm25_ctx_tuned_w1_full100 bm25_ctx_tuned_w2_full100; do
        echo "    $CFG:"
        python3 -m selectrag.runner "configs/$CFG.yaml" "$RUN_ID" | grep -E "recall@10|hit@10|mrr"
    done
else
    echo "  ⏭ 跳過:data/corpus 無語料檔"
fi
echo

echo "### 5e. dense 真實執行(fake embedder 建索引 → runner type=dense;無 GPU 亦可跑)"
DENSE_TMP="$(mktemp -d)"
python3 - "$DENSE_TMP/chunks.jsonl" <<'PY'
import json, sys
rows = [{"chunk_id": f"c{i}", "text": f"<正文 {i}>", "doc_title": f"<標題 {i}>", "heading_path": ["<節>"]}
        for i in range(12)]
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
PY
python3 tools/build_dense_index.py --chunks "$DENSE_TMP/chunks.jsonl" \
    --out "$DENSE_TMP/fake.npz" --embedder fake --context-fields doc_title
python3 - "$DENSE_TMP/fake.npz" <<'PY'
import sys
from pathlib import Path
from selectrag.retrieval.dense import load_index
ids, vecs, meta = load_index(Path(sys.argv[1]))
assert len(ids) == vecs.shape[0] and meta.context_fields == ("doc_title",), meta
print(f"  ✓ 索引讀回:{len(ids)} 筆,dim={meta.dim},context_fields={meta.context_fields}")
PY
if [ -d data/corpus ] && [ -f results/dense_index/bge_m3_plain.npz ] && [ -f results/dense_index/bge_m3_ctx.npz ]; then
    echo "  [對照] 100 題 v1-full100:bge-m3 純正文 / +文件脈絡(索引須先由桌機建好)"
    RUN_ID="${RUN_ID:-gauntlet-$(date +%s)}"
    for CFG in dense_bge_m3_full100 dense_bge_m3_ctx_full100 dense_bge_m3_ctx_w1_full100 dense_bge_m3_ctx_w2_full100; do
        echo "    $CFG:"
        python3 -m selectrag.runner "configs/$CFG.yaml" "$RUN_ID" | grep -E "recall@10|hit@10|mrr"
    done
else
    echo "  ⏭ 跳過真模型對照:results/dense_index/*.npz 不存在(桌機執行 tools/build_dense_index.py --embedder bge-m3)"
fi
echo

echo "### 5f. hybrid 真實執行(兩個相同 scripted base 融合 → 必須等於單一 scripted 的指標)"
HYB_TMP="$(mktemp -d)"
cat > "$HYB_TMP/hybrid.yaml" <<EOF
experiment_id: hybrid-scripted-check
eval_set: data/eval/eval_v0_synthetic.json
retriever:
  type: hybrid
  params:
    pool: 10
    rrf_k: 60
    retrievers:
      - {type: scripted, params: {script: data/eval/eval_v0_synthetic_retrievals.json}}
      - {type: scripted, params: {script: data/eval/eval_v0_synthetic_retrievals.json}}
k_values: [1, 3, 5, 10]
results_dir: $HYB_TMP
EOF
python3 -m selectrag.runner "$HYB_TMP/hybrid.yaml" check | grep -E "recall@10|mrr"
python3 - "$HYB_TMP/hybrid-scripted-check/check.json" results/synthetic-baseline/gauntlet.json <<'PY'
import json, sys
a, b = (json.load(open(p, encoding="utf-8"))["metrics"] for p in sys.argv[1:3])
assert a == b, f"hybrid(同一 base ×2) 指標應等於單一 base:{a} != {b}"
print("  ✓ 兩個相同 base 的 RRF 融合與單一 base 指標逐位相同")
PY
if [ -d data/corpus ] && [ -f results/dense_index/bge_m3_ctx.npz ]; then
    echo "  [對照] hybrid RRF 代表點(pool=20, rrf_k=60):"
    RUN_ID="${RUN_ID:-gauntlet-$(date +%s)}"
    python3 -m selectrag.runner configs/hybrid_rrf_full100.yaml "$RUN_ID" | grep -E "recall@10|hit@10|mrr"
else
    echo "  ⏭ 跳過真模型 hybrid:results/dense_index/bge_m3_ctx.npz 不存在"
fi
echo

echo "### 5g. rerank 真實執行(fake reranker 包 scripted base;無 GPU 亦可跑)"
RR_TMP="$(mktemp -d)"
python3 - "$RR_TMP/chunks.jsonl" <<'PY'
import json, sys
rows = [{"chunk_id": f"c{i}", "doc_id": "D", "text": f"<正文 {i}>", "doc_title": "<標題>", "heading_path": []}
        for i in range(12)]
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
PY
python3 - "$RR_TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
(tmp / "script.json").write_text(json.dumps({"<問題>": [f"c{i}" for i in range(12)]}), encoding="utf-8")
(tmp / "base.yaml").write_text(f"""experiment_id: rr-base
eval_set: {tmp}/eval.json
retriever:
  type: scripted
  params:
    script: {tmp}/script.json
k_values: [1, 3, 5, 10]
results_dir: {tmp}/r
""", encoding="utf-8")
from selectrag.evaluation.eval_set import EvalItem, freeze
(tmp / "eval.json").write_text(json.dumps(freeze([EvalItem("q1", "<問題>", frozenset({"c7"}))], "t", "t")), encoding="utf-8")
PY
python3 tools/rerank_grid.py "$RR_TMP/base.yaml" "$RR_TMP/chunks.jsonl" "$RR_TMP/grid" --candidates 5 12 --reranker fake | grep -E "^cand|reranker"
if [ -d data/corpus ] && [ -f results/dense_index/bge_m3_ctx.npz ] && [ -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3" ]; then
    echo "  [對照] rerank 代表點(candidates=20, ctx):"
    RUN_ID="${RUN_ID:-gauntlet-$(date +%s)}"
    python3 -m selectrag.runner configs/rerank_bge_full100.yaml "$RUN_ID" | grep -E "recall@10|hit@10|mrr"
else
    echo "  ⏭ 跳過真模型 rerank:dense 索引或 bge-reranker-v2-m3 快取不存在"
fi
echo

echo "### 5h. truth_dupes 真實執行(真值重複事實掃描,純標準庫;雲端可跑)"
if [ -f results/chunks.jsonl ] && [ -f data/eval/eval_v1.2_full100.json ]; then
    TD_TMP="$(mktemp -d)"
    python3 tools/truth_dupes.py data/eval/eval_v1.2_full100.json results/chunks.jsonl "$TD_TMP/td.json" > "$TD_TMP/out.txt"
    head -1 "$TD_TMP/out.txt"
else
    echo "  ⏭ 跳過:results/chunks.jsonl 或 eval_v1.2 不存在"
fi
echo

echo "### 5i. bench 真實執行(假 OpenAI 串流伺服器 → bench_serve → bench_compare;無 GPU 亦可跑)"
BENCH_TMP="$(mktemp -d)"
python3 - "$BENCH_TMP/p.jsonl" <<'PY'
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    for i in range(8):
        fh.write(json.dumps({"id": f"q{i}", "messages": [{"role": "user", "content": f"<問 {i}>"}]}) + "\n")
PY
python3 tools/fake_openai_server.py --port 0 > "$BENCH_TMP/srv.log" 2>&1 &
BENCH_SRV=$!
until grep -q "^PORT=" "$BENCH_TMP/srv.log"; do sleep 0.2; done
BENCH_PORT="$(sed -n 's/^PORT=//p' "$BENCH_TMP/srv.log")"
python3 tools/bench_serve.py "$BENCH_TMP/p.jsonl" "$BENCH_TMP/out" --url "http://127.0.0.1:$BENCH_PORT" \
    --model fake --engine fake --fmt x --concurrency 1 4 --warmup 2 --no-vram > "$BENCH_TMP/bench.txt"
BENCH_RC=$?
kill "$BENCH_SRV" 2>/dev/null; wait "$BENCH_SRV" 2>/dev/null || true
[ "$BENCH_RC" -eq 0 ] || { echo "bench_serve 失敗(rc=$BENCH_RC)"; exit 1; }
head -2 "$BENCH_TMP/bench.txt"
python3 tools/bench_compare.py "$BENCH_TMP/out/fake_x_c1.json" "$BENCH_TMP/out/fake_x_c4.json" > "$BENCH_TMP/cmp.txt"
BC_RC=$?
[ "$BC_RC" -eq 0 ] && grep -q "^簡體字元佔比:A " "$BENCH_TMP/cmp.txt" || { echo "bench_compare 層失敗(rc=$BC_RC 或缺簡體佔比行)"; cat "$BENCH_TMP/cmp.txt"; exit 1; }
head -1 "$BENCH_TMP/cmp.txt"; tail -1 "$BENCH_TMP/cmp.txt"
echo

echo "### 5j. judge 真實執行(假 OpenAI judge 模式 → judge_claims → calib_sheet build/score;無 GPU 亦可跑)"
JG_TMP="$(mktemp -d)"
python3 - "$JG_TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
texts = {f"q{i}": f"每日 {100 + i} 點。條件為 {i} 項。" for i in range(8)}
(tmp / "bench.json").write_text(json.dumps({"requests": [{"id": k, "text": v} for k, v in texts.items()]}, ensure_ascii=False), encoding="utf-8")
with (tmp / "p.jsonl").open("w", encoding="utf-8") as fh:
    for k in texts:
        fh.write(json.dumps({"id": k, "messages": [{"role": "system", "content": "<s>"}, {"role": "user", "content": f"問題:<問 {k}>\n\n參考段落:\n[1] <段 {k}>"}]}, ensure_ascii=False) + "\n")
PY
python3 tools/fake_openai_server.py --port 0 > "$JG_TMP/srv.log" 2>&1 &
JG_SRV=$!
until grep -q "^PORT=" "$JG_TMP/srv.log"; do sleep 0.2; done
JG_PORT="$(sed -n 's/^PORT=//p' "$JG_TMP/srv.log")"
printf 'model: fake\nrevision: x\nurl: http://127.0.0.1:%s\nseed: 42\nmax_tokens: 64\n' "$JG_PORT" > "$JG_TMP/judge.yaml"
python3 tools/judge_claims.py "$JG_TMP/j.json" --bench "$JG_TMP/bench.json" --prompts "$JG_TMP/p.jsonl" --config "$JG_TMP/judge.yaml" > "$JG_TMP/j.txt"
JG_RC=$?
[ "$JG_RC" -eq 0 ] && python3 tools/calib_sheet.py build "$JG_TMP/j.json" "$JG_TMP/calib" --n 4 --negatives 3 --seed 1 > "$JG_TMP/b.txt"
JG_RC2=$?
[ "$JG_RC2" -eq 0 ] && python3 tools/judge_claims.py "$JG_TMP/cj.json" --claims "$JG_TMP/calib/calib_claims.jsonl" --config "$JG_TMP/judge.yaml" > "$JG_TMP/cj.txt"
JG_RC3=$?
# 分層抽樣(方案 B):同一份 judge 結果給兩次當兩格式,去重後配額不足即全收,不得崩潰
[ "$JG_RC3" -eq 0 ] && python3 tools/calib_sheet.py build "$JG_TMP/j.json" "$JG_TMP/j.json" "$JG_TMP/calib_s" --stratify --negatives 2 --seed 1 > "$JG_TMP/bs.txt"
JG_RC4=$?
[ "$JG_RC4" -eq 0 ] && grep -q "stratum" "$JG_TMP/calib_s/calib_claims.jsonl" || JG_RC4=1
kill "$JG_SRV" 2>/dev/null; wait "$JG_SRV" 2>/dev/null || true
[ "$JG_RC" -eq 0 ] && [ "$JG_RC2" -eq 0 ] && [ "$JG_RC3" -eq 0 ] && [ "$JG_RC4" -eq 0 ] || { echo "judge 層失敗(rc=$JG_RC/$JG_RC2/$JG_RC3/$JG_RC4)"; exit 1; }
head -1 "$JG_TMP/j.txt"; head -1 "$JG_TMP/b.txt"; head -1 "$JG_TMP/cj.txt"; head -1 "$JG_TMP/bs.txt"
echo

echo "### 5k. 逐字命中前置層真實執行(verbatim_grid 對合成校準表;前置層命中不打 LLM)"
VG_TMP="$(mktemp -d)"
python3 - "$VG_TMP" <<'PY'
import csv, sys
from pathlib import Path
tmp = Path(sys.argv[1])
with (tmp / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["row_id", "question", "context", "claim", "你的判定", "備註"])
    for i in range(6):
        ctx = f"[1] 規定第 {i} 項為每日 {100 + i} 點且限一次\n[2] 其他段落甲乙丙丁戊己"
        w.writerow([f"t{i}", "<q>", ctx, f"規定第 {i} 項為每日 {100 + i} 點且限一次", "支持", ""])
        w.writerow([f"n{i}", "<q>", ctx, f"規定第 {i} 項為每日 {101 + i} 點且限一次", "不支持", ""])  # 數字竄改
PY
python3 tools/verbatim_grid.py "$VG_TMP/sheet.csv" --thresholds 0.5 0.7 0.9 > "$VG_TMP/vg.txt"
VG_RC=$?
[ "$VG_RC" -eq 0 ] || { echo "verbatim_grid 層失敗(rc=$VG_RC)"; cat "$VG_TMP/vg.txt"; exit 1; }
grep -q "誤放(不支持)" "$VG_TMP/vg.txt" && awk 'NR>1 && NR<5 && $3!=0 {bad=1} END {exit bad}' "$VG_TMP/vg.txt" \
  || { echo "數字守門未擋住合成負例"; cat "$VG_TMP/vg.txt"; exit 1; }
tail -1 "$VG_TMP/vg.txt"
echo

echo "### 5l. demo 真實執行(假 OpenAI 串流伺服器 + scripted preset → demo_app --check;不起 gradio、無 GPU 亦可跑)"
DM_TMP="$(mktemp -d)"
python3 - "$DM_TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
(tmp / "chunks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
    {"chunk_id": "c1", "text": "<甲>", "doc_title": "<文件A>", "heading_path": "<節1>"},
    {"chunk_id": "c2", "text": "<乙>", "doc_title": "<文件B>", "heading_path": ""},
    {"chunk_id": "c3", "text": "<丙>", "doc_title": "<文件C>", "heading_path": ""},
]) + "\n", encoding="utf-8")
# eval set:沿用合成 v0 的結構驗證,直接複製 repo 內合成檔
(tmp / "eval.json").write_text(Path("data/eval/eval_v0_synthetic.json").read_text(encoding="utf-8"), encoding="utf-8")
items = json.loads((tmp / "eval.json").read_text(encoding="utf-8"))["items"]
q = items[0]["question"]
# 兩個 preset 皆帶分數腳本(collect 需要);只有 right 設門檻 0.50。
# 第 1 題:left top-1 0.90(不拒)、right top-1 0.10(拒答不打 LLM);其餘 eval 題與語料外題也要有劇本
off_q = "<語料外問題>"
left = {q: [["c1", 0.90], ["c3", 0.20]]}
right = {q: [["c2", 0.10], ["c1", 0.05]]}
for it in items[1:]:
    left[it["question"]] = [["c1", 0.80]]
    right[it["question"]] = [["c1", 0.70]]
left[off_q], right[off_q] = [["c3", 0.30]], [["c3", 0.01]]
(tmp / "left_script.json").write_text(json.dumps(left, ensure_ascii=False), encoding="utf-8")
(tmp / "right_script.json").write_text(json.dumps(right, ensure_ascii=False), encoding="utf-8")
(tmp / "off.json").write_text(json.dumps({"items": [{"id": "off-1", "question": off_q}]}, ensure_ascii=False), encoding="utf-8")
for name in ("left", "right"):
    (tmp / f"{name}.yaml").write_text(
        f"experiment_id: {name}\neval_set: {tmp}/eval.json\nretriever:\n  type: scripted\n"
        f"  params:\n    script: {tmp}/{name}_script.json\n    scored: true\nk_values: [1, 3]\nresults_dir: {tmp}\n",
        encoding="utf-8")
PY
python3 tools/fake_openai_server.py --port 0 > "$DM_TMP/srv.log" 2>&1 &
DM_SRV=$!
until grep -q "^PORT=" "$DM_TMP/srv.log"; do sleep 0.2; done
DM_PORT="$(sed -n 's/^PORT=//p' "$DM_TMP/srv.log")"
cat > "$DM_TMP/demo.yaml" <<EOF2
presets:
  left: $DM_TMP/left.yaml
  right: $DM_TMP/right.yaml
chunks: $DM_TMP/chunks.jsonl
eval_set: $DM_TMP/eval.json
prompt:
  top_k: 2
  system: "<s>"
  template: "問題:{question}\n\n參考段落:\n{context}"
llm:
  url: http://127.0.0.1:$DM_PORT
  model: fake
  max_tokens: 32
  seed: 42
abstain:
  right: 0.50
EOF2
python3 tools/demo_app.py "$DM_TMP/demo.yaml" --check > "$DM_TMP/demo.txt" 2>&1
DM_RC=$?
DM_RC2=0
python3 tools/demo_app.py "$DM_TMP/demo.yaml" --cite-rate 1 --out "$DM_TMP/cite.json" > "$DM_TMP/cite.txt" 2>&1 || DM_RC2=$?
# 必要成分命中:清單涵蓋第 1 題、另放一題不存在的 id 驗 B8 留空不 raise;
# 表頭必須多兩欄,且未給 --components 時逐字不變。假伺服器回覆含 <甲> 故成分甲必命中。
python3 - "$DM_TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
q = json.loads((tmp / "eval.json").read_text(encoding="utf-8"))["items"][0]["id"]
(tmp / "components.json").write_text(json.dumps({
    "version": "gauntlet", "items": {
        q: {"components": [
              {"name": "<成分甲>", "aliases": ["<回覆", "<甲>"]},
              {"name": "<成分乙>", "aliases": ["<絕不出現的表述>"]},
              {"name": "<成分丙>", "aliases": ["<絕不出現>"], "forbidden": ["<回覆"]}],
            "forbidden": ["<回覆:"]},
        "<不存在的題號>": [{"name": "<成分丁>", "aliases": ["<丁>"]}],
    }}, ensure_ascii=False), encoding="utf-8")
PY
# 另起一台帶 --cite 的假伺服器:預設回覆無 [n] → 無有效引用句 → 成分層永遠零列,
# 斷言會形同未執行(2026-09-20 首次真正跑到此層才發現)。
python3 tools/fake_openai_server.py --port 0 --cite > "$DM_TMP/srv_cite.log" 2>&1 &
CP_SRV=$!
until grep -q "^PORT=" "$DM_TMP/srv_cite.log"; do sleep 0.2; done
CP_PORT="$(sed -n 's/^PORT=//p' "$DM_TMP/srv_cite.log")"
sed "s|http://127.0.0.1:$DM_PORT|http://127.0.0.1:$CP_PORT|" "$DM_TMP/demo.yaml" > "$DM_TMP/demo_cite.yaml"
CP_RC=0
python3 tools/support_sheet.py "$DM_TMP/demo_cite.yaml" "$DM_TMP/comp.xlsx" \
  --limit 1 --band all --sample 0 --preset left --components "$DM_TMP/components.json" \
  > "$DM_TMP/comp.txt" 2>&1 || CP_RC=$?
kill "$CP_SRV" 2>/dev/null; wait "$CP_SRV" 2>/dev/null || true
CP_OUT="$DM_TMP/comp.xlsx"; [ -f "$CP_OUT" ] || CP_OUT="$DM_TMP/comp.csv"
[ "$CP_RC" -eq 0 ] && grep -q "成分清單 2 題" "$DM_TMP/comp.txt" && python3 -c "
import sys
from openpyxl import load_workbook
ws = load_workbook(sys.argv[1]).active
head = [c.value for c in next(ws.iter_rows(max_row=1))]
assert head[-6:] == ['必要成分命中', '缺漏成分', '禁止表述命中', '題層禁止命中(整題歸零)', '命中但未引用', '命中但引用不支撐'], head
rows = [[c.value for c in r] for r in ws.iter_rows(min_row=2)]
assert rows, 'no rows'
assert all(r[-6] == '<成分甲>' for r in rows), rows
assert all(r[-5] == '<成分乙>' for r in rows), rows
assert all(r[-4] == '<成分丙>' for r in rows), rows
# 題層禁止(懲罰型)與成分層禁止(保護型)必須分開落地
assert all(r[-3] == '<回覆:' for r in rows), rows
# 成分甲的別名「<回覆」在引用句裡、也在所引段落裡 → 兩個支撐欄都空
assert all((r[-2] or '') == '' for r in rows), rows
assert all((r[-1] or '') == '' for r in rows), rows
print(f'[components] {len(rows)} 列,成分三欄 + 題層禁止 + 兩個支撐欄落地')
" "$CP_OUT"   || { echo "components 層失敗(rc=$CP_RC)"; cat "$DM_TMP/comp.txt"; exit 1; }
grep "^\[support\]" "$DM_TMP/comp.txt"
# 棄答列與藥名對照欄(2026-09-22):
# 另建最小情境,讓真值 chunk **確實在取回結果內**(既有劇本回 c1,而合成 eval 的真值是
# doc_001_*,truth_rank 會恆為空,斷言形同未執行——「腳本裡有這行」不等於「執行過」)。
# 段落含一個拉丁詞而回答沒有 → 「段落有但未照抄」必須標出它。
python3 - "$DM_TMP" <<'PY2'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
q = json.loads((tmp / "eval.json").read_text(encoding="utf-8"))["items"][0]["question"]
(tmp / "chunks2.jsonl").write_text(
    json.dumps({"chunk_id": "doc_001_0", "text": "<段落> methotrexate <說明>",
                "doc_title": "<文件A>", "heading_path": ""}, ensure_ascii=False) + "\n",
    encoding="utf-8")
(tmp / "script2.json").write_text(json.dumps({q: [["doc_001_0", 0.90]]}, ensure_ascii=False),
                                  encoding="utf-8")
(tmp / "left2.yaml").write_text(
    f"experiment_id: left2\neval_set: {tmp}/eval.json\nretriever:\n  type: scripted\n"
    f"  params:\n    script: {tmp}/script2.json\n    scored: true\nk_values: [1]\nresults_dir: {tmp}\n",
    encoding="utf-8")
PY2
sed -e "s|left: $DM_TMP/left.yaml|left: $DM_TMP/left2.yaml|" \
    -e "s|chunks: $DM_TMP/chunks.jsonl|chunks: $DM_TMP/chunks2.jsonl|" \
    "$DM_TMP/demo.yaml" > "$DM_TMP/demo_abst.yaml"
AB_RC=0
python3 tools/support_sheet.py "$DM_TMP/demo_abst.yaml" "$DM_TMP/abst.xlsx" \
  --limit 1 --band all --sample 0 --preset left > "$DM_TMP/abstrows.txt" 2>&1 || AB_RC=$?
AB_OUT="$DM_TMP/abst.xlsx"; [ -f "$AB_OUT" ] || AB_OUT="$DM_TMP/abst.csv"
[ "$AB_RC" -eq 0 ] && python3 -c "
import sys
from openpyxl import load_workbook
ws = load_workbook(sys.argv[1]).active
head = [c.value for c in next(ws.iter_rows(max_row=1))]
assert head[-4:] == ['列型', '外來詞(段落與題目皆無)', '段落有但未照抄', '棄答時真值最佳名次'], head
rows = [[c.value for c in r] for r in ws.iter_rows(min_row=2)]
# 核心修復:假伺服器回覆無 [n] → 過去整個組合零列、完全看不見(發現 47)
assert len(rows) == 1, rows
assert rows[0][-4] == '無引用回答', rows[0]
assert rows[0][-1] == 1, rows[0][-1]   # 真值 doc_001_0 排第一 → 該取到卻沒作答
print('[abstain-rows] 1 列,列型與真值名次落地(過去此組合零列)')
" "$AB_OUT" || { echo "棄答列層失敗(rc=$AB_RC)"; cat "$DM_TMP/abstrows.txt"; exit 1; }
# 同情境改用 --cite 伺服器 → 有作答列,驗兩個藥名對照欄
python3 tools/fake_openai_server.py --port 0 --cite > "$DM_TMP/srv_cite2.log" 2>&1 &
TM_SRV=$!
until grep -q "^PORT=" "$DM_TMP/srv_cite2.log"; do sleep 0.2; done
TM_PORT="$(sed -n 's/^PORT=//p' "$DM_TMP/srv_cite2.log")"
sed "s|http://127.0.0.1:$DM_PORT|http://127.0.0.1:$TM_PORT|" "$DM_TMP/demo_abst.yaml" > "$DM_TMP/demo_terms.yaml"
TM_RC=0
python3 tools/support_sheet.py "$DM_TMP/demo_terms.yaml" "$DM_TMP/terms.xlsx" \
  --limit 1 --band all --sample 0 --preset left > "$DM_TMP/termsrows.txt" 2>&1 || TM_RC=$?
kill "$TM_SRV" 2>/dev/null; wait "$TM_SRV" 2>/dev/null || true
TM_OUT="$DM_TMP/terms.xlsx"; [ -f "$TM_OUT" ] || TM_OUT="$DM_TMP/terms.csv"
[ "$TM_RC" -eq 0 ] && python3 -c "
import sys
from openpyxl import load_workbook
ws = load_workbook(sys.argv[1]).active
rows = [[c.value for c in r] for r in ws.iter_rows(min_row=2)]
assert rows, 'no rows'
assert all(r[-4] == '作答' for r in rows), rows
# 所引段落有 methotrexate、假伺服器回覆沒有 → 疑似譯寫(發現 42/46 的訊號)
assert all(r[-2] == 'methotrexate' for r in rows), [r[-2] for r in rows]
assert all((r[-3] or '') == '' for r in rows), [r[-3] for r in rows]  # 句中無憑空英文詞
assert all((r[-1] or '') == '' for r in rows), [r[-1] for r in rows]  # 作答列不填真值名次
print(f'[terms] {len(rows)} 列,外來詞/未照抄/真值名次三欄落地')
" "$TM_OUT" || { echo "藥名對照層失敗(rc=$TM_RC)"; cat "$DM_TMP/termsrows.txt"; exit 1; }
echo
kill "$DM_SRV" 2>/dev/null; wait "$DM_SRV" 2>/dev/null || true
# left 打 LLM(回覆 <回覆:…>);right top-1 0.10 < 門檻 0.50 → 拒答文字,且 fake server 只收到 left 的 1 筆正式請求
[ "$DM_RC" -eq 0 ] && grep -q "^\[left\]" "$DM_TMP/demo.txt" && grep -q "回答:<回覆:" "$DM_TMP/demo.txt" \
  && grep -q "回答:\[拒答\]" "$DM_TMP/demo.txt" || { echo "demo 層失敗(rc=$DM_RC)"; cat "$DM_TMP/demo.txt"; exit 1; }
grep -E "^\[(left|right)\]|預熱" "$DM_TMP/demo.txt"
# 引用有效率模式(scripted 只寫第 1 題劇本,故 --cite-rate 1):JSON 落地、兩個 preset 皆有摘要、明細零回答/段落文字(假伺服器回覆無 [n],只驗管線;真值率在桌機量)
[ "$DM_RC2" -eq 0 ] && [ -f "$DM_TMP/cite.json" ] \
  && [ "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d['per_preset']))" "$DM_TMP/cite.json")" = "2" ] \
  && [ "$(grep -c '"answer"\|"text"' "$DM_TMP/cite.json")" = "0" ] \
  || { echo "demo cite-rate 層失敗(rc=$DM_RC2)"; cat "$DM_TMP/cite.txt"; exit 1; }
grep "^\[cite-rate\]" "$DM_TMP/cite.txt"
# 拒答門檻網格:collect(不打 LLM)→ sweep;right 的 eval 題 0.10 / 語料外 0.01 → 存在分隔門檻,高原必須出現
python3 tools/abstain_grid.py collect "$DM_TMP/demo.yaml" "$DM_TMP/off.json" "$DM_TMP/abstain.json" --k 2 > "$DM_TMP/abst.txt" 2>&1 \
  && python3 tools/abstain_grid.py sweep "$DM_TMP/abstain.json" --grid 5 --max-false 0.0 --min-true 1.0 >> "$DM_TMP/abst.txt" 2>&1 \
  && python3 tools/abstain_grid.py cv "$DM_TMP/abstain.json" --k 2 --seed 0 --grid 5 --max-false 0.0 --min-true 1.0 >> "$DM_TMP/abst.txt" 2>&1 \
  && grep -q "^== right" "$DM_TMP/abst.txt" && grep -q "^高原(" "$DM_TMP/abst.txt" && grep -q "^held-out 平均誤拒" "$DM_TMP/abst.txt" \
  || { echo "abstain_grid 層失敗"; cat "$DM_TMP/abst.txt"; exit 1; }
grep -E "^\[collect\]|^高原|^中點|^held-out" "$DM_TMP/abst.txt"
echo


echo "### 6. 供應鏈稽核(只稽核本專案釘選的依賴,不掃系統套件)"
python3 -m pip_audit -r requirements.txt -r requirements-dev.txt
echo

echo "### 7. Suite health(隨機順序;seed 印出以便重現 flaky)"
SEED="${GAUNTLET_SEED:-$RANDOM}"
echo "  randomly-seed=$SEED"
python3 -m pytest tests -q -p randomly --randomly-seed="$SEED"
echo
echo "=== gauntlet 全層通過 ==="
