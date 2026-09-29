#!/usr/bin/env python3
"""手動變異測試:對檢索指標植入真實 bug,驗證測試套件能殺死每一個。

用法:python3 tools/mutants.py [pytest -k 表達式]
每個變異對應一種預先列出的失效模式;存活的變異代表測試有洞。
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

METRICS = Path("src/selectrag/evaluation/retrieval_metrics.py")
EVAL_SET = Path("src/selectrag/evaluation/eval_set.py")
EXPERIMENT = Path("src/selectrag/runner/experiment.py")
CONFIG = Path("src/selectrag/runner/config.py")
SCRIPTED = Path("src/selectrag/retrieval/scripted.py")
MAIN = Path("src/selectrag/runner/__main__.py")
NORMALIZE = Path("src/selectrag/corpus/normalize.py")
HEADINGS = Path("src/selectrag/corpus/headings.py")
CHUNKER = Path("src/selectrag/corpus/chunker.py")
LOADERS = Path("src/selectrag/corpus/loaders.py")
BM25 = Path("src/selectrag/retrieval/bm25.py")
DENSE = Path("src/selectrag/retrieval/dense.py")
NEIGHBOR = Path("src/selectrag/retrieval/neighbor.py")
HYBRID = Path("src/selectrag/retrieval/hybrid.py")
RERANK = Path("src/selectrag/retrieval/rerank.py")
MAIN_RUNNER = Path("src/selectrag/runner/__main__.py")
DOCUMENT = Path("src/selectrag/corpus/document.py")
TRUTH_DUPES = Path("tools/truth_dupes.py")
LANGDRIFT = Path("src/selectrag/perf/langdrift.py")
CALIB = Path("tools/calib_sheet.py")
DEMO_APP = Path("tools/demo_app.py")
SUPPORT = Path("tools/support_sheet.py")
TERMS = Path("src/selectrag/evaluation/terms.py")
ABSTAIN_AUDIT = Path("src/selectrag/evaluation/abstain_audit.py")
COMPONENTS = Path("src/selectrag/evaluation/components.py")
ABSTAIN = Path("src/selectrag/demo/abstain.py")
DEMO = Path("src/selectrag/demo/core.py")
BENCH = Path("src/selectrag/perf/bench.py")
FAITH = Path("src/selectrag/evaluation/faithfulness.py")

# (目標檔, 說明, 對應失效模式, 原字串, 變異字串)
MUTANTS: list[tuple[Path, str, str, str, str]] = [
    (
        METRICS,
        "recall 分母誤用 k 而非 |relevant|",
        "失效模式 1(公式錯)",
        "return hits / len(relevant)",
        "return hits / k",
    ),
    (
        METRICS,
        "recall 忽略 top-k 截斷",
        "失效模式 1(公式錯)",
        "hits = sum(1 for doc in retrieved[:k] if doc in relevant)",
        "hits = sum(1 for doc in retrieved if doc in relevant)",
    ),
    (
        METRICS,
        "reciprocal_rank 排名 off-by-one",
        "失效模式 1(公式錯)",
        "for rank, doc in enumerate(retrieved, start=1):",
        "for rank, doc in enumerate(retrieved, start=2):",
    ),
    (
        METRICS,
        "聚合用總和取代平均",
        "失效模式 3(除以錯的數)",
        "return fmean(recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases)",
        "return sum(recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases)",
    ),
    (
        METRICS,
        "k 邊界檢查放寬,k=0 不再報錯",
        "失效模式 2(邊界靜默)",
        'raise ValueError("k must be >= 1")\n    _check_inputs(retrieved, relevant)\n    hits',
        'raise ValueError("k must be >= 1") if k < 0 else None\n'
        "    _check_inputs(retrieved, relevant)\n    hits",
    ),
    (
        METRICS,
        "重複 id 檢查失效",
        "失效模式 5(重複計數)",
        "if len(set(retrieved)) != len(retrieved):",
        "if len(set(retrieved)) != len(retrieved) and False:",
    ),
    (
        EVAL_SET,
        "content hash 比對失效",
        "失效模式 1(凍結失效)",
        'if actual_hash != manifest["content_hash"]:',
        'if actual_hash != manifest["content_hash"] and False:',
    ),
    (
        EVAL_SET,
        "hash 改為對題目順序敏感",
        "失效模式 2(hash 不穩定)",
        "[_to_dict(item) for item in sorted(items, key=lambda i: i.id)],",
        "[_to_dict(item) for item in items],",
    ),
    (
        EVAL_SET,
        "item_count 比對失效",
        "失效模式 1(凍結失效)",
        'if manifest["item_count"] != len(items):',
        'if manifest["item_count"] < 0:',
    ),
    (
        EVAL_SET,
        "relevant_doc_ids 重複檢查失效",
        "失效模式 4(靜默去重)",
        "if len(set(doc_ids)) != len(doc_ids):",
        "if len(set(doc_ids)) != len(doc_ids) and False:",
    ),
    (
        EVAL_SET,
        "空 relevant_doc_ids 檢查失效",
        "失效模式 4(壞題目)",
        "if not item.relevant_doc_ids:",
        "if not item.relevant_doc_ids and False:",
    ),
    (
        EXPERIMENT,
        "覆蓋保護失效(結果被靜默蓋掉)",
        "失效模式 1(資料永久遺失)",
        "if path.exists():",
        "if path.exists() and False:",
    ),
    (
        EXPERIMENT,
        "eval set hash 記錄成空字串",
        "失效模式 2(可比性喪失)",
        "eval_set_hash=compute_content_hash(eval_set.items),",
        'eval_set_hash="",',
    ),
    (
        EXPERIMENT,
        "檢索深度誤用 min 而非 max(k)",
        "失效模式 5(k 與指標錯配)",
        "depth = max(config.k_values)",
        "depth = min(config.k_values)",
    ),
    (
        EXPERIMENT,
        "per_query 只記錄第一題",
        "失效模式 4(部分結果)",
        "per_query=tuple(per_query),",
        "per_query=tuple(per_query[:1]),",
    ),
    (
        CONFIG,
        "k 值下限檢查失效",
        "失效模式 5(壞設定)",
        "if any(k < 1 for k in k_values):",
        "if any(k < 0 for k in k_values):",
    ),
    (
        SCRIPTED,
        "未知問題靜默回傳空結果",
        "失效模式 4(漏題偽裝成檢索不到)",
        "if question not in self._script:",
        "if question not in self._script and False:",
    ),
    (
        MAIN,
        "未知檢索器型別靜默回退到空檢索器",
        "失效模式 3(設定與實際不符)",
        '    raise ValueError(f"unknown retriever type: {config.type!r}")',
        "    return ScriptedRetriever({})",
    ),
    (
        MAIN,
        "缺 script 參數不報錯",
        "失效模式 3(設定不完整)",
        "        if not script_path:",
        "        if not script_path and False:",
    ),
    (
        MAIN,
        "CLI 參數數量檢查失效",
        "失效模式 3(誤用不被擋)",
        "    if len(argv) < 2:",
        "    if len(argv) < 1:",
    ),
    (
        NORMALIZE,
        "NFKC 正規化失效(異體字未還原)",
        "失效模式 3(檢索比對不到)",
        'return re.sub(r"\\s+", " ", unicodedata.normalize("NFKC", text)).strip()',
        'return re.sub(r"\\s+", " ", text).strip()',
    ),
    (
        NORMALIZE,
        "行合併忽略標題邊界",
        "失效模式 1(跨條文黏合)",
        "if next_line is None or is_heading(next_line):",
        "if next_line is None:",
    ),
    (
        NORMALIZE,
        "短行不再視為段落結束",
        "失效模式 1(段落黏合)",
        "return len(line) < short_threshold",
        "return False",
    ),
    (
        HEADINGS,
        "日期防護失效",
        "失效模式 2(日期被當標題)",
        "if not stripped or _DATE.match(stripped):",
        "if not stripped:",
    ),
    (
        HEADINGS,
        "體系判別永遠回傳公文式",
        "失效模式 2(仿單切壞)",
        "return HeadingStyle.LEAFLET if leaflet > official else HeadingStyle.OFFICIAL",
        "return HeadingStyle.OFFICIAL",
    ),
    (
        CHUNKER,
        "短單元合併失效(回到碎片化)",
        "失效模式:語料形狀扭曲實驗",
        "if len(prev_body) >= min_chars or not shared or too_long:",
        "if True:",
    ),
    (
        CHUNKER,
        "合併跨越無共同前綴的邊界",
        "失效模式 5(citation 錯位)",
        "shared = _common_prefix(prev_path, path)",
        "shared = prev_path",
    ),
    (
        CHUNKER,
        "超長單元不再切分",
        "失效模式:chunk 大小失控",
        "return [text[i : i + max_chars] for i in range(0, len(text), step) "
        "if text[i : i + max_chars]]",
        "return [text]",
    ),
    (
        LOADERS,
        "HTML 正文壓成單行(切塊全毀)",
        "失效模式:結構喪失",
        'text = "\\n".join(piece for piece in pieces if piece)',
        'text = " ".join(piece for piece in pieces if piece)',
    ),
    (
        BM25,
        "IDF 退化為常數(常見詞與罕見詞同權重)",
        "失效模式 1(公式錯)",
        "math.log((n - count + 0.5) / (count + 0.5) + 1)",
        "1.0",
    ),
    (
        BM25,
        "文件長度正規化失效(b 形同 0)",
        "失效模式 1(公式錯)",
        "self._b * self._doc_len[index] / self._avgdl",
        "self._b",
    ),
    (
        BM25,
        "TF 飽和失效(分母漏掉詞頻)",
        "失效模式 1(公式錯)",
        "/ (count + norm)",
        "/ (1 + norm)",
    ),
    (
        BM25,
        "零分文件也入榜(零重疊查詢回垃圾)",
        "失效模式:recall 虛高",
        "if score > 0.0:",
        "if score >= 0.0:",
    ),
    (
        BM25,
        "top_k 截斷失效",
        "失效模式:k 與指標錯配",
        "scored[:top_k]",
        "scored",
    ),
    (
        BM25,
        "CJK bigram 退化為整段單一 token(中文查詢比對不到)",
        "失效模式:檢索無聲歸零",
        "tokens.extend(run[i : i + 2] for i in range(len(run) - 1))",
        "tokens.append(run)",
    ),
    (
        BM25,
        "查詢未正規化(全形查詢比對不到半形文件)",
        "失效模式:檢索無聲歸零",
        "query_tokens = tokenize(question)",
        "query_tokens = question.lower().split()",
    ),
    (
        BM25,
        "重複 id 檢查失效",
        "失效模式 5(重複計數)",
        "if len(set(ids)) != len(ids):",
        "if False:",
    ),
    (
        BM25,
        "空查詢不再報錯(壞題目偽裝成檢索不到)",
        "失效模式 4(壞題目)",
        'raise ValueError("query has no tokens after normalization")',
        "return []",
    ),
    (
        METRICS,
        "citation 分母誤用 |relevant| 而非 |cited|",
        "失效模式 1(公式錯)",
        "return sum(1 for doc in cited if doc in relevant) / len(cited)",
        "return sum(1 for doc in cited if doc in relevant) / len(relevant)",
    ),
    (
        METRICS,
        "空引用不再報錯(無據答案被記成 0 分混入平均)",
        "失效模式 4(壞資料靜默)",
        'raise ValueError("no citations: an uncited answer must be counted separately, '
        'not scored 0")',
        "return 0.0",
    ),
    (
        METRICS,
        "hit 恆報命中(檢索失敗被掩蓋)",
        "失效模式 1(公式錯)",
        "return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0",
        "return 1.0",
    ),
    (
        METRICS,
        "hit 忽略 top-k 截斷",
        "失效模式 1(公式錯)",
        "1.0 if any(doc in relevant for doc in retrieved[:k])",
        "1.0 if any(doc in relevant for doc in retrieved)",
    ),
    (
        EXPERIMENT,
        "hit 聚合誤用 recall 值",
        "失效模式 2(指標錯配)",
        'metrics[f"hit@{k}"] = fmean(q.hit[f"hit@{k}"] for q in per_query)',
        'metrics[f"hit@{k}"] = fmean(q.recall[f"recall@{k}"] for q in per_query)',
    ),
    (
        BM25,
        "context_fields 靜默失效(只索引正文)",
        "失效模式:對照組與控制組無差異",
        'entries.append((row["chunk_id"], index_text(row, context_fields, where)))',
        'entries.append((row["chunk_id"], row["text"]))',
    ),
    (
        BM25,
        "未知 context 欄位不再報錯",
        "失效模式 3(設定誤用不被擋)",
        "if field not in _CONTEXT_FIELDS:",
        "if False:",
    ),
    (
        BM25,
        "heading_path 串接退化為空字串",
        "失效模式:標題脈絡無聲丟失",
        'return " ".join(str(part) for part in value)',
        'return ""',
    ),
    (
        CHUNKER,
        "doc_title 退化為 doc_id(藥名丟失)",
        "失效模式:文件身分無聲丟失",
        "doc_title = next((ln for ln in lines if ln), document.doc_id)",
        "doc_title = document.doc_id",
    ),
    # ---- dense(2026-09-13)----
    (
        DENSE,
        "dense 排序方向反轉(最不相似排第一)",
        "失效模式 1(公式錯)",
        'order = np.argsort(-scores, kind="stable")[:top_k]',
        'order = np.argsort(scores, kind="stable")[:top_k]',
    ),
    (
        DENSE,
        "dense 平手改為不穩定排序(跨次不可重現)",
        "失效模式 2(不可重現)",
        'order = np.argsort(-scores, kind="stable")[:top_k]',
        'order = np.argsort(-scores, kind="quicksort")[::-1][:top_k][::-1]',
    ),
    (
        DENSE,
        "dense top_k 截斷失效",
        "失效模式:k 與指標錯配",
        'order = np.argsort(-scores, kind="stable")[:top_k]',
        'order = np.argsort(-scores, kind="stable")',
    ),
    (
        DENSE,
        "dense 查詢向量未 normalize(cosine 退化為內積)",
        "失效模式 1(公式錯)",
        "scores = self._matrix @ _l2_normalize(query)[0]",
        "scores = self._matrix @ query[0]",
    ),
    (
        DENSE,
        "dense 索引向量未 normalize(長向量恆勝)",
        "失效模式 1(公式錯)",
        "self._matrix = _l2_normalize(matrix)",
        "self._matrix = matrix",
    ),
    (
        DENSE,
        "dense 索引 model 身分核對失效(舊索引配新模型無聲通過)",
        "失效模式 3(設定與實際不符)",
        "if meta.model != model:",
        "if False and meta.model != model:",
    ),
    (
        DENSE,
        "dense 重複 id 檢查失效",
        "失效模式 5(重複計數)",
        "        if len(set(ids)) != len(ids):\n"
        '            raise ValueError("duplicate ids in index")',
        "        if False:\n" '            raise ValueError("duplicate ids in index")',
    ),
    # ---- neighbor(2026-09-14)----
    (
        NEIGHBOR,
        "鄰塊順序反轉(後一塊先於前一塊)",
        "失效模式 2(順序語意錯)",
        "            if i - d >= 0:\n                out.append(ids[i - d])\n"
        "            if i + d < len(ids):\n                out.append(ids[i + d])",
        "            if i + d < len(ids):\n                out.append(ids[i + d])\n"
        "            if i - d >= 0:\n                out.append(ids[i - d])",
    ),
    (
        NEIGHBOR,
        "鄰塊跨文件越界(以全域行序取鄰塊,文件末塊接到下一份文件)",
        "失效模式:citation 錯文件",
        "        doc, i = self._pos[chunk_id]\n        ids = self._docs[doc]",
        "        doc, i = self._pos[chunk_id]\n"
        "        ids = [c for d in self._docs.values() for c in d]\n"
        "        i = ids.index(chunk_id)",
    ),
    (
        NEIGHBOR,
        "去重失效(同一 chunk 重複入榜佔用 k 預算)",
        "失效模式 5(重複計數)",
        "                if cid not in seen:",
        "                if True:",
    ),
    (
        NEIGHBOR,
        "擴展後不截回 top_k(k 預算失守)",
        "失效模式:對照組數字虛高",
        "        return out[:top_k]",
        "        return out",
    ),
    (
        MAIN_RUNNER,
        "window=0 仍包裝(對照組被改變)",
        "失效模式 3(設定與實際不符)",
        "    if window == 0:\n        return base",
        "    if window < 0:\n        return base",
    ),
    # ---- hybrid(2026-09-14)----
    (
        HYBRID,
        "RRF 常數未進分母(退化為 1/rank)",
        "失效模式 1(公式錯)",
        "scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)",
        "scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / rank",
    ),
    (
        HYBRID,
        "RRF 排序方向反轉(分數低者在前)",
        "失效模式 1(公式錯)",
        "return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))",
        "return sorted(scores, key=lambda d: (scores[d], first_seen[d]))",
    ),
    (
        HYBRID,
        "平手改依最後出現(跨次不可重現的前兆)",
        "失效模式 2(不可重現)",
        "first_seen.setdefault(doc_id, (r_idx, rank))",
        "first_seen[doc_id] = (r_idx, rank)",
    ),
    (
        HYBRID,
        "候選池誤用 top_k(pool 變因失效)",
        "失效模式 5(k 與指標錯配)",
        "rankings = [list(r.retrieve(question, self._pool)) for r in self._retrievers]",
        "rankings = [list(r.retrieve(question, top_k)) for r in self._retrievers]",
    ),
    (
        HYBRID,
        "融合後不截回 top_k",
        "失效模式:對照組數字虛高",
        "return rrf_fuse(rankings, self._rrf_k)[:top_k]",
        "return rrf_fuse(rankings, self._rrf_k)",
    ),
    (
        MAIN_RUNNER,
        "巢狀 dense 不沿用注入的 embedder(測試替身失效、真模型被載入)",
        "失效模式 3(設定與實際不符)",
        "                embedder_factory,\n                reranker_factory,\n            )\n"
        "            for sub in nested",
        "                _default_embedder,\n                reranker_factory,\n            )\n"
        "            for sub in nested",
    ),
    # ---- rerank(2026-09-14)----
    (
        RERANK,
        "retrieve_scored 分數歸零(拒答門檻永遠觸發)",
        "失效模式 4(壞資料靜默)",
        "        return [(ids[i], float(scores[i])) for i in order[:top_k]]",
        "        return [(ids[i], 0.0) for i in order[:top_k]]",
    ),
    (
        BM25,
        "BM25 retrieve_scored 分數與 id 錯配(取排序前的分數)",
        "失效模式 2(語意錯)",
        "        return [(self._ids[index], score) for score, index in scored[:top_k]]",
        "        return [(self._ids[index], scored[0][0]) for score, index in scored[:top_k]]",
    ),
    (
        RERANK,
        "rerank 排序方向反轉(分數低者在前)",
        "失效模式 1(公式錯)",
        "order = sorted(range(len(ids)), key=lambda i: -scores[i])",
        "order = sorted(range(len(ids)), key=lambda i: scores[i])",
    ),
    (
        RERANK,
        "rerank 同分改為反序(不穩定排序的前兆)",
        "失效模式 2(不可重現)",
        "for i in order[:top_k]]",
        "for i in sorted(order, key=lambda i: (-scores[i], -i))[:top_k]]",
    ),
    (
        RERANK,
        "候選數誤用 top_k(candidates 變因失效)",
        "失效模式 5(k 與指標錯配)",
        "ids = list(self._base.retrieve(question, self._candidates))",
        "ids = list(self._base.retrieve(question, top_k))",
    ),
    (
        RERANK,
        "重排後不截回 top_k",
        "失效模式:對照組數字虛高",
        "        return [(ids[i], float(scores[i])) for i in order[:top_k]]",
        "        return [(ids[i], float(scores[i])) for i in order]",
    ),
    (
        RERANK,
        "分數筆數錯位不 raise(名次無聲全亂)",
        "失效模式 4(壞資料靜默)",
        "        if len(scores) != len(ids):",
        "        if False:",
    ),
    (
        RERANK,
        "NIM 回應忽略 index,依回傳順序填分數(名次無聲全亂)",
        "失效模式 1(公式錯)",
        '            scores[idx] = float(item["logit"])',
        '            scores[payload["rankings"].index(item)] = float(item["logit"])',
    ),
    (
        RERANK,
        "NIM 回應筆數不足不 raise",
        "失效模式 4(壞資料靜默)",
        "        if any(v is None for v in scores):",
        "        if False:",
    ),
    # ---- truth_dupes(2026-09-15)----
    (
        TRUTH_DUPES,
        "掃描不排除自身(每筆真值必列自己 1.0)",
        "失效模式:候選清單被自身污染",
        "for cid, g in grams.items() if cid != t]",
        "for cid, g in grams.items()]",
    ),
    (
        TRUTH_DUPES,
        "門檻改為嚴格大於(恰等於門檻者漏掉)",
        "失效模式 5(邊界錯)",
        "kept = [(cid, s) for cid, s in scored if s >= threshold]",
        "kept = [(cid, s) for cid, s in scored if s > threshold]",
    ),
    (
        TRUTH_DUPES,
        "排序方向反轉(最不相似排第一)",
        "失效模式 1(公式錯)",
        "kept.sort(key=lambda x: (-x[1], x[0]))",
        "kept.sort(key=lambda x: (x[1], x[0]))",
    ),
    (
        TRUTH_DUPES,
        "Jaccard 分母改用交集(恆為 1.0)",
        "失效模式 1(公式錯)",
        "return len(a & b) / union if union else 0.0",
        "return len(a & b) / len(a & b) if len(a & b) else 0.0",
    ),
    # ---- bench(2026-09-15)----
    (
        BENCH,
        "TTFT 以任何 chunk(含只有 role 的首 chunk)起算",
        "失效模式 1(公式錯:TTFT 系統性偏低)",
        "        if content:\n            if self._ttft is None:",
        "        if True:\n            if self._ttft is None:",
    ),
    (
        BENCH,
        "百分位改為插值下取(回傳值可能不在樣本中)",
        "失效模式 1(公式錯)",
        "    rank = max(1, math.ceil(p / 100.0 * len(s)))",
        "    rank = max(1, int(p / 100.0 * len(s)))",
    ),
    (
        BENCH,
        "吞吐分母改用延遲總和而非牆鐘時間(併發下虛高)",
        "失效模式 1(公式錯)",
        '        "tok_per_s": tokens_total / wall_s,',
        '        "tok_per_s": tokens_total / sum(lat),',
    ),
    (
        BENCH,
        "一致性比對忽略長度差(前綴相同即視為一致)",
        "失效模式 4(壞資料靜默)",
        "        if x == y:",
        "        if x.startswith(y) or y.startswith(x):",
    ),
    # ---- faithfulness(2026-09-16)----
    (
        FAITH,
        "拒答句不剔除(拒答被當主張灌進分母)",
        "失效模式 4(壞資料靜默)",
        "    return [s for s in _sentences(answer) if not _is_abstention_sentence(s)]",
        "    return list(_sentences(answer))",
    ),
    (
        FAITH,
        "supported 無證據編號不 raise(judge 說有據卻指不出來源)",
        "失效模式 4(壞資料靜默)",
        '    if verdict == "supported" and ev is None:',
        "    if False:",
    ),
    (
        FAITH,
        "κ 忽略期望一致項(退化為原始一致率)",
        "失效模式 1(公式錯)",
        "    return (po - pe) / (1.0 - pe)",
        "    return po",
    ),
    (
        FAITH,
        "合成負例可能等於原值(n=0 時 ×2 仍為 0)",
        "失效模式 5(邊界錯)",
        "    new = n + 1 if rng.random() < 0.5 or n == 0 else n * 2",
        "    new = n + 1 if rng.random() < 0.5 else n * 2",
    ),
    # ---- 逐字命中前置層(2026-09-17)----
    (
        FAITH,
        "覆蓋率門檻改成嚴格大於(門檻值本身不命中)",
        "失效模式 5(邊界錯)",
        "        if cov < min_coverage or not nums <= set(_NUMBER.findall(text)):",
        "        if cov <= min_coverage or not nums <= set(_NUMBER.findall(text)):",
    ),
    (
        FAITH,
        "數字守門失效(數字竄改的合成負例被前置層放行)",
        "失效模式 4(壞資料靜默)",
        "        if cov < min_coverage or not nums <= set(_NUMBER.findall(text)):",
        "        if cov < min_coverage:",
    ),
    (
        FAITH,
        "短句守門失效(短句碰巧命中即判 supported)",
        "失效模式 5(邊界錯)",
        "    if len(_norm(claim)) < _MIN_CLAIM_CHARS:",
        "    if False:",
    ),
    (
        FAITH,
        "多 chunk 取第一個命中而非覆蓋率最高",
        "失效模式 2(語意錯)",
        "        if best is None or key > best:",
        "        if best is None:",
    ),
    (
        FAITH,
        "覆蓋率分母錯(用 context 的 gram 數)",
        "失效模式 1(公式錯)",
        "    return len(g & _grams3(_norm(context))) / len(g)",
        "    return len(g & _grams3(_norm(context))) / max(len(_grams3(_norm(context))), 1)",
    ),
    (
        Path("tools/judge_claims.py"),
        "前置層命中不計入 summary(verbatim_hits 恆 0)",
        "失效模式 4(壞資料靜默)",
        "                n_verbatim += 1",
        "                n_verbatim += 0",
    ),
    (
        Path("tools/verbatim_grid.py"),
        "高原計算忽略誤放(任何門檻都算高原)",
        "失效模式 1(公式錯)",
        '    zero = [r["threshold"] for r in rows if r["false_hits"] == 0]',
        '    zero = [r["threshold"] for r in rows]',
    ),
    # ---- 簡體字元佔比(2026-09-17)----
    (
        LANGDRIFT,
        "分母改用全部字元(含 ASCII/標點,佔比被稀釋)",
        "失效模式 3(除以錯的數)",
        "    return sum(1 for c in cjk if c in SIMPLIFIED_ONLY) / len(cjk)",
        "    return sum(1 for c in cjk if c in SIMPLIFIED_ONLY) / max(len(text), 1)",
    ),
    (
        LANGDRIFT,
        "字表比對反向(繁體被當簡體)",
        "失效模式 2(語意錯)",
        "    return sum(1 for c in cjk if c in SIMPLIFIED_ONLY) / len(cjk)",
        "    return sum(1 for c in cjk if c not in SIMPLIFIED_ONLY) / len(cjk)",
    ),
    (
        LANGDRIFT,
        "含簡體題數改成全部題數",
        "失效模式 4(壞資料靜默)",
        "    ids = sorted(k for k, r in ratios.items() if r > 0)",
        "    ids = sorted(k for k, r in ratios.items() if r >= 0)",
    ),
    # ---- 拒答標記簡體正規化 + 分層抽樣(2026-09-17)----
    (
        FAITH,
        "拒答標記比對不做簡體正規化(参考資料未提及被當主張)",
        "失效模式 4(壞資料靜默)",
        "    t = normalize_variants(s)",
        "    t = s",
    ),
    (
        CALIB,
        "分層門檻改嚴格大於(覆蓋率恰等於門檻的逐字句掉到 paraphrase)",
        "失效模式 5(邊界錯)",
        "    if cov >= min_coverage:",
        "    if cov > min_coverage:",
    ),
    (
        CALIB,
        "去重失效(四格式同題同句重複進校準集)",
        "失效模式 4(壞資料靜默)",
        "                if key in seen:",
        "                if False:",
    ),
    (
        CALIB,
        "配額不裁(超額層全收)",
        "失效模式 2(語意錯)",
        "        chosen = members if q is None or len(members) <= q else rng.sample(members, q)",
        "        chosen = members",
    ),
    (
        CALIB,
        "分層 κ 用全體而非該層子集",
        "失效模式 1(公式錯)",
        "            sub = [p for p in pairs if strata.get(p[3]) == name]",
        "            sub = list(pairs)",
    ),
    # ---- demo core(2026-09-18)----
    (
        DEMO,
        "retriever 快取失效(每題重建 preset)",
        "失效模式 2(語意錯)",
        "        if preset not in self._cache:",
        "        if True:",
    ),
    (
        DEMO,
        "真值標色反向",
        "失效模式 2(語意錯)",
        "h.chunk_id in truth, h.chunk_id not in other",
        "h.chunk_id not in truth, h.chunk_id not in other",
    ),
    (
        DEMO,
        "only_this_side 漏算(恆 False)",
        "失效模式 4(壞資料靜默)",
        "h.chunk_id in truth, h.chunk_id not in other",
        "h.chunk_id in truth, False",
    ),
    (
        DEMO,
        "引用越界不報(0 與 n+1 也算有效)",
        "失效模式 5(邊界錯)",
        "    valid = {i for i in found if 1 <= i <= n_chunks}",
        "    valid = set(found)",
    ),
    (
        DEMO,
        "串流錯誤改 raise(單題失敗讓 demo 崩潰)",
        "失效模式 4(壞資料靜默)",
        '        yield f"[錯誤] {e}"',
        "        raise",
    ),
    (
        DEMO,
        "presets < 2 不擋",
        "失效模式 3(設定不完整)",
        "    if len(presets) < 2:",
        "    if len(presets) < 1:",
    ),
    (
        DEMO,
        "句號後的 [n] 不歸前一句(未標引用句被高估)",
        "失效模式 5(邊界錯)",
        "        m = _TRAILING_CITE.match(answer, i)",
        "        m = None",
    ),
    (
        DEMO,
        "句子切分漏 !?(只切 。)",
        "失效模式 5(邊界錯)",
        '_SENT_END = re.compile(r"[。!?]")',
        '_SENT_END = re.compile(r"[。]")',
    ),
    (
        DEMO,
        "cites_truth 恆 True(引用任何段落都算含真值)",
        "失效模式 2(語意錯)",
        "        cites_truth=bool(valid & truth_ranks),",
        "        cites_truth=bool(valid),",
    ),
    (
        DEMO,
        "truth_cite_rate 分母改用全部回答",
        "失效模式 2(語意錯)",
        '        "truth_cite_rate": truth_hits / len(truth_pool) if truth_pool else 0.0,',
        '        "truth_cite_rate": truth_hits / n if n else 0.0,',
    ),
    (
        DEMO_APP,
        "cite-rate 遇 LLM 錯誤不回 1(靜默跳過)",
        "失效模式 4(壞資料靜默)",
        'LLM 錯誤:{final[:80]}")\n                return 1',
        'LLM 錯誤:{final[:80]}")\n                continue',
    ),
    (
        SUPPORT,
        "high 帶邊界寫成嚴格大於(恰等於門檻的句子兩帶都不收,抽樣框漏列)",
        "失效模式 5(邊界錯)",
        "return [r for r in scored if float(r[COVERAGE_COL]) >= min_coverage]",
        "return [r for r in scored if float(r[COVERAGE_COL]) > min_coverage]",
    ),
    # ---- 必要成分命中(2026-09-20)----
    (
        COMPONENTS,
        "空成分清單靜默回全部命中(壞清單與真的全答到分不出來)",
        "失效模式 4(壞資料靜默)",
        'raise ValueError("empty component specs")',
        "return (), ()",
    ),
    (
        COMPONENTS,
        "只認第一個等價表述(措辭變體被誤判為漏答)",
        "失效模式 2(語意錯)",
        "any(_contains(norm_answer, normalize_glyphs(alias)) for alias in spec.aliases)",
        "_contains(norm_answer, normalize_glyphs(spec.aliases[0]))",
    ),
    (
        COMPONENTS,
        "引用支撐恆為空(completeness 與 support 的矛盾被掩蓋)",
        "失效模式 4(壞資料靜默)",
        "if not any(_contains(joined, normalize_glyphs(a)) for a in spec.aliases):",
        "if False:  # noqa",
    ),
    (
        SUPPORT,
        "未引用判定只看當前句而非全題引用句(同題其他句的支撐被忽略)",
        "失效模式 2(語意錯)",
        "uncited = unsupported_hits(hit, cited_sentences, spec_item.components)",
        "uncited = unsupported_hits(hit, [sup.sentence], spec_item.components)",
    ),
    (
        SUPPORT,
        "引用不支撐欄恆為空(引用錯置型從欄位底下溜過去)",
        "失效模式 4(壞資料靜默)",
        "mismatch = tuple(n for n in in_segments if n not in uncited)",
        "mismatch = ()",
    ),
    (
        SUPPORT,
        "兩欄不互斥(未引用的成分同時出現在引用不支撐欄,重複計數)",
        "失效模式 5(重複計數)",
        "in_segments = unsupported_hits(hit, cited_segments, spec_item.components)",
        "in_segments = unsupported_hits(hit, [], spec_item.components)",
    ),
    (
        COMPONENTS,
        "數字邊界檢查被拿掉(1500 誤中 15000、650 誤中 6500/1650)",
        "失效模式 5(邊界錯)",
        "    if not head_digit and not tail_digit:\n        return needle in haystack",
        "    if True:\n        return needle in haystack",
    ),
    (
        COMPONENTS,
        "邊界比對只看第一個出現位置(前面有不合格的出現就整個漏判)",
        "失效模式 5(邊界錯)",
        "        start = i + 1",
        "        return False",
    ),
    (
        COMPONENTS,
        "CJK 部首不折疊(語料的異體字「延⻑」與模型輸出的「延長」判成不同)",
        "失效模式 2(語意錯)",
        'folded = "".join(_RADICALS.get(ch, ch) for ch in text)',
        "folded = text",
    ),
    (
        COMPONENTS,
        "省掉末尾的 NFKC(去空白讓 combining mark 貼上前字,正規化不再冪等)",
        "失效模式 5(邊界錯)",
        'return unicodedata.normalize("NFKC", folded)',
        "return folded",
    ),
    (
        COMPONENTS,
        "千分位改脫所有逗號(句中逗號被吃掉,跨句子的表述會假命中)",
        "失效模式 2(語意錯)",
        r'_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")',
        '_THOUSANDS = re.compile(r",")',
    ),
    (
        COMPONENTS,
        "禁止表述被忽略(「每日2粒」的四倍劑量錯誤拿到滿分)",
        "失效模式 2(語意錯)",
        "if any(_contains(norm_answer, normalize_glyphs(bad)) for bad in spec.forbidden):",
        "if False:  # noqa",
    ),
    (
        COMPONENTS,
        "禁止表述命中後仍繼續比對 alias(答錯與答對同時成立)",
        "失效模式 5(邊界錯)",
        "            blocked.append(spec.name)\n            continue",
        "            blocked.append(spec.name)",
    ),
    (
        COMPONENTS,
        "禁止表述與正解重疊不被擋(該成分永遠不可能命中而無徵兆)",
        "失效模式 3(設定誤用不被擋)",
        "f\"forbidden overlaps aliases for component {raw['name']!r} of item {item_id}: \"",
        '"",  # noqa',
    ),
    (
        COMPONENTS,
        "缺漏成分被算進 completeness 的分子(漏答仍得滿分)",
        "失效模式 5(邊界錯)",
        "return len(hit) / len(specs)",
        "return 1.0 if hit else 0.0",
    ),
    (
        COMPONENTS,
        "同題重複成分名不被擋(清單筆誤導致成分被重複計數)",
        "失效模式 3(設定誤用不被擋)",
        'raise ValueError(f"duplicate component name in item {item_id}")',
        "pass  # noqa",
    ),
    (
        COMPONENTS,
        "題層禁止表述被忽略(整題答到別的標的仍拿滿分)",
        "失效模式 2(語意錯)",
        "    if item_forbidden_hits(answer, item):\n        return 0.0",
        "    if False:  # noqa\n        return 0.0",
    ),
    (
        COMPONENTS,
        "題層禁止表述與正解重疊不被擋(該題永遠歸零而無徵兆)",
        "失效模式 3(設定誤用不被擋)",
        'raise ValueError(f"item forbidden overlaps aliases in item {item_id}: {sorted(overlap)}")',
        "pass  # noqa",
    ),
    (
        COMPONENTS,
        "舊式 list 清單被當成 dict 讀(既有清單載入即壞)",
        "失效模式 3(設定誤用不被擋)",
        "    if isinstance(raw, Mapping):",
        "    if True:",
    ),
    (
        SUPPORT,
        "題層禁止與成分層禁止寫進同一欄(答不全與答錯題被統計成同一個數字)",
        "失效模式 2(語意錯)",
        '                            ",".join(item_bad),',
        '                            "",',
    ),
    (
        SUPPORT,
        "無引用句的組合不再留列(棄答整類回到看不見的狀態)",
        "失效模式 4(整類錯型被靜默剔除)",
        "            if not sups:",
        "            if not sups and False:",
    ),
    (
        SUPPORT,
        "棄答列的引用編號回到寫死空字串(引了段落卻說沒提到的訊號整批遺失)",
        "失效模式 4(整類訊號被靜默剔除)",
        '",".join(str(c) for c in sorted(cited_indices(answer, len(hits))[0])),',
        '"",',
    ),
    (
        SUPPORT,
        "棄答列引用編號不排除越界值(與句級的有效性判準不一致)",
        "失效模式 5(邊界錯)",
        "cited_indices(answer, len(hits))[0]",
        "cited_indices(answer, 99)[0]",
    ),
    (
        SUPPORT,
        "棄答與無引用回答不再分型(兩種成因被統計成同一個數字)",
        "失效模式 2(語意錯)",
        '"棄答" if is_abstain(answer, ABSTAIN) else "無引用回答",',
        '"棄答",',
    ),
    (
        SUPPORT,
        "分帶不再跳過無覆蓋率的列(棄答列混進覆蓋率分帶並使 float 崩潰)",
        "失效模式 5(邊界錯)",
        '    scored = [r for r in rows if r[COVERAGE_COL] != ""]',
        "    scored = list(rows)",
    ),
    (
        TERMS,
        "外來詞比對不再把題目當來源(整欄誤報:答句重述題目藥名即被標)",
        "失效模式 4(訊號被雜訊淹沒)",
        "    known = _keys([*segments, question])",
        "    known = _keys(segments)",
    ),
    (
        TERMS,
        "未照抄不再扣掉句子已寫的詞(恆回段落全部拉丁詞)",
        "失效模式 4(訊號被雜訊淹沒)",
        "    said = _keys([sentence])",
        "    said = set()",
    ),
    (
        TERMS,
        "詞集比對不再 casefold(Aspirin 與 aspirin 被當成兩個詞)",
        "失效模式 5(邊界錯)",
        "    return {t.casefold() for text in texts for t in latin_terms(text)}",
        "    return {t for text in texts for t in latin_terms(text)}",
    ),
    (
        TERMS,
        "詞長下限失效(mg / IU 等單位與單字母編號被當成藥名)",
        "失效模式 4(訊號被雜訊淹沒)",
        '_LATIN = re.compile(r"[A-Za-z]{3,}")',
        '_LATIN = re.compile(r"[A-Za-z]+")',
    ),
    (
        ABSTAIN_AUDIT,
        "棄答判定改為整段相等(第一句作答、第二句宣稱缺席的那型漏掉)",
        "失效模式 2(語意錯)",
        "    return abstain_marker in answer",
        "    return abstain_marker == answer",
    ),
    (
        ABSTAIN_AUDIT,
        "空真值集不再報錯(壞題目與真的沒取到混為一談)",
        "失效模式 4(壞題目)",
        "    if not relevant_doc_ids:",
        "    if not relevant_doc_ids and False:",
    ),
    (
        SUPPORT,
        "只有產生列的題目才進 dump(整題消失時失去唯一的回溯路徑)",
        "失效模式 2(語意錯)",
        "            if dump_to is not None:",
        "            if dump_to is not None and False:",
    ),
    (
        SUPPORT,
        "未給成分清單時也追加兩欄(既有產出的表頭被破壞)",
        "失效模式 3(設定誤用不被擋)",
        "return HEADER if components is None else [*HEADER, *COMPONENT_HEADER]",
        "return [*HEADER, *COMPONENT_HEADER]",
    ),
    (
        SUPPORT,
        "未知 preset 靜默跳過(跨格式比較少跑一邊而無徵兆)",
        "失效模式 3(設定誤用不被擋)",
        'raise KeyError(f"unknown preset {name!r}',
        "pass  # noqa",
    ),
    # ---- 拒答門檻(2026-09-19)----
    (
        DEMO,
        "逐字覆蓋用全部段落而非所引段落(引錯段落也算高覆蓋)",
        "失效模式 2(語意錯)",
        "join(text_by_rank[c] for c in cites)) if cites else None",
        "join(text_by_rank.values())) if cites else None",
    ),
    (
        DEMO,
        "csv 的列索引被標成字元位移(79% 語料的出處語意錯誤)",
        "失效模式 2(語意錯)",
        '    if hit.source_type == "csv":',
        "    if False:",
    ),
    (
        DEMO,
        "多列 chunk 的列範圍 off-by-one",
        "失效模式 5(邊界錯)",
        'rows = f"{start}" if end - start <= 1 else f"{start}–{end - 1}"',
        'rows = f"{start}" if end - start <= 1 else f"{start}–{end}"',
    ),
    (
        DEMO,
        "引用標記只剝一次(巢狀殘餘拼出新標記)",
        "失效模式 5(邊界錯)",
        "        if stripped == text:",
        "        if True:",
    ),
    (
        DEMO,
        "低覆蓋句標記門檻改 <=(恰等於門檻也標,與 low_coverage 判準不一致)",
        "失效模式 5(邊界錯)",
        "        if cov < min_coverage:",
        "        if cov <= min_coverage:",
    ),
    (
        DEMO,
        "低覆蓋標記加在所有引用句(不分覆蓋率,旗標失去意義)",
        "失效模式 2(語意錯)",
        '        cov = coverage3(_strip_cites(raw), "\\n".join(text_by_rank[c] for c in cites))',
        "        cov = 0.0",
    ),
    (
        DEMO,
        "低覆蓋句判定恰等於門檻也算低(< 改 <=)",
        "失效模式 5(邊界錯)",
        "    low = sum(1 for x in cited if x.coverage is not None and x.coverage < min_coverage)",
        "    low = sum(1 for x in cited if x.coverage is not None and x.coverage <= min_coverage)",
    ),
    (
        ABSTAIN,
        "交叉驗證用全部列搜門檻(held-out 洩漏)",
        "失效模式 2(語意錯)",
        "        train = [r for g in range(k) if g != f for r in folds[g]]",
        "        train = [r for g in range(k) for r in folds[g]]",
    ),
    (
        ABSTAIN,
        "交叉驗證分折不分層(語料外全落同一折)",
        "失效模式 2(語意錯)",
        "            folds[i % k].append(r)",
        '            folds[0 if kind == "offcorpus" else i % k].append(r)',
    ),
    (
        DEMO,
        "should_abstain 恰等於門檻也拒(< 改 <=)",
        "失效模式 5(邊界錯)",
        "    return top < min_score",
        "    return top <= min_score",
    ),
    (
        DEMO,
        "pool 忽略 retrieve_scored(分數永遠 None → 門檻設定即 raise)",
        "失效模式 4(壞資料靜默)",
        '        scored_fn = getattr(retriever, "retrieve_scored", None)',
        "        scored_fn = None",
    ),
    (
        ABSTAIN,
        "無結果(top1 None)不算拒答",
        "失效模式 4(壞資料靜默)",
        "    return sum(1 for r in rows if r.top1 is None or r.top1 < threshold) / len(rows)",
        "    return sum(1 for r in rows if r.top1 is not None and r.top1 < threshold) / len(rows)",
    ),
    (
        ABSTAIN,
        "高原只取最低門檻(區間退化為一點)",
        "失效模式 2(語意錯)",
        "    return min(ok), max(ok), (min(ok) + max(ok)) / 2",
        "    return min(ok), min(ok), min(ok)",
    ),
    (
        DEMO_APP,
        "demo 忽略拒答門檻(永遠打 LLM)",
        "失效模式 2(語意錯)",
        "    if thr is not None and should_abstain(hits, thr):",
        "    if False and thr is not None and should_abstain(hits, thr):",
    ),
    (
        LANGDRIFT,
        "ratio_max 改回傳平均",
        "失效模式 1(公式錯)",
        "    ratio_max = max(ratios.values()) if ratios else 0.0",
        "    ratio_max = (sum(ratios.values()) / len(ratios)) if ratios else 0.0",
    ),
]


def _abort_on_signal(signum: int, frame: FrameType | None) -> None:
    """把 SIGTERM 轉成例外,好讓 finally 的還原區塊有機會執行。

    預設情況下 SIGTERM 直接終止進程,會把變異留在原始碼裡污染後續所有執行。
    """
    raise KeyboardInterrupt(f"received signal {signum}")


def find_residue(originals: dict[Path, str]) -> list[str]:
    """偵測變異點異常——原字串應恰好出現一次。

    出現 0 次代表上次執行未還原(殘留會讓後續所有結果失效)或腳本已過期;
    出現多次代表變異點不夠精確,會一次改動多處。
    不改用「變異字串是否出現」來判斷:變異字串常是原始碼的合法子字串,會誤報。
    """
    findings = []
    for target, label, _mode, old, _new in MUTANTS:
        count = originals[target].count(old)
        if count != 1:
            findings.append(f"{label} → {target}(原字串出現 {count} 次,應為 1)")
    return findings


# pytest 失敗時的「檔案:行號: 例外類型」行(即使 addopts 的 -q 疊成 -qq 也會印,2026-09-24 實測)
_EXC_LINE = re.compile(r"^\S+\.py:\d+: \w+")


def failure_summary(returncode: int, stdout: str, stderr: str) -> str:
    """把一次失敗濃縮成一行可稽核的原因:哪個測試失敗 + 什麼例外。

    存在理由(2026-09-24):桌機 gauntlet 第 4b 層報 140/162、雲端同程式碼 85/162;
    單獨重跑一個「桌機被殺」的變異,桌機本機也存活——擊殺是假的,但原因查不出來,
    因為過去只留結束碼。沒有 FAILED 行(沒選到測試、收集中斷)時退回「結束碼 + 最後訊息」。
    """
    lines = stdout.splitlines()
    failed = next((ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))), "")
    exc = next((ln for ln in reversed(lines) if _EXC_LINE.match(ln)), "")
    if failed:
        return f"{failed}  [{exc}]" if exc else failed
    tail = next((ln for ln in reversed((stdout + "\n" + stderr).splitlines()) if ln.strip()), "")
    return f"rc={returncode} {tail}".strip()


def run_suite(selector: str | None) -> tuple[bool, str]:
    """回傳 (是否通過, 失敗原因)。通過 = 變異存活;selector 為 pytest -k 表達式。"""
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "-x", "-p", "no:randomly"]
    if selector:
        cmd += ["-k", selector]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""
    return False, failure_summary(result.returncode, result.stdout, result.stderr)


def main() -> int:
    selector = sys.argv[1] if len(sys.argv) > 1 else None
    if selector:
        print(f"[僅對子集執行:-k {selector}]")
    signal.signal(signal.SIGTERM, _abort_on_signal)

    # 目標檔一律從 MUTANTS 推導——手寫清單漏掉新檔會在殘留偵測階段 KeyError(2026-09-13 實證)
    targets = sorted({target for target, *_ in MUTANTS})
    originals = {t: t.read_text(encoding="utf-8") for t in targets}
    residue = find_residue(originals)
    if residue:
        print("✗ 變異點異常(殘留未還原,或腳本已過期),所有結果將不可信:")
        for item in residue:
            print(f"  - {item}")
        print("  殘留 → git checkout src/ 還原;程式碼已重構 → 更新本腳本的變異點。")
        return 3

    # 基線檢查:未植入任何變異時這組測試必須是綠的。run_suite 把非零結束一律當「殺死」,
    # 基線若本來就紅,每個變異都會被記成擊殺,報出一個假的高分而報告上看不出異常。
    ok, why = run_suite(selector)
    if not ok:
        print("✗ 基線不是綠的:未植入任何變異時這組測試已失敗,所有擊殺都會是假的。中止。")
        print(f"  {why}")
        return 4

    killed = 0
    survivors: list[str] = []

    try:
        for target, label, mode, old, new in MUTANTS:
            original = originals[target]
            if original.count(old) != 1:
                print(f"✗ 變異點不唯一或不存在,腳本已過期:{label}")
                return 2
            target.write_text(original.replace(old, new), encoding="utf-8")
            survived, why = run_suite(selector)
            target.write_text(original, encoding="utf-8")
            if survived:
                survivors.append(f"{label}({mode})")
                print(f"✗ 存活 {label} — {mode}")
            else:
                killed += 1
                print(f"✓ 殺死 {label} — {mode}")
                print(f"    ← {why}")  # 可稽核:假擊殺會在這一行露出不相干的測試或例外
    finally:
        for target, text in originals.items():
            target.write_text(text, encoding="utf-8")

    total = len(MUTANTS)
    print(f"\nmanual mutation: {killed}/{total} killed")
    if survivors:
        print("存活變異(測試有洞):")
        for s in survivors:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
