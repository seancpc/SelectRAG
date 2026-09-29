# SelectRAG Architecture Decision Memo

**English** | [繁體中文](DECISION_MEMO.md) · *Translated from the Chinese original; if the two differ, the Chinese version is authoritative.*

> **Purpose of this document**: the core deliverable of this project: an evidence-based answer to "which RAG configuration should we use?"
> Format commitments: each axis's conclusion is written as a trade-off, not a winner; each must include a "boring-but-safe" option;
> the Honesty section must list "combinations expected to win that lost".
> **Status**: axes A / B / C and judge phase 1 closed (2026-09-18); axes D / E out of scope (reasons in §5, §6).

---

## 0. Executive Summary

**Constraints**: one RTX 4090 (24 GB), 8,029 chunks of Traditional Chinese public medical documents (NHI reimbursement rules, drug package inserts, fee schedules), answers must cite sources.
The question is not "can RAG be done", but which cell to pick on each of three axes (retrieval strategy, inference engine, numeric format) under this budget, and what each cell costs.
All conclusions come from the same 100-question human-labelled ground-truth eval set (v1.2) and controlled comparisons on the same machine, re-runnable with one command.

**Which to pick** (one recommended and one boring-but-safe option per axis):

| Axis | Recommended | Key numbers | Boring-but-safe |
|---|---|---|---|
| A Retrieval | hybrid (BM25 + bge-m3, RRF) + bge-reranker-v2-m3 | recall@10 0.903 → 0.973, recall@3 0.808 → 0.941; cost: when sharing the single card with generation, retrieval per question 13 ms → 1501 ms (§2.7) | BM25 + tuned document context: zero GPU, 13 ms per question, recall@10 0.903 |
| B Engine | Do FP8 first, then talk engines | vLLM vs TRT-LLM differ by ≤ 10%; FP8 ×1.4–1.7 | vLLM bf16: one pip line, no conversion needed |
| C Format | vLLM + GPTQ-Int4 (official pre-quantized) | throughput ×2.36 (1000 vs 424 tok/s), weights 5.3 GiB, faithfulness indistinguishable from bf16 under the judge | Same as above: vLLM bf16 |

Stacking the three axes: on a single card, "hybrid + reranker retrieval, vLLM running GPTQ-Int4" is this project's answer; compared with the safest configuration, maintaining one extra vector index,
one cross-encoder and one set of quantized weights buys +0.13 top-rank hit rate and ×2.4 generation throughput.

**Costs and limits** (without these, the numbers above would mislead): ① quantization changes outputs: GPTQ's verbatim agreement with bf16 is only 0.25, and the judgement "no quality difference"
rests on the faithfulness judge, whose κ on the stratified calibration set is only 0.488, so it can serve only as a **relative comparison across the four formats**, not an absolute quality guarantee;
② W4 quantization amplifies Simplified-Chinese drift (questions containing Simplified characters: bf16 2 → GPTQ 6); deployments for Traditional Chinese users need post-processing; ③ all retrieval numbers are upper-bound estimates on the same 100 questions,
with no held-out set; ④ TensorRT-LLM cannot load HF-format AWQ / GPTQ (format is tied to engine), and NV-Embed-v2 is incompatible with the pinned transformers; both are listed in the Honesty section;
⑤ axis D (RAG paradigm) and axis E (fine-tune) are deliberately not done; the scope is left to the retrieval and inference layers.

## 1. Problem Setting and Evaluation Method

- **Scenario**: Q&A over public medical documents (NHI reimbursement rules, drug package inserts, fee schedules, announcements),
  sources are mandatory: citation correctness weighs the same as retrieval quality
- **Corpus**: 8,029 chunks (CSV 6,350 / PDF 1,660 / HTML 19), structure-first chunking + length cap
- **Evaluation**: 100 questions with human-judged ground truth (v1-full100, hash bdea697a); changes after freezing must be recorded;
  metrics recall@k / hit@k / MRR, re-runnable with one command (`python3 -m selectrag.runner <config> <run_id>`)
- **Why report both recall and hit**: when the ground-truth set contains equivalent chunks, recall is diluted;
  hit@k answers "was the answer retrieved at all". The two are interlocked by the invariant hit == (recall > 0)
- **Known limitations of the eval set (honest disclosure)**: candidate enumeration was assisted by multi-round rewritten BM25 queries,
  so there is a sampling bias toward lexical overlap; equivalent chunks found by dense retrieval but absent from the ground-truth set will be underestimated;
  later axis A experiments must manually audit dense's "wrong" hits

## 2. Axis A: Retrieval Strategy Selection

### 2.1 Done: BM25 baseline vs document context (2026-09-01)

| Configuration | recall@10 | hit@10 | MRR | Notes |
|---|---|---|---|---|
| BM25 body only (control) | 0.829 | 0.860 | 0.714 | CJK bigram + ASCII words, standard library only |
| BM25 + doc_title + heading_path | **0.905** | **0.920** | 0.732 | index build <1s, query <1ms/question |

**Finding 1: chunk bodies often lack the document's identity**. In drug package inserts the drug name appears only on the document's first line; after chunking, the body
(indications, dosage) contains no drug name at all: the correct chunk for 「四環素對哪些微生物有效」 ("which microorganisms is tetracycline effective against") does not contain the three characters 「四環素」 ("tetracycline").
This is the dominant root cause of baseline misses (3/5 of missed questions); injecting doc_title into the index fixes most of it.

**Field-split ablation (2026-09-29, v1.2, default k1/b, `configs/bm25_{title,heading}_only_full100.yaml`)**:

| Index fields | recall@10 | hit@10 | MRR |
|---|---|---|---|
| Body only | 0.817 | 0.860 | 0.724 |
| + doc_title | 0.856 | 0.880 | 0.726 |
| + heading_path | 0.832 | 0.870 | 0.719 |
| + both | 0.903 | 0.920 | 0.732 |

**Finding 59 (Fact)**: doc_title is the larger single source (+0.039 vs +0.015), consistent with the drug-name diagnosis; using both gives +0.086, more than the sum of the individual gains +0.054, i.e. a positive interaction.
heading_path's contribution to the gain is not negligible: removing it drops the ctx gain from 0.086 to 0.039, more than half. So the defect recorded in SOURCES (body text misclassified as a heading, which enters the index but not the context) may indeed cause part of the gain to be "visible to retrieval, invisible to the model".
Also: body-only is 0.829 on v1 and 0.817 on v1.2 (difference 0.012, mainly from ground truth added in v1 → v1.1); article #1 originally stated "v1.2 difference ≤ 0.003", which holds only for the four representative points; corrected.

**Finding 2: the cost is a slight loss of early precision** (at 50 questions recall@3 0.776→0.773). Document context raises similarity between
chunks of the same document: deep recall improves, head ranking blurs slightly; this is where a reranker is expected to help.

**Finding 3: list of BM25's structural weaknesses** (the expected sources of dense/hybrid gains):
- Chinese–English vocabulary gap: 「雌激素經皮貼片」 ("estrogen transdermal patch") cannot find 「Estradiol Transdermal Patch」
- Insufficient lexical overlap: 「同意書如何處理」 ("how to handle the consent form") does not match 「簽名黏貼於病歷表」 ("signed and pasted into the medical record")
- Crowding by synonymous documents: chunks for other brands with the same ingredient cannot get into the top-10

### 2.2 BM25 k1/b grid tuning (2026-09-01)

Two grid rounds (coarse 6×5 + plateau refinement), 100 questions, target recall@10 with MRR as tie-breaker; take the plateau midpoint, not the boundary:

| Configuration | Parameters | recall@10 | hit@10 | MRR | vs literature default (1.5/0.75) |
|---|---|---|---|---|---|
| Body only (tuned) | k1=0.5, b=0.1 | 0.870 | 0.900 | 0.772 | recall +4.2 pts |
| +document context (tuned) | k1=0.7, b=0.5 | 0.905 | 0.920 | **0.808** | recall flat, MRR +0.08 |

**Finding 4: the literature default is not a good default on this corpus**. b (length normalization) is the main factor: chunk
length in this corpus is already constrained by build parameters (max_chars=800, bounded CSV rows), so further length penalty actually hurts long chunks containing complete
provisions; term-frequency noise from Chinese bigrams also favors low k1 (fast saturation). The ctx version, whose index text is lengthened by document context,
tolerates b better (plateau spans b 0.5~0.75).

**Honest method disclosure**: tuning and evaluation use the same 100 questions (no held-out); the numbers risk overfitting
and should be read as "upper-bound estimates for this corpus + this question set"; for the dense comparison, both sides use their own tuned best values to be fair.
### 2.3 dense (bge-m3) first numbers (2026-09-14, 4090): sampling-bias audit done, citable

bge-m3 (`BAAI/bge-m3` @ 5617a9f6, sentence-transformers 5.7.0, cosine, offline index build of 8029 entries in 40 seconds),
100 questions, same eval set and same context-field rules as BM25 (`index_text` shared):

| Configuration (eval set **v1.1**, hash 859b0d7a, 127 ground truths) | recall@1 | recall@10 | hit@10 | MRR |
|---|---|---|---|---|
| BM25 body-only tuned (k1=0.5/b=0.1) | 0.626 | 0.870 | 0.900 | 0.777 |
| BM25 +ctx tuned (k1=0.7/b=0.5) | 0.666 | 0.905 | 0.920 | **0.808** |
| bge-m3 body only (untuned) | 0.624 | 0.887 | 0.910 | 0.780 |
| bge-m3 +ctx (untuned) | 0.589 | **0.906** | 0.920 | 0.761 |

(The v1 → v1.1 difference is within 0.01: the 4 newly added ground truths were not retrieved in the top-10 by either side and only enlarged the denominator;
on v1, dense +ctx recall@10 was 0.911 and BM25 +ctx tuned 0.905. §2.1 and §2.2 remain v1 numbers.)

**Finding 5 (Fact)**: dense with zero tuning already matches tuned BM25; +ctx has the highest recall@10 in the table but by only 0.006
(1 question out of 100), which does not constitute a conclusion.
**Finding 6 (Fact)**: the context fields affect dense opposite to BM25: recall@10 goes up, recall@1 and MRR go down.
Judgement (to be verified): heading text mixed into the embedding dilutes body semantics and hurts top-1; BM25 is term-weighted and does not get diluted.
**Honest method disclosure**: eval set ground truth was built by exhaustive multi-round rewritten BM25 queries (§10 2026-08-31); chunks hit by dense but missed by BM25
may be unlabelled ground truth. **Audit result (2026-09-14, `tools/audit_dense_hits.py` + human judgement)**:
the 97 top-5 candidates of the 16 questions dense missed entirely, plus 69 dense-only top-2 candidates on questions it hit, **all 166 judged "no"**:
the BM25-built ground truth has no sampling bias detectable by dense, and every dense-only result was wrong; the eval set stays at v1, no re-run.
**Reverse audit (2026-09-14, first batch of 25 questions)**: for each question, list all chunks of the document holding the ground truth (for large documents, ground truth ±8),
474 rows, 439 judged by hand → **4 unlabelled ground truths + 1 cross-question unlabelled** (all cases of "the same fact appears in multiple places in the corpus"; 3 of them had been
wrongly judged "no" in earlier batches because excerpts were truncated at 160 characters, and were overturned once the full text was shown). The forward audit (top-5) indeed could not see them: blind spot confirmed.
Handling: added to the eval set and frozen as v1.1, all configurations re-run (see §10 2026-09-14); reverse audit of the remaining 75 questions is on the To do list.
**Side finding: the heading path carries the answer**: in two ground truths the body is fragmented or answerless, and the heading_path is the clean answer.
This means part of the ctx vs plain gap is not an embedding or term gap but "whether the heading path is counted in the index";
later ablations must separate this factor when interpreting the ctx gain.

**Finding 7: dense's failure modes on this corpus are highly concentrated (a by-product of the audit, more useful than the main conclusion)**:
1. **Drug-name blind, sentence-pattern hit** (the largest group): package-insert sections such as half-life, storage conditions and side-effect tables are highly isomorphic; dense learns what a section looks like
   but does not treat the drug name as a hard constraint (Cephalexin half-life → top four are half-life sections of four other drugs). Remedy direction: hybrid + rerank
2. **Off by one adjacent chunk** (7 entries): the location is right, but the chunk boundary falls one chunk before or after the answer. This is a chunking problem, not an embedding problem;
   overlap / parent-child retrieval should absorb it directly; the cheapest improvement
3. **Same semantic field, wrong document**: recognizes 「語言治療項目」 ("speech therapy items") but cannot tell whether it is fee schedule 44xxx or rehabilitation ward PACST
4. **Whole-question collapse** (3 questions): neither top-5 set is anywhere close; 1 of them collapses only in the ctx group, suspected to be the doc_title prefix pulling in large-document semantics
There are also homograph false hits (「疱」) and 「早熟 vs 早產兒」 ("precocious vs premature infant").
Distractor additions: 3 nearby-code entries + 7 adjacent-chunk entries added to `data/eval/hard_negatives.json` (41 in total),
the former for reranker evaluation, the latter for chunking ablation. These four failure modes are the entry points for later axis A ablations.

### 2.4 chunking ablation: retrieval-time neighbor expansion (2026-09-14): conclusion: with the k budget locked, both sides regress

Targets Finding 7 type 2 (off by one adjacent chunk, 7 entries). Ground truth is a hash of chunk content, so re-chunking would invalidate all ground truth;
hence the remedy sits at retrieval time: `NeighborExpand` inserts same-document ±window neighbor chunks after a hit, deduplicates, then **truncates back to top_k**
(base also sees only top_k, so the k budget equals the control). window=0 is the control.

| Configuration (v1.1) | window | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| BM25 +ctx tuned | 0 | **0.808** | **0.854** | **0.905** | **0.808** |
| | 1 | 0.711 | 0.796 | 0.884 | 0.791 |
| | 2 | 0.711 | 0.736 | 0.839 | 0.782 |
| bge-m3 +ctx | 0 | **0.810** | **0.854** | **0.906** | **0.761** |
| | 1 | 0.684 | 0.765 | 0.868 | 0.734 |
| | 2 | 0.679 | 0.739 | 0.830 | 0.727 |

**Finding 8 (Fact)**: under a fixed k budget, neighbor expansion makes both BM25 and dense regress across the board, worse as window grows;
recall@1 is unchanged (the first rank is always the original hit). **Judgement**: expansion pushes ground truths originally at ranks 2~5 off the list,
a loss far larger than the gain from the 7 adjacent misses (for dense those 7 could recover at most hit@10 +0.07; measured hit@10 actually fell 0.02).
The "k budget" mechanism predicted by the pre-mortem holds, just in the opposite direction from expected.
**This remedy is rejected**; if retried, only two variants remain: ① expand only the top m (m=1~2), ② place neighbors at the tail instead of cutting in;
both sacrifice fewer original hits under the same k budget, but the gain ceiling is still those 7 entries, so they rank below hybrid + rerank.
Method disclosure: `tools/gauntlet.sh` 5d re-runs six BM25 configurations each time (including w1/w2), reproducible in the cloud container; the two dense configurations run on the desktop.

### 2.5 hybrid (RRF rank fusion): first numbers (2026-09-14, 4090, v1.1)

base = BM25 +ctx tuned and bge-m3 +ctx (best of each family); `HybridRRF`: take pool entries from each side → RRF fusion → truncate back to top_k.
**First compute the union ceiling** (pre-mortem requirement: the fusion gain must exceed "just enlarging one side's pool"):

| Depth N | BM25 recall@N | bge-m3 recall@N | Union of both top-N recall | Union hit |
|---|---|---|---|---|
| 10 | 0.905 | 0.906 | **0.975** | 0.98 |
| 20 | 0.938 | 0.958 | 0.975 | 0.98 |
| 40 | 0.968 | 0.980 | 0.985 | 0.99 |

**Finding 9 (Fact)**: the union recall of both top-10s is 0.975, 0.07 higher than either side alone: the questions the two retrievers miss barely overlap,
which is the entire theoretical room for hybrid; increasing depth to 40 adds only another 0.01.

Grid pool ∈ {10, 20, 40} × rrf_k ∈ {10, 60, 100}, output fixed at top_k=10:

| pool | rrf_k | recall@3 | recall@10 | hit@10 | MRR |
|---|---|---|---|---|---|
| 10 | 10 / 60 / 100 | 0.855 | **0.965** | **0.97** | 0.836 |
| 20 | 60 | 0.865 | 0.951 | 0.96 | 0.838 |
| 40 | 60 | 0.875 | 0.951 | 0.97 | **0.840** |

**Finding 10 (Fact)**: the three rrf_k cells give nearly identical numbers (plateau); pool is the main factor, and pool=10 has the highest recall@10;
enlarging pool lowers it by 0.014, while recall@3 / MRR rise slightly with pool. **Judgement**: with a larger pool, chunks that are "mid-ranked on both sides"
accumulate two votes and overtake ground truths that are "1st on one side, not in the other side's pool"; a small pool effectively requires the other side to also rank it near the front before granting a second vote.
**Pre-mortem check passed**: hybrid reaches 0.965 under a k=10 budget, higher than single-side recall@20 looking at 20 entries (0.958); the gain is not from "looking at more".
Fine pool grid (8 / 10 / 12 / 15, rrf_k=60): recall@10 0.963 / 0.965 / 0.963 / 0.963, MRR 0.837 / 0.836 / 0.837 / 0.839,
recall@3 0.842 / 0.855 / 0.855 / 0.865; 8~15 is the plateau. **The representative point is the plateau midpoint pool=12, rrf_k=60**
(`configs/hybrid_rrf_full100.yaml`): recall@3 0.855 / recall@10 0.963 / hit@10 0.97 / MRR 0.837.
**Finding 11 (Fact)**: versus the best single side, hybrid gives recall@10 +0.057, recall@3 +0.045, MRR +0.029 (vs BM25):
the largest single gain on axis A so far; what it absorbs is exactly Finding 7's "drug-name blind" and "same semantic field": BM25's term constraint supplies the entity sensitivity dense lacks.
Method disclosure: grid and representative point both use the same 100 questions, no held-out, so numbers are upper-bound estimates; the wide pool plateau (8~15) reduces overfitting concerns.

### 2.6 reranker (bge-reranker-v2-m3 cross-encoder): first numbers (2026-09-14, 4090, v1.1)

base = hybrid representative point (pool=12, rrf_k=60); `RerankRetriever`: base fetches candidates items → cross-encoder scores them → truncated back to top_k=10.
Model `BAAI/bge-reranker-v2-m3 @ 953dc6f6`, zero tuning; two rerank text variants: plain (body text) and ctx (body text + doc_title, same rule as the index side).
Grid candidates ∈ {10, 20, 30} × {plain, ctx}; in addition, 42 hard negatives (adjacent chunks / same semantic field, wrong document) are used for a "top ground-truth score > distractor score" check:

| candidates | Text | recall@3 | recall@10 | hit@10 | MRR | ms/query* |
|---|---|---|---|---|---|---|
| 10 | plain / ctx | 0.941 / 0.943 | 0.963 / 0.963 | 0.97 | 0.903 / 0.920 | 143 / 146 |
| 20 | plain / ctx | 0.936 / 0.943 | 0.971 / **0.975** | 0.98 | 0.898 / **0.921** | 256 / 262 |
| 30 | plain / ctx | 0.936 / 0.943 | 0.961 / 0.965 | 0.97 | 0.897 / 0.920 | 263 / 269 |

\* Full-chain time per query (BM25 + dense + rerank, including base), not pure reranker overhead; broken out separately in the §5.4 performance measurement.
hard negatives: ground truth wins 38/42 = 0.90 for plain, 39/42 = 0.93 for ctx.

**Finding 12 (Fact)**: For the hybrid representative point, the reranker lifts recall@3 0.855 → 0.943 (+0.088), MRR 0.837 → 0.921 (+0.084),
recall@10 0.963 → 0.975 (+0.012) — the largest **head** gain on axis A so far; hybrid already pulls the ground truth into the top 20, and the reranker pushes it into the top 3.
**Finding 13 (Fact)**: ctx makes almost no difference to recall (≤ 0.008) and 0.02 difference to MRR — at the rerank stage doc_title helps with "ranking it first", not with "retrieving it at all";
this is a different mechanism from adding context on the index side in §2.1 (recall +0.076); interpret them separately.
**Finding 14 (Judgement)**: candidates is the only major factor and is hill-shaped — at 10 there are too few candidates, at 30 the extra distractors pulled in outrank the ground truth;
the ctx row's recall@3 is flat at 0.943 across all three cells, meaning head ranking is already saturated within 10 items. **Representative point: candidates=20 + ctx**
(`configs/rerank_bge_full100.yaml`; the official record `results/rerank-bge-full100/v11.json` is bit-identical to grid cand20_ctx):
recall@1 0.789 / recall@3 0.943 / recall@10 0.975 / hit@10 0.98 / MRR 0.921.
Trade-off (decided): relative to candidates=10, ~116 ms more per query buys recall@10 +0.012; the primary metric of axis A is recall, latency is left to the §5.4 measurement.
Method disclosure: same 100 questions, no held-out, upper-bound estimate; the distractors used for the hard-negative check come from the spot-check process, not an independent set.
**NIM comparison (2026-09-15, passed the second timebox; process in §7 item 4)**: NeMo Retriever Text Reranking NIM
`nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:1.8.0` (digest `9b7e2163…`); on the 4090 the profile selector picked the L40S
TRT FP8 engine (`l40sx1-trt-fp8`, sm_89 compatible), called over HTTP via `/v1/ranking`, same grid:

| candidates | Text | recall@3 bge / NIM | recall@10 bge / NIM | MRR bge / NIM | ms/query* bge / NIM |
|---|---|---|---|---|---|
| 10 | plain / ctx | 0.941 / 0.928 ・ 0.943 / 0.926 | 0.963 / 0.963 ・ 0.963 / 0.963 | 0.903 / 0.903 ・ 0.920 / **0.931** | 143 / 122 ・ 146 / 117 |
| 20 | plain / ctx | 0.936 / 0.928 ・ **0.943** / 0.926 | 0.971 / 0.966 ・ 0.975 / 0.975 | 0.898 / 0.902 ・ 0.921 / 0.917 | 256 / 207 ・ 262 / 215 |
| 30 | plain / ctx | 0.936 / 0.928 ・ 0.943 / 0.926 | 0.961 / 0.956 ・ 0.965 / 0.975 | 0.897 / 0.901 ・ 0.920 / 0.917 | 263 / 216 ・ 269 / 219 |

hard negatives are identical item-by-item for both models (plain 38/42, ctx 39/42).
**Finding 15 (Fact)**: recall@10 is a tie (both 0.975 at cand20 ctx); bge is consistently 0.017 higher on recall@3 (same across all three candidates values, about 2 questions);
NIM is about 18% faster per query (TRT FP8 engine), but this is full-chain time including base, so the pure reranker gap is larger.
**Judgement**: the axis A representative point stays bge-reranker-v2-m3 (head recall is the primary metric); NIM is a viable alternative, suited to "latency-first" deployments.
**hard negatives check by type (2026-09-15, v1.2, 89 items; cand=20)** — number of items where top ground-truth score > distractor score:

| Type (count) | bge plain / ctx | NIM plain / ctx |
|---|---|---|
| near-code-same-remark, adjacent code with the same remark (41) | 40 / 40 | 39 / 40 |
| adjacent-chunk, adjacent chunks (8) | 7 / 7 | 7 / 7 |
| same-condition-other-drug (1) | 0 / 0 | 0 / 0 |
| same-context-other-number (1) | 0 / 0 | 0 / 1 |
| right-number-wrong-scenario (1) | 0 / 1 | 0 / 0 |
| Remaining 22 types (37) | 37 / 37 | 37 / 37 |
| Total | 84 / **85** (0.94 / 0.96) | 83 / **85** (0.93 / 0.96) |

**Finding 16 (Fact)**: The "adjacent code, same remark" family (Jaccard 0.91–0.94, the highest surface similarity in the whole table) is separated by both rerankers;
the three variables code + resource-use group + reasonable quantity are not hard for a cross-encoder; both models lose on the same few items — 1 adjacent chunk and 3 semantic ones
(same condition other drug, same context other number, right number wrong scenario). This is **model-independent**, the residual limit of rerankers on this corpus, and is logged as an observation in §7.
Same grid v1.2 cand20: bge plain / ctx recall@3 0.934 / 0.941, recall@10 0.969 / 0.973; NIM 0.927 / 0.925, 0.964 / 0.973.
Method disclosure: NIM goes over HTTP, bge runs as an in-process CrossEncoder, so ms/query includes different call overheads and the latency comparison is rough;
the two profiles differ in precision (NIM FP8, bge FP16/FP32), so the head difference may partly come from quantization; no same-precision comparison was done.

### 2.7 Axis A conclusion (2026-09-15)

**Question**: On a single 4090 with a corpus of 8,029 chunks of Chinese public medical documents, which retrieval configuration should be used?

| Configuration (eval set **v1.2**, hash 4c0761ca, 100 questions / 129 ground truths, top_k=10) | recall@3 | recall@10 | MRR | GPU | Extra components | Latency per query* |
|---|---|---|---|---|---|---|
| BM25 body-only, tuned | — | 0.870† | 0.777† | None | None | <1 ms |
| **BM25 +ctx, tuned (boring-but-safe)** | 0.808 | 0.903 | 0.808 | None | None | <1 ms |
| bge-m3 +ctx (zero tuning) | 0.808 | 0.904 | 0.761 | Index time | Vector index (.npz, 8029×1024) | ~30 ms |
| hybrid RRF (pool=12) | 0.853 | 0.961 | 0.837 | Index time | Both of the above + fusion | ~30 ms |
| **hybrid + bge-reranker-v2-m3 (cand=20, ctx) recommended** | **0.941** | **0.973** | **0.921** | Query time | The above + cross-encoder | ~260 ms |
| hybrid + NIM reranker (cand=20, ctx) | 0.926† | 0.975† | 0.917† | Query time | The above + Docker/NGC | ~215 ms |

\* Latency is full-chain time averaged over 100 questions (desktop 4090, including base retrieval), a **rough estimate**; the "demo measurement" below is an independent measurement under different conditions, and the two are not interchangeable.

**Retrieval latency per query (demo measurement, 2026-09-19)**: `tools/demo_app.py --cite-rate 0` timed each of the same 100 questions,
measuring the whole of `RetrieverPool.hits()` (BM25/dense retrieval → RRF → cross-encoder → assemble Hit):

| Configuration | Mean per query | Multiple of BM25 |
|---|---|---|
| BM25 +ctx, tuned | **13.4 ms** | 1× |
| hybrid + bge-reranker-v2-m3 | **1501 ms** | **×112** |

**Difference from the 260 ms in the table above (must read)**: the two were measured under different conditions; neither corrects the other. The demo measurement is **single-query, unbatched,
and shares the same 4090 with vLLM** (`--gpu-memory-utilization 0.75`, with the server running concurrently);
the table above is a rough estimate from the §2.6 rerank grid path. **My judgement (no controlled experiment done)**: the gap mainly comes from the fixed cost of two GPU forward passes (bge-m3 query embedding and cross-encoder)
without batching on single queries, plus GPU contention with vLLM; separating the two needs a separate "exclusive GPU × batch size" comparison, listed as follow-up.

**What this number is for**: it is the real perceived latency of **single-card co-located deployment** (retrieval and generation sharing one card),
i.e. the actual conditions of the demo. The cost of the recommended configuration's recall@10 +0.07 and recall@3 +0.13 is, under these conditions, about 1.5 seconds more per query.
If retrieval and generation are on separate cards, or queries can be batched, this number would drop substantially (not measured).
† v1.1 numbers (not rerun on v1.2); the v1.1 → v1.2 differences for the four representative points are all ≤ 0.003 and in the same direction (the 2 newly added ground truths never entered top-10, they only enlarge the denominator).

**Decision**: recommend **hybrid (BM25 +ctx ⊕ bge-m3 +ctx, RRF pool=12) + bge-reranker-v2-m3 reranking the top 20, with doc_title in the text**.
The reason is that the two gains come from different sources and stack: hybrid fixes "was it retrieved at all" (recall@10 +0.057; the two retrievers' misses barely overlap),
the reranker fixes "is it ranked near the top" (recall@3 +0.088, MRR +0.084); combined, recall@10 0.903 → 0.973 and recall@3 0.808 → 0.941 (v1.2).

**Boring-but-safe: BM25 +ctx, tuned**. Zero GPU, pure standard library, 1 second to build the index, 1 ms per query, recall@10 0.905 already within 0.07 of the recommended configuration;
if the deployment has no GPU or the requirement is only "the answer is somewhere in the top 10", this is the right answer, and dense alone is not stronger than it (0.906 vs 0.905).

**Should we add a vector database?** There is no reason to use dense alone (+0.001 over BM25); the only reason is hybrid — the vector index is there to complement BM25, not replace it.
At 8029 items, loading the whole `.npz` into memory is enough; a standalone vector database is not a necessary component at this scale (revisit at >10⁶).

**Is the reranker worth it?** Yes, provided head metrics matter (when generation is fed only the top 3~5 items, recall@3 is the hit rate);
the cost is query-time GPU and latency — the §2.6 path roughly estimates +230 ms per query, but **when sharing a single card with generation the demo measured 13.4 ms → 1501 ms (×112)**;
before going live you must re-measure under your own deployment conditions and cannot quote either number directly. NIM is a same-tier alternative, about 18% faster with recall@3 0.017 lower, suited to latency-first environments that already have NGC infrastructure.

**Rejected alternatives**: neighbour-chunk expansion (§2.4: both sides regress under the k budget); score-weighted fusion (needs normalization and α, confounded attribution);
indexing heading_path on its own (§7 item 1); dense alone (gain 0.001).

**Scope / limits of applicability (honest disclosure)**: ① all numbers are upper-bound estimates on the same 100 questions, no held-out; ② the corpus is Chinese government public documents,
where English drug names mixed into Chinese text are the main source of failure modes; the relative strength of BM25/dense may differ in other domains; ③ there are two latency figures under different conditions: the 260 ms in the table is a rough estimate over a different call path, and the demo measurement of 1501 ms is single-query unbatched and shares the card with vLLM;
neither is the value under "exclusive GPU + batching";
④ the two reranker models differ in precision (FP8 vs FP16), no same-precision comparison was done; ⑤ the ground-truth set is AI-generated questions + human judgement, and reverse spot-checking has only completed batch 1.


## 3. Axis B: inference engine (vLLM vs TensorRT-LLM)

### 3.1 bf16 first round (2026-09-15, 4090, online serving mode)

Setup: `Qwen/Qwen2.5-7B-Instruct @ a09a3545` bf16; vLLM 0.22.1 (`--max-model-len 4096 --gpu-memory-utilization 0.90 --generation-config vllm`)
vs TensorRT-LLM 1.2.1 (`trtllm-serve serve --max_seq_len 4096`, PyTorch backend, no engine build needed);
prompt set `results/prompts_v12.jsonl` (100 questions assembled into prompts via hybrid + bge reranker top-3, same file for both engines);
client `tools/bench_serve.py`: streaming, `temperature 0 / seed 42 / max_tokens 256`, 10 warm-up questions not counted, concurrency 1 / 4 / 8 with 100 questions each.

| Engine | c | TTFT p50 / p95 | Latency p50 / p95 | tok/s | req/s | Peak VRAM |
|---|---|---|---|---|---|---|
| vLLM | 1 | 0.087 / 0.171 s | 0.73 / 1.28 s | 57 | 1.28 | 23.5 GB* |
| vLLM | 4 | 0.054 / 0.058 | 0.75 / 1.36 | 214 | 4.87 | 23.5 GB* |
| vLLM | 8 | 0.054 / 0.061 | 0.74 / 1.35 | **424** | 9.65 | 23.4 GB* |
| TRT-LLM | 1 | 0.090 / 0.171 | 0.78 / 1.35 | 53 | 1.21 | 22.0 GB |
| TRT-LLM | 4 | 0.056 / 0.072 | 0.77 / 1.39 | 210 | 4.72 | 21.9 GB |
| TRT-LLM | 8 | 0.059 / 0.064 | 0.81 / 1.45 | 385 | 8.79 | 21.9 GB |

\* vLLM's peak VRAM is the reservation from `gpu_memory_utilization 0.90`, not actual demand; both engines show "weights + reserved KV", not comparable demand figures.
Output averages 45 tokens (the system prompt limits answers to three sentences, far below max_tokens), so throughput numbers are for a **short-answer RAG scenario** and must not be compared with long-generation benchmarks.

**Finding 17 (Fact)**: Same model, same card, same prompt set: the two engines in bf16 differ by ≤ 10% on TTFT, latency and throughput at all three concurrency levels
(c=8: vLLM 424 vs TRT-LLM 385 tok/s; c=1: 57 vs 53), with vLLM slightly ahead; latency p50 barely changes from c=1 to c=8,
the 4090 is far from saturated under this load (vLLM reports 27 concurrent sequences fit at 4096 context).
**Finding 18 (Fact)**: consistency check: 87/100 questions are verbatim identical; 13 diverge, with the first divergent character at p50 position 36 and output length difference p95 of 8 characters —
divergence occurs late in the answer and is token flipping from numerical differences under greedy decoding, not a systematic behavioural difference.
**Judgement**: in bf16, "TRT-LLM is faster than vLLM" does not hold on a 4090 with short-answer load; TRT-LLM's value depends on FP8 (§3.2) and axis C.

**Process honest disclosure**: ① in the first round, vLLM c=1 stalled for about 5 minutes on each of three questions (log: `Triton kernel JIT compilation during inference … latency spike`);
5 warm-up questions did not cover the JIT shapes, so the round was discarded (evidence kept in `results/bench-modelgencfg/vllm_bf16_c1_stalled.json`) and warm-up was raised to 10 questions;
② vLLM by default applies the model's `generation_config.json` (`repetition_penalty 1.05` etc.), a different condition from TRT-LLM;
results were only recorded after rerunning with `--generation-config vllm` (old results kept in `results/bench-modelgencfg/`);
③ whether TRT-LLM supports `stream_options.include_usage`: it does (token counts come from usage, same as vLLM).

### 3.2 FP8 rows (2026-09-15; ModelOpt 0.37 `FP8_DEFAULT_CFG`, calibration = 512 corpus items, the same HF checkpoint fed to both engines)

| Engine | c | TTFT p50 / p95 | Latency p50 / p95 | tok/s | req/s | vs same-engine bf16 |
|---|---|---|---|---|---|---|
| vLLM | 1 | 0.064 / 0.111 s | 0.52 / 1.00 s | 80 | 1.78 | ×1.40 |
| vLLM | 4 | 0.036 / 0.040 | 0.50 / 0.94 | 327 | 7.20 | ×1.53 |
| vLLM | 8 | 0.038 / 0.047 | 0.53 / 1.00 | 610 | 13.54 | ×1.44 |
| TRT-LLM | 1 | 0.051 / 0.097 | 0.48 / 0.92 | 88 | 1.91 | ×1.66 |
| TRT-LLM | 4 | 0.034 / 0.054 | 0.47 / 0.98 | 346 | 7.58 | ×1.65 |
| TRT-LLM | 8 | 0.037 / 0.052 | 0.51 / 0.98 | **645** | 14.11 | ×1.68 |

Peak VRAM for both engines is about 22.7–23.0 GB (reserved value, same note as §3.1).

**Finding 19 (Fact)**: FP8 on Ada (sm_89) **works on both engines** — the project-kickoff open item "vLLM FP8 on Ada to be verified" is answered: it works;
the same ModelOpt checkpoint does not need separate conversion per engine. FP8 speed-up over bf16: vLLM ×1.4–1.5, TRT-LLM ×1.65–1.7;
under FP8 TRT-LLM overtakes vLLM by about 6% (c=8: 645 vs 610 tok/s), and latency p50 drops from 0.74 s to 0.5 s.
**Finding 20 (Fact, warning for axis C)**: for the same engine, bf16 → FP8 greedy output exact-match rate is only 45% (TRT-LLM) / 51% (vLLM),
first divergent character p50 at positions 25–28, output length difference p95 reaching 25–33 characters — quantization changed the content of nearly half the answers,
while switching engines (§3.1, 87%) changed only a tenth. **Axis C cannot report FP8 quality without the §5.3 faithfulness judge**; speed numbers alone are misleading.
Addendum: under FP8, cross-engine (vLLM FP8 vs TRT-LLM FP8) match rate is 47/100, length difference p95 32 — in bf16 the two engines agreed 87%,
under FP8 it drops to the same order as quantization itself; the FP8 kernel numerical paths differ between the engines, and "the same checkpoint" does not mean "the same outputs".

### 3.3 Axis B conclusion (2026-09-15)

| Configuration | c=8 tok/s | c=1 latency p50 | Deployment complexity | Notes |
|---|---|---|---|---|
| **vLLM bf16 (boring-but-safe)** | 424 | 0.73 s | One pip line, main venv, no conversion | Output is the original model behaviour |
| TRT-LLM bf16 | 385 | 0.78 s | Separate venv (Py3.12, CUDA 13 .so paths, OpenMPI) | No speed advantage |
| vLLM FP8 | 610 | 0.52 s | Same as above + one ModelOpt conversion (TRT venv) | Quality pending §5.3 |
| **TRT-LLM FP8 (fastest)** | 645 | 0.48 s | Heaviest | +6% over vLLM FP8 |

**Quality addendum (2026-09-17, §6.5 judge v2)**: vLLM bf16 faithfulness 0.959 vs vLLM FP8 0.951, a difference of 1 claim, within the judge's inconsistency rate — FP8's speed gain has no measurable quality cost (short answers).

**Decision**: under short-answer RAG load on a 4090, the effect of engine choice (≤ 10%) is far smaller than the effect of numeric format (×1.4–1.7).
If you can do only one thing, do FP8 rather than switching engines; if you are already on vLLM, the 6% from switching to TRT-LLM is not worth maintaining another venv.
A legitimate case for TRT-LLM could only appear on axis C (more formats, KV cache quantization) or under longer generation loads.
**Scope / limits of applicability**: single consumer card, 4096 context, average output 45 tokens, concurrency ≤ 8 (not saturated); under long outputs and high concurrency the scheduling differences between engines may widen, not tested in this round.
To do: a quality conclusion on FP8 only after the axis C §5.3 faithfulness judge.

### 3.4 Note: long-output load (2026-09-16; system prompt changed to detailed paragraph-by-paragraph answers, max_tokens 1024, average output about 310 tokens)

| Engine / format | c=1 latency p50 / p95 | c=8 TTFT p50 / p95 | c=1 tok/s | c=4 tok/s | c=8 tok/s |
|---|---|---|---|---|---|
| vLLM bf16 | 5.02 / 7.89 s | 0.107 / 0.291 s | 59 | 219 | 406 |
| TRT-LLM bf16 | 5.26 / 8.47 | 0.131 / **0.719** | 55 | 201 | 365 |
| vLLM FP8 | 3.26 / 5.48 | 0.040 / 0.055 | 83 | 325 | 614 |
| TRT-LLM FP8 | 3.19 / 5.47 | 0.038 / 0.053 | 89 | 347 | 635 |

**Finding 21 (Fact)**: with outputs 7 times longer, c=8 throughput for all four groups is almost the same as for short answers (406 / 365 / 614 / 635 vs 424 / 385 / 610 / 645);
bf16 engines differ by 10%, FP8 by 3%, and the FP8 speed-up of ×1.5–1.7 is unchanged — the §3.3 conclusion holds under long outputs.
The only new signal: TRT-LLM bf16 at c=8 has TTFT p95 of 0.72 s (vLLM 0.29 s), the only group of the four showing queueing,
but under FP8 both return to 0.05 s; concurrency 8 still does not make the engines' scheduler differences visible in throughput.

## 4. Axis C: quantization formats (AWQ / GPTQ / FP8)

### 4.0 Pre-flight gates (2026-09-17)

- Format sources: AWQ / GPTQ use Qwen's official pre-quantized checkpoints (calibration data is whatever Qwen used, which differs from this project's FP8 calibration on 512 corpus samples; every table states which); FP8 uses the ModelOpt checkpoint from §3.
- **Gate C-1 (Fact)**: `trtllm-serve serve Qwen/Qwen2.5-7B-Instruct-AWQ` fails to load: the PyTorch backend's `load_weights_fused_gate_up_helper`
  asserts that every layer has a `weight`, while an AutoAWQ checkpoint only has `qweight / qzeros / scales`. TRT-LLM × AWQ (HF format) = N/A.
  This is the first empirical evidence for the founding hypothesis that "format is bound to engine": the same AWQ weight file runs on vLLM (§day-one check 2) but not on TRT-LLM.
  The proper AWQ path for TRT-LLM is quantizing and exporting ourselves with ModelOpt (not done; listed as an axis C to-do).
- **Gates C-2 / C-3 (Fact)**: vLLM 0.22.1 loads both AWQ (`awq_marlin`, MarlinLinearKernel) and GPTQ-Int4 (`auto_gptq` → MarlinLinearKernel) successfully;
  both run with dtype float16 (not bf16), weights 5.29 / 5.27 GiB.
- **Precondition check (Fact)**: all four formats share the same tokenizer: bf16 / FP8 / AWQ / GPTQ produce identical per-question token counts for the 100 prompts
  (total 90,404, vocab 151,665, per-question count hash 5ef906dc identical for all four), so there is no confounder on the prompt side.

### 4.1 Four formats within vLLM (2026-09-17, 4090; short answers use the §3.1 settings, long outputs the §3.4 settings; bf16 / FP8 rows quoted from §3)

| Format | Source / kernel | Weights GiB | KV cache tokens | c=1 latency p50 / p95 | c=1 tok/s | c=4 tok/s | c=8 tok/s | Long output c=8 tok/s | vs bf16 (c=8) |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | Official @ a09a3545 | 14.29 | 113,488 | 0.73 / 1.28 s | 57 | 214 | 424 | 406 | ×1.00 |
| FP8 | ModelOpt, self-calibrated on 512 samples | 8.17 | 223,632 | 0.52 / 1.00 | 80 | 327 | 610 | 614 | ×1.44 |
| AWQ (W4A16) | Official @ b250375, awq_marlin | 5.29 | 273,808 | 0.32 / 0.56 | 122 | 560 | 1000 | 915 | **×2.36** |
| GPTQ-Int4 | Official @ e9c932ac, gptq→marlin | 5.27 | 274,224 | 0.34 / 0.58 | 126 | 566 | 1000 | 920 | **×2.36** |

Weights GiB and KV cache tokens are copied from the vLLM startup log (`Model loading took` / `GPU KV cache size`, `gpu-memory-utilization 0.90`);
all four formats were re-launched with identical settings on 2026-09-17 and transcribed (the previously reported bf16 KV value of 111,216 was wrong; the log value 113,488 is authoritative). AWQ / GPTQ group_size follows the official checkpoint setting (128), and both go through the Marlin kernel:
their near-identical speed is no coincidence; it is the same W4A16 GEMM path, and the only difference is the quantization algorithm (calibration method), not the execution path.

| Quality / consistency (vs vLLM bf16) | bf16 | FP8 | AWQ | GPTQ |
|---|---|---|---|---|
| faithfulness (judge v2, short answers) | 0.959 (121 claims) | 0.951 (122 claims) | **0.938** (129 claims) | 0.960 (125 claims) |
| contradicted rate | 0.041 | 0.049 | 0.062 | 0.040 |
| Refusal questions | 3 | 3 | 1 | 3 |
| Verbatim agreement vs bf16 | — | 0.51 | 0.27 | 0.25 |
| First divergent character p50 / p95 | — | 28 / — | 19 / 57 | 21 / 70 |
| Output length difference p95 | — | 25–33 | 34 | 33 |
| Simplified-character share: questions containing Simplified / max per-question share | 2 / 0.032 | 1 / 0.167 | **9 / 0.227** | 6 / 0.133 |

**Finding 24 (Fact)**: both W4 formats reach ×2.36 bf16's c=8 throughput (1000 vs 424 tok/s), and c=1 latency p50 drops from 0.73 s to 0.32 s,
a further 1.6× over FP8 (×1.44); long-output c=8 is also ×2.25 (915–920 vs 406). KV cache capacity is 2.4× (274k vs 113k tokens; FP8 sits in between at 224k with 8.17 GiB weights),
so the concurrency a single card can absorb goes from 27 streams to 67. All speed numbers between AWQ and GPTQ differ by ≤ 5%, within measurement noise.
**Finding 25 (Fact)**: the deeper the quantization, the further the output drifts from bf16: verbatim agreement FP8 0.51 → W4 0.25–0.27, and the first divergent character moves from character 28 up to characters 19–21.
But the faithfulness the judge sees does not track agreement: GPTQ 0.960 and bf16 0.959 are indistinguishable under the judge, while AWQ 0.938 is 2.1 percentage points lower (8 contradicted claims vs 5).
**Judgement (needs further verification)**: AWQ's 3-claim gap sits at the edge of judge v2's disagreement rate (κ 0.845, calibration set of 48), and AWQ refused only 1 question (the other three refused 3):
answering 2 more questions exposes 8 more claims to judging, so part of the lower score is "answering more" rather than "answering more wrongly"; telling them apart requires a human to inspect AWQ's 8 contradicted claims.
This round draws no "AWQ has worse quality" conclusion; it only records "AWQ is the only one of the four formats below 0.95".

**Human review results (2026-09-17, `tools/contradicted_sheet.py` lays out AWQ's 8 contradicted claims against bf16 on the same questions)**:

| | Judge says contradicted | Actually wrong | Retrieval accomplice | Judge misjudgement |
|---|---|---|---|---|
| bf16 | 5 | 0 | 1 | 4 |
| AWQ | 8 | 2 | 1 | 5 |

All 5 bf16 claims fall on the same 4 questions as AWQ's, and all are judge misjudgements or retrieval accomplices. AWQ's 2 genuine errors: one answered a note's 「另計」 ("billed separately") as 「包含」 ("included") (the opposite direction);
one was a wrong abstention: the ingredient-content line in mixed Chinese/English on the corpus's first line states it explicitly, yet AWQ answered 「未提及」 ("not mentioned") while bf16 answered correctly (mixed Chinese/English lines are a suspected point of post-quantization degradation, to be tracked separately).
1 retrieval accomplice: two package inserts ended up in the same context, and both models made exactly the same cross-document splice; not attributable to quantization.
**Judgement update**: AWQ's real gap to bf16 is 2 claims (about 1.6 percentage points), still within judge noise; §4.2's "AWQ quality pending human review" becomes "gap of 2 claims, sources known".
**GPTQ follow-up review (2026-09-29, same tool, `results/judge/vllm_gptq.json` vs `vllm_bf16_v2.json`, 5 contradicted claims judged one by one by hand)**:
genuinely wrong 0, retrieval accomplice 0, judge misjudgement 4, pipeline artifact 1.
- `b1-q02#0`, `b2-q05#0`, `b2-q05#1`: judge misjudgements, shared across all three formats (the same sentences in bf16 and AWQ were also judged contradicted). This is direct evidence that judge error is "fixed per sentence, independent of format" (Finding 56).
- `b3-q16#0`: the judge **judged it the wrong way round**. GPTQ wrote 「不超過 60%」 ("not exceeding 60%"), a correct negation of the fixed-provider definition (>60%); bf16 wrote 「未明顯集中…大於 60%」 ("not clearly concentrated… greater than 60%"), which contradicts itself, yet was judged supported. The judge has no ability to discern numeric logical relations (greater than / not exceeding), the same type as the unit misplacement in §6.5.
- `b4-q12#1`: the Simplified 「参考资料未提及」 ("not mentioned in the references") dangles after a correct answer with no referent. The GPTQ judge file (2026-09-17 02:53 UTC) predates the Simplified normalization of refusal markers (5f60f9e, same day 15:40 UTC); under the current code this sentence would leave the denominator. Classified as a **pipeline artifact**, not a model error.
- `b2-q04` (found while tracing retrieval-accomplice questions; judged genuinely wrong by hand on 2026-09-29): GPTQ's only claim, 「Tetracycline 治療淋病的口服用法為每日2粒,每6小時一次,為時5天」 ("oral Tetracycline for gonorrhea: 2 capsules per day, every 6 hours, for 5 days"), was judged **supported** by the judge.
  The ground-truth chunk (`衛署藥製字第015791號115.08.10#0004#52ac5dad`) reads 「淋病–口服,2粒,每6小時一次,為時5天」 ("gonorrhea – oral, 2 capsules, every 6 hours, for 5 days"), meaning 2 capsules per dose.
  GPTQ added "per day" on its own, turning the daily dose from 8 capsules into 2, 4× too little, and the sentence is self-contradictory; the same chunk's 「成人每日處方限量可達4g」 ("adult daily prescription limit up to 4g") supports the 8-per-day reading.
  This is a judge **false-supported**, the same type as the unit misplacement (per day / per dose) in §6.5.

**Finding 56 (Fact)**: GPTQ's reviewed real gap to bf16, **counting only the contradicted side**, is 0 claims.
The supported side was separately reviewed for dose/numeric sentences (next paragraph): both formats let through 3 each. The judge misjudgement rate across the three formats is 4/5, 5/8, 4/5, and 3 misjudged sentences overlap across formats, supporting "symmetric error".

**Supported-side dose-sentence follow-up review (2026-09-29, `results/judge/review_supported_dose.xlsx`, judged row by row by hand)**:
Filter: judged supported, and the sentence contains "number + unit" (粒/錠/mg/ml/次/小時/日/週/%… i.e. capsule/tablet/mg/ml/times/hour/day/week/%…) or 「每日/每天/每次/每週」 ("per day / daily / per dose / per week"). Numbers written in Chinese numerals are missed, so this is a lower-bound sample.
75 claims in total, covering 36 questions: of bf16's 39, 3 are unsupported; of GPTQ's 36, 3 are unsupported; judge false-supported rate about 8%.

| Question | Format | Error type | Mechanism |
|---|---|---|---|
| `b2-q04` | GPTQ | Unit misplacement | 「每日2粒,每6小時一次」 ("2 capsules per day, every 6 hours"), daily dose 4× too low |
| `b4-q05` | GPTQ | Frequency (indication misattribution) | Moves anxiety's 「分2至3次」 ("in 2 to 3 divided doses") onto insomnia |
| `b3-q15` | GPTQ | Date-role misplacement | Announcement date written as effective date; the second half of the sentence corrects itself |
| `b2-q09` | bf16 | Entity misplacement | 「每次注射需間隔四週」 ("each injection must be four weeks apart") is verbatim in the corpus, but refers to the LAR long-acting form; the question asks about take-home self-injection |
| `b3-q16` | bf16 | Mutually exclusive categories merged | 「未明顯集中…大於60%」 ("not clearly concentrated… greater than 60%"), the same claim as calib v2 r30 |
| `b3-q19` | bf16 | Condition omitted (borderline) | Drops 「連續照護滿三個月」 ("three full months of continuous care"), loosening when a claim may be filed |

**AWQ supported-side dose-sentence follow-up review (2026-09-29, `results/judge/review_supported_dose_awq.xlsx`, same filter)**:
35 claims in total, covering 31 questions; 32 supported, 3 unsupported, false-supported rate 8.6%.

| Question | Error type | Mechanism |
|---|---|---|
| `b2-q09` | Entity misplacement | 「每次注射間隔為四週」 ("injections are four weeks apart") belongs to the LAR long-acting form; same question, same error as bf16, and the judge let it through both times |
| `b3-q15` | Date-role misplacement | Announcement date 114.11.18 written as the start of implementation; same question, same error as GPTQ; bf16 got it right |
| `b4-q11` | Condition attribution misplaced (borderline) | Accelerated bone age belongs under 「II. 治療條件」 ("II. treatment conditions") but was written into the diagnostic criteria |

Only 1 Simplified sentence (`b3-q09` 「治疗」, "treatment"), with correct content and a correct verdict. n=1 is not enough to rule out "the judge lets Simplified text through leniently"; we can only say this subset did not show it.
Note: in this round's context the same chunk for `b3-q13` was complete (「115-08-19 修訂」, "revised 115-08-19"), whereas in the previous two audits it was truncated to 「5-08-19」; the same passage is truncated to different degrees in different retrieval results, recorded as a retrieval-layer observation item.

**Simplified-to-Traditional re-judging (2026-09-29, `results/judge/s2t_rejudge.json`)**:
- **Sample**: every claim in the three formats' judge files containing Simplified-only characters (`langdrift.SIMPLIFIED_ONLY`), 16 in total: AWQ 8, GPTQ 6, bf16 2.
- **Method**: each claim was re-judged in the same run as the original sentence and an OpenCC s2t Traditional version; the config drops `verbatim_min_coverage`, disabling the pre-filter layer so only the LLM judges; conversion uses `opencc-python-reimplemented==0.1.7`, installed in a temporary venv and not added to requirements.
- **Result**: 15/16 verdicts unchanged (AWQ 7 supported + 1 contradicted, GPTQ 5 + 1, all unchanged). The only flip is bf16 `b2-q01#1` 「参考资料未提及进一步处理方式」 ("the references do not mention further handling"), which went supported → contradicted after conversion, the opposite direction to "Simplified is judged more harshly".

**Finding 58 (Fact)**: within this sample, the judge does not judge more harshly because of Simplified characters; the reverse mechanism "quantized formats are underrated due to Simplified-Chinese drift" was not observed.
Limits: n=16 and the character list is a lower bound, so this only rules out a large bias, not a small one. AWQ's only contradicted claim containing Simplified text remains contradicted after conversion. Its claim_id has been checked as `b4-q12#1` (the Letrozole question, 「参考资料未提及其他药物的併用情况」, "the references do not mention concomitant use with other drugs"), which is exactly the one originally suspected of being "disturbed by Simplified text": the cause of the wrong verdict is not the glyphs.

**Finding 57 (Fact)**: the judge is blind in both directions to the type "all the words are there, the relation is wrong". On the contradicted side it judges correct sentences wrong (`b3-q16` GPTQ); on the supported side it lets wrong ones through.
Four of the six are mirror images of known weaknesses: per dose / per day, numeric logic, date semantics, indication misattribution. The only new type is entity misplacement (`b2-q09`), exactly what the §6.5 decomposed question Q ("which entity does the claim refer to") is meant to handle.
**Judgement**: misjudgements on the contradicted side across the three formats are 4/5, 5/8, 4/5, and for supported dose sentences 3/39, 3/35, 3/36; both sides are close.
Judge error is independent of format, and this is now measured on bf16 / AWQ / GPTQ rather than merely assumed; FP8 was not reviewed on either side.
The reverse mechanism "the judge is harsher on Simplified text" has been tested with Traditional-conversion re-judging (see paragraph below, Finding 58) and was not observed in this sample.
Reviewed gap (vs bf16): GPTQ is 0 on both sides; AWQ has 2 more on the contradicted side, and dose sentences are 3 vs 3.
Dose-sentence errors across the three formats overlap on two questions (`b2-q09` bf16 + AWQ, `b3-q15` GPTQ + AWQ); these errors are shared by the model and systematically let through by the judge, not introduced by quantization.
Counter-evidence remains: Simplified-Chinese drift in 9 AWQ questions and 6 GPTQ questions, and the judging pipeline's handling of Simplified text once directly changed the denominator (`b4-q12#1`). Symmetry stays at "partially supported, not fully verified".
One more quantization side effect is recorded: **Simplified-Chinese drift (quantified 2026-09-17, row four of `tools/bench_compare.py`, deterministic Simplified-only character list of 523 characters, zero dependencies)**:
questions containing Simplified characters: bf16 2 / FP8 1 / AWQ 9 / GPTQ 6 (of 100); for AWQ, up to 22.7% of the CJK characters in a single question are Simplified. bf16 itself has 2 questions with 1 character each;
W4 quantization amplifies this 3–4.5×; matched characters were hand-checked and are all genuine Simplified (属、体、时、兰、状、线…). It is unrelated to faithfulness, counted separately,
and not included in the quality score; if deployment targets Traditional Chinese users, this is another cost of the W4 formats, removable at zero cost by output post-processing (Simplified-to-Traditional conversion).
Honest disclosure of the process: the first version of the character list wrongly included 「症」 (a legitimate Traditional character, 2,455 occurrences in the corpus), producing 20 false bf16 hits; it was caught by "character-by-character checking of bf16 hits + corpus scan" and removed.

### 4.2 Axis C conclusion (2026-09-17, short answers + long outputs; vLLM single engine)

| Configuration | c=8 tok/s | c=1 latency p50 | faithfulness | How obtained | Notes |
|---|---|---|---|---|---|
| vLLM bf16 (boring-but-safe) | 424 | 0.73 s | 0.959 | Official weights run directly | Baseline |
| vLLM FP8 | 610 | 0.52 | 0.951 | ModelOpt self-calibration (TRT venv) | One-time conversion; usable across engines |
| **vLLM GPTQ-Int4 (recommended)** | 1000 | 0.34 | 0.960 | Official pre-quantized | faithfulness indistinguishable from bf16 under the judge, fastest, weights 5.3 GiB |
| vLLM AWQ | 1000 | 0.32 | 0.938 | Official pre-quantized | Same speed as GPTQ; after review the real gap is 2 claims, within noise (§4.1) |

**Decision**: under the 4090 short-answer RAG load, W4A16 (Marlin) is this project's single largest source of speedup (×2.36), larger than switching engines (≤10%) and FP8 (×1.4);
GPTQ-Int4's faithfulness is indistinguishable from bf16 under the judge (the judge is for relative comparison, κ 0.488, see §6.5), making it the recommended format.
**2026-09-20 re-check of this conclusion by human error typing** (§6.7): on the same 20 questions side by side, bf16 fabrication 4/18, GPTQ 3/18; the Wilson intervals overlap heavily, so no difference is detected at n=18 either; but the two have **different error types** (bf16 leans toward citation/section attribution, GPTQ toward adjacent-clause misattribution), listed as an observation item for axis C round two. FP8's role becomes the middle option for "need to span vLLM / TRT-LLM and don't want to do W4 calibration ourselves".
**Scope / limits of applicability**: single consumer card, Marlin kernel (sm_80+), outputs of 45–310 tokens, concurrency ≤ 8; AWQ / GPTQ calibration data is what Qwen officially used rather than this corpus,
a different source from FP8 (512 samples of this corpus), so the four-format quality comparison is not strictly controlled. TRT-LLM × W4 untested (gate C-1 N/A; ModelOpt-AWQ route is a to-do).
**To do**: ①~~Human review of AWQ's 8 contradicted claims~~ (done, see §4.1) ②Fill in the FP8 weights / KV cache columns ③KV cache FP8 quantization, round two ④TRT-LLM ModelOpt-AWQ route.
## 5. Axis D: RAG paradigm (naive / agentic / long-context) — out of scope (decided 2026-09-18)

**Why not done**: the rule agreed at project founding, "cut axis D first if over time", was triggered: axes A / B / C and judge calibration used up the whole budget (founded 2026-08-27, three and a half weeks).
This memo's question is "which configuration to use on a single card"; axis D answers "whether to use RAG at all", which is the subject of a different memo.
**Partial evidence already in hand** (not enough for a conclusion, direction only): axis C KV cache capacity (bf16 113k / GPTQ 274k tokens, §4.1) shows that at a 4096 context
the concurrency a single card can absorb is far from saturated, so stuffing long context directly is feasible capacity-wise; but judge stage one only calibrated short answers (§6.5), and long-context output faithfulness
has no credible yardstick, so even if axis D were done its quality could not be compared. **Precondition**: judge stage two (atomic claims for long answers + citation accuracy).

## 6. Axis E: RAG vs fine-tune (QLoRA) — out of scope (decided 2026-09-18)

**Why not done**: same budget as above; moreover axis E's comparison basis is "generation quality", and §6.5 showed the current judge can only do relative comparison (κ 0.488);
using it to compare the absolute quality of RAG against a fine-tuned model would give untrustworthy conclusions. Axis E requires a credible quality yardstick first, which is judge stage two's job,
not something answered just by getting QLoRA running. **Preconditions already decided**: fine-tuning is folded into axis E rather than a separate project (2026-08-27);
if restarted, the control group reuses axis C's vLLM GPTQ-Int4 configuration, the fine-tune target is Qwen2.5-7B-Instruct @ a09a3545, and calibration/training data starts from the same 512 samples as FP8.


## 6.5 §5.3 faithfulness judge calibration record (2026-09-17)

judge = `Qwen/Qwen2.5-14B-Instruct-AWQ @ 53953585` (vLLM, json_schema constrained, seed 42); calibration set = 38 claims from 30 short-answer questions
+ 10 synthetic negatives (one number altered), judged by humans in a binary scheme (34 supported / 14 unsupported), with the judge's three classes collapsed into two for comparison.

| prompt | κ | Agreement | Synthetic negatives caught by judge | supported recall | not_supported recall | Disagreements |
|---|---|---|---|---|---|---|
| v1 | 0.789 | 0.917 | 9/10 | 0.97 | 0.79 | 4 |
| **v2 (live)** | **0.845** | 0.938 | **10/10** | 0.97 | 0.86 | 3 |

**The four v1 disagreement rows**: synthetic r23 (tampered + mutually exclusive categories merged) judge supported; real r30 (pure merge) and r34 (adjacent-clause misattribution) judge supported;
one row human-supported but judge contradicted (list of alcohol reactions). Human error-type tally: 14 unsupported = 8 numeric tampering + 6 structural
(adjacent-clause misattribution 1, grounded fact plus fabricated explanation 1, cross-document splice 2, mutually exclusive categories merged 2).
**Finding 22 (Fact)**: judge v1 caught all 8/8 numeric tamperings and missed all 3/3 structural errors; "numeric all right, structural all wrong" still lets overall κ reach 0.79,
and only per-error-type stats reveal the hole. The four structural errors (chunking boundaries, cross-document splices, causal fabrication, logical merging) are exactly what pure string matching cannot catch, and also the most common on the RAG generation side.
Action: prompt v2 adds explicit rules targeting "merged conditions" and "subject attribution" (prompts are production code; the evidence is the four rows above), and calibration was rerun;
the pass bar is unchanged (κ ≥ 0.8 and synthetic ≥ 9/10). Three judge edge cases are also recorded for observation: paraphrase drift (r49), absence statements (r13), cross-language drug names (r12); v1 judged all supported, agreeing with humans.
**v2 result**: κ 0.845, synthetic 10/10, and the new rules did not wrongly kill any previously agreeing row; three disagreements remain: ① the judge wrongly kills one row: passage [1]'s chunk starts at
「Disulfiram的作用,如腹部痙攣…」 ("the effect of Disulfiram, such as abdominal cramps…"), with the subject (alcohol + this drug taken together) cut off into the previous chunk; without cross-chunk common sense the judge declares a conflict, a judge false positive caused by the chunking boundary;
② r30 (mutually exclusive categories merged) and r34 (adjacent-clause misattribution) are still missed: residuals of the 14B judge on this corpus, listed as known limits, with no third round.
**Judge go-live conditions**: prompt v2, `Qwen2.5-14B-Instruct-AWQ @ 53953585`, seed 42, json_schema; disagreement rate about 6%,
so a faithfulness gap < 0.06 between two configurations is treated as indistinguishable.

**Short-answer faithfulness (v2 judge, 100 questions, same prompt set and answers as §3)**

| Configuration | Claims | Refusal answers | faithfulness | unsupported | contradicted |
|---|---|---|---|---|---|
| vLLM bf16 | 121 | 3 | 0.959 | 0.000 | 0.041 (5 claims) |
| vLLM FP8 | 122 | 3 | 0.951 | 0.000 | 0.049 (6 claims) |

**Finding 23 (Fact)**: the bf16 → FP8 faithfulness difference is 0.008 (1 claim), far below the judge disagreement rate. FP8 changes half the greedy outputs verbatim (§3.2),
but what changes is wording and length, not whether claims are grounded; in the short-answer setting FP8's quality cost is not measurable. Long answers and citation accuracy are left to stage two.

**Judge v2 residuals (2026-09-17, revealed by the axis C AWQ review)**: of 8 contradicted claims, human review found 5 misjudgements (false-contradicted 5/8),
which the v2 calibration κ 0.845 did not catch because the 48-row calibration set contained too few verbatim-match sentences. Three types and countermeasures:

| Type | Cases | Mechanism | Countermeasure |
|---|---|---|---|
| Verbatim match still judged contradicted | 2 | Claim is an original context sentence, yet the judge judges it the wrong way round; the same sentence split into two claims in bf16 got one supported and one contradicted, so verdicts are unstable | **Deterministic pre-filter layer**: if the normalized claim is a substring of the context or its character 3-gram coverage passes a threshold, judge it supported without sending it to the LLM; threshold grid-swept on the calibration set, take the plateau midpoint (next round) |
| Headless chunk | 1 | Main clause cut off by chunking, judge cannot see the causality; the following sentence is enough to carry it | Not fixed in the judge; assigned to the axis A chunking (parent-child) to-do; only count misjudgements of the "incomplete context" type |
| Abstention verdicts off | 2 | One wrong abstention judged correct, one correct abstention judged wrong: the judge cannot tell "the corpus really lacks it" from "the model didn't read it" | prompt v3: for "not mentioned" claims, ask instead "does the passage contain that information for <the subject the question refers to>", subject attribution before presence |

**Verbatim pre-filter layer goes live (2026-09-17, commit aa16c84)**: deterministic layer: if a claim's character 3-gram coverage against a single chunk is ≥ 0.925
(midpoint of the zero-let-through plateau 0.85–1.00 on the 48-row calibration set), all digit strings appear in that chunk, and the claim is ≥ 6 characters → directly supported, not sent to the LLM.
Real-data validation: AWQ rerun faithfulness 0.938 → **0.946**, contradicted 8 → 7, and the one taken over is exactly the verbatim-match one, as predicted;
the cost is as measured: it only takes over 1 claim. This layer is a zero-let-through safety short-circuit; the main fix still relies on prompt v3.

**Stratified calibration set v3 and prompt v2 baseline (2026-09-18)**: following plan B, stratified sampling after deduplicating the four formats' judge results
(`calib_sheet.py build --stratify`; verbatim 4 / paraphrase 12 / low 12 / abstention all 7 / synthetic negatives 7, 42 rows total, human binary labels;
after Simplified normalization of refusal markers the abstention stratum contains 4 Simplified 「参考資料未提及」 ("not mentioned in the references"), kept as claims to observe how the judge rules on abstention sentences).

| Configuration | n | κ | agreement | Too strict (human supports, judge against) | Too lenient (human doesn't support, judge lets through) |
|---|---|---|---|---|---|
| v2 on old random set (48 rows) | 48 | 0.845 | — | — | — |
| **v2 on stratified set (42 rows)** | 42 | **0.488** | 0.786 | 4 | 5 |

By stratum: verbatim 4/4 (taken by the pre-filter layer), negative 6/7, abstention 0.417 (7 rows), paraphrase −0.125, low −0.125 (12 rows each, errors in both directions roughly half-and-half, cancelling out).
**Finding 26 (Fact)**: the same judge goes from κ 0.845 on the random set → 0.488 on the stratified set; the old κ was propped up by verbatim-type and easy supported sentences, and stratified sampling exposes the weaknesses.
Types of the nine disagreements: too strict: corpus paraphrase judged contradicted ×2 (two wordings of the same fact both judged against), granularity (known limit of the whole-sentence-semantics decision), correct abstention judged the wrong way round;
too lenient: adjacent-clause misattribution, unit misplacement ×2 (same number, different measurement basis; one real, one synthetic), wrong abstention let through, false claim of existence let through (listing attachments the passage doesn't have).
**Judgement**: all three v2 rules target "too lenient", while half the stratified set's errors are "too strict"; prompt v3 must tighten both ways: abstention sentences first ask subject attribution,
paraphrases must not be judged against because of wording, units/measurement bases are included in numeric checks, false claims of existence are judged against.
Labeling criteria ruling (2026-09-18): wrong framing but facts present (listing nonexistent attachments) is judged unsupported (echoing the existing paired rule "no false claims of existence");
abstention-type claims are judged by "whether the corpus truly lacks that information for this question's subject", not by the literal wording of the "rules"; set-style closing sentences keep whole-sentence semantics and are judged supported.

**prompt v3 acceptance (2026-09-18, below the bar)**: after adding four bidirectional rules (abstention subject attribution, paraphrase not judged against, units/measurement basis, false existence of listed items),
rerunning the same stratified set: κ 0.488 → 0.488; only 2 of 42 raw outputs differed (one evidence_chunk number, one JSON line break),
and none of the nine errors moved. The bar (κ ≥ 0.70, ≤ 2 each direction, negatives 7/7) was not met; per the agreement, no forced tuning.
**Finding 27 (Fact)**: explicit rules have zero leverage on the 14B judge for hard sentences: the two v1→v2 rules worked (κ 0.789→0.845, structural misses filled in),
the four v2→v3 rules did not; the difference is that v2 added items "the model didn't think to check", while v3 asks for discretion where "the model checked but judged the wrong way round", which rule text cannot push.

**Plan B1 decomposed-question pre-experiment (2026-09-18, `tools/decomp_probe.py`, below the bar)**: split the judge into three sub-questions, Q1 subject attribution / Q2 values and measurement basis /
Q3 listed items, with the verdict merged by code (contradicted can only come from a named point of difference). Trial on the 8 rows v2 got wrong on the stratified set + 7 negatives:
flipped correct 3 / 8, negatives 7 / 7, bar (≥ 6) not met, B closed.
Row by row: ① both abstention rows flipped correct: asked about "subject attribution" on its own, the 14B can tell presence/absence of "other drugs" and "limit on number of visits"; the only empirical case where decomposition moves what the four v3 rules could not;
② periodontal disease judged correct but via the wrong path (Q3 missed the nonexistent item, Q2 wrongly grabbed 「第一階段」 ("stage one") as a value), not counted; ③ adjacent-clause misattribution also wrong under decomposition (Q1 attributes the adjacent passage
to the subject; no fix on the judge side); ④ for both paraphrase rows Q1 found no subject and returned null → unsupported, and Q2 force-filled "2 values 1 different" for sentences with no numbers (the schema-forced filling foreseen in the pre-mortem
happened on the spot); ⑤ unit misplacement: even when Q2 explicitly asks to compare measurement bases it returns "0 different"; changed numbers are caught, changed "per day / per dose" is invisible; ⑥ the granularity row: decomposition inherently reads atomically,
conflicting with the whole-sentence-semantics decision, as expected; also, chunk numbers returned by Q1 included document codes (not passage numbers), so a production implementation must validate against the context.
**Decision**: the judge's role is **relative comparison across the four formats**, with κ 0.488 and the nine error types stated explicitly as the scope / limits of applicability; PROMPT_VERSION reverts to v2;
"route abstention-type claims through Q1 decomposition" (2/2, cost about one extra call for ~3% of claims) is listed as a judge stage two candidate.

One more is a granularity dispute (set-style closing 「三病症的治療與預防」 ("treatment and prevention of the three conditions"), while the package insert only lists prevention for two of them): **granularity decided as whole-sentence semantics (2026-09-17)**;
split_claims splits by sentence, the judge works per sentence with no further splitting, and set-style closings are judged supported; the cost is that partial over-generalization inside a sentence is missed, but this judge is used for format/engine comparison, and sentence-level noise is symmetric across formats.
Atomic decomposition (RAGAS-style) rejected: claim count balloons, splitting itself adds another LLM step to calibrate, and numbers would not be comparable with v1/v2.

## 6.6 Manual review of the demo citation layer (2026-09-19): verbatim coverage cannot serve as a proxy for support

The demo prompt v13_cite requires every sentence to carry a `[段落編號]` tag. The deterministic layer checks "valid / out of range / cites the ground truth or not" and computes, for each sentence,
"the fraction of its character 3-grams covered by **the cited passage**". 100 questions × 2 presets: citation compliance 100%, zero out-of-range;
cited sentences with coverage < 0.5 make up 0.321 (BM25) / 0.345 (hybrid).
From 40 questions × 2 presets (74 cited sentences), all 23 low-coverage sentences were extracted and manually classified one by one:

| Verdict | Count | Notes |
|---|---|---|
| Paraphrase (content supported, different wording) | 12 | The majority of low-coverage cases; not errors |
| Fabrication (cited passage lacks this content) | 11 | See subtypes below |
| Cross-passage synthesis | **0** | All 5 sentences citing multiple passages drew their "content entirely from one of them"; the citation list is partly decorative |

**Finding 28 (Fact)**: **Verbatim coverage cannot serve as a proxy metric for support.** Three counterexamples suffice to refute it: one sentence with coverage 0 is a verbatim-faithful paraphrase;
one with coverage 0.351 is a **wrong-drug hit** (the question asked about Metronidazole's neurological warnings; the citation went to an antidiarrheal package insert's 「疲勞、眩暈或昏睡」 ("fatigue, dizziness or drowsiness"));
one with coverage 0.484 is a **causal inversion** (corpus: 「因本品降血糖作用甚強,故須餐時服用」 ("because this product's glucose-lowering effect is very strong, it must be taken with meals") → sentence: 「因為此時藥物的降血糖作用最強」 ("because the drug's glucose-lowering effect is strongest at this time")).
This project therefore positions coverage as a **diagnostic flag** (ranking "which sentences need manual verification") and **forbids it from entering any score or configuration comparison**;
code and UI wording were changed accordingly to 「需查證的引用句 a/b(診斷旗標,非品質分數)」 ("cited sentences needing verification a/b (diagnostic flag, not a quality score)").

**Six subtypes of the 11 fabrications** (by danger): ① wrong-drug hit (verbatim match but from a different drug; the hardest to catch with a quick manual scan);
② causal inversion (almost fully matching literally; only the causal direction is wrong); ③ adjacent-clause misattribution (a reimbursement restriction taken from the tail sentence of the next drug; same type as in the reverse spot-check);
④ **citation misattribution, 4 cases** (content correct but wrong passage cited: the same sentence under the other preset cites correctly and is fully supported; strictly speaking not a hallucination,
**kept as its own category to avoid overestimating the hallucination rate**); ⑤ unit misplacement (「每次 2 粒」 ("2 tablets per dose") → 「每日 2 粒」 ("2 tablets per day"), a 4× difference in daily dose; same type as in the judge calibration set);
⑥ table-fragment misreading (a column number read as a reimbursement item).

**High-coverage band supplementary sample (2026-09-20, dispute 2)**: 20 cited sentences with coverage ≥ 0.5 were sampled (GPTQ, same two presets, seed 42)
and manually classified → **paraphrase 19 / synthesis 0 / fabrication 1 (0.050, Wilson 95% [0.009, 0.236])**. The two bands' coverage ranges do not overlap at all
(low band 0–0.489, high band 0.541–1.000), so the stratification is clean. After stratified weighting:

| Band | Sentences | Method | Fabrication rate |
|---|---|---|---|
| Coverage < 0.5 | 23 | Full count | 0.478 (11/23) |
| Coverage ≥ 0.5 | 51 | Sample of 20 | 0.050 (1/20) |
| **Weighted estimate** | 74 | — | **0.183, interval [0.155, 0.311]** |

(The interval carries only the high band's sampling uncertainty; the low band is a full count with no sampling error.)
**This estimate comes with a condition that must be stated**: the high-band sample falls on 18 questions, which **overlap by only 5 questions** with the 18 questions containing the low band's 23 sentences
(`b1-q04`, `b1-q19`, `b1-q22`, `b2-q05`, `b2-q11`); the 13 questions unique to the high band include more single-value fee-schedule questions (`b1-q21`, `b1-q23`, `b2-q02`),
which are easier. Hence 0.050 may be low for "the same 51 sentences", and the weighted estimate 0.183 should be read as a **point estimate, not an unbiased estimate**;
what is certain is that **the lower bound of 11/74 ≈ 0.149 has been pushed up to about 0.18, and the high-coverage band is indeed not error-free** (it still caught 1 case).

**Scope / limits of applicability (must read)**: the two presets have similar fabrication counts (5/5 vs 7/6);
**hybrid is not better**: better retrieval does not automatically improve generation faithfulness.
One more defect outside support is recorded: one sentence rendered methotrexate as a nonexistent drug name (correct: 甲氨蝶呤); other Chinese renderings in the same batch were correct, so it was judged a one-off generation error,
but in a medical setting its severity is no lower than fabrication, so it is listed as a separate observation item, "drug-name transliteration".

**Two internal controls** (clean cases where a retrieval difference directly caused a generation error): on the same question b1-q11, BM25 retrieved an adjacent clause and misattributed it,
while hybrid retrieved the correct clause; the same sentence under different presets (b2-q01) had one wrong and one correct citation number.

**Actions**: ① coverage demoted to a diagnostic flag (done); ② citation misattribution kept as its own category, not counted as hallucination; ③ "whether the citation **semantically** supports the sentence"
remains judge stage-2 work; this layer only does deterministic format checks; ④ the six subtypes are merged into the judge stage-2 error-type list;
⑤ the coverage screening threshold is narrowed to "< 0.1 goes to manual review first": the two-way counter-evidence from dispute 3 (see §6.7) shows it is ineffective against "adjacent but misattributed citations";
the 0.5 threshold is only suitable as a UI flag, not as the ordering for manual review.

## 6.7 Cross-format support pairing (2026-09-20, dispute 3): bf16 and GPTQ differ in error types, error counts are indistinguishable

Axis C's conclusion "GPTQ's faithfulness is indistinguishable from bf16 under the judge" (§4.2) is a relative comparison under judge κ 0.488.
Here it is re-checked with the deterministic citation layer + manual classification: same 20 questions, same `bm25_ctx_tuned` (zero-GPU retrieval, whole card for the model),
same seed, only the weight format changed, compared side by side per question.

| Format | Cited sentences | Paraphrase | Synthesis | Fabrication | Fabrication rate (Wilson 95%) |
|---|---|---|---|---|---|
| bf16 | 18 | 13 | 1 | 4 | 0.222 [0.090, 0.452] |
| GPTQ-Int4 | 18 | 15 | 0 | 3 | 0.167 [0.058, 0.392] |

**Finding 29 (Fact)**: **The two intervals overlap heavily; no significant difference was detected at n=18**. This must not be written as "the two are the same".
But **the errors are not on the same sentences**, which is more informative than the error counts:
- **Errors common to both formats** (2): `b1-q08` (Benzodiazepine passage), `b1-q12` (Stiripentol)
- **bf16 only** (2): `b1-q01` cited the Spironolactone side-effects passage while GPTQ cited the correct clause 5.1.1 (same sentence, the only difference is the citation number);
  `b1-q07` answered 【禁忌】 (Contraindications) content as 【警語】 (Warnings) (right drug, right corpus, wrong section: the mildest type among the verdicts; under partial scoring it could get 0.5)
- **GPTQ only** (1): `b1-q11` answered Climara as 「每 6 個月評估一次」 ("reassess every 6 months") (correct: limited to one patch per week), an adjacent-clause misattribution

**Judgement (to be verified)**: both of bf16's unique errors are of the **citation / section attribution** type; GPTQ's unique error is of the **adjacent-clause misattribution** type.
If quantization really has an effect, it should show up as one error type increasing rather than uniform overall degradation; n=18 is insufficient to test this, so it is listed as a named observation item for axis C round 2.

**Finding 30 (Fact): for the two questions all three runs got wrong, the lesion is in retrieval and chunking, not the model**:
- `b1-q12` (Stiripentol): all three runs were wrong, each differently (bf16 attached two clauses that don't belong to it; GPTQ and the high band each attached one),
  and none reached the correct answer, Dravet syndrome. **Diagnosis corrected on 2026-09-20**: the earlier statement "Dravet was cut off outside the chunk head"
  was wrong. On checking, **"Dravet" appears 0 times in the entire corpus**; the lesion is **characters dropped at the extraction stage**, not a chunk boundary.
  That chunk only retains `syndrome)病人`, and its `heading_path` is also mismatched to a biomarker-testing heading;
  in the same PDF, 19/1046 chunks (1.8%) contain the same kind of misplaced-parenthesis fragment, scattered across at least 12 pages
  (full audit in the "known extraction defects" section of `data/corpus/SOURCES.md`). The model cannot see the disease name and can only grab surrounding clauses.
  **Action**: no corpus rebuild, no eval set change: `clobazam`, `valproate` and `全身性強直陣攣` are all in the chunk,
  and together already form the complete correct answer, so `required_components` deliberately excludes Dravet (including it would reward hallucination:
  the only way for the model to score would be to generate, from parametric memory, content absent from the corpus).
- In `b1-q08`, the `[3]` fed in is the Benzodiazepine/opioid interaction passage, which **does not contain spironolactone at all**: the retrieval layer's fault.
Three models being wrong simultaneously is the cleanest form of evidence that "the generation side is not to blame"; tuning the model or prompt is useless for these two questions. What needs fixing is chunk boundaries
(axis A parent-child to do) and retrieval.

**Finding 31 (Fact): coverage gives two-way counter-evidence on the same data**:
the two cases at 0.023 / 0.025 are citation misattributions, and low coverage **caught them correctly**; the two cases at 0.922 / 0.952 are Stiripentol adjacent-clause misattributions, and high coverage **got both wrong**.
That is, coverage works for "completely unrelated citations" but not for "adjacent but misattributed citations", **and the latter is exactly the most common error type in this corpus**
(adjacent chunks in clause-style corpora are highly similar literally). The conclusion points the same way as Finding 28 but more strongly: fine as a filter, not as a score.

**Defining case of "cross-passage synthesis"** (first appearance of this label): `b1-q05` in `support_bf16`: 「牙齒永久性變色」 ("permanent tooth discoloration") and 「抑制骨骼生長」 ("inhibition of bone growth")
take the wording of `[2]`, while 「琺瑯質形成不全」 ("enamel hypoplasia") takes the wording of `[1]` (`[2]` writes 齒釉質發育不全), each passage contributing different vocabulary, both faithfully.
**On the same question GPTQ took all three terms from `[1]` and was judged paraphrase**: same question, same two context passages, bf16 fuses while GPTQ copies only one passage.
This is the cleanest **model behaviour** difference (rather than quality difference) across the three files, more worth tracking than fabrication rate, and is listed as an axis C round 2 observation item.

**Two question revisions confirmed effective**: after `b1-q10` added dapagliflozin / empagliflozin, bf16 correctly answered ≦40% and was not pulled toward the
Entresto ≦35% in the same context; after `b1-q21` added the effective-date restriction it answered 39176, with no interference from the old rate 17703.

**Newly found question defects (not the model's fault; fix the question or the scoring)**: `b1-q19` (fecal occult blood subsidy eligibility and frequency) gave three
different partial answers over three runs (bf16 only the 40–44 family-history group, GPTQ only the frequency, high band only the 45–74 group), **never complete once**:
the two eligible groups are in two chunks, so a single-sentence claim inherently cannot answer fully. The same type also applies to `b1-q22` (points + whether special materials are included), each of two runs answering half.
Action (to do): change to ask about a single group only, or list each component as required in scoring; otherwise these two questions can never get full marks, and the deduction isn't the model's fault.

**Two generation quality issues (do not affect support verdicts, but listed as prompt to-dos)**: ① sentence-initial meta preamble leakage
(「參考段落中的信息顯示」 ("the information in the reference passages shows")); ② methotrexate rendered as a nonexistent drug name (keeping the English would be correct):
same type as the existing rule of thumb "LLM output-layer wording discretion must always be converged into explicit rules"; the prompt should explicitly require **keeping drug names in the original**.
**Action (2026-09-22, `configs/demo.yaml` prompt v14_cite)**: both were written as explicit rules:
① 「藥品名、器材名、菌種名一律照抄參考段落的原始書寫形式(含英文與大小寫),即使看似拼寫有誤也照抄」 ("drug, device and organism names are always copied in the original written form of the reference passage (including English and capitalization), even if they appear misspelled")
② 「直接陳述答案,禁止以指涉參考段落或作答過程的語句開頭」 ("state the answer directly; do not open with a sentence referring to the reference passages or the answering process").
Only the demo prompt was changed; **`configs/prompt_v12.yaml` (the axis C experiment prompt) is untouched**, so all existing numbers in the memo are unaffected;
the cost is that the fourth round of demo/support outputs is **no longer verbatim-comparable** with the first three (which is exactly what is to be measured); the v13 original text is kept in a config-file comment for rollback.

## 6.8 First scoring run of required-component verdicts (2026-09-21): counting citation misattribution separately cuts the substantive error rate by nearly 40%

The list `data/eval/required_components_v1.json` (16 questions, 36 components) was run for the first time on a real model
(GPTQ-Int4, 100 questions × 2 presets, 172 cited sentences), with manual per-sentence classification.

| Verdict | Count | bm25_ctx_tuned | hybrid_rerank_bge |
|---|---|---|---|
| Paraphrase | 142 | 69 / 84 | 73 / 88 |
| Cross-passage synthesis | 2 | 1 | 1 |
| Fabrication | 28 | 14 | 14 |

**Finding 32 (Fact): citation misattribution must be counted separately, otherwise the hallucination rate is overestimated by nearly 40%**. Of the 28 "fabrications",
**10 are pure citation misattributions**: content correct, only the passage number mislabelled; the same sentence under the other preset often cites correctly.
With these separated, substantive errors are **18 / 172 = 10.5%**, not 28 / 172 = 16.3%.
The clearest evidence is `b3-q18`: the sentence-initial leftover 「人及其家屬…」 ("person and their family…") matches exactly how the correct chunk is cut off at its head;
**the model saw the right chunk and just mislabelled the number**; and both presets had the same sentence with the same error → unrelated to retrieval, a generation-side labelling problem.
(`b2-q04` hybrid also mislabelled the passage, but also had a 4× dose unit misplacement, so it is classed as a substantive error to avoid underestimating.)

**Composition of the 18 substantive errors** (by danger): wrong-drug hit 3 (antidiarrheal taken as Metronidazole, Triamcinolone as
Dexamethasone, Allopurinol as Furosemide), causal mislink or inversion 4, adjacent-clause misattribution 3, unit misplacement 2,
irrelevant retrieved passage 1, and 1 each of the rest (version mix-up, table column-number misreading, same-semantic-field misattribution, announcement date taken as effective date,
flattened conditional structure).

**Finding 33 (Fact): forbidden expressions hit 6/6 in the first run, zero collateral damage**. The 3 protective-type ones (「每日2粒」 ("2 tablets per day") ×2,
Climara 「每6個月評估一次」) and the 2 question-level ones (Stiripentol's adjacent clause) all hit real errors;
the 6th (「甲基軹酸酯」) is redundant insurance: that sentence had no English drug name to begin with and its component score was already 0.
**None hit a correct answer**; S-C5's criterion ("appears only in wrong answers") is validated in the first run.

**Finding 34 (Fact): the two presets have almost identical fabrication rates (14/84 vs 14/88), but err on different questions**.
There are 8 questions where one is right and the other wrong, split evenly: hybrid fixed `b1-q08` (Spironolactone; previously all three models were wrong across four files,
correct for the first time this round), `b3-q01`, `b3-q08`, `b3-q15`; bm25 won on `b2-q08`, `b2-q09`.
**`b2-q08` has one preset right and one wrong under the same context** (bm25 verbatim correct, hybrid causal inversion)
→ the difference is not entirely in retrieval. This points the same way as §6.7 Finding 29, "error counts indistinguishable but error types differ".

**Finding 35 (Judgement, to be verified): completeness and support can give opposite signals on the same question**.
In `b1-q19`, the bm25 row has completeness = 1 (all four components hit), but the only cited sentence in that row is
「補助對象每二年進行一次檢查」 ("eligible persons are screened once every two years"): components are computed on the **full answer** (`support_sheet.py:147` passes in `answer`),
while support only includes cited sentences, so the two have different populations. Whether to change components to count only cited sentences
is a design trade-off awaiting decision (changing it would count "correct but uncited" as a miss).

**Known defects of the verdict layer (exposed in the first run)**: 5 false misses, 2 of which were fixed on 2026-09-21 by adding aliases
(`b3-q05`'s 「最多服用3次」 ("take at most 3 times") split by an inserted character, and `b1-q22`'s paraphrase of 「特殊材質費」 ("special materials fee"));
**1 cannot be fixed with an alias**: in `b3-q19` the model wrote 「支付點數為1500,」 ("payment points are 1500,"); adding the alias `點數為1500`
would falsely match 「點數為15000」, requiring a numeric-boundary rule in the verdict layer.

**Sporadic but reproducible**: 5 cases of Simplified-Chinese drift; `b3-q18`'s 「參考段落中的信息顯示」 meta preamble
recurred on the same question and sentence as the previous round → reproducible behaviour, not random noise; a prompt to-do.
`b3-q01` confirmed the batch-3 annotation prediction: back then Paraflex 「與 penicillin 沒有交叉過敏」 ("no cross-allergy with penicillin")
was labelled a contradiction-type distractor with the prediction it would mislead the model; this round bm25 indeed used it and even derived a self-contradictory causal claim.

**Quality of the two syntheses**: `b4-q01` is genuine cross-passage fusion (「尿酸腎病治療」 ("treatment of uric acid nephropathy") from [1],
「化療產生之高尿酸血症」 ("chemotherapy-induced hyperuricemia") from [2]), but [1] also lists uric acid kidney stones and calcium oxalate kidney stones, which were missed,
and the added hyperuricemia is strictly speaking not yet a kidney condition: **the synthesis ability has appeared, but material-selection judgement has not caught up**.

## 6.9 Component verdicts round 2 (2026-09-21): same input, only the verdict layer changed, turning "fixed" from conjecture into evidence

Round 1 exposed 5 false misses and a design contradiction; after the fixes it was rerun with **exactly the same settings**
(GPTQ-Int4, seed 42, same corpus, same presets).

**Finding 36 (Fact): the model outputs of both rounds are verbatim identical, so causality is clean**. A programmatic comparison of the 172 rows on the six shared fields
`preset / item_id / 句子 / 引用編號 / 覆蓋率 / 所引段落` found **zero differences**.
The verdict column therefore carries over from round 1 (paraphrase 142 / synthesis 2 / fabrication 28), and every change in the component columns
can **only** be attributed to the verdict layer. This experimental design is more worth recording than the numbers: **to attribute an effect to a change in one layer,
the other layers' outputs must be verbatim comparable**; otherwise "fixed it" cannot be separated from "the model happened to get it right this time".

| Check | Result |
|---|---|
| 5 false misses | **All fixed** (`b1-q22` special materials, `b3-q05` 24-hour cap ×2, `b3-q19` payment points ×2) |
| Regressions | **Zero** changed from hit to miss, **zero** new false hits (all 30 rows with components checked) |
| Bare-number alias collateral damage | None (`b4-q15`'s `650` still hits, and the cited passage indeed contains 「發現費650點」 ("detection fee 650 points")) |
| `b1-q19` contradiction | Flagged **three** items rather than the expected two: the cited sentence 「補助對象每二年進行一次檢查」
  also lacks "45-74 years", so of the four components only "frequency" has citation support |

**Finding 37 (Fact): a single "citation support" column has a structural blind spot; splitting it into two columns is needed to cover citation misattribution**.
The first version of the column checked "whether the component appears in a **sentence with a citation tag**", so it could catch
"component only in an uncited sentence" (`b1-q19`), but not "the sentence has a citation, but the **cited passage** does not contain the component",
and the latter is exactly the 10 citation misattributions of §6.8 Finding 32. Two real examples: in `b3-q12`, both components (recommended dose and injection interval)
were hit and the column was empty, but the cited `[3]` is a hemophilia dosing table that does not contain `10-40 IU/Kg` at all; `b2-q04` is the same type.
Action: split into two mutually exclusive columns:

| Column | Meaning | Corresponding support error type |
|---|---|---|
| Hit but **uncited** | Component appears only in sentences without a citation tag | Population difference between completeness and support |
| Hit but **citation not supporting** | Component is in a cited sentence, but the cited passage does not contain it | **Citation misattribution** |

Mutual exclusivity is guaranteed by "anything already in the uncited column does not enter the citation-not-supporting column", preventing double counting of a component.
Both use the same matching function, only the match target changes (cited sentence → cited passage).

**The verdict layer's three rules have now taken shape** (all exposed by scoring runs, not designed in advance):
① **Numeric boundary**: when an alias's edge is a digit, the adjacent character in the answer must not be a digit. `b3-q19`'s
「支付點數為1500,」 cannot be fixed by an alias (`點數為1500` would falsely match `點數為15000`) and must be solved in the verdict layer;
once the rule took effect, the aliases for the three payment-point items were reverted to bare numbers, which **covers more word orders and makes the list simpler**.
② **Glyph normalization yes, semantic normalization forbidden** (N4 of §6.6).
③ **Two kinds of "unsupported" in separate columns** (Finding 37).

**Still not measured**: the actual count of the citation-misattribution type must wait for round 3 (the first real-model run with the new columns).
The first two rounds proved the output stable, so round 3's differences will likewise be a pure verdict-layer effect.
(Round 3 results in §6.10.)

**Counting conventions for citation misattribution** (both are correct; don't mix them): **10 pure citation misattributions** (content correct, only the number mislabelled)
is the numerator of Finding 32; `b2-q04` is excluded because it also has a 4× dose unit misplacement and was classed as a substantive error;
if the question is instead "how many have a citation tag pointing to a passage that does not contain the content", the answer is **11** (including `b2-q04`).
The new column measures the latter type, so flagging `b2-q04` is correct behaviour, not a contradiction of Finding 32.

## 6.10 Component judgement, round 3 (2026-09-21) — correctness of the new column on the real model: 2 hits, 0 false positives, 0 false negatives

Round 3 was not expected to reveal new errors (prior check in the cloud container: of the 11 citation misattributions only 2 fall within the 16-question component list,
and those 2 have coverage 0.000 / 0.111 / 0.200, already caught by the support sampling in §6.6).
**What this round tests is whether the new column itself produces false positives or false negatives**; the expectation was written down before the run and held.

**Finding 38 (Fact): the three rounds' outputs are verbatim identical; the new column hits 2/2 with zero false positives and zero false negatives**.
The output columns and the five original columns of the judgement layer show zero difference from round 2 (column 15 only had its header renamed, values unchanged);
paraphrase / synthesis / fabrication stay at 142 / 2 / 28. Manual row-by-row check of the 30 rows with a component hit:

| Row | Question | Citation | Flagged by new column | Judgement |
|---|---|---|---|---|
| r55 | `b2-q04` hybrid | `[2]` antibacterial spectrum, no dosage | dosing interval, course of treatment | ✓ correct |
| r110 | `b3-q12` | `[3]` hemophilia dosage table | recommended dose, injection interval | ✓ correct |
| Remaining 28 rows | — | cited passage does contain the hit component | empty | ✓ zero false positives |

**The cleanest contrast is r54 vs r55**: same question, same wrong sentence 「每日2粒…每6小時一次…5天」 ("2 tablets daily… once every 6 hours… 5 days"),
bm25 cites `[1]` (which really has a dosage passage) and the new column is empty; hybrid cites `[2]` (which does not) and the new column flags it.
The sentences are almost verbatim identical and differ only in the cited passage — **the new column's signal follows the passage entirely, not the sentence**,
which is exactly what Finding 37 designed it for.

**Finding 39 (Fact): mutual exclusivity of the two columns holds on real data, and they can split a single row into three states**.
Only 3 rows have values, and in none are both columns non-empty. `r33` is the only three-state row: three subject-type components fall into "hit but not cited",
the frequency component is cited and supported (both columns empty), and "hit but citation does not support" is empty — the four components are fully partitioned with zero overlap.

**Residual risk (Judgement, not quantified)**: "citation does not support" matches aliases against the passage source text, but the aliases were originally designed
from how the **model output** writes things. Part of this round's zero false positives was caught by the glyph normalization layer
(`r101`'s variant form 「延⻑」 ("extend") and `r122`'s 「點數:1500」 ("points: 1500") both hit thanks to NFKC + the CJK radical table + the thousands-separator rule);
but normalization only covers glyphs — **if a passage uses a semantically equivalent wording not in the aliases, you get a false positive of "citation actually correct but flagged as unsupported"**.
This is the inevitable cost of §6.6 N4 "semantic normalization prohibited", not a defect; the only mitigations are expanding the aliases or manual review,
and like the component-hit column this column **must not enter any score**.

## 6.11 Prompt v14, round 4 (2026-09-22) — two wording rules added, fixed one symptom, caused three regressions

v14 added two explicit wording rules to `configs/demo.yaml` (copy proper nouns verbatim, ban meta preambles).
124 of 178 rows differ in output from round 3; the verbatim-comparability premise of the first three rounds ends here, and what is compared is **the count of error types**.
**Conclusion: v14 is discarded**. Only one of four acceptance criteria passed.

| Item | Round 3 | v14 | Judgement |
|---|---|---|---|
| Meta preamble | 3 | **0** | ✅ the only one met |
| Drug-name transliteration | 1 | 2 (new error forms) | ❌ not zeroed; **zero reverse collateral damage** |
| Paraphrase / synthesis / fabrication | 142 / 2 / 28 | 133 / 1 / **44** | ❌ fabrication rate 16.3% → 24.7% |
| Component completeness | 51/62 = 82% | 45/64 = **70%** | ❌ |

**Finding 40 (Fact): the fabrication-rate regression is entirely in citation numbers; the content level is flat**.
Of the 44 fabrications, citation misattributions went 11 → **28**, non-citation fabrications 17 → 16. The new citation misattributions are almost all
"content correct, from another retrieved passage, but labelled [2] or [3]", very rarely wrongly labelled [1].
**Adding rules did not make the model less honest; it made it worse at counting numbers** — these two things get conflated under a single "fabrication rate",
proving again that the separate count of Finding 32 is necessary.

**Finding 41 (Fact): the "rule numbering confused with passage numbering" hypothesis was directly refuted by the file contents**.
The most intuitive suspect was that new rules written as `1. 2. 3.` would collide with `[1][2][3]`, but v14's system prompt is **prose with zero numbering**
(the `1.` pattern appears 0 times; the only square brackets are `[段落編號]` `[1]` `[3]`). The cost of checking was one regex;
had we run an isolation experiment on "changing the numbering format" without checking, we would have got an empty contrast with no treatment group.
The two surviving hypotheses became: **H1 position** (v14 pushed the citation rule from the last sentence of the system prompt to the middle),
**H2 length** (system prompt 134 → 248 characters, +85%, and citation is the only rule that must be executed sentence by sentence).
Both predict "errors land on later passage numbers", consistent with the measured distribution.

**Finding 42 (Judgement, strong evidence): the mechanism of misspelled drug names is character drift, not a translation decision, so prompt rules necessarily cannot govern it**.
v14 wrote methotrexate as 「甲基喋吟」 (a garbled rendering) and azathioprine as 「硫脈」 (a garbled rendering), and the same batch also produced garbage like 「C 肻口」
unrelated to drug names. All three are the same type: **character-level drift on rare characters under GPTQ-Int4**. A banned-expressions list cannot block it
(round 3's 「甲基軹酸酯」 did not hit at all this round — change the error form and it slips through), and a prompt can only govern "whether to translate",
not "writing a character wrong". **Resolution: route drug names entirely to a post-processing layer** (check drug names in the sentence against the cited passage and flag those not in the passage),
without relying on the model to obey rules; this also removes the rule in v14 that had no effect yet contributed 65 characters of system-prompt length.
**Reverse collateral damage confirmed zero by a programmatic scan of all 178 sentences**: every English word present in a sentence but absent from its cited passage comes from the question;
not one is a Chinese drug name from the passage forcibly translated into English.

**Finding 43 (Fact): v14 is not a regression across the board, and the details of "fix one symptom, break three places" are more useful than the total score**.
After the three preambles were removed: in `b2-q11` hybrid the preamble sentence had been the whole answer (all three drugs' English names correct),
so the whole sentence vanished and the question has no cited sentence left; the `b2-q11` bm25 answer drifted from clause II to clause IV;
in `b4-q13` bm25 the same sentence's citation changed from `[1]` to `[3]`, becoming a citation misattribution.
There were also four improvements: `b1-q19` bm25 for the first time fully answered both groups of subjects in one sentence (one group per passage, both cited —
the only synthesis this round, and the first complete answer to that question since round 1); `b1-q08` bm25 answered correctly for the first time;
`b3-q20` hybrid fixed condition flattening; `b2-q13` hybrid correctly covered clause X for the first time.

**Finding 44 (Fact): the xlsx has a structural traceback blind spot, and the same blind spot bit a second time**.
`tools/support_sheet.py` only writes sentences that "have a citation and coverage is not None"; the full answer is discarded after use —
the hybrid runs of `b2-q11` / `b4-q15` vanished from the sheet entirely, and "squeezed out by the preamble ban" versus "genuine abstention" are
**indistinguishable** on the sheet, so the -4 in component completeness could not be settled. The first time was Finding 37's population difference, same root cause.
Resolution: `--dump-answers` writes each (preset, item)'s full answer to jsonl, **independent of whether any row is produced**,
locked in as a test and mutation anchor (assertion: fake server reply with no `[n]` → zero rows, but the dump still has two entries).
It contains corpus source text, so it does not go into the repo.

**Next round v15_noprefix** (plan A approved 2026-09-22): v13 + **only the preamble-ban rule**,
deliberately inserted **before** the citation rule so that "label every sentence with [段落編號] (passage number)" returns to the last sentence of the system prompt (hedging H1),
and length falls back to 183 characters (hedging H2). Relative to v13 this is a single change, so attribution is clean; if citation misattribution returns to around 11 it is adopted;
if still high, even this rule is not worth it — revert to v13 and hand preambles to post-processing too.

## 6.12 Prompt v15, round 5 (2026-09-22) — citation misattribution falls back, but exposes a scoring blind spot present since round 1

v15 = v13 + only the preamble-ban rule, with the citation rule put back at the end of the system prompt. Comparing only the
172 (preset, question) combinations that have sentences in both rounds, to exclude interference from changes in sentence count:

| | v14 | v15 | Round 1 baseline |
|---|---|---|---|
| Fabrication | 43 | **30** | — |
| of which citation misattribution | 29 | **15** | 11 |
| of which non-citation | 14 | 15 | 17 |

**Finding 45 (Fact): citation misattribution falls back with the system-prompt structure, and the drop was not bought by turning answers into abstentions**.
In the questions holding v14's 29 citation misattributions, v15 has **no abstention at all**, so 15 is a genuine fix rather than removal from the denominator
(this reverse check is necessary — turning wrong sentences into abstentions also lowers the error rate, and that is worse behaviour).
Specific fixes: `b3-q12` hybrid, `b4-q13` bm25, the causal inversion in `b2-q08` hybrid,
the unit misplacement in `b2-q04` bm25. **H1/H2 are supported; v15 is adopted**.
Non-citation fabrication is flat but the affected questions changed (`r58` took Loperamide's mechanism as Metronidazole's,
`r117` picked a distractor as the injection interval, `r152` was right then wrong within the same answer),
showing this category is random generation-side error that does not move systematically with prompt structure.

**Finding 46 (Fact): decoding degradation around rare Chinese characters is a reproducible pattern, and question type forces it out**.
Real meta preambles: 0 sentences, but the cited sentence left in `b2-q11` hybrid is
「甲基軹酸鹵化物、硫 )))」 (a nonexistent drug name + garbage + truncation), with all three components missing.
bm25 on the same question also produced 「硫 )))嘌呤」, plus `r99`'s 「汎敏松」→「汃敏松」 and v14's 「肻口口」.
`b2-q11` is a hotspot because it **requires converting English drug names into Chinese** — exactly the pressure point for character drift (Finding 42).
A banned-expressions list cannot block it (a different error form each time); `terms.py`'s `missing_verbatim` was designed for this:
it does not guess what it was translated into, only reports "the passage has X, this sentence did not copy X verbatim".

**Finding 47 (Fact, most important): abstention sentences have been silently dropped since round 1, leaving 20–30 (preset, question) combinations out of every denominator**.
The root cause is a single behaviour: `sentence_support` in `src/selectrag/demo/core.py` does `continue` on sentences containing 「參考資料未提及」 ("not mentioned in the references").
Number of combinations with "no sentences at all" on the sheet: **round 1 32 / v14 25 / v15 21**.
All 21 in v15 are standard abstentions (zero exceptions), and 18 of v14's 25 are also abstentions in v15 —
so v14's "two vanished questions" are most likely filtered-out abstentions, and completeness and support have only ever been computed on "questions that were answered".

**Abstentions themselves can be wrong, and quite often**: of the 21 abstentions, **at least 11 have the ground-truth chunk within the retrieved results**,
some ranked first, some even with the model's cited number pointing right at that chunk (`b4-q21` bm25's 01040C points,
`b3-q04` Dextromethorphan 10mg in both presets). These are **wrong abstentions**, entirely invisible to the current scoring.
Resolution direction: put abstention into the sheet as a sentence type and apply the same support check ("does the cited passage really lack the answer"),
listing wrong abstention as a separate error type alongside fabrication.
This is the same shape as §6.8 Finding 32 (citation misattribution must be counted separately): **an error type merged into someone else's denominator disappears as a whole class**.

**Finding 48 (Judgement): the "missing restriction" in `b4-q15` is not an omission; it actively denies a fact present in the passage**.
The jsonl shows the full answer's second sentence is 「參考資料未提及其他限制。」 ("the references do not mention other restrictions.") — without a citation marker it does not enter the sheet,
while the cited `[1]` explicitly states 「每一個案每一院所支付以一次為限」 ("payment is limited to once per case per institution"). This is **the reverse of the same mechanism** as the existing observation "when a format slot lacks material, the LLM upgrades material to fill it": there it was "falsely claiming presence", here it is "claiming absence".
The other half of the paired rule (ban claiming absence) has not yet been written into this project's demo prompt.

**Correction (2026-09-22, manual review)**: the statement in §6.11 that "`r168`'s 23 weeks was probably borrowed from P4202C's claim interval"
**is wrong**. This round `r174` cites Annex 2, whose source text is 「可參用23週以內之院外CT、MRI或腹部超音波檢查結果」 ("outside-hospital CT, MRI or abdominal ultrasound results within 23 weeks may be used"),
so 23 weeks is the correct answer; the defect in those two v14 rows is **only in the citation number; the value is right**.

## 6.13 Round 6: new baseline (2026-09-22) — abstentions enter the sheet for the first time; one dump round catches three sheet-layer defects

**This is a new baseline**; the fabrication-rate denominators of the first five rounds are not comparable with it (approved 2026-09-22 not to back-label the first five rounds).
206 rows = 185 answers + 21 abstentions. The prompt is still v15_noprefix; only the sheet changed.

**Finding 49 (Fact): "same outputs, only the sheet changed" established by two-way comparison**. The 185 answer rows, with citation markers stripped sentence by sentence,
compared against the jsonl answers: **0 rows mismatch**; the jsonl itself has **zero differences** from round 5's 200 entries.
The following three acceptance checks can therefore be attributed to the sheet layer, independent of the model.

**Finding 50 (Fact): reverse acceptance holds — the "foreign term" column has 0 rows, matching manual recomputation**.
This column's acceptance was deliberately designed in reverse: v14 had scanned all 178 sentences and confirmed that English words "present in the sentence but absent from the passage" **all come from the question**,
so once the question is listed as a legitimate source this column should be nearly empty. Manual recomputation from jsonl against passages and questions gave 0 rows, matching the sheet.
**"The column has no signal" and "the column is broken" can therefore be told apart** — this design is more useful than forward acceptance.

**Finding 51 (Fact): wrong abstentions settled at 11/21, and most cite exactly the passage containing the answer**.
Of the 21 abstentions, 11 have a ground-truth rank; manual one-by-one confirmation shows **all are wrong abstentions** (the ground-truth passage really contains the answer);
the other 10 did not retrieve the ground truth, a retrieval failure rather than a generation failure. This whole class was previously in no denominator (Finding 47).
**2026-09-29 exclusion check**: the 11 ground-truth chunks have zero overlap with the 7 questions affected by "heading_path containing >40 characters of body text misclassified" (a known extraction defect in `data/corpus/SOURCES.md`),
so it is not the case that the answer was hidden in the heading and invisible to the model; wrong abstention remains attributed to the generation side.

**Finding 52 (Fact): `--dump-answers` caught three sheet-layer defects in its first round, two of them "function right, call site wrong"**.
Without the jsonl, all three would be invisible:

| Defect | Root cause | Impact |
|---|---|---|
| **Mixed type invisible** | Extra rows are produced only under `if not sups`. `b4-q15` hybrid is "first sentence cited + second sentence claims absence"; sups is non-empty so it never triggers | The Finding 48 type remains invisible on the sheet |
| **Abstention rows drop the citation number** | The citation column of abstention rows is hard-coded to `""` | Of the 22 containing the abstention phrase, **21 carry `[n]`**; the strong signal "cited a passage yet says it is not mentioned" is lost wholesale |
| ~~Rank miscomputed~~ | **Does not hold**, see below | — |

The first two are the same type: **`is_abstain` tests "contains" rather than "equals" precisely for the mixed type; the function layer is right, but the call site uses `if not sups`
and shuts it out**. The corresponding property guards the function's semantics but cannot guard the call site — **an invariant's scope is where it is called**;
this is the most worth-remembering lesson of this round.

**One correction (2026-09-22, refuted on verification)**: a manual report said "`b1-q24`'s rank should be 1 not 2, because the question has two ground truths".
Checking eval set v1.2: `b1-q24`'s `relevant_doc_ids` has **only `#0000`**, while bm25 ranks `#0001` 1st and
`#0000` 2nd, so `truth_rank` returning 2 is correct. Whether `#0001` (the announcement page the question came from) should count as ground truth is a **domain determination**;
including it would require changing the eval set (v1.3, new hash, rerun axis A representative points), not a defect of this column.
**The correct semantics of this column is "minimum rank of the ground truth recorded in the eval set"**; its accuracy is capped by the completeness of the eval set's ground truth.

**Recorded limitation**: the "in passage but not copied verbatim" column has **141 rows with values** on the real corpus; its false-positive rate is too high to be useful as a filter
(it flags whenever a passage lists several drug names and the sentence answers only one). It is effective for the `b2-q11` type (English drug names written as Chinese garbage),
but needs manual filtering of most of the noise first; **it must not enter any score, nor should it be used for ranking**.

**Mixed-type rows deliberately not done (approved 2026-09-23)**: Finding 52's first defect — "first sentence cited + second sentence claims absence"
still invisible on the sheet — **was evaluated and not fixed**. The reason is return on effort: only **1 case** was measured (`b4-q15` hybrid),
and `--dump-answers` already makes it visible (it was found manually from the jsonl), so adding rows to the sheet would only present again
what is already visible; the cost would be sentence-splitting logic (cross-layer import of `core`'s private function, or writing a second copy, or changing the public API and breaking
the "do not change core" constraint) plus a 100-minute gauntlet run. The second defect (citation number in abstention rows) was fixed instead,
because it masked 21 entries and needs no sentence splitting. **The two defects in the same fix spec differed in cost-effectiveness by an order of magnitude; bundling them into one question was wrong**.
Re-evaluation trigger: if a "ban claiming absence" rule is later added and the mixed-type count rises, do it then.

## 6.14 Making the ground-truth criterion explicit (2026-09-23) — `#0001` for `b1-q24` is not added; the criterion is fixed as "text contains the answer"

The sixth-round manual report claimed "`b1-q24` has two ground-truth chunks, and the rank should be 1"; verification overturned the rank part (correction in §6.13),
but left a real question: should `#0001` be added to the ground truth. Verification took four steps (a single-question replay of the Decision 5 construction process):
① reduce the question to its answer element (application deadline = one date string) ② inspect the candidate chunk's `text` and `heading_path` **separately**
③ criterion: a chunk counts as equivalent only if its text contains the answer ④ grep the whole corpus for equivalent phrasings to confirm there is no third chunk.

**Finding 53 (Fact): the answer for `#0001` exists only in heading_path; its text is contact details and file downloads**.
Full-corpus scan: 「115-09-01」 appears in text only in `#0000`; 「115年9月1日」 ("September 1, ROC year 115") appears in the text of two other chunks,
but they are respectively **the revision date of a drug checklist** and **the effective date of a dental program** — the same date by coincidence, different events, natural hard negatives.
So under the "text contains the answer" criterion, this question has **zero missed ground truth**, the `truth_rank` of 2 is correct, and nothing needs re-running.

**Decision (approved 2026-09-23): do not add `#0001`; the criterion is made explicit as "ground-truth chunk = text contains the answer (can be answered on its own)"**.
The rejected alternative criterion was "retrieval-layer equivalence" (the heading is in BM25 context_fields, so the retriever finds it via the answer string):
it was rejected because the purpose of recall is to predict "generation can get the answer", while `make_messages` only sends text —
given `#0001`, the model cannot answer the date. Adding it would decouple retrieval scores from end-to-end performance,
and would amount to using the ground-truth set to absorb chunking damage (an answer cut into the heading = another facet of beheaded chunks, Honesty section item 9),
masking the value of the axis A parent-child to-do. **The lesion is in chunking, and so is the fix — not in the annotation.**

**Methodological takeaway**: this criterion was an implicit premise of Decision 5 all along; this case forced it to be written down. Any future dispute over "does this chunk count as ground truth"
goes through the four steps above first; step ②, "look at text and heading separately", is the key — the two are mixed into the same index at the retrieval layer,
but only text exists at the generation layer, so looking at them mixed inevitably leads to misjudgement.

## 6.15 prompt v16 round seven (2026-09-23) — the no-claiming-absence rule backfired catastrophically; discarded, reverted to v15

v16 = v15 + one rule: "before answering 「參考資料未提及」 ("not mentioned in the references"), you must check paragraph by paragraph that the information really is absent;
content already stated in the reference paragraphs must not be claimed as not mentioned". Retrieval results for the 200 combinations, compared hit by hit, were **identical** to round six;
all changes were on the generation side. **Conclusion: v16 is discarded; the demo prompt is finalized as v15_noprefix.**

**Finding 54 (Fact): the rule's effect was completely inverted — abstentions 22 → 66, and all 44 new ones are "questions that were answered normally becoming pure abstentions"**.
Typing (judged per sentence after stripping citation markers): run6 = 21 pure abstentions + 1 trailing sentence; run7 = **65 pure abstentions** + 1 trailing sentence.
The 44 new ones include questions answered correctly in round six (Tetracycline mechanism, LVEF≤40%, coagulation factor dosage…),
abstention rate 11% → 33%. Meanwhile **zero of the existing 11 wrong abstentions were fixed**; the trailing sentence of the only target case, `b4-q15`,
merely changed from 「未提及其他限制」 ("no other restrictions mentioned") to 「未提及。」 ("not mentioned."). **This verdict needs no manual review**: refusing questions that could previously be answered
is a regression regardless of the new answer's content, so a mechanical comparison settles it.

**Finding 55 (Judgement, strong evidence): for a 7B quantized model, this kind of meta-check instruction actually acts as string priming, not a behavioural check**.
The rule text **repeated** 「參考資料未提及」 yet again in the system prompt (three times in total), and drew attention toward
"the possibility that it cannot be found"; the model's response was not to perform the "paragraph-by-paragraph check", but to **output the repeatedly emphasized string more often**.
This mirrors the same family of mechanisms as the existing slot-filling observation: the rule text itself becomes the prompt.
Combined with the v14 lesson, the feasible room for demo prompt changes narrows to: **only "format/phrasing" constraints can be added (the no-preamble rule succeeded);
"ask the model to check itself" meta constraints cannot (verbatim drug names failed, no-claiming-absence backfired)**.

**Handling**: ① `configs/demo.yaml` reverted wholesale to v15 (output behaviour = round six, no verification rerun needed)
② guarding against wrong abstentions **gives up on the prompt layer; decided to go via the deterministic layer** — the abstention rows of the support table + the ground-truth rank column
(built 2026-09-22) can already bring "should have retrieved but abstained" to the front of human attention; in demo presentations wrong abstentions 11/21
are listed as a known limitation, stated as is ③ mixed-type count run7 = 3 (run6 = 2), below the re-evaluation threshold; the no-fix decision stands.

**Failure log for expectation management**: the pre-work expectation was "wrong abstentions go down, not expected to reach zero"; the actual direction was the exact opposite.
The error was extrapolating "rules can narrow discretion" (prior experience that holds for output **format**) to "rules can add checking behaviour" —
the latter does not hold for a model of this size, and no prior round of data supported that extrapolation.

## 7. Honesty section (combinations expected to win that lost)

1. **"Indexing heading_path fixes missed retrievals" — overturned by evidence** (2026-09-01). Error analysis pointed to "keyword not in body text";
   the intuitive fix was to index the heading path; pre-work verification found the missing drug names were not even in the heading path, only in the document's first line. Without verifying,
   we would have built an ineffective experimental arm that could show no difference. Fix: backfill doc_title at build time.
2. **"BM25 with textbook defaults (k1=1.5, b=0.75) is good enough" — lost by 4.2 percentage points** (2026-09-01).
   In this corpus chunk length is already constrained by the chunking parameters, so length normalization instead penalizes long chunks containing complete provisions;
   term-frequency noise in Chinese bigrams favours a low k1. Lesson: a baseline's "defaults" are also an unverified assumption.
3. **"Three property invariants are enough to lock down hit@k" — mutation testing proved otherwise** (2026-09-01).
   ≥recall, monotonicity and range are all one-sided bounds; a wrong implementation that "always reports a hit" passes them all;
   only re-running mutation testing on the property subset alone revealed it. Fix: add the two-sided lock hit == (recall > 0). Methodological lesson:
   a bound on one side must be paired with one on the other.
4. **"NIM cannot run on WSL2" — the stop-loss call of the first timebox was overturned by the second timebox** (2026-09-15).
   At gate 2 the container failed on startup with `CUDA failed to initialize [[ Named symbol not found (error 500) ]]`, TRT segfault;
   one fix attempt, "remove /usr/local/cuda/compat", did not work; the two hours ran out, and it was recorded as a "WSL2 + Docker Desktop environment-layer problem" stop-loss.
   After an extra one-hour timebox, the first action, "update Docker Desktop" (Engine 24.0.7 → 29.8.0), fixed it directly; the same command brought the service up.
   Lesson: the stop-loss call conflated "host environment unsupported" with "host software too old"; a version check is the cheapest and should be the first suspect for any container GPU problem.
   The NGC support matrix's 23 profiles include no GeForce, but the 4090 uses the L40S engine directly via sm_89 compatibility; "not in the support matrix" does not mean "cannot run".
5. **"NV-Embed-v2 can be compared side by side with bge-m3" — failed to load under the pinned transformers 4.57.6; abandoned** (2026-09-15).
   Model (remote code @ 3fa59658) download and weight loading were fine (fp16, 4 shards), inference crashed immediately: first issue,
   `DynamicCache.get_usable_length` had been renamed, patched back with a shim; second issue, the remote code's hand-written decoder loop does not pass
   `position_embeddings`, while in transformers ≥ 4.48 Mistral attention has the upper layer compute rotary and pass it in,
   so `cos, sin = position_embeddings` unpacks None. Fixing this means editing the model file; downgrading transformers conflicts with the
   4.57 bound to vLLM 0.22.1, and a third venv just for one comparison arm is not worth it. Another practical cost: the NIM reranker, when resident, takes 18.5 GB VRAM
   (Triton workspace) and cannot coexist with the 7B generation model, while bge-reranker in-process takes about 2 GB.
   Conclusion: axis A embedding stays bge-m3 only; the NV signal is carried by the NIM reranker comparison (§2.6).
   Lesson: compatibility of a `trust_remote_code` model is determined by "the transformers version the model file targets", independent of pinning; day-one verification must come before pinning.
6. **"TensorRT-LLM will be clearly faster than vLLM on the 4090" — lost under bf16, only 6% ahead under FP8** (2026-09-15).
   At project start axis B was treated as the main NVIDIA storyline, expecting the engine gap to be the main variable; measured across three concurrency levels, the bf16 gap was ≤ 10% with vLLM slightly ahead,
   only FP8 let TRT-LLM overtake by 6%, while FP8 itself brings ×1.4–1.7. The real variable is numeric format, not engine;
   a workload of 4096 context, 45-token short answers and concurrency ≤ 8 is also far from exposing scheduler differences between the two.
   Another surprise: FP8 changed nearly half of greedy outputs (§3.2 Finding 20); a "speed table" without a quality table would mislead.
- (to be backfilled with later experiments)
7. **"FP8 is the best format on Ada" — overtaken again by W4A16 at ×1.6** (2026-09-17). Axis B treated FP8 as the endpoint (×1.4);
   once axis C ran the official AWQ / GPTQ, c=8 throughput was 1000 tok/s vs FP8's 610; and GPTQ's faithfulness of 0.960 is indistinguishable from bf16 under the judge.
   The original assumption "4-bit must cost more quality than 8-bit" was not seen by the judge; what was seen was only "verbatim less like bf16" (agreement rate 0.25 vs 0.51),
   and the two are not the same thing — this is the project's second "output changed ≠ quality dropped" (the first was §3.2 Finding 20 vs the §3.3 quality addendum).
8. **"Four more explicit rules will fix the judge" — zero leverage** (2026-09-18). After the stratified calibration set knocked v2's κ from 0.845 back to 0.488,
   v3 wrote one rule for each of the nine error types; re-running the 42 items changed not a single verdict. Lesson: rules can fix "didn't check", not "checked and judged the wrong way round";
   the latter needs a structural change (split into sub-questions) or a model change, not rewording.

9. **"Since heading_path helped retrieval, using it as generation context should help too" — abandoned after spot checks, without even running the experiment** (2026-09-21).
   The citation misattributions (10/28) in the first component scoring round pointed to a causal chain: the `1.` numbering in regulations is exactly the heading style,
   body lines were misclassified as headings by `parse_heading` → `chunker.py` puts heading lines only into `heading_path`, not `text`
   → BM25's `context_fields` adds it to the index (**retrieval can find it**) → but `make_messages` only sends `h.text`
   (**the model cannot see it**). The most direct fix is to merge `heading_path` into the context, and it is reversible and needs no corpus rebuild.
   **Before acting we first verified the assumption "the paths are of good enough quality", and the assumption was overturned**: spot-checking 12 chunks already gave hard evidence —
   `1 MB` (a file size) treated as a heading, `5 ∼ 10倍之多。` (body text) treated as a heading, a biomarker-testing path attached to epilepsy drug provisions;
   across the whole corpus, the median character 2-gram overlap between the last-level heading and `text` is only **5.6%**, and 44.6% have no overlap at all
   (this metric overestimates mismatch — headings are inherently more abstract than body text — but the hard evidence from spot checks is unaffected).
   **Conclusion: heading_path is useful for retrieval and harmful for generation**. Retrieval only needs lexical overlap (a drug name in an upper-level heading is enough),
   generation needs semantic correctness; feeding 57.5% low-overlap paths in as context is feeding noise that "looks like context",
   and the model will trust it. Axis A's ctx gain (0.829→0.905) still holds, but it is a **bag-of-words effect**,
   and does not mean those paths are suitable as context. Lesson: the same field has different quality bars at different layers of the pipeline;
   "useful at layer A" does not imply "useful at layer B".
   **Cost assessment**: the root fix requires fixing `parse_heading` → recomputing 8,029 chunks → invalidating all axis A/B/C numbers,
   with no way to know in advance whether recall goes up or down. Handling: first trial-fix only the document with the largest gap and measure the recall change (see §2.x, to be filled),
   and let the result decide whether to pay the rebuild cost.
   **Trial-fix experiment done (2026-09-21, see `data/corpus/SOURCES.md`)**: adding two rules, "heading length cap + full-stop block", to the document with the largest gap
   took the gap from 32.6% → 10.2%, and swallowed body text returned to `text`.
   But the threshold is extremely sensitive (`max_len=40` rescues `Dravet`, `45` does not), the gap is non-monotonic,
   and **the more is blocked, the emptier `heading_path` becomes** — axis A's ctx gain relies precisely on headings entering the index.
   Measuring the net effect requires looking at recall, and after recomputation `chunk_id` changes so ground truth no longer matches
   → **the cost of scouting is already close to the cost of the root fix**. Not fixing stands; this number itself is the cost assessment.
   (Incidental correction: the gap ratio was previously computed as 22.3%, with a denominator of "gap + coverage", but chunks overlap,
   so summed coverage exceeds the full-text length; the correct value with `len(document.text)` as denominator is **31.6%**.)

## 8. Appendix

- How to re-run: the config files under `configs/` + `bash tools/gauntlet.sh` (quality gate)
- eval set version history: v1-batch1~4 → v1-full100 (hash bdea697a) → v1.1 (859b0d7a, reverse spot-check batch 1 +4 ground truth)
  → **v1.2 (4c0761ca, 2026-09-15, truth_dupes scan +2 ground truth, 129 items)**; hard negatives 42 → 89
  (three primary label types: adjacent code with same note 41 / same code old rate 1 / adjacent chunk 1; they differ in difficulty for the reranker and are counted separately);
  §2.3–2.6 numbers are v1.1; the four representative points in §2.7 were re-run with v1.2 on 2026-09-15 (difference ≤ 0.003)
- Axis C re-run results `results/bench/vllm_{awq,gptq}[-long]_c*.json`, `results/judge/vllm_{awq,gptq}.json`
- Ground-truth duplicate scan: `python3 tools/truth_dupes.py data/eval/eval_v1.2_full100.json results/chunks.jsonl <out.json>`
  (on v1.2: 129 ground truth / 37 with candidates / 80 candidates / 22 already ground truth)
- Development history: `docs/devlog.en.md` (English) / `docs/devlog.md` (Chinese)
