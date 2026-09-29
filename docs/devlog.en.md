# SelectRAG development log

**English** | [繁體中文](devlog.md) · *Translated from the Chinese original; if the two differ, the Chinese version is authoritative.*

> Purpose of this material: the project narrative for a personal job-hunting website. Failures and detours are the most convincing part, so they are kept as they were.

## 2026-08-31 — First real eval numbers: BM25 baseline on a 25-question real eval set

**Trigger**: With the corpus pipeline (8,029 chunks) and the BM25 retriever done, what was missing was "real questions + human-judged ground truth".
Without it, every retrieval number was just smoke from synthetic data. Goal: build batch 1 (25 questions) of the 100-question eval set and freeze it.

**What we tried**:
- First designed an "AI writes questions + BM25-assisted exhaustive search for equivalent chunks + human row-by-row judgement" process (option A),
  rejecting purely human-written questions (5× the cost) and automatic LLM ground-truth judgement (ground-truth contamination cannot be fixed after the fact).
- Produced a 255-row annotation sheet (25 questions × about 10 candidates each), judged ✓/✗ row by row by hand.
- The judging itself caught three toolchain traps: ① the preview showed only 160 characters, so a source chunk whose answer sat at character 219 was wrongly marked ✗;
  ② one chunk's answer (the submission deadline) existed only in the heading path, not the body, while the index only took the body;
  ③ the same procedure code had two point values in different effective periods, so a question that did not specify the period had two mutually exclusive correct answers — the question was rewritten to converge.

**Key trade-offs**: The ground-truth set is not "one question, one chunk" — the same information often appears in multiple documents
(package inserts for other brands with the same ingredient; reimbursement programmes and the fee schedule restating each other). If equivalent chunks are left unlabelled,
a later dense retriever that finds them gets wrongly marked as incorrect, directly distorting the axis A comparison.
So we preferred having a human scan 200+ extra ✗ rows over missing any ✓.

**Outcome**: Batch 1 frozen (hash aa91a0b26af1), BM25 baseline:
recall@1=0.66, recall@3=0.82, recall@10=0.86, MRR=0.786.
All four misses were successfully attributed, and each represents one structural weakness:
- q04 crowding by synonymous documents (same-ingredient chunks from other brands can't get into the top-10)
- q07/q24 loss of heading information (answer is in heading_path, index only takes the body) — same root cause twice
- q11 Chinese–English vocabulary gap (「雌激素經皮貼片」 ("estrogen transdermal patch") vs "Estradiol Transdermal Patch")
- q16 insufficient lexical overlap (「如何處理」 ("how to handle") vs 「簽名黏貼於病歷表」 ("sign and attach to the medical record"))

This list of weaknesses is the hypothesis source for "where the expected gains come from" in the later dense/hybrid comparison experiments —
the baseline's weaknesses are the experiments' selling points. Open issues: whether heading_path should go into the index (pending comparison experiment);
sampling bias from building ground truth with BM25 assistance (batch 2 mitigates this with multi-round keyword-rewritten queries).

## 2026-09-01 — The day a hypothesis was overturned by evidence: heading_path can't fix the misses, doc_title can

**Trigger**: Miss attribution for BM25 on the 50-question eval set showed "the answer's keywords are not in the chunk body" as the dominant failure
(drug names from package inserts and section semantics are not in the post-chunking body). Original plan: add heading_path to the index.

**What we tried**: Verified before starting — and the premise was overturned: the "drug name" the missed questions lacked is not in heading_path at all,
nor in doc_id (that is the licence number); it is only on the **first line** of each document. Implementing without verifying
would have produced an invalid experimental arm that shows no difference. Direction changed to: add doc_title to the Chunk schema
(extracted from the document's first line at build time), and give BM25 a context_fields option that can attach doc_title and heading_path.
Added alongside: the hit@k metric — when the ground-truth set contains several equivalent chunks, recall is diluted
(finding 1 of 5 scores only 0.2); hit@k answers "was the answer retrieved at all".

**Key trade-offs**: ① The control arm must not be contaminated — context_fields defaults to empty, and the change passes only if the control arm's 50-question numbers
are bit-for-bit identical to before; ② mutation testing re-run on the property subset alone revealed that hit's three
invariants were all one-sided (≥recall, monotone, value range) and could not stop an implementation that "always reports a hit" —
only adding the two-sided lock hit == (recall > 0) killed it; ③ rejected option: derive the title from chunk sequence #0000
(zero schema change, but relies on implicit ordering coupling).

**Outcome** (see screenshots/20260901_bm25_ctx對照_gauntlet全過.png):
50-question comparison, recall@10 0.844 → 0.916, hit@10 0.88 → 0.92; of three targeted missed questions two were fixed
(q03 recall 0.2→0.8, q04 0→1) plus one unexpected gain; q06 still misses (kept honestly).
The cost was as the pre-mortem predicted: recall@3 dipped slightly (0.776→0.773) — document context improves deep recall
while slightly hurting early precision, which is exactly the first trade-off in the axis A memo.
gauntlet: 208 tests, mutation 45/45, supply chain at zero, all layers passed.

## 2026-09-01 — All 100 eval-set questions frozen: the foundation for axis A experiments is done

**Trigger**: The "100 questions, human-judged ground truth, frozen in batches" goal set at project start was completed in four batches.

**Process numbers**: The four batches produced 1,151 candidate annotation rows, all human-judged; the suggestion-column error rate was
1/255, 1/298, 3/301, 13/297 — broken down, the jump in batch 4 was 10 rows from two bad questions
(the same rule appears repeatedly in 5~7 chunks in the corpus, so a single correct answer can't hold); real judgement differences were only 3 rows.

**Process evolution (each batch more accurate than the last)**: batch 1 single-round queries → batch 2 three rounds of rewritten queries to widen the equivalent-candidate net
→ batch 3 excerpts show full text (eliminating "presumed ✓") → batch 4 a "review priority" column (must-read/spot-check) halving human effort.
The lesson of the two bad questions became a question-writing rule: before writing a question, scan "how many times the answer sentence appears in the corpus"; if more than twice, ask for a unique value instead.

**Final numbers (100 questions, v1-full100, hash bdea697a)**:
BM25 body-only recall@10=0.829 / hit@10=0.860; +doc_title+heading_path 0.905 / 0.920.
By-product: 31 human-labelled hard negatives (after cleanup) (contradiction type, text fully correct but wrong drug, number correct but wrong entity),
ready-made test items for later faithfulness evaluation — the "text fully correct, source wrong" trap requires even a human to use the
chunk_id to judge correctly, which is exactly what RAG citation discipline has to guard against.

## 2026-09-01 — Textbook defaults are not defaults: BM25 k1/b grid tuning

**Trigger**: After the 100-question eval set was done, a fair dense comparison needed a "tuned" BM25 —
the textbook defaults k1=1.5 / b=0.75 had never been validated on this corpus.

**What we tried**: A coarse 6×5 grid (k1 0.5–2.2 × b 0–1) found the body-only optimum on the b=0 boundary;
following the "take the plateau midpoint, not the boundary" principle, a second finer round (k1 0.2–0.7 × b 0–0.3) confirmed a plateau spanning b 0–0.1, k1 0.4–0.6.
The ctx version (+document context) had a wider plateau instead (k1 0.3–1.1 × b 0.5–0.75), and the textbook defaults were already on it —
explaining why it "didn't seem to need tuning". When the runner gained k1/b pass-through, the first test version was a no-op (the assertion had no discriminating power);
we caught it ourselves and rewrote it as "ranking flips when b=0" before it counted.

**Key trade-offs**: ① Use recall@10 to pick the plateau, MRR as the tie-breaker; ② tuning and evaluation share the same 100 questions
with no held-out set — we chose to disclose this honestly as an "upper-bound estimate" rather than carve out a validation set (splitting 100 questions would leave both sides too small);
③ tuned values do not change the code defaults; they live in a separate config file, so the control arm's definition stays uncontaminated.

**Outcome**: body-only k1=0.5/b=0.1 → recall@10 0.829→0.870; +ctx k1=0.7/b=0.5 → MRR 0.732→0.808.
The root cause is b: chunk length is already bounded by the chunking parameters, so an extra length penalty actually hurts long chunks containing complete provisions.
Open: when dense comes in, both sides must be "tuned vs tuned" to be fair.

## 2026-09-08 — Quantization format effects can't be separated from the engine: trade-offs in axis C experiment design

**Trigger**: While surveying the NVIDIA stack we found a structural problem in axis C (AWQ / GPTQ / FP8) —
every quantization number is the product of "format × engine kernel × engine scheduling"; FP8's official main path is
ModelOpt → TensorRT-LLM, vLLM's support for Ada FP8 was unconfirmed, and the two engines don't even share "the same weights"
(checkpoints need separate conversion). Without fixing the engine, axis C would steal axis B's conclusions.

**Three options**: ① single engine, three formats (lock TRT-LLM; cleanest causality, but conclusions hold only for TRT-LLM);
② full matrix of two engines × three formats, with cells that can't run honestly marked N/A (most complete; "format is bound to engine" is itself
knowledge an SA should give customers, but ×2 time and still no diagonal comparisons); ③ boring-but-safe: a 2×2 on the AWQ/GPTQ intersection,
FP8 only as a single-point note on engines that support it (guaranteed to finish, but downgrades the most NVIDIA-flavoured cell).

**Key trade-offs**: Decided "② as the target, ③ as the floor" — first finish the 2×2 intersection as a guaranteed deliverable, then add FP8 and
the empty cells; reporting iron rule: compare formats only within one engine and engines only within one format; no cross-engine rankings.
Before starting we wrote a pre-mortem: the most likely challenge six months later is "the FP8 win happens to equal the engine gap measured in axis B";
the defence is labelling the engine on every table and never ranking across engines. Another often-overlooked dimension: quantization affects not only speed but also
generation quality, so the §5.3 faithfulness judge became a prerequisite of axis C — otherwise the only conclusion would be the useless "smaller is faster".

**Decided in the same session**: axis A tests both embeddings (bge-m3 saves VRAM; NV-Embed-v2 carries the NV signal but is 7B-class,
index built offline by default); NIM / NeMo Retriever enter the experiment with a two-hour timebox — if consumer cards aren't supported, switch to an open-source reranker.
Open: everything awaits day-one verification on the 4090 desktop (the seven-step order is written into requirements-gpu.txt).

## 2026-09-13 — First cross-machine reproduction: everything developed on cloud CPU matched on the 4090 desktop with one command

**Trigger**: The GPU-free parts of axis A (corpus, eval set, BM25 baseline, gauntlet) were all done in the Claude Code
cloud container; not a single line had run on the desktop. Today's goal was to prove "GPU-free conclusions carry over unchanged to the GPU machine";
otherwise later dense / hybrid numbers would have no credible control arm.

**What we tried**: ① git clone only brought the code and SOURCES.md — the corpus files were gitignored all along and lived only in the disposable container;
we briefly assumed "cloning should include them". ② Re-fetching via SOURCES.md was rejected: chunk_id is the sha256 of doc_id + sequence + body,
the NHI Administration HTML had been updated, and mismatched re-fetched bytes would invalidate the entire 100-question ground-truth set. ③ In the end the cloud packed the 42 original files
(79 MB; attachment limit 30 MiB per file, split into three parts); the desktop merged them, compared sha256 and then extracted.
④ Three small environment-layer pitfalls: `wsl` defaulted to docker-desktop instead of Ubuntu, Ubuntu 24.04 has no python3.11
(deadsnakes PPA), and gauntlet piped through `| tail` produced 10 minutes of zero output, leading to a mistaken Ctrl+C.

**Key trade-offs**: The desktop already had a separate conda environment; we chose "a single WSL distro, one venv per project"
over another distro — after `conda deactivate`, build `.venv` with python3.11, package versions pinned by requirements.
The corpus backup issue surfaced: gitignore keeps the repo clean, at the cost of the corpus having no version-controlled copy, relying only on manual backups.

**Outcome**: 8029 chunks (csv 6350 / html 19 / pdf 1660), 123/123 ground truths present, the four BM25 configs'
recall@10 0.8287 / 0.8703 / 0.9047 / 0.9047 identical to the cloud digit for digit, gauntlet passed all layers (218 tests,
99% coverage, 47/47 mutation, pip-audit at zero). Credibility of cross-machine reproduction comes from three things: all package versions pinned,
the corpus verified by hash, and the baseline being one command. Next: day-one GPU package verification and pinning `requirements-gpu.txt`.

## 2026-09-14 — Three axis A ablations in one day: neighbour-chunk expansion rejected, hybrid fixes misses, reranker fixes ordering

**Trigger**: dense (bge-m3)'s first recall@10 of 0.911 beat tuned BM25 by only 0.006, and spot-checking failure modes found four classes
(drug-name blindness, adjacent chunks, same semantic field but wrong document, whole-question collapse). The axis A question shifted from "is a vector DB worth it" to "which countermeasure eats which failure class".
Decided order: chunking first, then hybrid + rerank, each step benchmarked on the same 100 questions (v1.1; after reverse spot-check, ground truth 123→127).

**What we tried**: ① retrieval-time neighbour-chunk expansion (bring back the 1~2 chunks before and after each hit) — BM25 and dense regressed across the board,
recall@3 dropping 0.10~0.13: with the k budget locked, the gain from 7 adjacent misses couldn't offset original hits being pushed off the list; rejected the same day.
② hybrid: first computed "union of both top-10s recall 0.975" before the grid — the union is 0.07 higher than either side alone, proving the two retrievers miss almost non-overlapping questions,
which is the theoretical room for fusion; in the RRF grid the three rrf_k cells formed a plateau, pool was the main variable and a small pool was actually better; representative point pool=12 gave recall@10 0.963.
③ reranker (bge-reranker-v2-m3) re-ranking the top N of hybrid: N ∈ {10,20,30} was mountain-shaped, peaking at 20; recall@3 0.855 → 0.943,
MRR 0.837 → 0.921, recall@10 only +0.012 — hybrid gets things into the top 20, the reranker pushes them into the top 3; the two work on different metrics.
The ground truth beat 42 hard negatives in 93% of checks.

**Key trade-offs**: Before each ablation we wrote a falsifiable hypothesis and a pre-mortem — hybrid's pre-mortem required
"the gain must exceed a single side with an enlarged pool"; hybrid@10 (0.963) beat single-side recall@20 (0.958), ruling out "just looking at more".
The reranker's trade-off is latency for recall: candidates 20 vs 10 costs ~116 ms more per question for recall@10 +0.012; axis A's main metric is recall, latency is left to the performance axis.
Another signal to read separately: doc_title raises recall on the indexing side but only raises MRR on the reranking side — the same field is a different mechanism at the two stages.

**Outcome**: axis A mainline numbers settled at BM25 0.905 → dense 0.911 → hybrid 0.963 → reranker 0.975 (recall@10),
head recall@3 0.855 → 0.943. Same question set, no held-out, all marked as upper-bound estimates.
Open: NIM reranker two-hour timebox, axis A trade-off conclusion, reverse spot-check batches 2~4, a scanning tool for "how many highly similar chunks a ground-truth chunk has in the corpus".

## 2026-09-15 — Axis B finished in one day: the engine isn't the variable, the numeric format is

**Trigger**: At project start axis B (vLLM vs TensorRT-LLM) was treated as the main line of the NVIDIA narrative, expecting TRT-LLM to be clearly faster on the 4090.
Prerequisites took the whole day: two NIM reranker timeboxes (the first misjudged the root cause, the second found Docker Desktop was too old), NV-Embed-v2 abandoned due to
two incompatibilities between remote code and transformers 4.57, and ModelOpt FP8 conversion succeeded first try (512 calibration corpus samples, 46 seconds).

**What we tried**: ① the measurement tool first went through gauntlet in the cloud with a fake OpenAI streaming server covering SSE parsing, TTFT, percentiles and consistency comparison
(all 80 mutations killed; the property subset revealed three core-semantics blind spots, then invariants were added); ② on the desktop's first round, vLLM c=1 stalled 5 minutes on each of three questions —
the log pointed to Triton JIT compiling kernels during inference; 5 warm-up questions didn't cover the shapes, so the round was discarded and changed to 10; ③ vLLM by default applies the model's
generation_config (repetition_penalty 1.05), a different condition from TRT-LLM, so we re-ran with `--generation-config vllm`;
④ bf16 on both engines differed by ≤ 10% across three concurrency levels; ⑤ the same FP8 checkpoint ran on both engines, with ×1.4–1.7 speed-up.

**Key trade-offs**: The easiest mistake is crediting throughput to the engine — the pre-mortem required recording output token counts and
consistency comparison for every question in advance, which proved both engines did the same work (length difference p50 0), so the numbers are comparable. The other is FP8 quality: same-engine bf16→FP8
greedy outputs were verbatim identical only half the time, so a speed table on its own would mislead; hence the conclusion explicitly states "axis C quality criteria are a prerequisite for the FP8 conclusion".

**Outcome**: Under a short-answer RAG load on the 4090, "do FP8 first, then talk engines"; boring-but-safe is vLLM bf16 (424 tok/s @ c=8),
fastest is TRT-LLM FP8 (645), but only 6% more than vLLM FP8 — not worth maintaining another venv. The "vLLM support for Ada FP8 to be verified" listed at project start
now has an answer. Open: one round of long-output load, axis C faithfulness judge, a third-venv route for NV-Embed-v2.


## 2026-09-17 — Axis C: FP8 overtaken by 4-bit, then the judge is contradicted by manual review

**Trigger**: Axis B treated FP8 (×1.4) as the end of the road on Ada; axis C had to answer whether AWQ / GPTQ were worth maintaining yet another format.
The original plan was to calibrate all three formats ourselves with ModelOpt on the same corpus (strictly controlled); after evaluation we switched to Qwen's official pre-quantized checkpoints:
two fewer conversions, at the cost of calibration data not sharing a source with FP8, which every table in the report states. TRT-LLM failed outright loading HF-format AWQ
(the PyTorch backend only recognizes `weight`, while AutoAWQ uses `qweight/qzeros/scales`), so the matrix shrank from 2×4 to four formats on the single vLLM engine —
the first empirical confirmation of the founding hypothesis "format is tied to engine".

**What we tried**: ① First verified that the four formats share the same tokenizer (per-question token counts identical across 100 questions), ruling out prompt-side confounding; ② for each format ran short answers
at three concurrency levels, long output at c=8, the faithfulness judge, and verbatim agreement rate against bf16; ③ both W4A16 formats reached 1000 tok/s throughput at c=8, ×2.36 over bf16,
×1.6 over FP8, with AWQ and GPTQ within ≤ 5% of each other (same Marlin kernel; the only difference is the calibration algorithm); ④ quality: GPTQ 0.960 tied with bf16 0.959,
AWQ 0.938 was the only one below 0.95; ⑤ laid out AWQ's 8 contradicted claims in an xlsx next to bf16 on the same questions for manual review —
2 genuine errors, 1 retrieval accomplice, **5 judge misjudgements**; and all 5 of bf16's contradicted claims were the same kind of misjudgement or accomplice.

**Key trade-offs**: The first counterintuitive result was "output change ≠ quality loss" showing up for the second time: the deeper the quantization, the lower the verbatim agreement (FP8 0.51 → W4 0.25),
yet the judge sees no difference. The second was that judge v2's calibration κ 0.845 failed to catch false-contradicted 5/8 — the calibration set had too few verbatim-hit sentences,
so κ only proves "agreement on the calibration set". The misjudgements fall into three types, each with a different remedy: verbatim-hit cases should not rely on the prompt but be blocked by a deterministic pre-filter;
truncated-head chunks are a chunking disease, assigned to axis A parent-child; abstention verdicts must first ask about subject attribution, then about presence. On granularity, between "whole-sentence semantics" and
"atomic decomposition (RAGAS-style)" we chose the former: this judge exists to compare formats and engines, sentence-level noise is symmetric across formats, and atomic decomposition would inflate the claims,
add another splitting step that needs calibration, and make numbers incomparable with v1/v2.

**Outcome**: Recommended vLLM GPTQ-Int4 (×2.36, quality tied, weights 5.3 GiB, 2.5× KV cache capacity); FP8 is demoted to the middle option for "needs to work across engines
but doesn't want to do its own W4 calibration"; AWQ's real gap of 2 claims is within noise, conclusion unchanged. Two quantization side effects were also recorded: a mixed Chinese-English ingredient line
got a wrong abstention from AWQ (bf16 answered correctly), and one question's output drifted into Simplified characters. Left over: judge verbatim pre-filter layer, prompt v3 abstention rules, separate Simplified-Chinese drift count,
TRT-LLM ModelOpt-AWQ route, second round of KV cache quantization.

### ❌ Failure log
- judge v2's κ 0.845 was taken as evidence of being "trustworthy", while the actual false rate on the contradicted class was 5/8; κ only answers for the calibration set's distribution,
  and sentence types the calibration set doesn't cover (verbatim hits, abstentions) still go wrong. Lesson: the calibration set must be stratified by "judgement difficulty type", not just randomly sampled.
- The first version of the prompt_tokens verification command grabbed the wrong field (bench didn't record prompt_tokens); the correct verification artifact was offline per-question counting with each checkpoint's tokenizer.

## 2026-09-17 — Verbatim pre-filter layer: measure before building, found the gain smaller than imagined, still built it as a zero-let-through safety layer

**Trigger**: The AWQ review revealed that 5 of the judge's 8 contradicted claims were misjudgements, two of which were claims taken verbatim from corpus sentences.
The intuitive fix was "if the claim is a substring of the context, judge it supported directly without sending it to the LLM", expected to eliminate most misjudgements.

**What we tried**: Before building, measured the distribution of 3-gram coverage against human labels on the 48-row manual calibration table — the result contradicted the intuition:
① synthetic negatives that change only one number still had coverage of 0.70–0.73, and adjacent-clause misattribution reached 0.84, so coverage alone is not safe;
② genuinely verbatim claims are rare: of the calibration set's 34 supported claims only 3 were above 0.85, and of AWQ's 5 misjudgements only 1 was truly verbatim,
the other was a paraphrased sentence (0.57). The design became "coverage ≥ threshold + number guard + short-sentence guard + judged against a single chunk",
with the threshold taken by grid scan at the midpoint 0.925 of the zero-let-through plateau (0.85–1.00). Went through old-coder: in round one the mutation "change threshold to strictly greater than" survived
(nothing pinned that equal-to-threshold must hit); in round two the property subset killed only 2/7, after adding three invariants 5/7, with the 2 survivors being tool-layer counting.

**Key trade-offs**: After measuring it was clear this layer can't fix the main misjudgements; the reason to build it anyway is that it has zero let-through, zero cost, and deterministically seals off the judge's
instability on "verbatim copied sentences" (the same sentence split into two claims judged differently); fixing the main errors relies on prompt v3's abstention subject-attribution rule.
Without measurement we would have built a "let through at coverage 0.7" version, putting all the number-tampered negatives into supported — the one error the judge can least afford.

**Outcome**: `verbatim_hit` entered the judge pipeline (disabled when the config key is missing, old behavior unchanged), `verbatim_grid` rescans the threshold whenever the calibration table changes version;
gauntlet 91/91 mutations all killed. Left over: prompt v3, Simplified-Chinese drift column, calibration set stratified by judgement difficulty type.

### ❌ Failure log
- Used `git checkout <檔>` to revert a one-off mutation, which also wiped the uncommitted implementation in the same file; had to rewrite it once to get back to green.
  Revert one-off mutations with a reverse sed or look at `git diff` first; never check out the whole file.

## 2026-09-18 — The judge's real κ is 0.49, not 0.85: stratified sampling, two prompt versions, one split-question attempt, and finally accepting the limit

**Trigger**: The AWQ quality review found that 5 of the 8 claims the judge marked contradicted were the judge's own mistakes, yet the calibration κ 0.845 looked very healthy.
The suspect was not the judge but the calibration set: 30 randomly sampled questions, dominated by verbatim-type and easy supported sentences, which propped κ up.

**What we tried**: ① Proposed three stratified-sampling options and chose "four strata, equal quotas" — using deterministic rules (3-gram coverage, abstention wording), the four formats'
claims were deduplicated and split into verbatim / paraphrase / low-overlap / abstention strata with a fixed number drawn from each, and 42 rows were relabelled manually; v2 dropped to κ 0.488 on the new set, with errors split into
too strict 4 (paraphrased sentences judged the wrong way round, correct abstentions judged the wrong way round) and too lenient 5 (unit misplacement, falsely claiming presence, wrong abstention). Two bugs were caught along the way:
the Simplified 「参考資料未提及」 ("not mentioned in the references") bypassed the Traditional refusal marker and was sent for judging as a claim, and claim_id collisions between same-question same-index claims across the four formats made score silently drop 8 rows.
② prompt v3 wrote one explicit rule for each of the nine errors and reran the 42 rows: not a single verdict changed, only one evidence_chunk number and one JSON line break differed.
③ Split into three sub-questions (subject attribution / values and unit of measure / enumerated items), asked separately and merged in code: 3 of 8 flipped to correct, negatives 7/7. Abstention type 2/2 flipped to correct;
but unit misplacement stayed invisible even when explicitly asked to compare units of measure, and sentences without numbers were forced by the schema to produce fake values (the failure mode written in the pre-mortem appeared on the spot),
and adjacent-clause misattribution was just as wrong with split questions.

**Key trade-offs**: Adding rules from v1→v2 worked, v2→v3 did not; the difference is that the former supplied "things the model didn't think to check", while the latter needed discretion over "checked but judged the wrong way round",
and rule text can't move discretion. Splitting questions only helps for the "subject attribution" type. Switching to a bigger judge would reopen the §5.3 decision and void every calibration number.
Of the three options, we chose the least flattering but most honest: the judge is repositioned as "relative comparison among the four formats", κ 0.488 and the error types are listed explicitly as scope / limits of applicability, and the prompt is reverted to v2.

**Outcome**: Axis C conclusions are unaffected (error types are symmetric across formats); "route abstention-type through subject-attribution split questions" is listed as a phase-two candidate;
the stratified sampling, verbatim pre-filter layer and split-question probe tools stay in the repo. Today's most valuable output is not any κ, but knowing why the old κ was an overestimate.

### ❌ Failure log
- prompt v3: four rules, zero verdict changes. Assumed explicit rules were all-powerful; in fact they only help with "didn't check".
- Split question B1: the pre-mortem anticipated "the model fills it in anyway even though the schema allows an empty array", and it appeared in the first round; anticipating it is not preventing it, it must be blocked in the merge rules.
- First version of stratified sampling had claim_id collisions; score silently lost 8 rows and it was only noticed at n=34. "Row counts don't match" is the cheapest detection signal; from now on score should assert directly on n.

## 2026-09-18 — Demo up and running: a dry-run that only watched pydantic missed the entire dependency tree

**Trigger**: Before release we needed an interface that can show "configuration differences" live. Settled on two side-by-side columns: same question, one retrieval preset per column,
eval set ground truth in green boxes, chunks retrieved only by that column in yellow boxes, each streaming a vLLM answer below. Pure logic goes into gauntlet; the UI layer is verified by a real `--check` run.

**What we tried**: The cloud container can't reach HF, so all we could do was `pip install --dry-run` to check whether gradio would touch pydantic (vLLM had already bumped it to 2.13.5, a known conflict point).
The dry-run reported no conflict, so we pinned 6.27.0. On installing on the desktop, huggingface-hub was pulled from 0.36 to 1.32, transformers 4.57.6 requires <1.0,
and the dense retriever crashed on import — the bm25 column was fine, the hybrid column died during construction.

**Key trade-offs**: Three paths. Downgrading transformers would ripple into vLLM's pins, rejected; a separate venv for the demo would prevent it from using the main venv's retrievers directly, rejected;
what remained was finding a gradio version that satisfies both sides. Checked PyPI metadata version by version: 6.17.3 is the last version accepting hf-hub 0.x; from 6.18.0 it hard-requires 1.2+.
Re-pinned to 6.17.3 and explicitly pinned hf-hub 0.36.2; `--check` passed first time.

**Outcome**: Demo runs. VRAM 23879/24564 MiB; with vLLM 0.75 + bge-m3 + reranker on the same card only 685 MiB remain — usable but no headroom.
First question (b1-q05, tetracycline in pregnancy): both columns' top-2 hit the ground truth but in opposite order, and the third result in each retrieved a different yellow box — exactly the difference we wanted to show.
Lesson recorded as environment pitfall F9: a dry-run must look at the upper bounds across the whole dependency tree, not stare at one known conflicting package.

### ❌ Failure log
- High-risk assumption ① "gradio 6.27.0 has no conflict with the main venv, verifiable by dry-run" was written too narrowly: the verification method covered only pydantic, and what overturned it was precisely a package not included in the verification.

## 2026-09-18 — Citation defense: the UI line "valid citations [] / invalid []" was always empty, because the prompt never asked for citations

**Trigger**: Reviewing the finished product before recording the demo, we noticed the citation check on the answer's last line had never lit up — `cited_indices` was pre-wired for judge phase two,
but the demo prompt deliberately shared its source with the experiment prompt, which doesn't ask for [n] markers. In practice, RAG without citations is untraceable, so we decided to add them.

**What we tried**: Added one line to the prompt: "end each sentence with [段落編號] (paragraph number) marking its basis; if unable to answer, write only 『參考資料未提及』 ("not mentioned in the references") and cite nothing";
core gained cite_stats (valid / out-of-range / whether ground truth is cited / number of uncited sentences) and a batch summary, and demo_app gained a `--cite-rate` mode to measure 100 questions.
In the first gauntlet round 5l died in the harness (the scripted playbook only had question 1); in the second round the property subset let all 4 new anchors survive;
after adding "independent recomputation" invariants, the third round killed the two core-semantic anchors for sentence boundaries and the denominator.

**Key trade-offs**: Add the citation rule to the demo prompt or the experiment prompt? Adding it to the experiment prompt would force rerunning all 200-plus bench and judge numbers for axes B/C,
rejected; the demo prompt forks from here, stated explicitly in the config file. Whether a citation "semantically supports the sentence" is left to judge phase two; this round only does deterministic format and ground-truth mapping.

**Outcome**: 7B-GPTQ compliance 100%, zero out-of-range; questions with ground truth in top-3 were 84 vs 97, exactly reproducing axis A's hit@3; when ground truth is in top-3, the rate of citing it was 0.90 on both sides —
"citing the right paragraph" is a model capability, independent of retrieval configuration. 30% of sentences had no citation — a screenshot the next day revealed a sentence-splitting defect (the model puts [n] after the period); after the fix, remeasuring gave only 0.019 / 0.000, almost every sentence cited. Next: retrieval-confidence threshold refusal (a wrong retrieval like b2-q05 still gets a "valid citation").

### ❌ Failure log
- Initially wrote the truth_cite_rate denominator as "answers with a valid citation", so open questions (no ground truth) would drag the rate down; corrected the spec publicly and added a has_truth field.
- The property strategy built CiteStats directly without going through cite_stats, so it could never kill "cites_truth always True" — that survival is expected and covered by hand-computed cases;
  what actually needed adding was core semantics like "sentence boundary" and "denominator" that a property can independently recompute.

## 2026-09-19 — Refusal threshold: one side textbook-clean, the other with no plateau at all

**Trigger**: The two questions before recording the demo (b2-q05, b3-q08) exposed "wrong retrieval, equally confident answer". We wanted a deterministic defense:
don't send to the LLM if the top-1 relevance score is below a threshold. No hand-guessing the threshold: grid search over 100 questions + 20 out-of-corpus questions, taking the plateau midpoint.

**What we tried**: First exposed the reranker and BM25 scores from the retrievers (the protocol only returned ids; scores were discarded after computation); wrote 20 out-of-corpus questions for manual per-question review
(2 rejected: one had a frequency phrasing structurally identical to the reimbursement rules yet wasn't marked as a lure, one was a volatile fact; 2 unexpected lure questions were additionally labelled). Collected scores once on the desktop and scanned 40 grid cells offline.

**Key trade-offs**: Search both columns or only the reranker? Chose both, because "BM25 has no calibratable confidence" should be a measured conclusion, not an assumption.
Result: on the reranker side, all 20 out-of-corpus questions had top-1 below 0.026, only 1 of 97 questions with ground truth was below 0.10, plateau as wide as 0.03–0.59;
on the BM25 side scores drifted between 7–167 with query length, the so-called plateau was a single cell, and the next cell already refused wrongly 3.6%. The threshold is applied only on the reranker side.

**Outcome**: hybrid + reranker threshold 0.308; none for BM25, and the demo narration gains a line on "the hidden cost of the boring-but-safe configuration".
Honest limitation: of the 3 questions where top-3 missed the ground truth, the reranker still gave > 0.85 on 2 — wrongly retrieved but semantically similar questions that the threshold can't separate;
these need the judge or better retrieval, not something a threshold can fix.

### ❌ Failure log
- S1 committed after running only pytest; two existing mutation anchors went stale due to the refactor, and gauntlet's residue detection blocked the entire layer; the mutation layer is not something to "run last".
- The first version of the out-of-corpus questions labelled "how often should it be checked" as a pure fact question; manual review pointed out it is structurally identical to package insert frequency phrasing — the question author is the one most blind to lure surfaces.

## 2026-09-19 — Trying to shore up one contested point, we dug up the demo's most valuable finding

**Trigger**: Self-checking before interviews for what in the demo could still be picked apart. Listed nine points, of which "valid citation ≠ correct citation" was the one we were least confident about —
the UI showing "citation [1] (contains ground truth)" only proves the number exists, not that the sentence really comes from that paragraph. Added a deterministic check layer:
for each sentence compute "the fraction of character 3-grams covered by the cited paragraph", zero LLM cost.

**What we tried**: On the desktop, low-coverage sentences were 30%, far above my assumed 0.1 reading line. Wrote a sampling tool that outputs low-coverage sentences together with the cited paragraphs as a table,
and manually judged each as "paraphrase / synthesis / fabrication". All 23 judged: paraphrase 12, fabrication 11, synthesis 0.

**Key trade-offs**: The plan was to decide between "tune the threshold" and "change the metric" based on the proportion, but three counterexamples overturned the whole direction —
the one with coverage 0 was a verbatim-faithful paraphrase, while the one with coverage 0.351 was the wrong drug (asking about drug A's neurological warning, it cited an antidiarrheal's package insert for "fatigue, dizziness"),
and the 0.484 one was a causal inversion (describing a drug property as a property of a time point). **High or low coverage is simply uncorrelated with correctness**; tuning the threshold would be wasted effort.
Decided not to treat it as a score, only as a ranking for "this sentence needs manual verification", and changed all wording in code and UI.
Another judgement: 4 of the 11 fabrications were actually "right content, wrong paragraph cited" — the same sentence in another configuration cited correctly and was fully grounded —
so it became its own category, otherwise the hallucination rate would be reported as worse than it is.

**Outcome**: 11/74 ≈ 15% of cited sentences have no support in the cited paragraph, and this is a **lower bound** (only the 23 low-coverage sentences were checked).
Both retrieval configurations had similar fabrication counts, so "get retrieval right and generation will be faithful" is false. This finding is far more valuable than the contested point we set out to shore up:
it measures a real defect on the RAG generation side with a rerunnable method, rather than being yet another pretty number.

### ❌ Failure log
- My 0.1 reading line rested on the assumption "low coverage ≈ fabrication", which was itself wrong; fortunately it was written up as a high-risk assumption requiring verification before building, so it wasn't directly adopted as a metric.
- Zero synthesis cases was a surprise: when the model cites multiple paragraphs, the content actually all comes from one of them. The citation list is partly decorative, something nobody had thought to check.


## 2026-09-20 — The gauntlet needs a gauntlet too: the testing tooling failed to produce a valid result four rounds in a row

**Trigger**: Adding a "required component hit" check for multi-component questions (questions like "what are the eligible recipients **and** the frequency" that require answering two independent things).
None of the existing metrics could see this defect — the retrieval layer had complete ground truth, the faithfulness layer found the half that was answered well-grounded, and nobody was measuring "incomplete answer".
The code itself took two days to write, but it only counts as done once it passes the gauntlet, and the gauntlet failed to produce a single valid result four rounds in a row.

**What we tried**: In round one, upon reaching the mutation layer, two gaps in the design were pointed out (a forbidden-phrasing column; a normalization layer should replace per-question enumerated variants),
so we aborted and redid it. Round two was 139/140, and the survivor was an anchor I had just added — an idempotency hole originally found **randomly** by a property test;
after fixing the implementation only the property guarded it, and with a different seed it couldn't produce that character sequence, so mutation testing suddenly reported a survivor.
Round three finally reached the new component layer and hit two bugs in a row: that section was placed after the fake LLM server had already been shut down (`Connection refused`),
and the fake server's default reply contained no citation markers, so that layer always had zero rows — **the `assert rows` I wrote had never actually executed in the previous two rounds,
yet I had treated it in the report as a verified layer**. In round four, to save effort, I chained commit, push and gauntlet into one `&&` chain thrown into the background,
committing three mistakes at once: unable to observe progress, leading to misjudging that the commit hadn't happened; the pass gate took `tail`'s exit code (a pattern I had myself written into the rules
as explicitly forbidden); and the log filename collided with a leftover file from three weeks earlier, nearly treating that old "all layers passed" as this round's result.
Finally, when the outer shell of that chain was killed, gauntlet as a child process went down with it, SIGTERM didn't let the restore logic finish, and mutation residue was left behind.

**Key trade-offs**: Every failed round offered two paths: "loosen the assertion so it passes" or "fix the testing tool itself". The incentive to loosen was strong —
that `assert rows` in round three would have gone green immediately if changed to "accept zero rows, only check the header". But that trades an assertion that can never fail
for green on the report, which is worse than no assertion: it makes people think that layer is guarding something. The four rules distilled into the standing rules over those four rounds
are all about **the trustworthiness of the testing tooling**, not the product code — anchor ordering, property counterexamples must be frozen into hand-computed tests,
assertions must be confirmed to have actually executed (the criterion is seeing its own printed message in the log), and background execution wraps only a single command.

**Outcome**: Code and checklist complete (16 questions, 36 components). The most valuable design point is **separating the two kinds of forbidden phrasing**:
「每日2粒」 ("2 tablets daily") would falsely satisfy the component 「2粒」 ("2 tablets"), so it is the protective type, attached to that component and zeroing it; 「迷走神經刺激術」 ("vagus nerve stimulation") matches no alias,
and what it proves is that the whole answer went to a different drug, so it is the penalizing type, zeroing the whole question. Sharing one field would count "incomplete answer" and "answered the wrong question" as the same number.
All four real test cases behaved as designed. The real takeaway was realizing that **the gauntlet needs a gauntlet too** —
an assertion that never executed, an invariant guarded only by random testing, a log that collided with an old file,
can each turn "all layers passed" into a sentence with no content.

### ❌ Failure log
- For two rounds in a row, treated an assertion that never executed as verified. The criterion was wrong: I checked "the line is in the script",
  not "its printed message is in the log". If an earlier layer hits EXIT=1 first, the later assertion is dead code.
- Just after writing "the gate always takes the tool's own exit code" into the rules, within the next hour I wrote `pytest | tail && git commit` myself.
  Writing a rule into the docs doesn't mean following it, especially when "just chaining it up in passing".
- While checking a question's components I also checked the corpus and found that the required keyword (a syndrome name) appears **0 times in the entire corpus** —
  characters were lost at the PDF extraction stage (19/1046 chunks of the same document have the same kind of bracket misalignment).
  I had intended to list it as a required component, which would have rewarded hallucination: the model's only way to score would be to generate, from parametric memory, content that doesn't exist in the corpus.
  Also corrected a diagnosis previously written into the memo — I had judged it a "chunking boundary problem", but the actual lesion is further upstream.

## 2026-09-23: Three-stage governance of wrong abstention — visibility succeeded, prompt treatment backfired, finally settled on a deterministic guard

**The problem**: During the sixth round of manual-review scoring, we found the evaluation table had always been "missing an entire class of errors". When the RAG system answers 「參考資料未提及」 ("not mentioned in the references")
(abstention), sentence-level support analysis skips the whole sentence, so every round 20~30 "question × retrieval configuration" combinations vanished from the table entirely —
it had been that way since the first round, and we never noticed. Worse, abstention itself can be wrong: in 11 of 21 abstentions, the correct paragraph containing the answer
**was right there in the model's context**, mostly the very paragraph the model itself had cited — it cited [3], then said [3] doesn't mention it.
In a demo this kind of "wrong abstention" looks worse than a wrong answer: the user watches the system hold the answer and say there is no answer.

**What we did (three stages)**:
Stage one first made it **visible**: the evaluation table gained an "abstention row" row type, plus a deterministically computed "best ground-truth rank" column —
on abstention it automatically marks what rank retrieval gave the correct paragraph. "Should have had it but abstained" now surfaces into human view on its own,
no longer depending on someone happening to dig through raw output. This step also forced the ground-truth criterion to be made explicit (it counts as ground truth only if the body text contains the answer;
a chunk where the answer is only in the heading path doesn't count — because the generation layer can't read the heading at all).
Stage two tried **treating it with the prompt**: the existing rule pushed only one direction ("anything not in the paragraphs is answered as not mentioned"), so we added the reverse
paired rule "content already stated in the reference paragraphs must not be claimed as not mentioned; before answering not mentioned, confirm paragraph by paragraph".
Single change, citation rule kept at sentence end, same seed and same retrieval as the previous round, so the net effect can be measured in one round.
Stage three was cleanup: the measured result decided a rollback, and the guarding duty shifted to the deterministic layer built in stage one.

**Result**: Stage two **backfired catastrophically**. Retrieval results for all 200 combinations were compared one by one with the previous round and were identical,
so all change was on the generation side and attribution was clean — and the generation-side change was: abstentions 22 → 66, and the 44 new ones were **all**
questions that had been answered normally (mostly correctly) turning into pure abstentions, abstention rate 11% → 33%; of the original 11 wrong abstentions **zero were fixed**;
even for the original case the rule targeted, the final sentence only changed from "no other restrictions mentioned" to "not mentioned".
In hindsight the mechanism is clear: the rule **repeated the string** 「參考資料未提及」 once more in the system prompt,
and steered the model's attention toward "the possibility of not finding it". A quantized 7B model's response to this kind of "please self-check" meta instruction
is not to perform the check but to **output the repeatedly emphasized string more often** — the rule text itself became the priming source.
Fortunately the verdict needed no manual review: "refusing questions that were previously answerable" is a regression regardless of the new answer content, and mechanical comparison settles it;
from receiving the data to pushing the rollback took under an hour.

**What to do differently (final solution)**:
1. **The feasible space at the prompt layer converges to one empirical law**: for a model of this size, **format/phrasing constraints can be added**
   (the ban on meta preambles succeeded, 3→0 with zero side effects), **meta constraints that "ask the model to self-check" cannot be added**
   (verbatim drug-name copying failed, banning claims of absence backfired, two empirical cases). Rules can constrain "how to write", not add "think before writing".
2. **Guarding against wrong abstention is settled on the deterministic layer**: the abstention row + ground-truth rank column is a check that doesn't depend on the model obeying any rule;
   each round it automatically surfaces suspect rows, and humans only judge the one step "does the paragraph really contain the answer". Behavior we can't fix becomes a process we can guard.
3. **Experimental discipline was the only part that didn't slip once this time; keep it**: single change, same seed, verify retrieval is identical row by row before attributing to generation,
   expectations (including failure expectations) written before starting, rollback text preloaded in config-file comments. When the backfire appeared, rollback took ten minutes.
4. External narrative (interviews) told as it is: wrong abstention 11/21 is a known limitation, the prompt treatment was falsified experimentally,
   and the system brings it into routine monitoring with deterministic tools — "I measured it, I can't fix it, but I can guard it".

### ❌ Failure log
- The pre-work expectation was "wrong abstentions go down, not expected to reach zero"; the actual direction was the complete opposite. The mistake was extrapolating "rules can constrain output format"
  to "rules can add checking behavior" — the latter doesn't hold for a quantized 7B model, and at the time of extrapolating there wasn't a single round of data supporting it,
  only "it sounds reasonable". Reasonableness is not evidence.
- This rule was a verified success pattern on larger models (paired rules sealing off both directions),
  but when ported directly, "model size" wasn't treated as a precondition. A success carries its scope of applicability with it, and the scope doesn't automatically move house with it.

## 2026-09-29: One sentence in an article, "the judge's errors are symmetric", took six rounds of audit to go from assumption to measurement

**The problem**: English article #2 concludes "W4 quantization changes the output a lot, but quality doesn't change", based on the local 14B judge's
faithfulness scores (bf16 0.959, GPTQ 0.960, AWQ 0.938). An external reviewer pointed out that this conclusion actually rests on one unmeasured sentence:
"the judge is suitable for relative comparison between formats because its errors are symmetric". The reviewer also identified four concrete holes:
① only AWQ's contradicted claims were audited, compared against unaudited bf16 (later verification showed bf16 had in fact been audited in the same batch, the article just didn't say so);
② abstentions have no claims to judge, so the whole sentence leaves the denominator, and the score is computed on a subset that varies by format;
③ garbled drug names near rare characters (nonexistent drug names, mojibake) are not "contradictions", and the judge's framework can't see them at all;
④ W4 raises Simplified-Chinese drift from 2 questions to 6~9. If the judge is stricter on Simplified text, the quantized formats would actually be **underestimated**.
In other words, the quality evidence for the recommended conclusion was half measurement, half assumption.

**What we did (every step was "reviewer question → find the cheapest measurement → run on desktop → manual verdict → record")**:
1. **Fix the wording first, then add evidence**: the article states that the judge sees all three retrieved paragraphs (not only the cited one) and judges by whole sentence;
   abstention and entity degradation are listed as "not covered by the score"; symmetry is rewritten as an "unverified assumption". First make the article not lie, then decide which measurements to add.
2. **Supplementary audit of GPTQ contradicted** (5 claims, no GPU needed, laid out as a review table from the existing judge file): genuine errors 0, judge misjudgements 4,
   of which 3 were **the same sentence** misjudged in all three formats bf16 / AWQ / GPTQ — the first direct evidence of symmetry.
   The remaining 1 turned out to be a pipeline artifact: GPTQ's judge file predated the "Simplified abstention marker normalization" fix by half a day,
   and under the current code that sentence would never be sent to the judge.
3. **Chasing a "missing question" uncovered a blind spot**: to find out why a certain question didn't appear in GPTQ's contradicted list, we pulled up its verdict,
   and found GPTQ wrote "2 tablets daily, once every 6 hours" and the judge ruled supported; checking the package insert in the corpus, the original is "2 tablets, once every 6 hours"
   (2 tablets per dose), so the daily dose was 4× too low. **Auditing only contradicted means only checking what the judge caught, not what it let through.**
4. **Supplementary audit of dosage sentences on the supported side** (35~39 per format, filtered by regex for "number + unit" or "daily / per dose"):
   errors the judge let through: bf16 3/39, GPTQ 3/36, AWQ 3/35, nearly identical across the three formats.
   All nine errors are the same type: the literal text is all in the paragraph but the relation is wrong (daily/per-dose swapped, frequency transplanted between indications, announcement date taken as effective date,
   mutually exclusive categories merged, condition omitted); the only new type was one entity misplacement (answering a long-acting injection's dosing interval with that of the regular formulation).
   Two of the questions had two formats making the same error, and the judge let both through, showing these are errors shared by the model, not introduced by quantization.
5. **Re-judging with Simplified converted to Traditional** (requires starting the 14B judge, the only step needing GPU): took all 16 claims across the three formats containing Simplified-only characters,
   and in the same round judged both the "original sentence" and the "OpenCC Traditional version", with the verbatim pre-filter layer turned off, to avoid coverage rising after conversion and bypassing the LLM,
   which would measure the pre-filter rather than the judge. The conversion package was installed in a temporary venv in `/tmp`, not polluting the main environment.

**Result**:
- 15 of 16 verdicts unchanged; the only flip was one bf16 sentence, and the direction was **stricter** after conversion to Traditional, opposite to the assumption.
  The AWQ Letrozole abstention (`b4-q12#1`) originally suspected of "Simplified interference" was confirmed, after checking the claim_id, to still be judged contradicted after conversion;
  the cause of the error is not the glyphs.
- Judge error is close across all three formats and both directions: contradicted-side misjudgements 4/5, 5/8, 4/5; supported-side dosage sentences let through 3/39, 3/35, 3/36.
  "Symmetric error" on bf16 / AWQ / GPTQ **went from assumption to measurement**; FP8 was not audited, stated in the text.
- Post-audit gap (vs bf16): GPTQ 0 on both sides; AWQ 2 more on the contradicted side. The GPTQ recommendation **holds, and on firmer ground than before**,
  but the wording was tightened accordingly: the article doesn't say "GPTQ is as good as bf16", only "within the range this judge can measure there is no gap, and what it can't measure is listed clearly".
- Memo gains Findings 56~58; the article's Result 3 went from "score + a one-line assurance" to "audit tables for both directions + three explicit not-covered items".

**What to do differently**:
1. **Always audit the judge in both directions together**: looking only at sentences judged contradicted audits the judge's "accusations", not its "judgement".
   The supported side needs at least a high-risk subset sampled (dosage, values, dates, conditions); otherwise let-through error types will never appear on any table.
2. **Before using a judge for relative comparison, measure symmetry first; don't write symmetry first**. The two cheapest ways to measure:
   see whether misjudged sentences overlap across formats (the same sentence wrong in every format = error tied to the sentence, not the format);
   do paired re-judging on the suspected bias source (same round, change only one variable, turn off confounding pre-filter layers).
3. **Evaluation artifacts must record the code version that produced them**. This time the pipeline artifact was found only by matching file timestamps against commit times;
   if the judge output file carried its own commit hash, "code at judging time ≠ current code" would be obvious at a glance.
4. **Every adjective in the article must point back to a table**. Words like "symmetric", "indistinguishable", "no gap":
   before writing them, ask which audit table supports it, which direction it covers, and what it doesn't cover.

### ❌ Failure log
- When the draft wrote "its errors are symmetric", the only evidence in hand was the κ value and the one-sided AWQ review, without a single direct measurement.
  The sentence was used to support the conclusion but had no support itself — and the counter-evidence (Simplified-Chinese drift) was sitting in the same article; at the time the two weren't connected.
- The audit table listed only contradicted claims, and mentally I treated it as "audited". Only while chasing an unrelated question
  did I stumble on a 4× dosage error judged supported. **The direction of the audit is itself a variable that can be missed.**
- In the three-format audit, the reviewer and I each got a denominator wrong once (bf16 dosage sentences were 39, written as 36).
  The habit of recounting row by row against the original table before recording caught it, but it's a reminder: every time a number is transcribed from conversation into a document, check it against the source file.

