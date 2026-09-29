# Your BM25 Baseline Is Weaker Than It Should Be — and That Distorts Every RAG Comparison

> **2026-09-18, postscript 2026-09-29.** Numbers are from the frozen 100-question eval set
> (v1-full100 for the first two tables; the field ablation is on v1.2, where body-only
> BM25 scores 0.817 rather than 0.829 because later ground-truth additions enlarged the denominator).
> Series: *SelectRAG — choosing a RAG configuration by experiment, not by faith* (1 of 2)

Every "dense beats BM25" chart has a hidden assumption: that the BM25 column was
built competently. In this post I show, on a Traditional-Chinese medical corpus
(8,029 chunks from Taiwan's NHI reimbursement rules, drug package inserts, and
fee schedules), that two cheap, CPU-only fixes moved BM25 from recall@10 =
0.829 to 0.905 (hit@10 0.860 to 0.920) — the kind of gap that is usually presented as the *contribution of a
vector database*.

## The setup

- **Corpus**: 8,029 chunks (CSV fee-schedule rows, PDF package inserts and
  reimbursement rules, HTML announcements), structure-first chunking with a
  length cap.
- **Eval set**: 100 questions with human-adjudicated ground truth, frozen with a
  content hash. Ground truth is a *set* of equivalent chunks per question —
  the same rule often appears verbatim in multiple documents, and penalizing a
  retriever for finding "the other correct copy" silently biases every
  comparison downstream.
- **Metrics**: recall@k and hit@k side by side. With multi-chunk ground truth,
  recall is diluted when only one of five equivalent chunks is found; hit@k
  answers the question RAG actually cares about — *did the answer reach the
  context window?* The two are locked together by an invariant:
  `hit == (recall > 0)`.
- **Reproducibility**: one command per experiment
  (`python3 -m selectrag.runner <config> <run_id>`), results carry the eval-set
  hash, and a mutation-tested code base (100/100 mutants killed at time of
  writing, including a separate run against the property-based tests alone)
  backs the numbers.

## Fix 1: your chunks don't know which document they came from

Error analysis of the baseline's misses showed a dominant pattern: **the drug
name never appears in the chunk body**. A package insert says "2 Indications:
effective against Gram-positive and Gram-negative bacteria…" — the word
*tetracycline* lives only in the document's first line. Chunk-level indexing
amputates document identity.

The obvious fix — index the heading path — **does not work**, and this is the
part worth stealing: we verified before building, and the missing drug names
were not in the heading path either. They were in the document title line only.
So each chunk now carries a `doc_title` field (extracted at corpus-build time),
and the BM25 index optionally concatenates `doc_title + heading_path + body`.
Heading-path indexing on its own was rejected as a *fix for the missing drug
names*, because the names are not there. It still sits in the index, so we
split the gain by field afterwards (v1.2, default k1/b, same chunks and
ground truth):

| Indexed fields | recall@10 | hit@10 | MRR |
|---|---|---|---|
| body | 0.817 | 0.860 | 0.724 |
| body + doc_title | 0.856 | 0.880 | 0.726 |
| body + heading_path | 0.832 | 0.870 | 0.719 |
| body + doc_title + heading_path | **0.903** | **0.920** | 0.732 |

The title is the larger single source (+0.039 against +0.015), which fits
the drug-name diagnosis. The two together give +0.086, more than the sum of
the parts (+0.054). The title names the document, and the heading names the
section within it. Neither alone narrows a query to the right chunk as well
as both do.

| Config | recall@10 | hit@10 | MRR |
|---|---|---|---|
| BM25, body only | 0.829 | 0.860 | 0.714 |
| BM25 + doc context | **0.905** | **0.920** | 0.732 |

The cost, predicted in the design pre-mortem and observed on schedule: early
precision dips slightly (recall@3 −0.003) because same-document chunks become
more similar to each other. That is a reranker's job to claw back — and now we
know exactly how much headroom it has.

## Fix 2: the textbook k1/b defaults are wrong for this corpus

A two-round grid search (60 combinations, minutes on CPU) over k1 × b, picking
the *plateau midpoint* rather than the best cell:

| Config | tuned params | recall@10 | MRR | vs. defaults (1.5 / 0.75) |
|---|---|---|---|---|
| body only | k1=0.5, b=0.1 | 0.870 | 0.772 | recall +4.2 pts |
| + doc context | k1=0.7, b=0.5 | 0.905 | **0.808** | recall flat, MRR +0.08 |

Why the defaults fail here: **b (length normalization) is the culprit**. Chunk
lengths are already bounded by the chunking parameters, so penalizing longer
chunks mostly punishes the ones that contain a complete clause. And
character-bigram tokenization (our no-dependency choice for Chinese) produces
noisy term frequencies that favor a low k1 (fast tf saturation). Notably, the
doc-context variant is *robust* to b — its longer index texts sit on a wide
plateau that happens to include the textbook defaults, which is why it looked
"already tuned".

Honesty note: tuning and evaluation share the same 100 questions (no held-out
split). Read these as upper-bound estimates; the dense comparison will tune
both sides before comparing.

## What this means for the A-axis comparison

The baseline that dense retrieval must beat is now **0.905 / 0.920**, not
0.829 / 0.860 — and the remaining misses form a precise target list:
cross-lingual vocabulary gaps (「雌激素經皮貼片」 vs "Estradiol Transdermal
Patch"), low lexical overlap paraphrases, and same-ingredient crowding. If
dense retrieval wins, it must win *there*, and we can check. That is the whole
point of building the boring baseline properly first.

## Postscript: how the axis ended (added 2026-09-18)

The check was run. On the same eval set (v1.2), bge-m3 with the same document
context scored recall@10 **0.904** against tuned BM25's **0.903** — dense alone
bought 0.001. Where it paid off was *complementarity*: the two retrievers miss
almost disjoint sets of questions, so a hybrid (reciprocal-rank fusion) reached
0.961, and a cross-encoder reranker on the hybrid's top 20 lifted recall@3 from
0.808 to 0.941 (recall@10 0.973). The memo's recommendation is therefore
"hybrid + reranker" — and its *boring-but-safe* option is still the tuned BM25
from this post: zero GPU, one millisecond, within 0.07 of the best configuration
on recall@10. Nobody would have known that without fixing the baseline first.

## Postscript: a known extraction defect (added 2026-09-29)

A later corpus audit found that, in the 41 non-CSV documents, 31.6% of the
source text is not in any chunk's body. Most of that is ordinary heading
lines, which the chunker moves into `heading_path` by design. Some of it,
though, is answer text that the heading parser mistook for a heading. Such
text is in the BM25 index through `heading_path`, but it is not in the
passages sent to the generator. So part of the doc-context gain above may
come from text that retrieval can see and the model cannot. The retrieval
numbers themselves are unaffected. Whether the gain carries over to answer
quality is a separate question. On the eval set, only one question is known
to be affected so far. The measurements and the reason we did not re-chunk
are in the corpus source list, `data/corpus/SOURCES.md`, under "known
extraction defects".

Full tables, the rejected alternatives (adjacent-chunk expansion, heading-path
indexing, dense alone) and the honesty section are in the
[decision memo](../DECISION_MEMO.md).

*Next in series: [The engine is not the variable — vLLM vs TensorRT-LLM vs
four numeric formats on one RTX 4090](02_engine_and_format_en.md).*
