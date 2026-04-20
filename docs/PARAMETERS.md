# Tunable parameters

Authoritative commented list of **every** workflow variable: [workflow_config_full.env](../workflow_config_full.env) (section banners `# ---------- ... ----------` match the headings below). This document adds **tuning ranges**, **constraints**, **how caps chain across stages**, and links to notebooks. For BioASQ-oriented paths and Docker, see [BioASQ docs/USAGE.md](https://github.com/fulaibaowang/BioASQ/blob/main/docs/USAGE.md) and the script index [BioASQ scripts/public/README.md](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/README.md).

## Quick index

| First knobs | Section |
|-------------|---------|
| `HAVE_GROUND_TRUTH` | [Ground truth](#ground-truth-eval-metrics) |
| `WORKFLOW_OUTPUT_DIR`, `INPUT_JSONL`, `INPUT_BATCH_JSONLS`, `DOCS_JSONL`, `TOP_K`, `RECALL_KS` | [Shared paths and retrieval depth](#shared-paths-and-retrieval-depth) |
| `BM25_INDEX_PATH`, `DENSE_INDEX_DIR` | [Index paths](#index-paths) |
| Multi-field query fusion | [Query text fields](#query-text-fields-optional) |
| HNSW query/build | [Dense retrieval (HNSW)](#dense-retrieval-hnsw) |
| Cross-encoder pool, post-RRF fusion | [Stage 2 rerank and post-rerank fusion](#stage-2-rerank-and-post-rerank-fusion) |
| `RERANK_TSTAR_*` | [Post-fusion score cutoff](#post-fusion-score-cutoff-t-star) |
| Snippet windows, CE, final fusion | [Snippet-RRF route](#snippet-rrf-route-optional) |
| `POST_RERANK_DOC_POOL`, `EVIDENCE_TOP_K*` | [Evidence (contexts)](#evidence-contexts) |
| `GENERATION_*`, backends | [Answer generation (LLM)](#answer-generation-llm) |

---

## Ground truth (eval metrics)

Corresponds to `# ---------- Ground truth (eval metrics) ----------` in [workflow_config_full.env](../workflow_config_full.env).

Set `HAVE_GROUND_TRUTH=0` to skip evaluation metrics for BM25, Dense, Hybrid, and Rerank when you have no qrels.

---

## Shared paths and retrieval depth

Corresponds to `# ---------- Shared (paths + retrieval depth) ----------` in [workflow_config_full.env](../workflow_config_full.env).

### Query inputs (JSONL)

The pipeline reads **query streams only from `.jsonl`** (one JSON object per line, with optional `query_id` / `query_text` / `query_type` plus optional task-specific blobs). Set **at least one** of:

| Variable | Role |
|----------|------|
| `INPUT_JSONL` | Primary query file (optional if all splits are listed in batches) |
| `INPUT_BATCH_JSONLS` | Space-separated list of additional query `.jsonl` files |

Legacy names `TRAIN_JSON` / `TEST_BATCH_JSONS` are accepted by the shell driver only when they point to `.jsonl` paths. Wrapped task JSON (`{"questions":[...]}`) must be converted to `.jsonl` first; for BioASQ use [`bioasq_json_to_queries_jsonl.py`](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/format/bioasq_json_to_queries_jsonl.py) (see [BioASQ scripts/public/README.md](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/README.md)).

### Unified retrieval depth (`TOP_K`)

`TOP_K` is the single config for “how many documents per query” at the workflow level. Stage scripts use different CLI flags internally (`k_eval`, `topk`, `cap`, `candidate_limit`), but the orchestration sets them from `TOP_K` unless you set stage overrides (`BM25_TOP_K`, `DENSE_TOP_K`, `HYBRID_*`, `RERANK_CANDIDATE_LIMIT`, etc.).

**Commonly required / shared**

| Env var | Description | Example |
|---------|-------------|---------|
| `WORKFLOW_OUTPUT_DIR` | Base output path for all stages | `output/workflow_run` |
| `INPUT_JSONL` | Primary query stream | `/path/to/train.jsonl` |
| `INPUT_BATCH_JSONLS` | Space-separated extra query `.jsonl` files | `/path/to/a.jsonl /path/to/b.jsonl` |
| `TOP_K` | Retrieval depth (default 5000) | `1000` or `5000` |
| `RECALL_KS` | Comma-separated K values for recall metrics | `50,100,200,300,400,500` |
| `BM25_INDEX_PATH` | Terrier index directory | Path to index |
| `DENSE_INDEX_DIR` | Dense HNSW index directory | Path to index |
| `DOCS_JSONL` | Corpus JSONL for reranker and evidence lookup | Path to docs |
| `RUN_BASELINE` | Build baseline evidence / generation | `1` |
| `RUN_SNIPPET_RRF` | Enable snippet-RRF route | `0` |

Exhaustive `BM25_*`, `DENSE_*`, `HYBRID_*`, … names are listed under each `# ----------` block in [workflow_config_full.env](../workflow_config_full.env).

### Caps and internal flags (per stage)

| Stage | Internal flag(s) | Set from env |
|-------|------------------|--------------|
| BM25 | `k_eval` | `TOP_K` or `BM25_TOP_K` |
| Dense | `topk` | `TOP_K` or `DENSE_TOP_K` |
| Hybrid | `bm25_topk`, `cap`, `k_max_eval` | `TOP_K` or `HYBRID_*` |
| Reranker | `candidate_limit` | `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)` clamped to 30–2000 |

The reranker cannot use more documents than hybrid produced; for small corpora set `TOP_K` lower (for example 1000) and a smaller `RECALL_KS` so metrics stay meaningful.

### Recommended operating ranges

| Parameter | Suggested range | Default | Constraint |
|-----------|------------------|---------|------------|
| `TOP_K` | 1000 – 5000 | 5000 | Retrieval depth for BM25 and Dense |
| `HYBRID_CAP` | 1000 – 2000 | `TOP_K` | ≤ `TOP_K` |
| `RERANK_CANDIDATE_LIMIT` | 200 – 1000 | `TOP_K` (clamped [30, 2000]) | ≤ `HYBRID_CAP` |
| `pool_top_rerank` | 50 – 200 | 50 | ≤ effective rerank depth |
| `pool_top_hybrid` | 50 – 200 | 50 | ≤ `HYBRID_CAP` |

Chains: `pool_top ≤ RERANK_CANDIDATE_LIMIT ≤ HYBRID_CAP ≤ TOP_K` (see constraints below).

### Parameter constraints (data-flow)

The pipeline cascades `TOP_K` (default 5000) into stage-specific defaults. These invariants avoid silent truncation or degraded results.

```
TOP_K ──► BM25_TOP_K ──────────┐
     ──► DENSE_TOP_K ──────────┤
                                ├──► HYBRID_CAP ──► RERANK_CANDIDATE_LIMIT ──► pool_top_rerank
                                │                                               pool_top_hybrid ◄── HYBRID_CAP
                                │
                                └──► (RRF fusion top-N union) ──► evidence top-K ──► generation
```

| Constraint | Why | What happens on violation |
|------------|-----|---------------------------|
| `RERANK_CANDIDATE_LIMIT ≤ HYBRID_CAP` | Reranker reads hybrid runs | Pipeline enforces `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)` |
| `pool_top_rerank ≤ RERANK_EFFECTIVE` | Fusion draws from reranker output | Silent truncation |
| `pool_top_hybrid ≤ HYBRID_CAP` | Fusion draws from hybrid output | Silent truncation |
| `RERANK_EFFECTIVE ≥ 30` | Minimum reranker input | Pipeline clamps to 30 with a warning |
| `RERANK_EFFECTIVE ≤ 2000` | Cost cap | Pipeline clamps to 2000 with a warning |
| `ef_search ≥ DENSE_TOP_K` | HNSW needs ef ≥ k | Auto-promoted: `ef = max(meta, topk)` |
| `k_feedback ≥ fb_docs` | RM3 feedback | Not enforced — pool must cover `fb_docs` |

#### Default safety check

With default `TOP_K=5000`:

| Variable | Resolved value | Safe? |
|----------|----------------|-------|
| `HYBRID_CAP` | 5000 | ≥ `pool_top_hybrid` (50) |
| `RERANK_CANDIDATE_LIMIT` | 5000 → clamped 2000 | ≥ `pool_top_rerank` (50) |
| `pool_top_rerank` | 50 | ≤ 2000 |
| `pool_top_hybrid` | 50 | ≤ 5000 |
| `ef_search` | max(100, 5000) = 5000 | ≥ `DENSE_TOP_K` |

#### Potential issues when overriding defaults

1. **`TOP_K` < 50**: May cascade below default `pool_top_rerank` / `pool_top_hybrid`; fusion uses fewer docs (warnings).
2. **`RERANK_CANDIDATE_LIMIT` < 50**: After clamping, reranker output may be smaller than `pool_top_rerank`; pipeline warns.
3. **`DENSE_EF_CAP` too low**: If `DENSE_EF_CAP < DENSE_TOP_K`, deep recall degrades.
4. **`HYBRID_CAP` ≪ `RERANK_CANDIDATE_LIMIT`**: Effective rerank input becomes `min` of the two.

---

## Index paths

Corresponds to `# ---------- Index paths ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Variable | Role |
|----------|------|
| `BM25_INDEX_PATH` | Terrier index directory (BM25 + RM3) |
| `DENSE_INDEX_DIR` | Directory produced by `build_dense_hnsw_index_from_jsonl_shards.py` |

---

## Query text fields (optional)

Corresponds to `# ---------- Query text fields (optional) ----------` in [workflow_config_full.env](../workflow_config_full.env).

Comma-separated `BM25_QUERY_FIELD`, `DENSE_QUERY_FIELD`, `RERANK_QUERY_FIELD` run per-field subdirectories and optional weighted RRF fusion back into canonical `bm25/runs`, `dense/runs`, `rerank/runs`. See comments in the env file for `*_QUERY_FUSION_WEIGHTS`, `*_QUERY_FUSION_K_RRF`, and `*_QUERY_BODY_WEIGHT`.

---

## BM25 + RM3

Corresponds to `# ---------- BM25 + RM3 ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `fb_docs` | 5 – 20 | **20** | Feedback documents for term extraction |
| `fb_terms` | 10 – 30 | **30** | Expanded terms to add |
| `fb_lambda` | 0.6 – 0.8 | **0.6** | Interpolation weight (0 = pure RM3, 1 = pure original) |

**Decision:** Aggressive config (`fb_docs=20, fb_terms=30, fb_lambda=0.6`) achieved the highest recall while balanced configs had marginally higher MAP@10 but lower recall.

See [notebooks/bm25_test.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/bm25_test.ipynb) for the RM3 parameter sweep.

---

## Dense retrieval (HNSW)

Corresponds to `# ---------- Dense ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `M` | — | **32** | HNSW graph degree (common value, not swept) |
| `ef_construction` | — | **200** | HNSW build-time quality (common value, not swept) |
| `ef_search` | 5000 – 20000 | **max(meta, DENSE_TOP_K)** | Runtime auto-promoted to ≥ topk; capped by `DENSE_EF_CAP` if set |
| `batch_size` | — | **128** (index) / **256** (retrieval) | Embedding batch size (not swept) |
| `max_seq_length` | — | **512** | Encoder truncation length (~2.8% of docs truncate at 512) |
| `hnsw_space` | — | **cosine** | Distance metric |

**Decision:** Only `ef_search` was swept (5000/10000/20000); differences in MeanR@5000 were marginal (0.845 → 0.850). The retrieval script auto-promotes `ef_search` to `max(meta_value, topk)` so a small meta default does not limit deep recall.

See [notebooks/dense_test.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/dense_test.ipynb) for details.

### Dense models tested

| Model | Result |
|-------|--------|
| [abhinand/MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1) | **Default** — used in production pipeline |
| [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | Worse hybrid recall than MedEmbed |

See [notebooks/hybrid_pubmedbert.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/hybrid_pubmedbert.ipynb) for the model comparison.

---

## Hybrid retrieval (RRF)

Corresponds to `# ---------- Hybrid ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `K_RRF` | 30, 60, 100, 150, 200 | **150** | RRF constant in `1 / (k_rrf + rank)` |
| BM25 weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for BM25 RRF scores |
| Dense weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for Dense RRF scores |

**Decision:** Equal weights (`1.0 / 1.0`) with `K_RRF=150` selected via grid search (25 configs). Best MeanR@2000 = 0.9012.

See [notebooks/hybrid.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/hybrid.ipynb) for the RRF grid search.

---

## Stage 2 rerank and post-rerank fusion

Corresponds to `# ---------- Reranker + Evidence (global: PubMed/literature corpus) ----------` and `# ---------- RRF fusion (Hybrid + Rerank, top-10) ----------` in [workflow_config_full.env](../workflow_config_full.env).

### Cross-encoder rerank

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `--candidate-limit` | — | **2000** | Stage-1 candidates per query (clamped to [30, 2000]) |
| `--model` | MiniLM, BGE v2 | **BAAI/bge-reranker-v2-m3** | Cross-encoder model |
| `--model-device` | cpu / cuda / mps / auto | **auto** | Device selection |
| `--model-batch` | — | **16** | Cross-encoder batch size |
| `--model-max-length` | 200, 512 | **512** | Token truncation length |
| `--ks-recall` | — | 50,100,200,300,400,500,1000,2000,5000 | Recall K values for evaluation |

| Model | max_length | MAP@10 (test avg) |
|-------|------------|-------------------|
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 512 | 0.385 |
| BAAI/bge-reranker-v2-m3 | 200 | 0.351 |
| **BAAI/bge-reranker-v2-m3** | **512** | **0.418** |

**Decision:** `candidate-limit=2000` from recall curves (95% of max hybrid recall). BGE v2 at `max_length=512` is the default reranker.

See [notebooks/analyze_results.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/analyze_results.ipynb) for recall curves and comparisons.

### Post-rerank fusion

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `k_rrf` (fusion) | 30, 60 | **60** | RRF constant for BGE + Hybrid fusion |
| `w_bge` | 0.5 – 1.0 | **0.8** | BGE reranker weight |
| `w_hybrid` | 0.0 – 0.5 | **0.2** | Hybrid stage-1 weight |
| `pool_top_rerank` | 50, 100, 200 | **50** | Top-K from reranker for fusion pool |
| `pool_top_hybrid` | 20, 50, 100, 200 | **50** | Top-K from hybrid for fusion pool |

**Decision:** Reranker-dominant fusion (`w_bge=0.8, w_hybrid=0.2`) outperforms pure reranker output. Optimal for MAP@10: `k_rrf=30, pool_rerank=50, pool_hybrid=50`. Optimal for Recall@50: `k_rrf=60, pool_rerank=100, pool_hybrid=200`.

See [notebooks/analyze_workflow_results.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/analyze_workflow_results.ipynb) for the fusion sweep.

---

## Post-fusion score cutoff (t-star)

Corresponds to `# ---------- Post-fusion global rerank score cutoff (t*) ----------` in [workflow_config_full.env](../workflow_config_full.env).

Optional filter after post-rerank RRF fusion (t*): `RERANK_TSTAR_ENABLE`, `RERANK_TSTAR`, floor/cap variables as documented in the env block. When enabled, writes `rerank/post_rerank_fusion_tstar/` and `rerank/post_rerank_fusion_snippet_tstar/`.

---

## Snippet-RRF route (optional)

Corresponds to `# ---------- Snippet RRF route (--snippet-rrf) ----------` in [workflow_config_full.env](../workflow_config_full.env).

The snippet-RRF route adds snippet window reranking and final doc/snippet fusion:

- `rerank/post_rerank_fusion_snippet/` → `snippet/snippet_rerank/`
- Doc-side fused runs + `snippet/snippet_rerank/` → `snippet/snippet_doc_fusion/`
- Evidence / generation: `evidence/evidence_snippet/`, `generation/generation_snippet/`

### Snippet extraction + window rerank

| Parameter | Suggested range | Default | Notes |
|-----------|-----------------|---------|-------|
| `SNIPPET_N_DOCS` | 50 – 200 | **100** | Top docs per query used for windowing |
| `SNIPPET_WINDOW_SIZE` | 2 – 5 | **3** | Sentences per window |
| `SNIPPET_WINDOW_STRIDE` | 1 – 2 | **1** | Sliding stride in sentences |
| `SNIPPET_TOP_W` | 4 – 16 | **8** | Top windows per doc after Stage A |
| `SNIPPET_DENSE_MODEL` | — | **abhinand/MedEmbed-small-v0.1** | Stage A dense scorer |
| `SNIPPET_DENSE_BATCH` | — | **256** | Stage A batch size |
| `SNIPPET_CE_MODEL` | — | **BAAI/bge-reranker-v2-m3** | Stage B cross-encoder |
| `SNIPPET_CE_BATCH` | — | **84** | CE batch size |
| `SNIPPET_CE_MAX_LENGTH` | 200 – 512 | **512** | CE truncation length |

### Final fusion (`snippet/snippet_doc_fusion`)

| Parameter | Suggested range | Default | Notes |
|-----------|-----------------|---------|-------|
| `SNIPPET_FINAL_POOL` | 50 – 200 | **SNIPPET_N_DOCS** | Pool size on both sides of final fusion |
| `SNIPPET_RRF_K` | 30 – 100 | **60** | RRF constant in \(1 / (k + rank)\) |
| `SNIPPET_RRF_W_DOCS` | 0.5 – 0.9 | **0.8** | Weight on doc-side post-rerank fusion |
| `SNIPPET_RRF_W_SNIPPET` | 0.1 – 0.5 | **0.2** | Weight on `snippet/snippet_rerank` |

See [notebooks/snippet_extraction.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/snippet_extraction.ipynb) and [notebooks/snippet_extraction_MedCPT.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/snippet_extraction_MedCPT.ipynb).

---

## Listwise reranking (optional)

Corresponds to `# ---------- Listwise Reranking (RankZephyr, separate container) ----------` in [workflow_config_full.env](../workflow_config_full.env).

Listwise stages run in a dedicated flow (`Dockerfile.listwise`, scripts under `listwise_script/`). Variables such as `RUN_LISTWISE`, `LISTWISE_*`, and listwise fusion weights are documented in that env block.

---

## Evidence (contexts)

Corresponds to `# ---------- Evidence (post-rerank JSON + contexts JSONL) ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Parameter | Suggested range | Default | Notes |
|-----------|-----------------|---------|-------|
| `POST_RERANK_DOC_POOL` | 10 – 200 | **30** | Max docs per query in `post_rerank_*.jsonl`; snippet route may merge `doc_snippet_windows` when `snippet/snippet_rerank/windows/{split}.jsonl` exists |
| `EVIDENCE_TOP_K` / `EVIDENCE_TOP_K_BASELINE` / `EVIDENCE_TOP_K_SNIPPET` | 1 – `POST_RERANK_DOC_POOL` | **10** | Max docs per question in `evidence/*/_contexts.jsonl` |
| `SNIPPET_CONTEXT_TOP_WINDOWS` | **1 or 2** | **2** | Top CE windows per doc; with 2, second kept only if disjoint from the first |

---

## Answer generation (LLM)

Corresponds to `# ---------- Generation (LLM answers from contexts JSONL) ----------` in [workflow_config_full.env](../workflow_config_full.env).

| Parameter | Range tested | Default | Notes |
|-----------|--------------|---------|-------|
| `--temperature` | 0.0, 0.3, 0.8 | **0.0** | LLM sampling temperature |
| `--top-p` | — | **1.0** | Nucleus sampling |
| `--max-contexts` | — | **10** | Cap on evidence passages per question |
| `--max-chars-per-context` | — | **1300** | Truncation length per context |
| Model | — | **llama3.3:latest** | Ollama model (override via `GENERATION_MODEL` / provider) |

**Decision:** Temperature differences were marginal; default `temperature=0.0` for deterministic output.

See [notebooks/generation_test.ipynb](https://github.com/fulaibaowang/BioASQ/blob/main/notebooks/generation_test.ipynb) for temperature and prompt tests.

---

## Retrieval pipeline summary

| Stage | Method | Key defaults |
|-------|--------|--------------|
| 1a | BM25 + RM3 | `fb_docs=20, fb_terms=30, fb_lambda=0.6` |
| 1b | Dense (MedEmbed) | `M=32, ef_construction=200, ef_search=max(meta,topk)` |
| 2 | Hybrid RRF | `K_RRF=150, weights=1.0/1.0, cap=TOP_K` |
| 2 | Rerank (BGE v2) | `candidate_limit` clamped [30,2000], `max_length=512` |
| 2 | Fusion (BGE + Hybrid) | `k_rrf=60, w_bge=0.8, w_hybrid=0.2, pool_top_rerank=50, pool_top_hybrid=50` (pools 200 when snippet route is active) |
| 2.5 (optional) | Snippet rerank + fusion | `SNIPPET_N_DOCS=100, window=3/1, top_w=8, final_pool=SNIPPET_N_DOCS, weights=0.8/0.2` |
| 3 | Generation | `temperature=0.0, max_contexts=10` |

For multi-query fusion, HyDE, and deduplication, see [MULTI_QUERY_HYDE.md](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/query_parsing/MULTI_QUERY_HYDE.md) in the BioASQ repository.
