# Tunable Parameters

Pipeline config (env vars and script mapping): [scripts/public/README.md](../README.md).

## Recommended Operating Ranges

| Parameter | Suggested Range | Default | Constraint |
|-----------|----------------|---------|------------|
| `TOP_K` | 1000 – 5000 | 5000 | Retrieval depth for BM25 and Dense |
| `HYBRID_CAP` | 1000 – 2000 | TOP_K | `≤ TOP_K` |
| `RERANK_CANDIDATE_LIMIT` | 200 – 1000 | TOP_K (clamped [30, 2000]) | `≤ HYBRID_CAP` |
| `pool_top_rerank` | 50 – 200 | 50 | `≤ RERANK_CANDIDATE_LIMIT` |
| `pool_top_hybrid` | 50 – 200 | 50 | `≤ HYBRID_CAP` |

Each stage's output feeds the next, so values must satisfy: `pool_top ≤ RERANK_CANDIDATE_LIMIT ≤ HYBRID_CAP ≤ TOP_K`. Staying within the suggested ranges guarantees no constraint violations.

## BM25 + RM3

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `fb_docs` | 5 – 20 | **20** | Feedback documents for term extraction |
| `fb_terms` | 10 – 30 | **30** | Expanded terms to add |
| `fb_lambda` | 0.6 – 0.8 | **0.6** | Interpolation weight (0 = pure RM3, 1 = pure original) |

**Decision:** Aggressive config (`fb_docs=20, fb_terms=30, fb_lambda=0.6`) achieved the highest recall while balanced configs had marginally higher MAP@10 but lower recall.

See [notebooks/bm25_test.ipynb](../../../notebooks/bm25_test.ipynb) for the RM3 parameter sweep.

## Dense Retrieval (HNSW)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `M` | — | **32** | HNSW graph degree (common value, not swept) |
| `ef_construction` | — | **200** | HNSW build-time quality (common value, not swept) |
| `ef_search` | 5000 – 20000 | **max(meta, DENSE_TOP_K)** | Runtime auto-promoted to `≥ topk`; stored as 100 in meta.json. Capped by `DENSE_EF_CAP` if set |
| `batch_size` | — | **128** (index) / **256** (retrieval) | Embedding batch size (not swept) |
| `max_seq_length` | — | **512** | Encoder truncation length (~2.8% of docs truncate at 512) |
| `hnsw_space` | — | **cosine** | Distance metric |

**Decision:** Only `ef_search` was swept (5000/10000/20000); differences in MeanR@5000 were marginal (0.845 → 0.850). Other HNSW parameters use standard values. The retrieval script auto-promotes `ef_search` to `max(meta_value, topk)` so the stored default of 100 does not limit deep recall.

See [notebooks/dense_test.ipynb](../../../notebooks/dense_test.ipynb) for details.

### Dense Models Tested

| Model | Result |
|-------|--------|
| [abhinand/MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1) | **Default** — used in production pipeline |
| [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) | Worse hybrid recall than MedEmbed |

**Decision:** MedEmbed is the default dense model. PubMedBERT was tested in a separate hybrid pipeline and produced lower recall.

See [notebooks/hybrid_pubmedbert.ipynb](../../../notebooks/hybrid_pubmedbert.ipynb) for the model comparison.

## Hybrid Retrieval (RRF)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `K_RRF` | 30, 60, 100, 150, 200 | **150** | RRF constant in `1 / (k_rrf + rank)` |
| BM25 weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for BM25 RRF scores |
| Dense weight | 1.0, 2.0, 3.0 | **1.0** | Weight multiplier for Dense RRF scores |

**Decision:** Equal weights (`1.0 / 1.0`) with `K_RRF=150` selected via grid search (25 configs). Equal weights beat both BM25-heavy (2:1, 3:1) and Dense-heavy (1:2, 1:3) configurations. Best MeanR@2000 = 0.9012. `K_RRF` has a small effect — 60 and 200 are nearly identical to 150.

See [notebooks/hybrid.ipynb](../../../notebooks/hybrid.ipynb) for the RRF grid search.

## Stage 2 Rerank (Cross-Encoder)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `--candidate-limit` | — | **2000** | Stage-1 candidates per query to rerank (clamped to [30, 2000]) |
| `--model` | MiniLM, BGE v2 | **BAAI/bge-reranker-v2-m3** | Cross-encoder model |
| `--model-device` | cpu / cuda / mps / auto | **auto** | Device selection |
| `--model-batch` | — | **16** | Cross-encoder batch size |
| `--model-max-length` | 200, 512 | **512** | Cross-encoder token truncation length |
| `--ks-recall` | — | 50,100,200,300,400,500,1000,2000,5000 | Recall K values for evaluation |

### Reranker Models Tested

| Model | max_length | MAP@10 (test avg) |
|-------|-----------|-------------------|
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 512 | 0.385 |
| BAAI/bge-reranker-v2-m3 | 200 | 0.351 |
| **BAAI/bge-reranker-v2-m3** | **512** | **0.418** |

**Decision:** `candidate-limit=2000` chosen from recall curves — the smallest K where hybrid recall reaches 95% of maximum (P=0.95). BGE v2 at `max_length=512` is the default reranker, outperforming MiniLM (MAP@10 0.418 vs 0.385) and BGE at `max_length=200` (0.351).

See [notebooks/analyze_results.ipynb](../../../notebooks/analyze_results.ipynb) for recall curves, reranker comparison, and BM25/Dense/Hybrid baselines.

### Post-Rerank Fusion

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `k_rrf` (fusion) | 30, 60 | **60** | RRF constant for BGE + Hybrid fusion |
| `w_bge` | 0.5 – 1.0 | **0.8** | BGE reranker weight |
| `w_hybrid` | 0.0 – 0.5 | **0.2** | Hybrid stage-1 weight |
| `pool_top_rerank` | 50, 100, 200 | **50** | Top-K from reranker for fusion pool |
| `pool_top_hybrid` | 20, 50, 100, 200 | **50** | Top-K from hybrid for fusion pool |

**Decision:** Reranker-dominant fusion (`w_bge=0.8, w_hybrid=0.2`) outperforms pure reranker output. Optimal for MAP@10: `k_rrf=30, pool_rerank=50, pool_hybrid=50`. Optimal for Recall@50: `k_rrf=60, pool_rerank=100, pool_hybrid=200`.

See [notebooks/analyze_workflow_results.ipynb](../../../notebooks/analyze_workflow_results.ipynb) for the fusion sweep.

## Snippet-RRF route (optional)

The snippet-RRF route adds a snippet window reranking stage and a second fusion stage:

- `rerank/post_rerank_fusion_snippet/` → `snippet/snippet_rerank/` (window extraction + two-stage window selection + CE rerank)
- Doc-side fused runs + `snippet/snippet_rerank/` → `snippet/snippet_doc_fusion/` (final RRF fusion)
- Evidence/generation can then use `evidence/evidence_snippet/` + `generation/generation_snippet/`.

### Snippet extraction + window rerank (`snippet/snippet_rerank/`)

| Parameter | Suggested Range | Default | Notes |
|-----------|----------------|---------|-------|
| `SNIPPET_N_DOCS` | 50 – 200 | **100** | Top docs per query (from `rerank/post_rerank_fusion_snippet`) used for windowing |
| `SNIPPET_WINDOW_SIZE` | 2 – 5 | **3** | Sentences per window |
| `SNIPPET_WINDOW_STRIDE` | 1 – 2 | **1** | Sliding stride in sentences |
| `SNIPPET_TOP_W` | 4 – 16 | **8** | Top windows per doc kept after Stage A |
| `SNIPPET_DENSE_MODEL` | — | **abhinand/MedEmbed-small-v0.1** | Stage A dense scorer for windows |
| `SNIPPET_DENSE_BATCH` | — | **256** | Stage A dense batch size (tune for GPU memory) |
| `SNIPPET_CE_MODEL` | — | **BAAI/bge-reranker-v2-m3** | Stage B cross-encoder reranker for windows |
| `SNIPPET_CE_BATCH` | — | **84** | CE batch size (tune for GPU memory) |
| `SNIPPET_CE_MAX_LENGTH` | 200 – 512 | **512** | CE truncation length |

### Final fusion (`snippet/snippet_doc_fusion/`)

| Parameter | Suggested Range | Default | Notes |
|-----------|----------------|---------|-------|
| `SNIPPET_FINAL_POOL` | 50 – 200 | **SNIPPET_N_DOCS** | Pool size on both sides of final fusion |
| `SNIPPET_RRF_K` | 30 – 100 | **60** | RRF constant in \(1 / (k + rank)\) |
| `SNIPPET_RRF_W_DOCS` | 0.5 – 0.9 | **0.8** | Weight on doc-side post-rerank fusion (`rerank/post_rerank_fusion_snippet`) |
| `SNIPPET_RRF_W_SNIPPET` | 0.1 – 0.5 | **0.2** | Weight on `snippet/snippet_rerank` |

### Snippet evidence contexts

| Parameter | Suggested Range | Default | Notes |
|-----------|----------------|---------|-------|
| `SNIPPET_CONTEXT_TOP_WINDOWS` | **1 or 2** | **2** | Top CE windows per doc; with 2, second is kept only if disjoint from the first |

See also: [notebooks/snippet_extraction.ipynb](../../../notebooks/snippet_extraction.ipynb) and [notebooks/snippet_extraction_MedCPT.ipynb](../../../notebooks/snippet_extraction_MedCPT.ipynb) for window-size and fusion exploration.

## Answer Generation (LLM)

| Parameter | Range Tested | Default | Notes |
|-----------|-------------|---------|-------|
| `--temperature` | 0.0, 0.3, 0.8 | **0.0** | LLM sampling temperature |
| `--top-p` | — | **1.0** | Nucleus sampling (no truncation) |
| `--max-contexts` | — | **10** | Cap on evidence passages per question |
| `--max-chars-per-context` | — | **1300** | Truncation length per context |
| Model | — | **llama3.3:latest** | Ollama model |

**Decision:** Temperature differences are marginal (F_MRR: 0.4994 at 0.3 vs 0.4988 at 0.8; R_SU4_Rec: 0.3478 at 0.8 vs 0.3474 at 0.3). Script default is `temperature=0.0` for deterministic output. Prompt tests are to be updated.

See [notebooks/generation_test.ipynb](../../../notebooks/generation_test.ipynb) for temperature and prompt tests.

## Retrieval Pipeline Summary

| Stage | Method | Key Defaults |
|-------|--------|-------------|
| 1a | BM25 + RM3 | `fb_docs=20, fb_terms=30, fb_lambda=0.6` |
| 1b | Dense (MedEmbed) | `M=32, ef_construction=200, ef_search=max(meta,topk)` |
| 2 | Hybrid RRF | `K_RRF=150, weights=1.0/1.0, cap=TOP_K` |
| 2 | Rerank (BGE v2) | `candidate_limit=min(TOP_K,HYBRID_CAP) clamped [30,2000], max_length=512` |
| 2 | Fusion (BGE + Hybrid) | `k_rrf=60, w_bge=0.8, w_hybrid=0.2, pool_top_rerank=50, pool_top_hybrid=50` (`pool_top_rerank=200, pool_top_hybrid=200` when snippet route is active) |
| 2.5 (optional) | Snippet rerank + fusion | `SNIPPET_N_DOCS=100, window=3/1, top_w=8, final_pool=SNIPPET_N_DOCS, weights=0.8/0.2` |
| 3 | Generation (Llama 3.3) | `temperature=0.0, max_contexts=10` |

## Parameter Constraints

The pipeline cascades `TOP_K` (default 5000) into stage-specific defaults. The following invariants must hold to avoid silent truncation or degraded results.

### Data-flow constraints

```
TOP_K ──► BM25_TOP_K ──────────┐
     ──► DENSE_TOP_K ──────────┤
                                ├──► HYBRID_CAP ──► RERANK_CANDIDATE_LIMIT ──► pool_top_rerank
                                │                                               pool_top_hybrid ◄── HYBRID_CAP
                                │
                                └──► (RRF fusion top-N union) ──► evidence top-10 ──► generation
```

| Constraint | Why | What happens on violation |
|------------|-----|--------------------------|
| `RERANK_CANDIDATE_LIMIT ≤ HYBRID_CAP` | Reranker reads hybrid runs | Pipeline enforces `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)` |
| `pool_top_rerank ≤ RERANK_EFFECTIVE` | Fusion draws from reranker output | Silent truncation — fewer rerank docs in pool than requested |
| `pool_top_hybrid ≤ HYBRID_CAP` | Fusion draws from hybrid output | Silent truncation — fewer hybrid docs in pool than requested |
| `RERANK_EFFECTIVE ≥ 30` | Minimum reranker input | Pipeline clamps to 30 with a warning |
| `RERANK_EFFECTIVE ≤ 2000` | Reranking is expensive beyond this | Pipeline clamps to 2000 with a warning |
| `ef_search ≥ DENSE_TOP_K` | HNSW requires ef ≥ k for k results | Auto-promoted at runtime: `ef = max(meta, topk)` |
| `k_feedback ≥ fb_docs` | RM3 needs enough docs in feedback pool | Not enforced — BM25 feedback pool must cover fb_docs |

### Default safety check

With default `TOP_K=5000`:

| Variable | Resolved value | Safe? |
|----------|---------------|-------|
| `HYBRID_CAP` | 5000 | `≥ pool_top_hybrid (50)` |
| `RERANK_CANDIDATE_LIMIT` | 5000 → clamped 2000 | `≥ pool_top_rerank (50)` |
| `pool_top_rerank` | 50 | `≤ 2000` |
| `pool_top_hybrid` | 50 | `≤ 5000` |
| `ef_search` | max(100, 5000) = 5000 | `≥ DENSE_TOP_K` |

### Potential issues when overriding defaults

1. **Setting `TOP_K` < 50**: Cascades to `HYBRID_CAP` and `RERANK_CANDIDATE_LIMIT`, both of which may fall below the default `pool_top_rerank=50` / `pool_top_hybrid=50`. The pipeline now warns but does not error — fusion silently uses fewer docs.

2. **Setting `RERANK_CANDIDATE_LIMIT` < 50**: After clamping to [30, 2000], the reranker output may be smaller than `pool_top_rerank`. The pipeline warns via `WARNING: RRF_POOL_TOP_RERANK > RERANK_CANDIDATE_LIMIT`.

3. **Setting `DENSE_EF_CAP` too low**: If `DENSE_EF_CAP < DENSE_TOP_K`, deep recall degrades because HNSW can't return accurate results when `ef < k`. The runtime prints ef_search details but does not warn explicitly.

4. **Setting `HYBRID_CAP` much smaller than `RERANK_CANDIDATE_LIMIT`**: Pipeline uses `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)`, so an explicit `RERANK_CANDIDATE_LIMIT=2000` with `HYBRID_CAP=100` silently reduces the reranker input to 100.

