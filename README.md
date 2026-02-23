# Public scripts (retrieval pipeline)

Run the retrieval pipeline (BM25 → Dense → Hybrid, optionally Reranker) via the workflow script and config env.

## Workflow globals and naming

**Unified retrieval depth:** `TOP_K` is the single config for "how many docs per query" and is used by every stage. Scripts use different CLI flags internally (`k_eval`, `topk`, `cap`, `candidate_limit`), but the pipeline sets them all from `TOP_K` (or stage overrides like `BM25_TOP_K`, `DENSE_TOP_K`, etc.).

Required and common env (set by sourcing a config):

| Env var | Description | Example |
|---------|-------------|---------|
| `WORKFLOW_OUTPUT_DIR` | Base output path for all stages | `output/workflow_run` |
| `TRAIN_JSON` | Path to training questions JSON | `example/training14b_10pct_sample.json` |
| `TEST_BATCH_JSONS` | Space-separated paths to test batch JSONs | `bioasq/13B1_golden.json` |
| `TOP_K` | Retrieval depth for all stages (default 5000) | `1000` or `5000` |
| `RECALL_KS` | Comma-separated K values for recall metrics | `50,100,200,300,400,500` |
| `BM25_INDEX_PATH` | Terrier index directory | Path to index |
| `DENSE_INDEX_DIR` | Dense HNSW index directory | Path to index |
| `DOCS_JSONL` | JSONL corpus for reranker (optional) | Path to docs |

**Full options:** `workflow_config_full.env` lists every parameter with comments, including stage-specific overrides:

- **BM25:** `BM25_TOP_K`, `BM25_JAVA_MEM`, `BM25_THREADS`, `BM25_RM3_FEEDBACK_POOL`, `BM25_RM3_FB_DOCS`, `BM25_RM3_FB_TERMS`, `BM25_RM3_LAMBDA`, `BM25_INCLUDE_BASELINE`, `BM25_NO_EVAL`, `BM25_SAVE_RUNS`, `BM25_SAVE_PER_QUERY`, `BM25_SAVE_ZERO_RECALL`, …
- **Dense:** `DENSE_TOP_K`, `DENSE_EF_SEARCH`, `DENSE_EF_CAP`, `DENSE_BATCH_SIZE`, `DENSE_DEVICE`, `DENSE_MODEL_NAME`, `DENSE_NO_EVAL`, `DENSE_SAVE_PER_QUERY`
- **Hybrid:** `HYBRID_CAP`, `HYBRID_K_MAX_EVAL`, `HYBRID_MODE`, `HYBRID_K_RRF`, `HYBRID_W_BM25`, `HYBRID_W_DENSE`, `HYBRID_WEIGHTS`, `HYBRID_JOBS`, `HYBRID_NO_EVAL`, `HYBRID_NO_PLOTS`, …
- **Reranker:** `RERANK_CANDIDATE_LIMIT`, `RERANK_KS_RECALL`, `RERANK_MODEL`, `RERANK_MODEL_DEVICE`, `RERANK_MODEL_BATCH`, `RERANK_DISABLE_METRICS`, …

**Local config:** For machine-specific paths, use `scripts/private_scripts/config.env` (edit `REPO_ROOT` and paths there; same variable names as above).

## Caps and K (per stage)

| Stage | Internal flag(s) | Set from env |
|-------|-------------------|--------------|
| BM25 | `k_eval` | `TOP_K` or `BM25_TOP_K` |
| Dense | `topk` | `TOP_K` or `DENSE_TOP_K` |
| Hybrid | `bm25_topk`, `cap`, `k_max_eval` | `TOP_K` or `HYBRID_*` |
| Reranker | `candidate_limit` | Derived from `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)`, then clamped to 100–2000 |

Reranker can only use as many docs as hybrid produces; the pipeline clamps the value to at least 100 and at most 2000. For small corpora (e.g. &lt; 5k docs), set `TOP_K` lower (e.g. 1000) and a smaller `RECALL_KS` so metrics stay valid.

For typical tuning ranges (e.g. RM3, HNSW, RRF, reranker) and links to the notebooks, see [docs/PARAMETERS.md](../../../docs/PARAMETERS.md).

## Run format (TSV only)

All stages write runs as TSV with columns: `qid`, `docno`, `rank`, `score`. No parquet or JSON for runs. Same format everywhere for downstream consumption.

## Running the pipeline

1. Use or copy an example config:
   - `workflow_config_small.env` – small dataset (TOP_K=1000)
   - `workflow_config_full.env` – full parameter list and comments
   - `scripts/private_scripts/config.env` – local paths (edit `REPO_ROOT` and index paths)

2. Run with a config file (from repo root):
   ```bash
   cd /path/to/BioASQ
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/config.env
   ```
   Or: `./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -c scripts/public/workflow_config_small.env`

   To run only retrieval (BM25, Dense, Hybrid) and skip the reranker: add `--no-rerank`:
   ```bash
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -c config.env --no-rerank
   ```

   You can still source then run: `source scripts/public/workflow_config_small.env && ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh`

3. Outputs appear under `$WORKFLOW_OUTPUT_DIR/bm25/`, `dense/`, `hybrid/`. If `DOCS_JSONL` is set and you do not pass `--no-rerank`, the reranker step runs and writes to `rerank/`. Set `RERANK_DISABLE_METRICS=1` when you have no ground truth. If a stage's key output already exists (e.g. hybrid's `ranked_test_avg.csv`), that stage is skipped; when hybrid is done, the reranker uses hybrid results and does not rerun earlier stages.

## Prerequisites

- Python env with dependencies (pyterrier, hnswlib, sentence-transformers, pandas, etc.)
- Terrier index (BM25)
- Dense HNSW index (from `shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py`)
- Question JSONs with `questions` and optional `documents` (for evaluation)
