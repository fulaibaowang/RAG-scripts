# Public scripts (retrieval pipeline)

Run the pipeline via the workflow script and a config env. It supports:

- **Baseline route**: BM25 → Dense → Hybrid → Reranker → RRF fusion (`rerank_hybrid/`) → evidence + generation
- **Optional snippet-RRF route**: snippet window rerank (`snippet_rerank/`) → final fusion (`snippet_rrf/`) → snippet-based evidence + generation

## Workflow globals and naming

**Unified retrieval depth:** `TOP_K` is the single config for "how many docs per query" and is used by every stage. Scripts use different CLI flags internally (`k_eval`, `topk`, `cap`, `candidate_limit`), but the pipeline sets them all from `TOP_K` (or stage overrides like `BM25_TOP_K`, `DENSE_TOP_K`, etc.).

Required and common env (set by sourcing a config):

| Env var | Description | Example |
|---------|-------------|---------|
| `WORKFLOW_OUTPUT_DIR` | Base output path for all stages | `output/workflow_run` |
| `TRAIN_JSON` | Path to dev question sets JSON | `example/training14b_10pct_sample.json` |
| `TEST_BATCH_JSONS` | Space-separated paths to test batch JSONs | `bioasq/13B1_golden.json` |
| `TOP_K` | Retrieval depth for all stages (default 5000) | `1000` or `5000` |
| `RECALL_KS` | Comma-separated K values for recall metrics | `50,100,200,300,400,500` |
| `BM25_INDEX_PATH` | Terrier index directory | Path to index |
| `DENSE_INDEX_DIR` | Dense HNSW index directory | Path to index |
| `DOCS_JSONL` | JSONL corpus for reranker | Path to docs |
| `RUN_BASELINE` | Build baseline evidence/generation (`evidence_baseline/`, `generation_baseline/`) | `1` |
| `RUN_SNIPPET_RRF` | Enable snippet-RRF route (steps 6–7 + `evidence_snippet/`, `generation_snippet/`) | `0` |

**Full options:** `workflow_config_full.env` lists every parameter with comments, including stage-specific overrides:

- **BM25:** `BM25_TOP_K`, `BM25_JAVA_MEM`, `BM25_THREADS`, `BM25_RM3_FEEDBACK_POOL`, `BM25_RM3_FB_DOCS`, `BM25_RM3_FB_TERMS`, `BM25_RM3_LAMBDA`, `BM25_INCLUDE_BASELINE`, `BM25_NO_EVAL`, `BM25_SAVE_RUNS`, `BM25_SAVE_PER_QUERY`, `BM25_SAVE_ZERO_RECALL`, …
- **Dense:** `DENSE_TOP_K`, `DENSE_EF_SEARCH`, `DENSE_EF_CAP`, `DENSE_BATCH_SIZE`, `DENSE_DEVICE`, `DENSE_MODEL_NAME`, `DENSE_NO_EVAL`, `DENSE_SAVE_PER_QUERY`
- **Hybrid:** `HYBRID_CAP`, `HYBRID_K_MAX_EVAL`, `HYBRID_MODE`, `HYBRID_K_RRF`, `HYBRID_W_BM25`, `HYBRID_W_DENSE`, `HYBRID_WEIGHTS`, `HYBRID_JOBS`, `HYBRID_NO_EVAL`, `HYBRID_NO_PLOTS`, …
- **Reranker:** `RERANK_CANDIDATE_LIMIT`, `RERANK_KS_RECALL`, `RERANK_MODEL`, `RERANK_MODEL_DEVICE`, `RERANK_MODEL_BATCH`, `RERANK_DISABLE_METRICS`, …
- **Snippet-RRF:** `SNIPPET_N_DOCS`, `SNIPPET_WINDOW_SIZE`, `SNIPPET_WINDOW_STRIDE`, `SNIPPET_TOP_W`, `SNIPPET_DENSE_MODEL`, `SNIPPET_DENSE_DEVICE`, `SNIPPET_DENSE_BATCH`, `SNIPPET_CE_MODEL`, `SNIPPET_CE_DEVICE`, `SNIPPET_CE_BATCH`, `SNIPPET_CE_MAX_LENGTH`, `SNIPPET_FINAL_POOL`, `SNIPPET_RRF_K`, `SNIPPET_RRF_W_DOCS`, `SNIPPET_RRF_W_SNIPPET`, `SNIPPET_CONTEXT_TOP_WINDOWS`

**Local config:** For machine-specific paths, use `scripts/private_scripts/config.env` (edit `REPO_ROOT` and paths there; same variable names as above).

## Caps and K (per stage)

| Stage | Internal flag(s) | Set from env |
|-------|-------------------|--------------|
| BM25 | `k_eval` | `TOP_K` or `BM25_TOP_K` |
| Dense | `topk` | `TOP_K` or `DENSE_TOP_K` |
| Hybrid | `bm25_topk`, `cap`, `k_max_eval` | `TOP_K` or `HYBRID_*` |
| Reranker | `candidate_limit` | Derived from `min(RERANK_CANDIDATE_LIMIT, HYBRID_CAP)`, then clamped to 100–2000 |

Reranker can only use as many docs as hybrid produces; the pipeline clamps the value to at least 100 and at most 2000. For small corpora (e.g. &lt; 5k docs), set `TOP_K` lower (e.g. 1000) and a smaller `RECALL_KS` so metrics stay valid.

For typical tuning ranges (e.g. RM3, HNSW, RRF, reranker) and links to the notebooks, see [docs/PARAMETERS.md](docs/PARAMETERS.md).

## Scripts

- **Pipeline orchestrator**
  - [run_retrieval_rerank_pipeline.sh](run_retrieval_rerank_pipeline.sh) — end-to-end pipeline from BM25 retrieval to LLM generation.

- **Config templates**
  - [workflow_config_baseline.env](workflow_config_baseline.env), [workflow_config_snippet.env](workflow_config_snippet.env), [workflow_config_full.env](workflow_config_full.env) — example configs with defaults and comments.

- **Indexing** (`index/`)
  - [index/build_bm25_index_from_jsonl_shards.py](index/build_bm25_index_from_jsonl_shards.py) — build a Terrier BM25 index from JSONL shards.
  - [index/build_dense_hnsw_index_from_jsonl_shards.py](index/build_dense_hnsw_index_from_jsonl_shards.py) — build an HNSW dense index from JSONL shards.

- **Stage 1 retrieval** (`retrieval/`)
  - [retrieval/eval_bm25_rm3.py](retrieval/eval_bm25_rm3.py) — BM25 + RM3 retrieval and evaluation.
  - [retrieval/eval_dense.py](retrieval/eval_dense.py) — dense retrieval over an HNSW index.
  - [retrieval/eval_hybrid.py](retrieval/eval_hybrid.py) — hybrid RRF fusion of BM25 and dense runs.

- **Stage 2 reranking** (`rerank/`)
  - [rerank/rerank_stage2.py](rerank/rerank_stage2.py) — cross-encoder document reranking.
  - [rerank/rerank_rrf_hybrid.py](rerank/rerank_rrf_hybrid.py) — RRF fusion of hybrid and reranker scores (`rerank_hybrid/`).
  - [rerank/plot_rerank_eval.py](rerank/plot_rerank_eval.py) — plots recall/MAP curves from rerank outputs.

- **Stage 2.5 snippet + evidence** (`evidence/`)
  - [evidence/snippet_rerank.py](evidence/snippet_rerank.py) — snippet window extraction and two-stage dense + CE reranking.
  - [evidence/build_contexts_from_snippets.py](evidence/build_contexts_from_snippets.py) — build snippet-based evidence JSONL from `snippet_rrf/`.
  - [evidence/build_contexts_from_documents.py](evidence/build_contexts_from_documents.py) — build document-based evidence JSONL from `rerank_hybrid/`.
  - [evidence/post_rerank_json.py](evidence/post_rerank_json.py) — convert rerank TSV outputs into BioASQ-style JSON runs.

- **Stage 3 generation** (`generation/`)
  - [generation/generate_answers.py](generation/generate_answers.py) — LLM answer generation from evidence JSONL.
  - [generation/rescue_failed_generation.py](generation/rescue_failed_generation.py) — retry/repair failed generations.

- **Utilities**
  - [compare_result_dirs.py](compare_result_dirs.py) — compare metrics across two pipeline output directories.
  - [logging_config.py](logging_config.py) — shared logging configuration (LOG_LEVEL, LOG_FILE).
  - [retrieval_eval/common.py](retrieval_eval/common.py) — shared retrieval evaluation helpers (metrics, I/O).

## Run format (TSV only)

All stages write runs as TSV with columns: `qid`, `docno`, `rank`, `score`. No parquet or JSON for runs. Same format everywhere for downstream consumption.

## Snippet windows, run log, and logging

- **Snippet windows:** Snippet evidence uses windows written by **split** (logical id, e.g. `13B1_golden`). Files live under `snippet_rerank/windows/` as `{split}.jsonl`. The pipeline and `build_contexts_from_snippets.py` use this name; no separate "windows stem" is used.
- **Pipeline run log:** The script appends a run log to `$WORKFLOW_OUTPUT_DIR/pipeline_run.log` (override with `PIPELINE_RUN_LOG`). Each line has timestamp, step name, and duration or `skip`. A short config snapshot (steps, output dir, config file, `RUN_SNIPPET_RRF`) is written at start; an `end` line is written when the pipeline finishes.
- **Logging config:** Pipeline Python scripts (snippet_rerank, build_contexts_from_snippets, post_rerank_json, generation, etc.) read `LOG_LEVEL` (default `INFO`) and `LOG_FILE`. When `LOG_FILE` is set (default: `$WORKFLOW_OUTPUT_DIR/pipeline.log`), they add a file handler so script logs go there. Set `LOG_LEVEL=DEBUG` or unset `LOG_FILE` to change behaviour.
- **Model-loading progress:** The pipeline sets `HF_HUB_DISABLE_PROGRESS_BARS=1` and `TRANSFORMERS_VERBOSITY=error` so Hugging Face “Loading weights” / “Materializing param” lines do not flood sbatch `.err` or console. Override with `HF_HUB_DISABLE_PROGRESS_BARS=0` or `TRANSFORMERS_VERBOSITY=info` if you want progress output.

## Running the pipeline

1. Use or copy an example config:
   - `workflow_config_baseline.env` – baseline defaults (in this folder)
   - `workflow_config_full.env` – full parameter list and comments (in this folder)
   - `scripts/private_scripts/config.env` – local paths (edit `REPO_ROOT` and index paths)

2. Run with a config file (from repo root):
   ```bash
   cd /path/to/BioASQ
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/config.env
   ```
   Or: `./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -c scripts/public/shared_scripts/workflow_config_baseline.env`

   To run only retrieval (BM25, Dense, Hybrid) and skip the reranker: add `--no-rerank`:
   ```bash
   ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh -c config.env --no-rerank
   ```

   You can still source then run: `source scripts/public/shared_scripts/workflow_config_baseline.env && ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh`

4. Enable snippet-RRF (optional):
   - CLI: add `--snippet-rrf`
   - Or set in your config: `RUN_SNIPPET_RRF=1`
   - Outputs:
     - `snippet_rerank/` (snippet window rerank outputs)
     - `snippet_rrf/` (final fusion outputs)
     - `evidence_snippet/`, `generation_snippet/` (snippet-based evidence/generation)

3. Outputs appear under `$WORKFLOW_OUTPUT_DIR/bm25/`, `dense/`, `hybrid/`. If `DOCS_JSONL` is set and you do not pass `--no-rerank`, the reranker step runs and writes to `rerank/`. Set `RERANK_DISABLE_METRICS=1` when you have no ground truth. If a stage's key output already exists (e.g. hybrid's `ranked_test_avg.csv` or `metrics.csv`), that stage is skipped; when hybrid is done, the reranker uses hybrid results and does not rerun earlier stages.

## Prerequisites

- Python env with dependencies (pyterrier, hnswlib, sentence-transformers, pandas, etc.)
- Terrier index (BM25)
- Dense HNSW index (from `shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py`)
- Question JSONs with `questions` and optional `documents` (for evaluation)
