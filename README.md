# Public scripts (retrieval pipeline)

Run the pipeline via the workflow script and a config env. It supports:

- **Baseline route**: BM25 → Dense → retrieval fusion → cross-encoder → post-rerank RRF (`rerank/post_rerank_fusion/`) → evidence + generation
- **Optional snippet-RRF route**: snippet window rerank (`snippet/snippet_rerank/`) → final fusion (`snippet/snippet_doc_fusion/`) → snippet-based evidence + generation

## Quickstart

### First time on a new machine

**Recommended: Docker** — reproduces Java 21, CUDA 12.8 / PyTorch cu128, Terrier, and Python deps without hand-tuning a venv.

1. From the **BioASQ repo root** (a normal `git clone` so `.git` exists, or set paths in config to absolute locations):
   ```bash
   docker build -t bioasq-pipeline -f Dockerfile .
   ```
2. Mount the repo (or your data) and run the pipeline inside the container, e.g.:
   ```bash
   docker run --rm -it --gpus all \
     -v "$PWD:/app" -w /app \
     bioasq-pipeline \
     bash -lc './scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh \
       --config scripts/public/shared_scripts/workflow_config_baseline.env'
   ```
   Copy `workflow_config_baseline.env` to a private file, set `WORKFLOW_OUTPUT_DIR`, `INPUT_JSONL` and/or `INPUT_BATCH_JSONLS` (`.jsonl` query files; adapt-in from BioASQ JSON with `scripts/public/format/bioasq_json_to_queries_jsonl.py`), `BM25_INDEX_PATH`, `DENSE_INDEX_DIR`, and (for rerank+) `DOCS_JSONL`. Use `HAVE_GROUND_TRUTH=0` when you have no qrels.

**Local venv (optional)** — Python deps for the main image live at repo root: [requirements-docker-pytorch.txt](../../../requirements-docker-pytorch.txt) (PyTorch only; CUDA wheel index) and [requirements-docker.txt](../../../requirements-docker.txt) (everything else). Install a **matching** `torch` for your OS/GPU from [pytorch.org](https://pytorch.org), activate your venv (e.g. `source /path/to/.venv/bin/activate`), then from repo root `pip install -r requirements-docker.txt`. You still need **Java** and OS libraries the Dockerfile installs.

**Where outputs go** — under `$WORKFLOW_OUTPUT_DIR`: `retrieval/{bm25,dense,fusion}/`, `rerank/{cross_encoder,post_rerank_fusion,...}/`, optional `snippet/...`, then `evidence/...` and `generation/...`. Full layout: [docs/output.md](docs/output.md).

## Workflow globals and naming

**Unified retrieval depth:** `TOP_K` is the single config for "how many docs per query" and is used by every stage. Scripts use different CLI flags internally (`k_eval`, `topk`, `cap`, `candidate_limit`), but the pipeline sets them all from `TOP_K` (or stage overrides like `BM25_TOP_K`, `DENSE_TOP_K`, etc.).

Required and common env (set by sourcing a config):

| Env var | Description | Example |
|---------|-------------|---------|
| `WORKFLOW_OUTPUT_DIR` | Base output path for all stages | `output/workflow_run` |
| `INPUT_JSONL` | Primary query stream (optional if batches cover all splits) | `example/training14b_10pct_sample.jsonl` |
| `INPUT_BATCH_JSONLS` | Space-separated extra query `.jsonl` files | `bioasq/13B1_golden.jsonl` |
| `TOP_K` | Retrieval depth for all stages (default 5000) | `1000` or `5000` |
| `RECALL_KS` | Comma-separated K values for recall metrics | `50,100,200,300,400,500` |
| `BM25_INDEX_PATH` | Terrier index directory | Path to index |
| `DENSE_INDEX_DIR` | Dense HNSW index directory | Path to index |
| `DOCS_JSONL` | JSONL corpus for reranker | Path to docs |
| `RUN_BASELINE` | Build baseline evidence/generation (`evidence/evidence_baseline/`, `generation/generation_baseline/`) | `1` |
| `RUN_SNIPPET_RRF` | Enable snippet-RRF route (steps 6–7 + `evidence/evidence_snippet/`, `generation/generation_snippet/`) | `0` |

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

For multi-query fusion (running multiple query variants per stage and fusing with RRF), HyDE query preparation, and smart deduplication, see [../query_parsing/MULTI_QUERY_HYDE.md](../query_parsing/MULTI_QUERY_HYDE.md).

## Scripts

- **Pipeline orchestrator**
  - [run_retrieval_rerank_pipeline.sh](run_retrieval_rerank_pipeline.sh) — end-to-end pipeline from BM25 retrieval to LLM generation.

- **Config templates**
  - [workflow_config_baseline.env](workflow_config_baseline.env), [workflow_config_snippet.env](workflow_config_snippet.env), [workflow_config_full.env](workflow_config_full.env) — example configs with defaults and comments.

- **Indexing** (`index/`)
  - [index/build_bm25_index_from_jsonl_shards.py](index/build_bm25_index_from_jsonl_shards.py) — build a Terrier BM25 index from JSONL shards.
  - [index/build_dense_hnsw_index_from_jsonl_shards.py](index/build_dense_hnsw_index_from_jsonl_shards.py) — build an HNSW dense index from JSONL shards.

- **Stage 1: hybrid retrieval** (`retrieval/`)
  - [retrieval/eval_bm25_rm3.py](retrieval/eval_bm25_rm3.py) — BM25 + RM3 retrieval and evaluation.
  - [retrieval/eval_dense.py](retrieval/eval_dense.py) — dense retrieval over an HNSW index.
  - [retrieval/eval_hybrid.py](retrieval/eval_hybrid.py) — hybrid RRF fusion of BM25 and dense runs.

- **Stage 2 / 2b: document reranking + post-rerank fusion** (`rerank/`)
  - [rerank/rerank_stage2.py](rerank/rerank_stage2.py) — Stage 2 cross-encoder document reranking.
  - [rerank/rerank_rrf_hybrid.py](rerank/rerank_rrf_hybrid.py) — Stage 2b post-rerank RRF fusion of retrieval fusion (`retrieval/fusion/`) and cross-encoder scores (`rerank/post_rerank_fusion/`).
  - [rerank/plot_rerank_eval.py](rerank/plot_rerank_eval.py) — plots recall/MAP curves from rerank outputs.

- **Stage 3 / 3b: snippet-aware evidence and fusion** (`evidence/`)
  - [evidence/snippet_rerank.py](evidence/snippet_rerank.py) — Stage 3 snippet window extraction and two-stage dense + CE reranking.
  - [evidence/build_contexts_from_snippets.py](evidence/build_contexts_from_snippets.py) — Stage 3b snippet-based evidence JSONL from `snippet/snippet_doc_fusion/`.
  - [evidence/build_contexts_from_documents.py](evidence/build_contexts_from_documents.py) — Stage 3b document-based evidence JSONL from `rerank/post_rerank_fusion/`.
  - [evidence/post_rerank_json.py](evidence/post_rerank_json.py) — convert rerank TSV outputs into BioASQ-style JSON runs.

- **Stage 4: LLM answer generation** (`generation/`)
  - [generation/generate_answers.py](generation/generate_answers.py) — LLM answer generation from evidence JSONL (baseline and snippet).
  - [generation/rescue_failed_generation.py](generation/rescue_failed_generation.py) — retry/repair failed generations.

- **Utilities**
  - [compare_result_dirs.py](compare_result_dirs.py) — compare metrics across two pipeline output directories.
  - [logging_config.py](logging_config.py) — shared logging configuration (LOG_LEVEL, LOG_FILE).
  - [retrieval_eval/common.py](retrieval_eval/common.py) — shared retrieval evaluation helpers (metrics, I/O).

## Run format (TSV only)

All stages write runs as TSV with columns: `qid`, `docno`, `rank`, `score`. No parquet or JSON for runs. Same format everywhere for downstream consumption.

## Snippet windows, run log, and logging

- **Snippet windows:** Snippet evidence uses windows written by **split** (logical id, e.g. `13B1_golden`). Files live under `snippet/snippet_rerank/windows/` as `{split}.jsonl`. The pipeline and `build_contexts_from_snippets.py` use this name; no separate "windows stem" is used.
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

3. Outputs appear under `$WORKFLOW_OUTPUT_DIR/retrieval/{bm25,dense,fusion}/`. If `DOCS_JSONL` is set and you do not pass `--no-rerank`, the reranker runs and writes to `rerank/cross_encoder/`, then post-rerank fusion under `rerank/post_rerank_fusion/`. Set `RERANK_DISABLE_METRICS=1` when you have no ground truth. If a stage's key output already exists (e.g. fusion's `ranked_test_avg.csv` or `metrics.csv`), that stage is skipped; when retrieval fusion is done, the reranker uses those results and does not rerun earlier stages.

4. Enable snippet-RRF (optional):
   - CLI: add `--snippet-rrf`
   - Or set in your config: `RUN_SNIPPET_RRF=1`
   - Outputs:
     - `snippet/snippet_rerank/` (snippet window rerank outputs)
     - `snippet/snippet_doc_fusion/` (final fusion outputs)
     - `evidence/evidence_snippet/`, `generation/generation_snippet/` (snippet-based evidence/generation)

## Prerequisites

- Python env with dependencies (pyterrier, hnswlib, sentence-transformers, pandas, etc.)
- Terrier index (BM25)
- Dense HNSW index (from `shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py`)
- Question JSONs with `questions` and optional `documents` (for evaluation)
