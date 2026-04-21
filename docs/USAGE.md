# Usage Guide (shared pipeline)

**This file** documents **generic** indexing, per-stage CLIs, and placeholder paths. Commands assume your current working directory is the **RAG-scripts repository root** (the folder that contains `run_retrieval_rerank_pipeline.sh`, `index/`, `retrieval/`, etc.).

## Indexing

### BM25 index

Build a Terrier-based BM25 index from JSONL shards e.g pubmed corpus or [MS MARCO V2.1 document corpus](https://trec-rag.github.io/annoucements/2024-corpus-finalization/#where-can-i-find-the-corpus):

```bash
python index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/shards/*.jsonl" \
  --index_path "/path/to/indexes/my_bm25_index" \
  --threads 4 \
  --overwrite
```

**Arguments:**

- `--jsonl_glob`: Glob pattern for input JSONL files
- `--index_path`: Output Terrier index directory
- `--threads`: Number of indexing threads
- `--overwrite`: Recreate index if it exists

### Dense (HNSW) index

Build an HNSW dense vector index using SentenceTransformer embeddings:

```bash
python index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/path/to/shards/*.jsonl" \
  --out_dir /path/to/indexes/my_dense_index \
  --model_name "abhinand/MedEmbed-small-v0.1" \
  --device "cuda" \
  --batch_size 128 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100
```

**Arguments:**

- `--model_name`: SentenceTransformer model (default: MedEmbed-small-v0.1)
- `--device`: `cuda`, `cpu`, or `mps` (default: cuda)
- `--batch_size`: Embedding batch size (default: 128)
- `--M`: HNSW graph degree (default: 32)
- `--ef_construction`: HNSW construction parameter (default: 200)
- `--ef_search`: HNSW query-time parameter (default: 100)
- `--max_docs`: Limit index to N docs (for testing)
- `--dedup_pmids`: De-duplicate documents by PMID


## Full pipeline (orchestrator)

Run all stages with one script and a config file. The script skips stages whose outputs already exist. Use `--no-rerank` for retrieval only; use `--no-generation` to skip LLM generation while still building evidence.

```bash
./run_retrieval_rerank_pipeline.sh --config /path/to/your.env
```

Example templates in the repository root: [workflow_config_baseline.env](../workflow_config_baseline.env), [workflow_config_snippet.env](../workflow_config_snippet.env), [workflow_config_full.env](../workflow_config_full.env). Every variable is commented in [workflow_config_full.env](../workflow_config_full.env); tuning notes: [PARAMETERS.md](PARAMETERS.md). BioASQ + Docker walkthrough: [BioASQ docs/USAGE.md](https://github.com/fulaibaowang/BioASQ/blob/main/docs/USAGE.md). Public script index (format, adapt-out): [BioASQ scripts/public/README.md](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/README.md).

### Snippet-RRF route (optional)

```bash
./run_retrieval_rerank_pipeline.sh \
  --config workflow_config_baseline.env \
  --snippet-rrf
```

Or set `RUN_SNIPPET_RRF=1` (and `RUN_BASELINE` / `RUN_SNIPPET_RRF` as needed) in your env file.

## Retrieval

Replace paths below with your index, corpus JSONL, and train/test query `.jsonl` files.

### BM25 + RM3

```bash
python retrieval/eval_bm25_rm3.py \
  --index_path "/path/to/indexes/my_bm25_index/data.properties" \
  --train_json "/path/to/train_queries.json" \
  --test_batch_jsons \
    /path/to/test_batch_a.json \
    /path/to/test_batch_b.json \
  --out_dir "output/eval_bm25_rm3" \
  --threads 4 \
  --k_eval 5000 \
  --k_feedback 50 \
  --rm3_fb_docs 20 \
  --rm3_fb_terms 30 \
  --rm3_lambda 0.6 \
  --save_runs --save_per_query --save_zero_recall
```

**Key arguments:**

- `--k_eval`: Maximum documents to retrieve (default: 5000)
- `--k_feedback`: Documents used for RM3 feedback (default: 50)
- `--rm3_fb_docs`, `--rm3_fb_terms`, `--rm3_lambda`: RM3 tuning
- `--include_bm25`: Also output BM25-only baseline
- `--java_mem`: JVM heap size, e.g. `8g`

### Dense retrieval

```bash
python retrieval/eval_dense.py \
  --index_dir "/path/to/indexes/my_dense_index" \
  --train-jsonl "/path/to/train_queries.jsonl" \
  --test-batch-jsonls \
    /path/to/test_batch_a.jsonl \
    /path/to/test_batch_b.jsonl \
  --out_dir "output/eval_dense" \
  --topk 5000 \
  --ks "50,100,200,500,2000,5000" \
  --ef_search 100 \
  --batch_size 256
```

### Retrieval fusion (BM25 + dense)

```bash
python retrieval/eval_hybrid.py \
  --bm25_runs_dir "output/eval_bm25_rm3/runs" \
  --dense_root "output/eval_dense" \
  --train-jsonl "/path/to/train_queries.jsonl" \
  --test-batch-jsonls \
    /path/to/test_batch_a.jsonl \
    /path/to/test_batch_b.jsonl \
  --out_dir "output/eval_hybrid" \
  --mode "default" \
  --k_rrf 150 \
  --w_bm25 1.0 \
  --w_dense 1.0
```

### Stage 2 rerank (cross-encoder)

```bash
python rerank/rerank_stage2.py \
  --runs-dir "output/eval_hybrid/runs" \
  --docs-jsonl "/path/to/corpus.jsonl" \
  --train-jsonl "/path/to/train_queries.jsonl" \
  --test-batch-jsonls \
    /path/to/test_batch_a.jsonl \
    /path/to/test_batch_b.jsonl \
  --output-dir "output/eval_stage2_rerank" \
  --candidate-limit 2000 \
  --model "cross-encoder/ms-marco-MiniLM-L-12-v2" \
  --model-device "cpu" \
  --model-batch 16 \
  --model-max-length 512
```

## Evidence and generation (after reranking)

`run_retrieval_rerank_pipeline.sh` does **not** stop at the cross-encoder. It continues with:

1. **Post-rerank JSONL** — merge each query split with the ranked doc list from the chosen run TSV (`post_rerank_<split>.jsonl` under `rerank/post_rerank_fusion/` or the snippet fusion tree; see [output.md](output.md)). On the snippet route, optional `--windows-jsonl` also merges **pre-selected** CE windows into compact `doc_snippet_windows` (`--window-size`, `--top-windows`, same env names as `build_contexts_from_snippets.py`).
2. **Contexts JSONL** — attach evidence text from the corpus (`evidence/evidence_*/*_contexts.jsonl`), either full title+abstract per doc (`build_contexts_from_documents.py`) or snippet windows (`build_contexts_from_snippets.py` on the snippet route). Each question row carries `context_mode`: `document` or `snippet`.
3. **LLM generation** — `generate_answers.py` then optional `rescue_failed_generation.py` under `generation/generation_*`. Answer JSONL drops `doc_snippet_windows` and per-context `rejected_windows` (and `window_idx` inside `selected_windows`); `context_mode` is kept.

Use `--no-generation` (or `RUN_GENERATION_*=0`) to build evidence only. Env knobs (`POST_RERANK_DOC_POOL`, `EVIDENCE_TOP_K*`, `GENERATION_*`, backends): [PARAMETERS.md](PARAMETERS.md).

### Manual CLI (baseline: document contexts)

Assume a rerank run TSV (e.g. `output/.../rerank/post_rerank_fusion/runs/best_rrf_<split>_top....tsv`) and the same query `.jsonl` you used for retrieval:

```bash
python evidence/post_rerank_jsonl.py \
  --run-path "output/.../rerank/post_rerank_fusion/runs/best_rrf_my_split_top50.tsv" \
  --query-jsonl "/path/to/my_split.jsonl" \
  --output-path "output/.../rerank/post_rerank_fusion/post_rerank_my_split.jsonl" \
  --top-k 30

python evidence/build_contexts_from_documents.py \
  --post-rerank-jsonl "output/.../rerank/post_rerank_fusion/post_rerank_my_split.jsonl" \
  --corpus-path "/path/to/corpus.jsonl" \
  --output-path "output/.../evidence/my_split_contexts.jsonl" \
  --evidence-top-k 10
```

### Snippet route: JSONL shapes

**Post-rerank** (`post_rerank_jsonl.py` with `--windows-jsonl`): `doc_snippet_windows` is only the compact map `pmid → {"selected_windows": [{"window_idx", "ce_score", "sent_ids", "query_field"?}, ...]}` (no full max-pooled lattice). **Older** `post_rerank_*.jsonl` that stored `pmid → [ flat list of all pooled windows ]` are no longer supported for embedded snippet evidence; regenerate post-rerank with the current script, or point `build_contexts_from_snippets.py` at `--snippet-windows-dir` until you do.

**Contexts** (`*_contexts.jsonl`): full provenance remains on disk (`selected_windows`, `rejected_windows` on each context where applicable) plus top-level `doc_snippet_windows` when present, and `context_mode: "snippet"` or `"document"`.

**Answers** (`*_answers.jsonl`): after `generate_answers.py`, records omit `doc_snippet_windows` and context `rejected_windows`; `context_mode` is preserved.

Set **`GENERATION_BACKEND=openai_compat`**, **`GEN_API_BASE`** (OpenAI-compatible `.../v1` base URL for chat completions), and **`GEN_API_KEY`**, then:

```bash
export GENERATION_BACKEND=openai_compat
export GEN_API_BASE="https://openrouter.ai/api/v1"
export GEN_API_KEY="openrouter-api-key"
python generation/generate_answers.py \
  --input-path "output/.../evidence/my_split_contexts.jsonl" \
  --output-dir "output/.../generation" \
  --schemas-dir "/path/to/prompt/schemas" \
  --model "your-provider-model-id"
```

## Output

Directory layout and fusion naming: [output.md](output.md).

## Tuning

See [PARAMETERS.md](PARAMETERS.md) for ranges, constraints, and notebook links.
