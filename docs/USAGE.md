# Usage Guide (shared pipeline)

**This file** documents **generic** indexing, per-stage CLIs, and placeholder paths. For BioASQ-oriented Docker, indexes, and task workflows, see [docs/USAGE.md](../../../docs/USAGE.md) at the repository root.

## Data Preparation

### Parse PubMed XML to JSONL

Convert PubMed baseline XML files to JSONL format with title, abstract, and MeSH metadata:

```bash
python scripts/public/data/parse_pubmed_local.py \
  --input_dir /path/to/pubmed/baseline \
  --output_dir /path/to/pubmed/jsonl_shards \
  --skip_existing
```

**Arguments:**

- `--input_dir`: Directory containing PubMed XML baseline files
- `--output_dir`: Output directory for JSONL shards
- `--skip_existing`: Skip files that already exist in output

## Indexing & Retrieval

### BM25 Index

Build a Terrier-based BM25 index from JSONL shards:

```bash
python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
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

### Dense (HNSW) Index

Build an HNSW dense vector index using SentenceTransformer embeddings:

```bash
python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
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

## Evaluation

Replace paths below with your index, corpus JSONL, and train/test query `.jsonl` files.

### BM25 + RM3

```bash
python scripts/public/shared_scripts/retrieval/eval_bm25_rm3.py \
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
python scripts/public/shared_scripts/retrieval/eval_dense.py \
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
python scripts/public/shared_scripts/retrieval/eval_hybrid.py \
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
python scripts/public/shared_scripts/rerank/rerank_stage2.py \
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

## Full pipeline (orchestrator)

Run all stages with one script and a config file. The script skips stages whose outputs already exist. Use `--no-rerank` for retrieval only; use `--no-generation` to skip LLM generation while still building evidence.

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config /path/to/your.env
```

Example templates in this directory: [workflow_config_baseline.env](../workflow_config_baseline.env), [workflow_config_snippet.env](../workflow_config_snippet.env), [workflow_config_full.env](../workflow_config_full.env). Every variable is commented in [workflow_config_full.env](../workflow_config_full.env); tuning notes: [PARAMETERS.md](PARAMETERS.md). BioASQ + Docker walkthrough: [docs/USAGE.md](../../../../docs/USAGE.md). Public script index (format, adapt-out): [scripts/public/README.md](../../README.md).

### Snippet-RRF route (optional)

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh \
  --config scripts/public/shared_scripts/workflow_config_baseline.env \
  --snippet-rrf
```

Or set `RUN_SNIPPET_RRF=1` (and `RUN_BASELINE` / `RUN_SNIPPET_RRF` as needed) in your env file.

## Output

Directory layout and fusion naming: [output.md](output.md).

## Tuning

See [PARAMETERS.md](PARAMETERS.md) for ranges, constraints, and notebook links.
