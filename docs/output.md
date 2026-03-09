# Pipeline outputs and directories

Most stages in the pipeline write a `metrics.csv` summary, a `runs/` directory with TSV runs (`qid, docno, rank, score`), and optional per-query breakdowns under `$WORKFLOW_OUTPUT_DIR`.

We refer to three fusion steps:

- **Retrieval fusion**: BM25 + dense (RRF over first-stage runs).
- **Post-rerank fusion**: reranker + retrieval fusion (RRF over cross-encoder and retrieval fusion scores).
- **Evidence fusion**: document ranking + snippet ranking (final RRF over `rerank_hybrid` and `snippet_rerank`).

## Stage outputs

- **BM25 / Dense / Retrieval fusion** (standalone eval scripts and within the pipeline):
  - `bm25/`, `dense/`, `hybrid/` under `$WORKFLOW_OUTPUT_DIR` for the pipeline.
  - Each stage writes `metrics.csv`, `runs/`, `per_query/`, and `*_meta.json`.

- **Post-rerank fusion (document ranking)**:
  - `rerank/` – cross-encoder reranker outputs.
  - `rerank_hybrid/` – post-rerank fusion of retrieval fusion (`hybrid/`) and reranker scores (and optionally `rerank_hybrid_200/` when using a wider pool for snippet-RRF).

- **Snippet-RRF route (evidence fusion, when enabled)**:
  - `snippet_rerank/` – document-level runs from snippet window reranking.
  - `snippet_rrf/` – evidence fusion of `rerank_hybrid` (document ranking) and `snippet_rerank` (snippet ranking), used by snippet-based evidence.

- **Evidence and generation** (when `DOCS_JSONL` is set):
  - `evidence_baseline/` and `generation_baseline/` – baseline document-based contexts and LLM answers.
  - `evidence_snippet/` and `generation_snippet/` – snippet-based contexts and LLM answers (snippet-RRF route).

