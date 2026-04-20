# Pipeline outputs and directories

Most stages write a `metrics.csv` summary, a `runs/` directory with TSV runs (`qid, docno, rank, score`), and optional per-query breakdowns under `$WORKFLOW_OUTPUT_DIR`.

We refer to three fusion steps:

- **Retrieval fusion**: BM25 + dense (RRF over first-stage runs), stored under `retrieval/fusion/` (not “hybrid” on disk).
- **Post-rerank fusion**: cross-encoder + retrieval fusion (RRF over CE scores and retrieval fusion scores), under `rerank/post_rerank_fusion/` (and `rerank/post_rerank_fusion_snippet/` for the wider pool used by the snippet route).
- **Evidence fusion**: document ranking + snippet ranking (final RRF over doc-side post-rerank fusion runs and `snippet/snippet_rerank/` runs), output in `snippet/snippet_doc_fusion/`.

## Stage outputs (current layout)

- **BM25 / Dense / Retrieval fusion**
  - `retrieval/bm25/`, `retrieval/dense/`, `retrieval/fusion/` under `$WORKFLOW_OUTPUT_DIR`.
  - Each stage writes `metrics.csv`, `runs/`, `per_query/`, and `*_meta.json` as before.

- **Cross-encoder rerank + post-rerank fusion**
  - `rerank/cross_encoder/` – cross-encoder reranker outputs (TSVs, metrics, figures).
  - `rerank/post_rerank_fusion/` – post-rerank RRF of `retrieval/fusion/` + cross-encoder (default pool 50 for baseline).
  - `rerank/post_rerank_fusion_snippet/` – same fusion with pool 200 when the snippet route (or run-both) needs a wider doc pool.
  - Optional t* filtered runs: `rerank/post_rerank_fusion_tstar/`, `rerank/post_rerank_fusion_snippet_tstar/` when `RERANK_TSTAR_ENABLE=1`.

- **Snippet route (when `--snippet-rrf` / `RUN_SNIPPET_RRF=1`)**
  - `snippet/snippet_rerank/` – window extraction, CE rerank, `windows/` JSONL per split.
  - `snippet/snippet_doc_fusion/` – final RRF of doc-side fused runs and snippet-level runs (evidence fusion for the snippet path).

- **Evidence and generation** (when `DOCS_JSONL` is set)
  - `evidence/evidence_baseline/`, `generation/generation_baseline/` – baseline document contexts and answers.
  - `evidence/evidence_snippet/`, `generation/generation_snippet/` – snippet-route contexts and answers.

- **Listwise** (optional, separate stage; see `listwise_script/run_listwise_rerank.sh`)
  - `listwise_rerank/` – listwise runs, `listwise_fused/`, optional `listwise_fused_sliding/`.
  - `listwise_script/run_listwise_evidence_gen.sh` writes `evidence/evidence_listwise/`, `generation/generation_listwise/` (and `*_listwise_sliding/` when used).

## Migrating an existing workflow directory

If you have outputs from an older pipeline revision, rename/move under the same `$WORKFLOW_OUTPUT_DIR`:

| Old path | New path |
|----------|----------|
| `bm25/` | `retrieval/bm25/` |
| `dense/` | `retrieval/dense/` |
| `hybrid/` | `retrieval/fusion/` |
| `rerank/` (cross-encoder only) | `rerank/cross_encoder/` |
| `rerank_hybrid/` | `rerank/post_rerank_fusion/` |
| `rerank_hybrid_200/` | `rerank/post_rerank_fusion_snippet/` |
| `rerank_hybrid_tstar/` | `rerank/post_rerank_fusion_tstar/` |
| `rerank_hybrid_200_tstar/` | `rerank/post_rerank_fusion_snippet_tstar/` |
| `snippet_rerank/` | `snippet/snippet_rerank/` |
| `snippet_rrf/` | `snippet/snippet_doc_fusion/` |
| `evidence_baseline/` | `evidence/evidence_baseline/` |
| `evidence_snippet/` | `evidence/evidence_snippet/` |
| `generation_baseline/` | `generation/generation_baseline/` |
| `generation_snippet/` | `generation/generation_snippet/` |
| `evidence_listwise/` | `evidence/evidence_listwise/` |
| `generation_listwise/` | `generation/generation_listwise/` |

Example moves (adjust `OUT` to your `$WORKFLOW_OUTPUT_DIR`; create parent dirs with `mkdir -p` as needed):

```bash
OUT=/path/to/workflow_run
mkdir -p "$OUT/retrieval" "$OUT/rerank" "$OUT/snippet" "$OUT/evidence" "$OUT/generation"
[ -d "$OUT/bm25" ] && mv "$OUT/bm25" "$OUT/retrieval/bm25"
[ -d "$OUT/dense" ] && mv "$OUT/dense" "$OUT/retrieval/dense"
[ -d "$OUT/hybrid" ] && mv "$OUT/hybrid" "$OUT/retrieval/fusion"
# Cross-encoder was the old top-level rerank/ directory:
[ -d "$OUT/rerank" ] && [ ! -d "$OUT/rerank/cross_encoder" ] && mv "$OUT/rerank" "$OUT/rerank_ce_tmp" && mkdir -p "$OUT/rerank" && mv "$OUT/rerank_ce_tmp" "$OUT/rerank/cross_encoder"
[ -d "$OUT/rerank_hybrid" ] && mv "$OUT/rerank_hybrid" "$OUT/rerank/post_rerank_fusion"
[ -d "$OUT/rerank_hybrid_200" ] && mv "$OUT/rerank_hybrid_200" "$OUT/rerank/post_rerank_fusion_snippet"
[ -d "$OUT/rerank_hybrid_tstar" ] && mv "$OUT/rerank_hybrid_tstar" "$OUT/rerank/post_rerank_fusion_tstar"
[ -d "$OUT/rerank_hybrid_200_tstar" ] && mv "$OUT/rerank_hybrid_200_tstar" "$OUT/rerank/post_rerank_fusion_snippet_tstar"
[ -d "$OUT/snippet_rerank" ] && mv "$OUT/snippet_rerank" "$OUT/snippet/snippet_rerank"
[ -d "$OUT/snippet_rrf" ] && mv "$OUT/snippet_rrf" "$OUT/snippet/snippet_doc_fusion"
[ -d "$OUT/evidence_baseline" ] && mv "$OUT/evidence_baseline" "$OUT/evidence/evidence_baseline"
[ -d "$OUT/evidence_snippet" ] && mv "$OUT/evidence_snippet" "$OUT/evidence/evidence_snippet"
[ -d "$OUT/generation_baseline" ] && mv "$OUT/generation_baseline" "$OUT/generation/generation_baseline"
[ -d "$OUT/generation_snippet" ] && mv "$OUT/generation_snippet" "$OUT/generation/generation_snippet"
```

If `rerank/` already exists as the new layout, skip the rename that uses `rerank_ce_tmp`. Re-running the pipeline is simpler when in doubt; skip logic only recognizes the new paths.
