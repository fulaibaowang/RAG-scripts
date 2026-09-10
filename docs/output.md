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
  - `rerank/post_rerank_fusion/` – post-rerank RRF of `retrieval/fusion/` + cross-encoder (default pool 50 for the document route).
  - `rerank/post_rerank_fusion_snippet/` – same fusion with pool 200 when the snippet route (or run-both) needs a wider doc pool.
  - Optional t* filtered runs: `rerank/post_rerank_fusion_tstar/`, `rerank/post_rerank_fusion_snippet_tstar/` when `RERANK_TSTAR_ENABLE=1`.

- **Snippet route (when `--snippet-rrf` / `RUN_SNIPPET_RRF=1`)**
  - `snippet/snippet_rerank/` – window extraction, CE rerank, `windows/` JSONL per split.
  - `snippet/snippet_doc_fusion/` – final RRF of doc-side fused runs and snippet-level runs (evidence fusion for the snippet path).

- **Evidence and generation** (when `DOCS_JSONL` is set)
  - `evidence/evidence_baseline/`, `generation/generation_baseline/` – **document route** contexts and answers (one record per document/abstract). The `_baseline` suffix is a legacy on-disk name retained for back-compat with existing run trees.
  - `evidence/evidence_snippet/`, `generation/generation_snippet/` – snippet-route contexts and answers.
  - With `GENERATION_MODE=claims`, distillation intermediates (`<split>_claims_cache.jsonl`, `<split>_distilled_contexts.jsonl`) sit next to the contexts file, and answers are written as `<split>_distilled_answers.jsonl`.
  - With `CITATION_GRANULARITY=sentence`, the attribution stage writes `<split>_[distilled_]answers_attributed.jsonl` alongside the answers it read.

## JSONL schemas

Every JSONL file is **one JSON object per line**, and each stage passes unknown fields through
untouched. Question identity on the wire is always **`query_id`**, **`query_text`**, **`query_type`**,
on read and on write (nested **`bioasq`** and duplicate **`id` / `body` / `type`** fields are dropped
when loading). Legacy lines that only have `id` / `body` / `type` are still accepted on read and
normalized to `query_*`. Converting to and from a task's own submission JSON is the consuming repo's
job, not this pipeline's — for BioASQ see
[`bioasq_json_to_queries_jsonl.py`](https://github.com/fulaibaowang/BioASQ/blob/main/scripts/public/format/bioasq_json_to_queries_jsonl.py).

### Input query JSONL

Each line is a single question. **`query_id`** is required (after normalization from legacy `id` / `qid` / `bioasq.id` if needed). **`query_text`** is the retrieval topic. **`query_type`** is **optional** and task-specific — BioASQ uses it for the task label (`summary`, `yesno`, `factoid`, `list`); corpora with no such notion simply omit it. Optional gold for eval is a **`documents`** array of corpus `docno`s.

```json
{
  "query_id": "67d723d918b1e36f2e000039",
  "query_text": "Are there biomarkers of depression?",
  "query_type": "summary"
}
```

Docids are whatever your corpus uses — the pipeline never parses them. The examples below are
BioASQ/PubMed, matching the demo data and the Docker image. A TREC-RAG/ClimbMix line carries no
`query_type` and uses shard-style docids, and runs through the same stages unchanged:

```json
{
  "query_id": "rag2026-0",
  "query_text": "I'm on a hospital nursing DEI council that has to recommend a three-year plan..."
}
```

### Corpus JSONL

Each line is a document the indexes and evidence stages look up by `docno`:

```json
{ "docno": "your-id", "title": "...", "text": "..." }
```

Index build and a new-corpus checklist: [USAGE.md](USAGE.md#indexing).

### Post-rerank JSONL

Carries **`query_*`** plus retrieved **`doc_ids`** in rank order (no document URLs or text here).

```json
{
  "query_id": "680fe1e3353a4a2e6b00000f",
  "query_text": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?",
  "query_type": "yesno",
  "doc_ids": ["26173390", "28431642", "21453671", "30498395", "12741168"]
}
```

On the snippet route the same record also carries the selected windows per document:

```json
{
  "query_id": "680fe1e3353a4a2e6b00000f",
  "query_text": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?",
  "query_type": "yesno",
  "doc_ids": ["26173390", "28431642"],
  "doc_snippet_windows": {
    "26173390": [
      { "window_idx": 2, "ce_score": 12.5 },
      { "window_idx": 7, "ce_score": 9.1 }
    ],
    "28431642": [
      { "window_idx": 0, "ce_score": 11.0 }
    ]
  }
}
```

The exact shapes at each snippet-route step are in [USAGE.md](USAGE.md#snippet-route-jsonl-shapes).

### Generation output JSONL (`*_answers.jsonl`)

Written by `generation/generate_answers.py` from a **contexts** JSONL (`evidence/evidence_*/*_contexts.jsonl`). Each output line is the input record **plus** model fields. On success: **`ideal_answer`** (string), **`evidence_ids`** (strings matching context `id` values — `<docid>-<n>`, e.g. `PMID-1` on PubMed, `shard_00122_5199-1` on ClimbMix), and for `yesno` / `factoid` / `list` also **`exact_answer`**. On failure, those may be null and an **`error`** string is set.

An optional post-generation pass can attribute each answer *sentence* back to the documents behind it, adding an `answer_sentences` field and leaving the rest of the row untouched — see `CITATION_GRANULARITY` in [PARAMETERS.md](PARAMETERS.md).

```json
{
  "query_id": "680fe1e3353a4a2e6b00000f",
  "query_text": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?",
  "query_type": "yesno",
  "doc_ids": ["26173390", "28431642"],
  "contexts": [
    {
      "id": "26173390-1",
      "doc_id": "26173390",
      "text": "Title: …\n\nAbstract: …"
    }
  ],
  "ideal_answer": "No. SNPs are defined as common variants (often ≥1% frequency), whereas “mutation” often denotes rarer or pathogenic change; usage overlaps and context matters.",
  "evidence_ids": ["26173390-1", "28431642-1"]
}
```

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

## Logs, verbosity, and sidecar artifacts

Run files stay **TSV** under each stage’s `runs/` as in the introduction above (`qid`, `docno`, `rank`, `score`).

### Snippet windows

Snippet evidence uses per-split window files written under `snippet/snippet_rerank/windows/` as `{split}.jsonl` (logical split id, for example a golden batch name). The orchestrator and `build_snippet_contexts.py` use this layout; there is no separate “windows stem” setting.

### Pipeline run log

The shell orchestrator appends a line-oriented run log to `$WORKFLOW_OUTPUT_DIR/pipeline_run.log` (override with `PIPELINE_RUN_LOG`). Each line records timestamp, step name, and duration or `skip`. A short config snapshot (steps, output dir, config file, `RUN_SNIPPET_RRF`) is written at start; an `end` line is written when the pipeline finishes.

### Python logging

Pipeline Python steps (`rerank_snippets`, `build_snippet_contexts`, `build_retrieval_jsonl`, generation helpers, …) read `LOG_LEVEL` (default `INFO`) and `LOG_FILE`. When `LOG_FILE` is set (default: `$WORKFLOW_OUTPUT_DIR/pipeline.log`), they attach a file handler so script logs go there. Use `LOG_LEVEL=DEBUG` or unset `LOG_FILE` to change behaviour.

### Hugging Face / Transformers verbosity

The orchestrator sets `HF_HUB_DISABLE_PROGRESS_BARS=1` and `TRANSFORMERS_VERBOSITY=error` so model download and “Loading weights” output does not flood batch logs or the console. Set `HF_HUB_DISABLE_PROGRESS_BARS=0` or `TRANSFORMERS_VERBOSITY=info` if you want progress output.
