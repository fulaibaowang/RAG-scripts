# Public scripts (retrieval pipeline)

This directory is the **[RAG-scripts](https://github.com/fulaibaowang/RAG-scripts)** hybrid retrieval and reranking stack: BM25 + RM3, dense HNSW retrieval, retrieval fusion (RRF), cross-encoder reranking, optional post-rerank fusion, optional snippet-RRF, evidence construction, and LLM generation.

## What the pipeline does

- **Baseline route:** BM25 → Dense → retrieval fusion → cross-encoder → post-rerank RRF → baseline evidence → baseline generation.
- **Optional snippet-RRF route:** snippet window rerank → final doc/snippet fusion → snippet evidence → snippet generation.

```mermaid
flowchart TD
  BM25[BM25 + RM3] --> Dense[Dense]
  Dense --> Hybrid["Retrieval fusion (BM25 + dense)"]
  Hybrid --> Rerank["Cross-encoder rerank"]
  Rerank --> RH["Post-rerank fusion"]

  RH --> EB[Baseline evidence]
  EB --> GB[Baseline generation]

  RH --> SR["Snippet rerank"]
  SR --> RRF2["Evidence fusion"]
  RRF2 --> ES[Snippet evidence]
  ES --> GS[Snippet generation]
```

Output layout (directories, fusion names, run format, logs): [docs/output.md](docs/output.md).

## JSONL shapes (examples)

The pipeline uses **one JSON object per line** (JSONL). Wrapped BioASQ files with a top-level `questions` array are not accepted; convert with `scripts/public/format/bioasq_json_to_queries_jsonl.py` if needed. More detail: [docs/PARAMETERS.md](docs/PARAMETERS.md), [docs/USAGE.md](docs/USAGE.md).

### Input query JSONL

Each line is a single question. You must be able to resolve an **`id`** (`id`, `qid`, or `query_id` / `bioasq.id` after merge). The retrieval topic text is read from **`body`**, or **`query`**, or **`question`**. Optional **`type`** is a BioASQ task label (`summary`, `yesno`, `factoid`, `list`).

You can either use a flat BioASQ-shaped object or a thin wrapper with aliases:

```json
{
  "id": "680fe1e3353a4a2e6b00000f",
  "type": "yesno",
  "body": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?"
}
```

```json
{
  "query_id": "67d723d918b1e36f2e000039",
  "query_type": "summary",
  "query_text": "Are there biomarkers of depression?",
  "bioasq": {
    "id": "67d723d918b1e36f2e000039",
    "type": "summary",
    "body": "Are there biomarkers of depression?"
  }
}
```

### Post-rerank JSONL

Produced by `evidence/post_rerank_jsonl.py` (rerank TSV + the same query JSONL). Carries the question fields with gold/oracle keys stripped, plus **`doc_ids`**: the top-`k` **docno** strings from the run (for PubMed, numeric PMIDs), in rank order. There are **no** PubMed URLs in this file; those are added later when exporting to BioASQ JSON.

```json
{
  "id": "680fe1e3353a4a2e6b00000f",
  "type": "yesno",
  "body": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?",
  "doc_ids": ["26173390", "28431642", "21453671", "30498395", "12741168"]
}
```

On the snippet-RRF path, the same script may add **`doc_snippet_windows`** (per-doc snippet windows merged from optional `--windows-jsonl`).

### Generation output JSONL (`*_answers.jsonl`)

Written by `generation/generate_answers.py` from a **contexts** JSONL (e.g. output of `build_contexts_from_*.py`). Each output line is the input record **plus** model fields. On success: **`ideal_answer`** (string), **`evidence_ids`** (strings matching context `id` values, e.g. `PMID-1`), and for `yesno` / `factoid` / `list` also **`exact_answer`**. On failure, those may be null and an **`error`** string is set.

```json
{
  "id": "680fe1e3353a4a2e6b00000f",
  "type": "yesno",
  "body": "Is a single-nucleotide polymorphism (SNP) the same as a mutation?",
  "doc_ids": ["26173390", "28431642"],
  "contexts": [
    {
      "id": "26173390-1",
      "doc_id": "26173390",
      "text": "Title: …\n\nAbstract: …"
    }
  ],
  "ideal_answer": "No. SNPs are defined as common variants (often ≥1% frequency), whereas “mutation” often denotes rarer or pathogenic change; usage overlaps and context matters.",
  "evidence_ids": ["26173390-1", "28431642-1"],
  "exact_answer": "no"
}
```

## Quickstart

**Docker (recommended)** TODO: add demo here

```bash
docker build -t rag-scripts .
```

## Running the pipeline (high level)

1. Copy an example env ([workflow_config_baseline.env](workflow_config_baseline.env), [workflow_config_full.env](workflow_config_full.env)) or create your own.
2. Set `WORKFLOW_OUTPUT_DIR`, query `.jsonl` paths (`INPUT_JSONL` / `INPUT_BATCH_JSONLS`), index paths, and `DOCS_JSONL` for reranking or building evidence.
3. From **this directory**:

   ```bash
   ./run_retrieval_rerank_pipeline.sh --config /path/to/your.env
   ```

   Use `--no-rerank` for retrieval only; `--no-generation` to skip LLM calls; `RUN_SNIPPET_RRF=1` for the snippet route.

Stages whose key outputs already exist are skipped. Per-stage **standalone** commands: [docs/USAGE.md](docs/USAGE.md).

## Entrypoint scripts

| Role | Path |
|------|------|
| Orchestrator | [run_retrieval_rerank_pipeline.sh](run_retrieval_rerank_pipeline.sh) |
| BM25 index | [index/build_bm25_index_from_jsonl_shards.py](index/build_bm25_index_from_jsonl_shards.py) |
| Dense index | [index/build_dense_hnsw_index_from_jsonl_shards.py](index/build_dense_hnsw_index_from_jsonl_shards.py) |
| LLM answers | [generation/generate_answers.py](generation/generate_answers.py) |

Other stage scripts are invoked by the orchestrator; see [docs/USAGE.md](docs/USAGE.md) for direct CLI examples.

## Prerequisites

- Python environment with pipeline dependencies (PyTerrier, hnswlib, sentence-transformers, pandas, …).

Python dependencies are pinned in [requirements-docker-pytorch.txt](requirements-docker-pytorch.txt) and [requirements-docker.txt](requirements-docker.txt).

**Local venv (optional):** install a matching `torch` for your OS/GPU from [pytorch.org](https://pytorch.org), then `pip install -r requirements-docker-pytorch.txt` and `pip install -r requirements-docker.txt`. You still need Java and the system packages installed in the [Dockerfile](Dockerfile).

- Terrier BM25 index and dense HNSW index (see [docs/USAGE.md](docs/USAGE.md)).

## related repo

- [BioASQ](https://github.com/fulaibaowang/BioASQ/blob/main/README.md).
