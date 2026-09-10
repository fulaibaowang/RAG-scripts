# AGENTS.md

How to *change* the pipeline safely. For running it, start with [README.md](README.md).

## What this is

A retrieval-and-generation pipeline driven by **one shell orchestrator over an env config file**.
It is not a library — nothing here is imported as a package. Every stage is a standalone script that
reads files and writes files; the orchestrator's job is to pass the right paths between them.

It is **corpus-agnostic**. Docids are opaque strings the pipeline never parses, and documents can be
short abstracts or long web pages. If you find yourself writing code that assumes a docid *shape*, a
document *length*, or a particular task's field names, that is a bug — keep it out.

## The pipeline

Stages run in this order, each writing its own subdirectory of `$WORKFLOW_OUTPUT_DIR` (layout in
[docs/output.md](docs/output.md)):

1. BM25 (+RM3) retrieval — `retrieval/retrieve_bm25.py`
2. Dense retrieval — `retrieval/retrieve_dense.py`
3. Retrieval fusion — `retrieval/fuse_retrieval.py`
4. Cross-encoder rerank — `rerank/rerank_crossencoder.py`
5. Post-rerank fusion — `rerank/fuse_rerank.py`
6. Snippet windows + CE rerank — `evidence/rerank_snippets.py` (snippet route only)
7. Evidence fusion — `rerank/fuse_rerank.py` (snippet route only)

Then evidence (`evidence/build_doc_contexts.py`, `evidence/build_snippet_contexts.py`) and
generation (`generation/generate_answers.py`).

- **Three separate RRF fusions** (retrieval, post-rerank, evidence) are *not* interchangeable.
  Retrieval fusion joins two first-stage retrievers. Post-rerank fusion joins the cross-encoder's
  order with the first-stage order, so a document the reranker hates but retrieval loved is not
  simply discarded. Evidence fusion joins document-level and snippet-level rankings. Use those
  names in code and comments, not "hybrid".
- **`_baseline` in the evidence and generation paths is a legacy on-disk name for the document
  route.** It does not mean "a baseline system". Don't rename it — existing run trees depend on it.
- **`STAGE1_SOURCE`:** `rrf` (default, stages 1–3), `bm25` or `dense` (one retriever), or `external`
  (skip 1–3; stage a precomputed run from `STAGE1_RUN`).
- **`GENERATION_MODE=direct`** (default) is a no-op. **`claims`** distils contexts into slots so
  generation can ingest more evidence than fits a raw prompt. **`facets`** is CI-tested but not
  recommended — don't enable it or suggest it.
- **`claims` is ollama-only**, refused at config time when `GENERATION_BACKEND=openai_compat`. A
  port to chat completions is more than plumbing, and each of these fails quietly: `num_ctx` has
  no hosted equivalent (the truncation policy is to raise `GENERATION_NUM_CTX`, never cut slots);
  `think` is three-state and step-dependent (`distill_common.call_ollama`); and the claim-cache key
  (`distill_common.sha_key`) carries no model or backend, so a second backend would silently reuse
  claims across backends — and growing the key to fix that invalidates every banked cache in every
  consuming repo. Don't land a port without a back-to-back quality comparison. Details:
  [docs/PARAMETERS.md](docs/PARAMETERS.md#context-distillation-optional).

## One entrypoint

```bash
./run_retrieval_rerank_pipeline.sh --config /path/to/your.env
./run_retrieval_rerank_pipeline.sh --help     # every flag and env toggle — authoritative
```

The stage scripts also run standalone (see [docs/USAGE.md](docs/USAGE.md)), which is useful for
debugging one stage. **Don't assemble a workflow out of hand-run stages** — the path conventions
between stages are the orchestrator's job, and hand-running them is how run trees end up subtly
wrong in ways that only surface three stages later.

Config is a plain shell file `source`d with `set -a`, so **every variable in it is exported**, and
it overrides anything already in your environment. Command-line flags in turn override the config.

## The gotcha that costs the most time

**A stage whose key outputs already exist is skipped.** This makes re-running a long pipeline cheap,
but it means:

- Editing a stage's code and re-running the same config **changes nothing** — the stage is skipped
  because its outputs are still sitting there. "I fixed it and the fix didn't take" is almost always
  this, not a bug in the fix.
- Generation additionally checkpoints per completion (`GENERATION_CHECKPOINT=1`, default on) into
  `*_answers.jsonl.partial`, fingerprinted on input/model/params. Deleting the answers file but
  leaving the sidecar silently reuses the old completions.

To get a clean run, point `WORKFLOW_OUTPUT_DIR` somewhere new. It is the only reliable reset.

## Verifying a change without a GPU or an LLM

Don't hand-test generation against a real endpoint. There is a deterministic mock:

```bash
DEMO_DIR=demo OUT_ROOT=/tmp/genmodes bash ci/run_generation_modes.sh
```

This runs every `GENERATION_MODE` against `ci/mock_ollama.py` — no GPU, no network egress — and
asserts that answers come back error-free and that every distilled slot's `doc_id` is a real corpus
docno. That last assertion is the citation-lineage guarantee; if your change breaks it, the change
is wrong. `.github/workflows/ci.yml` runs the same script, the distillation cache-contract tests
(`python3 generation/test_distill_common.py`), and four end-to-end configurations (document route,
chunked documents, single-retriever stage-1, post-rerank fusion disabled) on public demo data in
Docker. Touching the orchestrator's stage wiring usually means adding a matrix entry.

## Conventions to preserve

- **Query identity on the wire is `query_id` / `query_text` / `query_type`**, on read and on write.
  Older `id` / `body` / `type` are normalized on read only — never reintroduce them downstream.
  `query_type` is **optional**; don't write code that requires it.
- **JSONL everywhere**: one JSON object per line, and every stage passes unknown fields through
  untouched. A consumer that ignores a field you added must see no change at all.
- **Context ids are `<docid>-<n>`**, with `doc_id` carrying the real corpus docid. Citation lineage
  back to a real docno is a property the pipeline guarantees and CI asserts.
- **Runs are TSVs** (`qid`, `docno`, `rank`, `score`) under each stage's `runs/`, alongside
  `metrics.csv`, `per_query/` and `*_meta.json`. See [docs/output.md](docs/output.md).
- **New behaviour goes behind a flag that defaults off**, so an existing config keeps producing
  byte-identical output. `GENERATION_MODE=direct` is the model to copy.
- **Never delete or rename** a config variable, an output path, or a JSONL field. Other consumers
  of this pipeline still depend on them.

## Where the answers are

| Question | Source |
|----------|--------|
| What flags and env toggles exist? | `./run_retrieval_rerank_pipeline.sh --help` |
| What does a knob do, what range is sane, how do caps chain? | [docs/PARAMETERS.md](docs/PARAMETERS.md) |
| How do I run one stage standalone? | [docs/USAGE.md](docs/USAGE.md) |
| What does a run write, and where? | [docs/output.md](docs/output.md) |
| What do the JSONL files look like? | [docs/output.md](docs/output.md) § JSONL schemas |
| What should I set if I don't know? | [README.md](README.md) § Recommended starting point |
| What is each directory for? | § Repo map, below |

When `--help` and the docs disagree, `--help` is right — then fix the docs.

## Repo map

| Directory | Contents |
|-----------|----------|
| `run_retrieval_rerank_pipeline.sh` | The orchestrator — the one entrypoint |
| `index/` | Build the Terrier BM25 and dense HNSW indexes from JSONL shards |
| `retrieval/` | Stage-1 BM25 and dense retrieval, retrieval RRF, multi-field query fusion |
| `rerank/` | Cross-encoder rerank, post-rerank fusion, t\* score cutoff, eval plots |
| `evidence/` | Turn ranked docids into contexts — document contexts, snippet windows, snippet rerank |
| `generation/` | Claim extraction and distillation, answer generation, sentence citation attribution |
| `conf/` | Example workflow configs; copy one and edit |
| `prompts/` | Generation system/user prompt templates and output schemas |
| `docs/` | Parameter reference, per-stage usage, output layout and JSONL schemas |
| `retrieval_eval/` | Shared metric helpers (recall/nDCG) used by the stage scripts |
| `analysis/` | Standalone post-hoc reports — recall by question, low-recall diagnosis, run comparison. Not part of the pipeline |
| `listwise_script/` | Optional LLM listwise reranker, an alternative to the cross-encoder. Not wired into the orchestrator |
| `ci/` | Mock LLM server and the generation-mode smoke script used by GitHub Actions |
| `utils/` | Logging setup |

## Style

Match the file you are editing. The shell is `set -e` bash with `[ ]` tests; the Python is stdlib
plus the pinned dependencies, argparse CLIs, no framework. There is no linter or formatter config,
so consistency with the surrounding code is the standard.
