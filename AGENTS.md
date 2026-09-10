# AGENTS.md

Orientation for coding agents working in this repository. Read by Cursor directly, and by Claude
Code through the `@AGENTS.md` import in `CLAUDE.md`. Humans should start with [README.md](README.md);
this file is the operating manual for *changing* the pipeline safely.

## What this is

A retrieval-and-generation pipeline driven by **one shell orchestrator over an env config file**.
It is not a library — nothing here is imported as a package. Every stage is a standalone script that
reads files and writes files; the orchestrator's job is to pass the right paths between them.

It is **corpus-agnostic**. Docids are opaque strings the pipeline never parses, and documents can be
short abstracts or long web pages. If you find yourself writing code that assumes a docid *shape*, a
document *length*, or a particular task's field names, that is a bug — keep it out.

## The pipeline

Seven numbered stages, each writing into its own subdirectory of `$WORKFLOW_OUTPUT_DIR`. Stages 6–7
are the optional snippet route; evidence and generation follow.

| # | Stage | Script | Reads | Writes |
|---|-------|--------|-------|--------|
| 1 | BM25 (+RM3) retrieval | `retrieval/retrieve_bm25.py` | query JSONL, Terrier index | `retrieval/bm25/` |
| 2 | Dense retrieval | `retrieval/retrieve_dense.py` | query JSONL, HNSW index | `retrieval/dense/` |
| 3 | Retrieval fusion (RRF) | `retrieval/fuse_retrieval.py` | runs from 1 + 2 | `retrieval/fusion/` |
| 4 | Cross-encoder rerank | `rerank/rerank_crossencoder.py` | stage-1 run, `DOCS_JSONL` | `rerank/cross_encoder/` |
| 5 | Post-rerank fusion (RRF) | `rerank/fuse_rerank.py` | runs from 3 + 4 | `rerank/post_rerank_fusion/` |
| 6 | Snippet windows + CE rerank | `evidence/rerank_snippets.py` | reranked docs, `DOCS_JSONL` | `snippet/snippet_rerank/` |
| 7 | Evidence fusion (RRF) | `rerank/fuse_rerank.py` | runs from 5 + 6 | `snippet/snippet_doc_fusion/` |
| — | Evidence (contexts) | `evidence/build_doc_contexts.py`, `evidence/build_snippet_contexts.py` | final run, `DOCS_JSONL` | `evidence/evidence_baseline/`, `evidence/evidence_snippet/` |
| — | Generation | `generation/generate_answers.py` | contexts JSONL, LLM endpoint | `generation/generation_baseline/`, `generation/generation_snippet/` |

Two things about that table are worth internalising, because they explain most of the code:

- **There are three separate RRF fusions** (stages 3, 5, 7) and they are *not* interchangeable.
  Stage 3 fuses two first-stage retrievers. Stage 5 fuses the cross-encoder's order with the
  first-stage order, so a document the reranker hates but retrieval loved is not simply discarded.
  Stage 7 fuses document-level and snippet-level rankings. `docs/output.md` names all three; use
  those names in code and comments, not "hybrid".
- **`_baseline` in the evidence and generation paths is a legacy on-disk name for the document
  route.** It does not mean "a baseline system". Don't rename it — existing run trees depend on it.

The first stage is swappable via `STAGE1_SOURCE`: `rrf` (default, stages 1–3), `bm25` or `dense`
(one retriever, skipping the other and the fusion), or `external` (skip stages 1–3 entirely and
stage a precomputed run from `STAGE1_RUN` — for hosted retrieval APIs).

Generation has an optional preceding stage. `GENERATION_MODE=claims` extracts claims from each
context and distils them into slots before the answer prompt, so generation can ingest evidence from
far more documents than would fit in a raw prompt. The default `direct` is a no-op that passes raw
contexts through.

**Distillation is ollama-only, and that is a constraint rather than an oversight.** Answer generation
speaks both backends; `GENERATION_MODE=claims|facets` is refused at config time when
`GENERATION_BACKEND=openai_compat`. Three things make a port to chat completions more than plumbing,
and all three fail quietly rather than loudly:

- **`num_ctx` has no hosted equivalent.** Hosted APIs fix the context window server-side, and the
  documented truncation policy — raise `GENERATION_NUM_CTX`, never cut slots or contexts — depends
  on being able to raise it. The orchestrator's pre-generation truncation warning rests on the same
  knob and would become noise.
- **`think` is three-state and step-dependent.** `True`, `False` and key-absent are three different
  behaviours, and the correct one differs per stage — see the contract on
  `distill_common.call_ollama`. `reasoning.effort` is not a faithful mapping, and a wrong choice
  costs output quality without raising anything.
- **The claim-cache key carries no model or backend** (`distill_common.sha_key`). That is sound only
  while one output tree implies one model. A second backend makes cross-backend reuse reachable
  inside a tree, so the key would have to grow — invalidating every banked cache in every consuming
  repo.

The code is the easy part. Landing it safely needs a back-to-back quality comparison between
backends, and that evidence belongs with whoever runs it.

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
is wrong. `.github/workflows/ci.yml` runs the same script plus four end-to-end configurations
(document route, chunked documents, single-retriever stage-1, post-rerank fusion disabled) on public
demo data in Docker. Touching the orchestrator's stage wiring usually means adding a matrix entry.

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

`GENERATION_MODE=facets` is accepted and exercised by CI, but is **not recommended**: it adds a
clustering stage and a second LLM pass for no measured quality gain on long-form synthesis. Don't
enable it in a config or suggest it as an improvement. Default/`direct` is the no-op path; switch to
`claims` when the raw prompt cannot hold the evidence (ollama-only — see above).

## Where the answers are

| Question | Source |
|----------|--------|
| What flags and env toggles exist? | `./run_retrieval_rerank_pipeline.sh --help` |
| What does a knob do, what range is sane, how do caps chain? | [docs/PARAMETERS.md](docs/PARAMETERS.md) |
| How do I run one stage standalone? | [docs/USAGE.md](docs/USAGE.md) |
| What does a run write, and where? | [docs/output.md](docs/output.md) |
| What should I set if I don't know? | [README.md](README.md) § Recommended starting point |
| What is each directory for? | [README.md](README.md) § Repo map |

When `--help` and the docs disagree, `--help` is right — then fix the docs.

## Style

Match the file you are editing. The shell is `set -e` bash with `[ ]` tests; the Python is stdlib
plus the pinned dependencies, argparse CLIs, no framework. There is no linter or formatter config,
so consistency with the surrounding code is the standard.
