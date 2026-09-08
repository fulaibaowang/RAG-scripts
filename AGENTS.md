# AGENTS.md

Orientation for coding agents (Claude Code, Cursor, …) working in **RAG-scripts**. Read by Cursor
directly and by Claude Code via the `@AGENTS.md` import in `CLAUDE.md`. Humans should start with
[README.md](README.md); this file is the operating manual for making *changes* here safely.

## What this repo is

A retrieval-and-generation pipeline: BM25 (+RM3) and dense HNSW retrieval → RRF fusion →
cross-encoder rerank → post-rerank fusion → evidence construction → LLM generation → optional
per-sentence citation attribution. It is a **pipeline, not a library** — nothing here is imported as
a package by downstream code; everything is driven by one shell orchestrator over env config.

It is **corpus-agnostic by design**. Docids are opaque strings the pipeline never parses. It has
been driven over PubMed abstracts, MS MARCO v2.1, and ClimbMix. If you find yourself adding code
that assumes a docid *shape*, an abstract-length document, or a specific task's field names, that
is a bug, not a feature — push it out to the consuming repo.

## The one rule that saves the most time

**This repo is a git subtree consumed by several projects** (BioASQ, dictycite, TREC-RAG work).
An edit here lands in all of them on their next `git subtree pull`. So:

- Keep changes **backward-compatible**. New behaviour goes behind a flag that defaults **off**, so
  an existing config produces byte-identical output. `GENERATION_MODE=direct` and
  `CITATION_GRANULARITY=answer` are the model to copy — both are no-op defaults.
- Never delete or rename a config variable, an output path, or a JSONL field because *your* caller
  stopped using it. Another repo still does.
- Don't add task-specific formatting. Converting answers into a task's submission wire format is
  each consuming repo's own adapt-out step.

## One entrypoint

```bash
./run_retrieval_rerank_pipeline.sh --config /path/to/your.env
./run_retrieval_rerank_pipeline.sh --help     # every flag and env toggle, authoritative
```

Everything in `retrieval/`, `rerank/`, `evidence/`, `generation/` is invoked *by* the orchestrator.
The stage scripts also run standalone (see [docs/USAGE.md](docs/USAGE.md)) — useful for debugging a
single stage, but **don't build a workflow out of hand-run stages**. Path conventions between stages
are the orchestrator's job, and hand-running them is how output trees end up subtly wrong.

Config is a plain shell env file, `source`d with `set -a`. That means **every variable in it is
exported**, and any variable already in your environment is overridden by the file. Flags passed on
the command line override the config.

## Stage skipping is the gotcha

**A stage whose key outputs already exist is skipped.** This is deliberate — it makes re-running a
long pipeline cheap — but it means:

- Editing a stage's code and re-running the same config **changes nothing**. The stage is skipped
  because its outputs are still there. Delete the stage's output directory, or point
  `WORKFLOW_OUTPUT_DIR` somewhere new.
- "It ran and my fix didn't take effect" is almost always this, not a bug in your fix.
- Generation additionally checkpoints per completion (`GENERATION_CHECKPOINT=1`, default on): an
  interrupted run resumes from `*_answers.jsonl.partial`, reusing rows fingerprinted on
  input/model/params. Clearing the answers file but leaving the sidecar will reuse old completions.

When you want a clean measurement, use a fresh `WORKFLOW_OUTPUT_DIR`. It is the only reliable reset.

## Verifying a change without a GPU or an LLM

Don't hand-test generation against a real endpoint. There is a deterministic mock:

```bash
git clone https://github.com/fulaibaowang/RAG-scripts-demo-data.git demo
DEMO_DIR=demo OUT_ROOT=/tmp/genmodes bash ci/run_generation_modes.sh
```

This exercises every `GENERATION_MODE` against `ci/mock_ollama.py` — no network egress — and asserts
that answers come back error-free and that every distilled slot's `doc_id` is a real corpus docno
(citation lineage). `.github/workflows/ci.yml` runs the same script plus four end-to-end pipeline
configurations (document route, chunked, BM25-only stage-1, no post-rerank fusion) on public demo
data in Docker. If you touch the orchestrator's stage wiring, expect to need a new matrix entry.

## Where the answers are

| Question | Doc |
|----------|-----|
| What does a knob do, what range is sane, how do caps chain across stages? | [docs/PARAMETERS.md](docs/PARAMETERS.md) |
| How do I run one stage standalone? | [docs/USAGE.md](docs/USAGE.md) |
| What directories/files does a run write, and what's the run format? | [docs/output.md](docs/output.md) |
| What are the flags and env toggles? | `./run_retrieval_rerank_pipeline.sh --help` |
| What should I set if I don't know? | [README.md](README.md) § Recommended starting point |
| What is each directory for? | [README.md](README.md) § Repo map |

Prefer the `--help` output over the docs when the two disagree, and then fix the docs.

## Conventions to preserve

- **Wire format for query identity is `query_id` / `query_text` / `query_type`**, on read and on
  write. Legacy `id` / `body` / `type` are normalized on read only. New stages read and write
  `query_*` — do not reintroduce the legacy names downstream.
- `query_type` is **optional**. Don't write code that requires it.
- **JSONL everywhere**: one JSON object per line, and every stage passes unknown fields through
  untouched. A consumer that ignores your new field must see no change.
- Context ids are `<docid>-<n>`; `doc_id` carries the real corpus docid. Citation lineage back to a
  real docno is a property worth protecting — CI asserts it.
- Retrieval runs are TSVs (`qid`, `docno`, `rank`, `score`) under each stage's `runs/` directory,
  alongside `metrics.csv`, `per_query/` and `*_meta.json`. See [docs/output.md](docs/output.md).

## Not recommended

`GENERATION_MODE=facets` is accepted but not recommended: it adds a clustering stage and a second
LLM pass without a measured quality gain on long-form synthesis. It stays in the tree for
reproducibility and is exercised by CI. **Don't enable it in a config, suggest it as an
improvement, or promote it in docs.** Use `claims`.

## Style

Match the file you're editing. The shell is `set -e` bash with `[ ]` tests; the Python is
stdlib-plus-the-pinned-deps, argparse CLIs, no framework. There is no linter or formatter config —
consistency with the neighbouring code is the standard.
