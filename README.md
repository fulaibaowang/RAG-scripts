# RAG-scripts

A retrieval, reranking and generation pipeline: BM25 + dense retrieval fused with RRF,
cross-encoder reranking, evidence construction, and LLM answers with citations back to real
corpus docids. Optional routes add snippet-level evidence and **context distillation**
(`GENERATION_MODE=claims`), so generation can use more evidence than fits in a raw prompt.

![Pipeline overview](docs/img/pipeline.png)

## Quickstart

### Try the demo

```bash
git clone https://github.com/fulaibaowang/RAG-scripts.git
git clone https://github.com/fulaibaowang/RAG-scripts-demo-data.git RAG-scripts/demo
cd RAG-scripts
docker run --rm \
  -v "$PWD:/work" \
  -e HF_HOME=/work/.hf_cache \
  -e HF_HUB_CACHE=/work/.hf_cache/hub \
  -e SENTENCE_TRANSFORMERS_HOME=/work/.hf_cache/sentence_transformers \
  --workdir /work \
  fulaibaowang/bioasq:08.03.26b200 \
  bash -c "./run_retrieval_rerank_pipeline.sh --config demo/config.env"
```

Outputs land in `demo/output/`.

### Run on your own data

You need a query JSONL and a corpus JSONL:

```json
{ "query_id": "q1", "query_text": "Are there biomarkers of depression?" }
{ "docno": "your-id", "title": "...", "text": "..." }
```

Build the BM25 and dense indexes from the corpus ([docs/USAGE.md](docs/USAGE.md#indexing)), then:

```bash
cp conf/workflow_config_document.env my_run.env
# edit my_run.env: WORKFLOW_OUTPUT_DIR, INPUT_JSONL, INPUT_BATCH_JSONLS,
#                  BM25_INDEX_PATH, DENSE_INDEX_DIR, DOCS_JSONL
# no gold documents[] on your queries? set HAVE_GROUND_TRUTH=0, or metrics come back all zero
./run_retrieval_rerank_pipeline.sh --config my_run.env
```

Run it inside the Docker image above, or a local env (see [Prerequisites](#prerequisites)). Useful
flags: `--no-rerank` (retrieval only), `--no-generation` (stop before the LLM), `--snippet-rrf`
(snippet route). `--help` lists them all.

**Stages whose outputs already exist are skipped.** To get a clean re-run, point
`WORKFLOW_OUTPUT_DIR` somewhere new.

## Recommended starting point

[docs/PARAMETERS.md](docs/PARAMETERS.md) documents every knob. If you have no reason to choose
otherwise, start here and change one thing at a time:

| Knob | Start with | Why |
|------|-----------|-----|
| `STAGE1_SOURCE` | `rrf` (default) | BM25 + dense fused beats either alone; `external` if a hosted first stage already gives you a run |
| Post-rerank fusion | on (default) | fuses the cross-encoder order with the stage-1 order rather than trusting either outright |
| `GENERATION_MODE` | `direct` (default) | `claims` when generation needs more evidence than fits a raw prompt, at the cost of an extra LLM pass. `facets` is not recommended |
| Snippet route | off | `RUN_SNIPPET_RRF=1` when passage-level evidence beats whole documents for your corpus. An alternative to the document route, not a companion to `claims` |

These defaults give you a sensible first run. They are not tuned settings.

## Pointing generation at an LLM

```bash
GENERATION_BACKEND=ollama                        # default; OLLAMA_URL defaults to a local ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
GENERATION_MODEL=llama3.3:latest
```

```bash
GENERATION_BACKEND=openai_compat                 # any OpenAI-compatible endpoint
GEN_API_BASE=https://openrouter.ai/api/v1
GENERATION_MODEL=meta-llama/llama-3.3-70b-instruct
```

API keys (`LLAMA_API_KEY`, `GEN_API_KEY`) come from the environment or a repo-root `.env`, never a
committed config. `GENERATION_MODE=claims` requires the ollama backend. Details:
[docs/PARAMETERS.md](docs/PARAMETERS.md#answer-generation-llm).

## Documentation

| | |
|-|-|
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | What every knob does and what range is sane |
| [docs/USAGE.md](docs/USAGE.md) | Index building and running each stage standalone |
| [docs/output.md](docs/output.md) | Output layout, run format, and JSONL schemas |
| [AGENTS.md](AGENTS.md) | Changing the pipeline (with or without a coding agent), repo map, CI |

## Prerequisites

Dense retrieval and cross-encoder reranking want a GPU; they run on CPU but slowly. The Docker image
above carries every dependency. For a local venv, install a matching `torch` from
[pytorch.org](https://pytorch.org), then `pip install -r requirements-docker-pytorch.txt -r requirements-docker.txt`;
you also need Java and the system packages from the [Dockerfile](Dockerfile).

## Related repos

This pipeline is shared by several projects that drive it over different corpora. Task-specific
submission formatting lives in those repos, not here.

- [BioASQ](https://github.com/fulaibaowang/BioASQ): biomedical question answering over PubMed.
- [trec-rag](https://github.com/fulaibaowang/trec-rag): TREC RAG, over the ClimbMix web corpus. *Going public soon.*
- [dictycite](https://github.com/fulaibaowang/dictycite): literature curation for dictyBase.

## License

Apache-2.0. See [LICENSE](LICENSE).
