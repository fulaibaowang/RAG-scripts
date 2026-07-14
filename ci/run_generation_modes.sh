#!/usr/bin/env bash
# CI: exercise the generation stage in all three GENERATION_MODE values
# (direct | claims | facets) against the deterministic mock LLM (ci/mock_ollama.py),
# reusing the prebuilt demo output tree so retrieval/rerank/evidence skip.
# No real LLM, no network egress; asserts answers exist error-free and that every
# distilled slot doc_id is a real corpus docno (citation lineage).
#
#   DEMO_DIR=demo OUT_ROOT=/tmp/genmodes bash ci/run_generation_modes.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${DEMO_DIR:-demo}"; DEMO_DIR="$(cd "$DEMO_DIR" && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/genmodes}"
PORT="${MOCK_PORT:-18765}"
mkdir -p "$OUT_ROOT"

python3 "$REPO_ROOT/ci/mock_ollama.py" "$PORT" & MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
sleep 1

# Prebuilt output tree (local dev convenience). In CI there is none: the direct-mode
# run builds retrieval/rerank/evidence from scratch and then seeds claims/facets.
BASE_TREE=""
[ -d "$DEMO_DIR/output_chunked" ] && BASE_TREE="$DEMO_DIR/output_chunked"

for MODE in direct claims facets; do
  OUT="$OUT_ROOT/$MODE"
  rm -rf "$OUT"; mkdir -p "$OUT"
  if [ -n "$BASE_TREE" ]; then
    cp -R "$BASE_TREE" "$OUT/output_chunked"
    # normalise pre-stage1 fusion TSV names in older prebuilt trees so the split resolves
    for runs in "$OUT/output_chunked/rerank/post_rerank_fusion/runs" \
                "$OUT/output_chunked/snippet/snippet_doc_fusion/runs"; do
      for f in "$runs"/best_rrf_queries_top*.tsv; do
        [ -f "$f" ] && mv "$f" "$runs/stage1_queries_top100.tsv"
      done
    done
  fi
  CFG="$OUT/config.env"
  sed -e "s|/work/demo|$DEMO_DIR|g" \
      -e "s|^REPO_ROOT=.*|REPO_ROOT=$REPO_ROOT|" \
      -e "s|^WORKFLOW_OUTPUT_DIR=.*|WORKFLOW_OUTPUT_DIR=$OUT/output_chunked|" \
      -e "s|^RUN_GENERATION_DOCUMENT=.*|RUN_GENERATION_DOCUMENT=1|" \
      -e "s|^RUN_GENERATION_SNIPPET=.*|RUN_GENERATION_SNIPPET=1|" \
      "$DEMO_DIR/config.chunked.env" > "$CFG"
  {
    echo "GENERATION_BACKEND=ollama"
    echo "GENERATION_MODEL=mockmodel"
    echo "GENERATION_CONCURRENCY=2"
    if [ "$MODE" != "direct" ]; then
      echo "GENERATION_MODE=$MODE"
      # exercise the post-hoc sentence-attribution stage, lexical-only (no torch),
      # with the per-sentence cite cap (TREC RAG 2026 allows at most 3)
      echo "CITATION_GRANULARITY=sentence"
      echo "CITATION_MOCK=1"
      echo "CITATION_MAX_CITES=3"
    fi
  } >> "$CFG"
  echo "=== GENERATION_MODE=$MODE ==="
  OLLAMA_URL="http://127.0.0.1:$PORT/api/generate" LLAMA_API_KEY=ci-mock \
    bash "$REPO_ROOT/run_retrieval_rerank_pipeline.sh" --config "$CFG"
  # first (direct) run built everything from scratch: reuse its tree for the other modes
  [ -z "$BASE_TREE" ] && BASE_TREE="$OUT/output_chunked"
done

# Config-time guard: CITATION_GRANULARITY=sentence + direct mode must refuse before any work
echo "=== guard: CITATION_GRANULARITY=sentence with GENERATION_MODE=direct must refuse ==="
if CITATION_GRANULARITY=sentence OLLAMA_URL="http://127.0.0.1:$PORT/api/generate" LLAMA_API_KEY=ci-mock \
    bash "$REPO_ROOT/run_retrieval_rerank_pipeline.sh" --config "$OUT_ROOT/direct/config.env" \
    >/dev/null 2>&1; then
  echo "FAIL: sentence+direct was not refused" >&2
  exit 1
fi
echo "guard OK (refused)"

python3 - "$OUT_ROOT" "$DEMO_DIR/data/docs_chunked.jsonl" <<'PY'
import json, glob, sys
out_root, corpus = sys.argv[1], sys.argv[2]
rows = {(json.loads(l).get("docno") or json.loads(l).get("docid")) for l in open(corpus)}

def check_answers(path):
    recs = [json.loads(l) for l in open(path)]
    errs = [r for r in recs if r.get("error")]
    assert recs and not errs, f"{path}: {len(recs)} answers, {len(errs)} errors"
    return 1

n = 0
for route in ("generation_baseline", "generation_snippet"):
    n += check_answers(f"{out_root}/direct/output_chunked/generation/{route}/queries_answers.jsonl")
    for mode in ("claims", "facets"):
        n += check_answers(f"{out_root}/{mode}/output_chunked/generation/{route}/queries_distilled_answers.jsonl")

slot_files = glob.glob(f"{out_root}/*/output_chunked/evidence/*/queries_distilled_contexts.jsonl")
assert len(slot_files) == 4, f"expected 4 distilled slot files (2 modes x 2 routes), got {len(slot_files)}"
for f in slot_files:
    for line in open(f):
        for c in json.loads(line).get("contexts", []):
            assert c.get("doc_id") in rows, f"{f}: doc_id {c.get('doc_id')} not a corpus docno"

# sentence attribution (CITATION_GRANULARITY=sentence, mock/lexical): every attributed row
# carries answer_sentences whose doc_ids are real corpus docnos. The deterministic mock LLM
# emits empty evidence_ids, so 0 citations is expected here — this smoke-checks the stage
# plumbing (segmentation, schema, real-docid resolution), not attribution quality (that is
# the banked byte-identical descent check, offline).
attr_files = glob.glob(f"{out_root}/*/output_chunked/generation/*/queries_distilled_answers_attributed.jsonl")
assert len(attr_files) == 4, f"expected 4 attributed answer files (2 modes x 2 routes), got {len(attr_files)}"
n_cites = 0
for f in attr_files:
    for line in open(f):
        r = json.loads(line)
        if not (r.get("ideal_answer") or "").strip():
            continue
        sents = r.get("answer_sentences")
        assert sents, f"{f}: qid {r.get('query_id')} missing answer_sentences"
        for s in sents:
            assert s.get("text", "").strip(), f"{f}: empty sentence text"
            assert len(s.get("doc_ids", [])) <= 3, \
                f"{f}: {len(s['doc_ids'])} doc_ids on one sentence despite CITATION_MAX_CITES=3"
            for d in s.get("doc_ids", []):
                assert d in rows, f"{f}: answer_sentences doc_id {d} not a corpus docno"
                n_cites += 1
print(f"generation-modes OK: {n} answer files error-free; all distilled slot doc_ids are corpus docnos; "
      f"{len(attr_files)} attributed files carry answer_sentences ({n_cites} real-docid citations)")
PY
