#!/usr/bin/env bash
#
# Run retrieval + rerank pipeline: BM25 -> Dense -> Hybrid (and optionally Reranker).
#
# Usage:
#   ./run_retrieval_rerank_pipeline.sh --config <path/to/config.env>
#   ./run_retrieval_rerank_pipeline.sh -c <path/to/config.env>
#   ./run_retrieval_rerank_pipeline.sh -c config.env --no-rerank   # stop at hybrid
#
# Or: source my.env && ./run_retrieval_rerank_pipeline.sh 
#
# Config file sets: WORKFLOW_OUTPUT_DIR, INPUT_JSONL / INPUT_BATCH_JSONLS (optional; .jsonl only), TOP_K,
# RECALL_KS, BM25_INDEX_PATH, DENSE_INDEX_DIR, DOCS_JSONL (optional),
# BM25_QUERY_FIELD / DENSE_QUERY_FIELD (comma-separated = multi-query RRF fusion per stage),
# and stage overrides (BM25_*, DENSE_*, HYBRID_*, RERANK_*). Evidence: POST_RERANK_DOC_POOL (docs written
# to post_rerank_*.jsonl), EVIDENCE_TOP_K and optional EVIDENCE_TOP_K_BASELINE / EVIDENCE_TOP_K_SNIPPET
# (caps in build_contexts_from_*). See workflow_config_full.env.
#
set -e

# Allow overriding the shared-scripts root via env (e.g. when vendoring into another project).
# If SHARED_SCRIPTS_DIR is set, treat it as SCRIPT_DIR; otherwise use this script's directory.
if [ -n "${SHARED_SCRIPTS_DIR:-}" ]; then
  SCRIPT_DIR="$SHARED_SCRIPTS_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Resolve REPO_ROOT portably by walking up until we find .git (works regardless of shared_scripts depth).
_d="$SCRIPT_DIR"
while [ "$_d" != "/" ]; do
  [ -d "$_d/.git" ] && break
  _d="$(dirname "$_d")"
done
if [ "$_d" = "/" ]; then
  echo "Warning: could not locate .git repo root; falling back to SCRIPT_DIR as working directory." >&2
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$_d"
fi

# Parse -c / --config, --no-rerank, --no-rrf-fusion, --snippet-rrf, --run-both-routes, --no-generation*,
# --generation-schemas-dir, -h / --help
CONFIG_FILE=""
_PIPELINE_GENERATION_SCHEMAS_DIR=""
# Allow environment to override defaults (and keep CLI flags as explicit overrides below).
RUN_RERANK="${RUN_RERANK:-1}"
RUN_RRF_FUSION="${RUN_RRF_FUSION:-1}"
SNIPPET_RRF=0
RUN_BOTH_ROUTES=0
RUN_GENERATION_BASELINE="${RUN_GENERATION_BASELINE:-1}"
RUN_GENERATION_SNIPPET="${RUN_GENERATION_SNIPPET:-1}"
BM25_QUERY_FIELD_ARG=""
DENSE_QUERY_FIELD_ARG=""
RERANK_QUERY_FIELD_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--config)
      [ -z "${2:-}" ] && { echo "Error: --config requires a path." >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    --no-rerank)
      RUN_RERANK=0
      shift
      ;;
    --no-rrf-fusion)
      RUN_RRF_FUSION=0
      shift
      ;;
    --snippet-rrf)
      SNIPPET_RRF=1
      shift
      ;;
    --run-both-routes)
      RUN_BOTH_ROUTES=1
      SNIPPET_RRF=1
      shift
      ;;
    --bm25-query-field)
      [ -z "${2:-}" ] && { echo "Error: --bm25-query-field requires a value." >&2; exit 1; }
      BM25_QUERY_FIELD_ARG="$2"
      shift 2
      ;;
    --dense-query-field)
      [ -z "${2:-}" ] && { echo "Error: --dense-query-field requires a value." >&2; exit 1; }
      DENSE_QUERY_FIELD_ARG="$2"
      shift 2
      ;;
    --rerank-query-field)
      [ -z "${2:-}" ] && { echo "Error: --rerank-query-field requires a value." >&2; exit 1; }
      RERANK_QUERY_FIELD_ARG="$2"
      shift 2
      ;;
    --no-generation)
      RUN_GENERATION_BASELINE=0
      RUN_GENERATION_SNIPPET=0
      shift
      ;;
    --no-generation-baseline)
      RUN_GENERATION_BASELINE=0
      shift
      ;;
    --no-generation-snippet)
      RUN_GENERATION_SNIPPET=0
      shift
      ;;
    --generation-schemas-dir)
      [ -z "${2:-}" ] && { echo "Error: --generation-schemas-dir requires a path." >&2; exit 1; }
      _PIPELINE_GENERATION_SCHEMAS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--config|-c <config.env>] [--no-rerank] [--no-rrf-fusion] [--snippet-rrf] [--run-both-routes] [--no-generation] [--generation-schemas-dir DIR] [--bm25-query-field F] ..."
      echo "  -c, --config PATH       Source PATH as config (env vars) before running."
      echo "  --no-rerank             Run only BM25, Dense, retrieval fusion; skip reranker even if DOCS_JSONL is set."
      echo "  --no-rrf-fusion         Disable post-rerank RRF fusion (retrieval fusion + cross-encoder) after reranker."
      echo "  --snippet-rrf           Snippet route: pool=200, snippet CE, doc+snippet fusion (snippet/); evidence/evidence_snippet/, generation/generation_snippet/."
      echo "  --run-both-routes        Run baseline and snippet routes; write evidence/evidence_baseline/, evidence/evidence_snippet/, generation/generation_baseline/, generation/generation_snippet/ (no overwrite)."
      echo "  --no-generation          Skip LLM generation (and rescue) for both baseline and snippet routes."
      echo "  --no-generation-baseline  Skip LLM generation for baseline route only (evidence still built)."
      echo "  --no-generation-snippet   Skip LLM generation for snippet route only (evidence still built)."
      echo "  --generation-schemas-dir DIR  Schema *.txt directory for generate_answers.py (overrides GENERATION_SCHEMAS_DIR in config)."
      echo "  --bm25-query-field F    Use F as query text for BM25 (overrides env). Comma-separated = multi-query RRF fuse."
      echo "  --dense-query-field F   Use F as query text for Dense (overrides env). Comma-separated = multi-query RRF fuse."
      echo "  --rerank-query-field F  Use F for reranker and snippet CE (overrides env). Comma-separated = multi-query RRF fuse."
      echo "  -h, --help              Show this help."
      echo ""
      echo "Env toggles:"
      echo "  RUN_BASELINE=0|1        Control baseline evidence/generation route (default 1)."
      echo "  RUN_SNIPPET_RRF=0|1     Control snippet-rrf route (steps 6–7, evidence/evidence_snippet/, generation/generation_snippet/)."
      echo "  RUN_RRF_FUSION=0|1      Control Hybrid+Rerank RRF fusion (default 1; 0 is same as --no-rrf-fusion)."
      echo "  RUN_GENERATION_BASELINE=0|1   Run generation for baseline route (default 1)."
      echo "  RUN_GENERATION_SNIPPET=0|1    Run generation for snippet route (default 1)."
      echo "  GENERATION_SCHEMAS_DIR       Directory of schema *.txt for LLM prompts (default: scripts/public/shared_scripts/prompts/schemas under repo root)."
      echo "  POST_RERANK_DOC_POOL         Max docs per query written into post_rerank_*.jsonl (default: 30)."
      echo "  EVIDENCE_TOP_K / EVIDENCE_TOP_K_BASELINE / EVIDENCE_TOP_K_SNIPPET   Max docs per question for contexts (build_contexts; default: EVIDENCE_TOP_K or 10)."
      echo ""
      echo "Example: $0 --config scripts/private_scripts/config.env"
      echo "Example: $0 -c config.env --no-rerank --bm25-query-field body_expansion_long --dense-query-field body"
      echo "Example: source workflow_config_small.env && $0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (use -h for help)" >&2
      exit 1
      ;;
  esac
done

if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
  echo "Loaded config: $CONFIG_FILE"
fi

# Route toggles (can be set in env; defaults are baseline on, snippet off unless --snippet-rrf was passed)
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SNIPPET_RRF="${RUN_SNIPPET_RRF:-${SNIPPET_RRF:-0}}"
DO_SNIPPET_RRF="${RUN_SNIPPET_RRF}"
# When both routes are requested, enable both RRF outputs (step 5 -> post_rerank_fusion, step 5b -> post_rerank_fusion_snippet) so baseline and snippet evidence both have runs
if [ "$RUN_BASELINE" = "1" ] && [ "$DO_SNIPPET_RRF" = "1" ]; then
  RUN_BOTH_ROUTES=1
fi

# Explicit args override env (used by run_query_field_sweep.sh so query fields are never lost)
[ -n "$BM25_QUERY_FIELD_ARG" ] && export BM25_QUERY_FIELD="$BM25_QUERY_FIELD_ARG"
[ -n "$DENSE_QUERY_FIELD_ARG" ] && export DENSE_QUERY_FIELD="$DENSE_QUERY_FIELD_ARG"
[ -n "$RERANK_QUERY_FIELD_ARG" ] && export RERANK_QUERY_FIELD="$RERANK_QUERY_FIELD_ARG"

# Optional: single switch to skip all eval metrics (run pipeline without ground truth)
if [ "${HAVE_GROUND_TRUTH:-1}" = "0" ]; then
  export BM25_NO_EVAL=1
  export DENSE_NO_EVAL=1
  export HYBRID_NO_EVAL=1
  export RERANK_DISABLE_METRICS=1
  echo "HAVE_GROUND_TRUTH=0: eval metrics disabled for BM25, Dense, Hybrid, Rerank"
fi

cd "$REPO_ROOT"

# Generation schema snippets: default portable path under shared_scripts; override with
# GENERATION_SCHEMAS_DIR in config or --generation-schemas-dir (CLI wins over config).
_DEFAULT_GENERATION_SCHEMAS_DIR="$REPO_ROOT/scripts/public/shared_scripts/prompts/schemas"
if [ -n "${_PIPELINE_GENERATION_SCHEMAS_DIR:-}" ]; then
  export GENERATION_SCHEMAS_DIR="$_PIPELINE_GENERATION_SCHEMAS_DIR"
else
  export GENERATION_SCHEMAS_DIR="${GENERATION_SCHEMAS_DIR:-$_DEFAULT_GENERATION_SCHEMAS_DIR}"
fi

# Required env (set by config file or by sourcing before run)
: "${WORKFLOW_OUTPUT_DIR:?Set WORKFLOW_OUTPUT_DIR (e.g. output/workflow_run)}"
: "${BM25_INDEX_PATH:?Set BM25_INDEX_PATH (Terrier index directory)}"
if [ -z "${DENSE_INDEX_GLOB:-}" ]; then
  : "${DENSE_INDEX_DIR:?Set DENSE_INDEX_DIR or DENSE_INDEX_GLOB (Dense HNSW index directory or shard glob)}"
fi

TOP_K="${TOP_K:-5000}"
RECALL_KS="${RECALL_KS:-50,100,200,300,400,500,1000,2000,5000}"

# Query inputs (.jsonl only). Canonical: INPUT_JSONL / INPUT_BATCH_JSONLS. Legacy env names via indirect expansion (avoid duplicating deprecated identifiers in this file).
_legacy_train_var=TRAIN_JSON
_legacy_batch_var=TEST_BATCH_JSONS
[ -z "${INPUT_JSONL:-}" ] && [ -n "${!_legacy_train_var:-}" ] && INPUT_JSONL="${!_legacy_train_var}"
[ -z "${INPUT_BATCH_JSONLS:-}" ] && [ -n "${!_legacy_batch_var:-}" ] && INPUT_BATCH_JSONLS="${!_legacy_batch_var}"
_have_pq=0
[ -n "${INPUT_JSONL:-}" ] && _have_pq=1
[ -n "${INPUT_BATCH_JSONLS:-}" ] && _have_pq=1
if [ "$_have_pq" != 1 ]; then
  echo "Error: set INPUT_JSONL and/or INPUT_BATCH_JSONLS (non-empty), or legacy TRAIN_JSON / TEST_BATCH_JSONS in your config." >&2
  exit 1
fi
case "${INPUT_JSONL:-}" in
  "") ;;
  *.jsonl) ;;
  *) echo "Error: INPUT_JSONL must end with .jsonl or be empty: ${INPUT_JSONL}" >&2; exit 1 ;;
esac
for _pq in ${INPUT_BATCH_JSONLS:-}; do
  [ -z "$_pq" ] && continue
  case "$_pq" in
    *.jsonl) ;;
    *) echo "Error: INPUT_BATCH_JSONLS entries must be .jsonl: $_pq" >&2; exit 1 ;;
  esac
done

# Stage-specific retrieval depth (default to TOP_K)
BM25_TOP_K="${BM25_TOP_K:-$TOP_K}"
DENSE_TOP_K="${DENSE_TOP_K:-$TOP_K}"
HYBRID_CAP="${HYBRID_CAP:-$TOP_K}"
HYBRID_K_MAX_EVAL="${HYBRID_K_MAX_EVAL:-$TOP_K}"
RERANK_CANDIDATE_LIMIT="${RERANK_CANDIDATE_LIMIT:-$TOP_K}"

# Reranker takes top K from hybrid; must be <= hybrid output, clamped to [30, 2000]
if [ "$RERANK_CANDIDATE_LIMIT" -le "$HYBRID_CAP" ]; then
  RERANK_RAW=$RERANK_CANDIDATE_LIMIT
else
  RERANK_RAW=$HYBRID_CAP
fi
if [ "$RERANK_RAW" -lt 30 ]; then
  RERANK_EFFECTIVE=30
  echo "Reranker candidate-limit clamped to minimum 30"
elif [ "$RERANK_RAW" -gt 2000 ]; then
  RERANK_EFFECTIVE=2000
  echo "Reranker candidate-limit clamped to maximum 2000"
else
  RERANK_EFFECTIVE=$RERANK_RAW
fi

RETRIEVAL_OUT="$WORKFLOW_OUTPUT_DIR/retrieval"
BM25_OUT="$RETRIEVAL_OUT/bm25"
DENSE_OUT="$RETRIEVAL_OUT/dense"
HYBRID_OUT="$RETRIEVAL_OUT/fusion"
RERANK_ROOT="$WORKFLOW_OUTPUT_DIR/rerank"
CROSS_ENCODER_OUT="$RERANK_ROOT/cross_encoder"
POST_RERANK_FUSION_OUT="$RERANK_ROOT/post_rerank_fusion"
POST_RERANK_FUSION_SNIPPET_OUT="$RERANK_ROOT/post_rerank_fusion_snippet"
POST_RERANK_FUSION_TSTAR_OUT="$RERANK_ROOT/post_rerank_fusion_tstar"
POST_RERANK_FUSION_SNIPPET_TSTAR_OUT="$RERANK_ROOT/post_rerank_fusion_snippet_tstar"
SNIPPET_ROOT="$WORKFLOW_OUTPUT_DIR/snippet"
SNIPPET_RERANK_OUT="$SNIPPET_ROOT/snippet_rerank"
SNIPPET_DOC_FUSION_OUT="$SNIPPET_ROOT/snippet_doc_fusion"

# BM25 method name used by hybrid stage: default to RM3, but switch to plain BM25 when RM3 is disabled.
if [ "${BM25_DISABLE_RM3:-0}" = "1" ]; then
  BM25_METHOD_FOR_HYBRID="BM25"
else
  BM25_METHOD_FOR_HYBRID="BM25_RM3"
fi

mkdir -p "$BM25_OUT" "$DENSE_OUT" "$HYBRID_OUT"

# Step count for progress (3 = retrieval only, 4 = + reranker, 5 = + RRF fusion; 7 = + snippet extraction + final RRF)
  if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "1" ]; then
  if [ "$RUN_RRF_FUSION" = "1" ]; then
    if [ "${DO_SNIPPET_RRF:-0}" = "1" ]; then
      TOTAL_STEPS=7
    else
      TOTAL_STEPS=5
    fi
  else
    TOTAL_STEPS=4
  fi
else
  TOTAL_STEPS=3
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Suppress Hugging Face model-loading progress (Loading weights / Materializing param) in .err logs
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# Run log and optional Python log file (set LOG_LEVEL=DEBUG etc. to tune)
PIPELINE_RUN_LOG="${PIPELINE_RUN_LOG:-$WORKFLOW_OUTPUT_DIR/pipeline_run.log}"
export LOG_FILE="${LOG_FILE:-$WORKFLOW_OUTPUT_DIR/pipeline.log}"
_log_run() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$PIPELINE_RUN_LOG"; }
# Config snapshot at start
_log_run "start" "steps=$TOTAL_STEPS" "out=$WORKFLOW_OUTPUT_DIR" "config=${CONFIG_FILE:-}" "RUN_SNIPPET_RRF=${DO_SNIPPET_RRF:-0}"

# ---------- Multi-query field helpers (comma-separated *_QUERY_FIELD -> sub-runs + RRF fuse) ----------
_trim_csv_field() {
  echo "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}
_query_field_has_comma() {
  case "${1:-}" in
    *,*) return 0 ;;
    *) return 1 ;;
  esac
}
_subdir_for_query_field() {
  local f="$1"
  f="${f//\//_}"
  echo "_sub_${f}"
}
# Sets global _MQUERY_FIELDS (array of trimmed fields) from comma-separated $1
_parse_query_field_csv() {
  _MQUERY_FIELDS=()
  local csv="$1"
  [ -z "$csv" ] && return 0
  local rest="$csv"
  local x
  while [ -n "$rest" ]; do
    x="${rest%%,*}"
    if [ "$x" = "$rest" ]; then
      rest=""
    else
      rest="${rest#*,}"
    fi
    x=$(_trim_csv_field "$x")
    [ -n "$x" ] && _MQUERY_FIELDS+=("$x")
  done
}
# Build comma-separated string from _MQUERY_FIELDS array (sets _MQUERY_LABELS)
_build_mquery_labels() {
  _MQUERY_LABELS=""
  local _sep=""
  for _f in "${_MQUERY_FIELDS[@]}"; do
    _MQUERY_LABELS="${_MQUERY_LABELS}${_sep}${_f}"
    _sep=","
  done
}
# Usage: _run_multi_query_fuse <out_runs_dir> <glob_pattern> <k_rrf> <weights_or_empty> <cap_or_empty> <labels_csv_or_empty> <body_weight_or_empty> -- <run_dir1> <run_dir2> ...
# Eval is auto-enabled when HAVE_GROUND_TRUTH!=0 and INPUT_JSONL/INPUT_BATCH_JSONLS are set.
# Set _FUSE_EXTRA_ARGS=(...) before calling to pass additional flags (e.g. --plot, --no-fused-all-plots, --recall-k-max).
_run_multi_query_fuse() {
  local _out="$1" _pat="$2" _k="$3" _weights="$4" _cap="$5" _labels="$6" _body_w="$7"
  shift 7
  if [ "${1:-}" != "--" ]; then
    echo "Error: _run_multi_query_fuse internal: expected -- before run-dirs" >&2
    exit 1
  fi
  shift
  [ "$#" -lt 1 ] && { echo "Error: _run_multi_query_fuse: no run-dirs" >&2; exit 1; }
  mkdir -p "$_out"
  local _args=(
    "$SCRIPT_DIR/retrieval/multi_query_fuse.py"
    --out-dir "$_out"
    --pattern "$_pat"
    --k-rrf "${_k:-60}"
    --run-dirs "$@"
  )
  [ -n "$_weights" ] && _args+=(--weights "$_weights")
  [ -n "$_cap" ] && _args+=(--cap "$_cap")
  [ -n "$_labels" ] && _args+=(--labels "$_labels")
  [ -n "$_body_w" ] && _args+=(--body-weight "$_body_w")
  if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ]; then
    [ -n "${INPUT_JSONL:-}" ] && _args+=(--train-jsonl "$INPUT_JSONL")
    [ -n "${INPUT_BATCH_JSONLS:-}" ] && _args+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
    [ -n "${RECALL_KS:-}" ] && _args+=(--ks "$RECALL_KS")
  else
    _args+=(--no-eval)
  fi
  [ "${#_FUSE_EXTRA_ARGS[@]}" -gt 0 ] 2>/dev/null && _args+=("${_FUSE_EXTRA_ARGS[@]}")
  python "${_args[@]}"
}

# ----- BM25 -----
BM25_ARGS=(
  --index_path "$BM25_INDEX_PATH"
  --out_dir "$BM25_OUT"
  --k_eval "$BM25_TOP_K"
  --ks "$RECALL_KS"
)
[ -n "${INPUT_JSONL:-}" ] && BM25_ARGS+=(--train_jsonl "$INPUT_JSONL")
[ -n "${INPUT_BATCH_JSONLS:-}" ] && BM25_ARGS+=(--test_batch_jsonls $INPUT_BATCH_JSONLS)
[ -n "${BM25_JAVA_MEM:-}" ] && BM25_ARGS+=(--java_mem "$BM25_JAVA_MEM")
[ -n "${BM25_THREADS:-}" ] && BM25_ARGS+=(--threads "$BM25_THREADS")
[ -n "${BM25_RM3_FEEDBACK_POOL:-}" ] && BM25_ARGS+=(--k_feedback "$BM25_RM3_FEEDBACK_POOL")
[ -n "${BM25_RM3_FB_DOCS:-}" ] && BM25_ARGS+=(--rm3_fb_docs "$BM25_RM3_FB_DOCS")
[ -n "${BM25_RM3_FB_TERMS:-}" ] && BM25_ARGS+=(--rm3_fb_terms "$BM25_RM3_FB_TERMS")
[ -n "${BM25_RM3_LAMBDA:-}" ] && BM25_ARGS+=(--rm3_lambda "$BM25_RM3_LAMBDA")
[ "${BM25_INCLUDE_BASELINE:-0}" = "1" ] && BM25_ARGS+=(--include_bm25)
[ "${BM25_DISABLE_RM3:-0}" = "1" ] && BM25_ARGS+=(--disable_rm3)
[ "${BM25_NO_EVAL:-0}" = "1" ] && BM25_ARGS+=(--no_eval)
[ "${BM25_SAVE_RUNS:-1}" = "1" ] && BM25_ARGS+=(--save_runs)
[ "${BM25_SAVE_PER_QUERY:-0}" = "1" ] && BM25_ARGS+=(--save_per_query)
[ "${BM25_SAVE_ZERO_RECALL:-0}" = "1" ] && BM25_ARGS+=(--save_zero_recall)
[ "${BM25_NO_EXCLUDE_TEST_QIDS:-0}" = "1" ] && BM25_ARGS+=(--no_exclude_test_qids)
[ -n "${BM25_QUERY_FIELD:-}" ] && ! _query_field_has_comma "$BM25_QUERY_FIELD" && BM25_ARGS+=(--query-field "$BM25_QUERY_FIELD")

if [ -f "$BM25_OUT/metrics.csv" ] || [ -n "$(find "$BM25_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
  echo "[1/$TOTAL_STEPS] BM25... (skip: output exists)"
  _log_run "step" "1" "BM25" "skip"
elif [ -n "${BM25_QUERY_FIELD:-}" ] && _query_field_has_comma "$BM25_QUERY_FIELD"; then
  echo "[1/$TOTAL_STEPS] BM25... (multi-query field fusion)"
  STEP_BM25_START=$(date +%s)
  _parse_query_field_csv "$BM25_QUERY_FIELD"
  if [ "${#_MQUERY_FIELDS[@]}" -lt 2 ]; then
    echo "Error: BM25_QUERY_FIELD must list at least two fields when using commas." >&2
    exit 1
  fi
  _BM25_FUSE_DIRS=()
  for _qf in "${_MQUERY_FIELDS[@]}"; do
    _bm25_sub="$BM25_OUT/$(_subdir_for_query_field "$_qf")"
    mkdir -p "$_bm25_sub"
    if [ -n "$(find "$_bm25_sub/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      echo "  BM25 sub-run skip (exists): $_bm25_sub"
    else
      _bm25_one=("${BM25_ARGS[@]}")
      # Replace --out_dir target with subdir
      _bm25_new=()
      _skip_next=0
      for _a in "${_bm25_one[@]}"; do
        if [ "$_skip_next" = "1" ]; then
          _bm25_new+=("$_bm25_sub")
          _skip_next=0
          continue
        fi
        if [ "$_a" = "--out_dir" ]; then
          _bm25_new+=("$_a")
          _skip_next=1
          continue
        fi
        _bm25_new+=("$_a")
      done
      _bm25_one=("${_bm25_new[@]}")
      _bm25_one+=(--query-field "$_qf" --no_eval --skip-empty-query-field)
      python "$SCRIPT_DIR/retrieval/eval_bm25_rm3.py" "${_bm25_one[@]}"
    fi
    _BM25_FUSE_DIRS+=("$_bm25_sub/runs")
  done
  mkdir -p "$BM25_OUT/runs"
  _build_mquery_labels
  _FUSE_EXTRA_ARGS=()
  _run_multi_query_fuse "$BM25_OUT/runs" "*.tsv" "${BM25_QUERY_FUSION_K_RRF:-60}" "${BM25_QUERY_FUSION_WEIGHTS:-}" "$BM25_TOP_K" "$_MQUERY_LABELS" "${BM25_QUERY_BODY_WEIGHT:-}" -- "${_BM25_FUSE_DIRS[@]}"
  STEP_BM25_END=$(date +%s)
  echo "[timing] BM25 step: $((STEP_BM25_END-STEP_BM25_START))s"
  _log_run "step" "1" "BM25" "$((STEP_BM25_END-STEP_BM25_START))s"
else
  echo "[1/$TOTAL_STEPS] BM25..."
  STEP_BM25_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_bm25_rm3.py" "${BM25_ARGS[@]}"
  STEP_BM25_END=$(date +%s)
  echo "[timing] BM25 step: $((STEP_BM25_END-STEP_BM25_START))s"
  _log_run "step" "1" "BM25" "$((STEP_BM25_END-STEP_BM25_START))s"
fi

# ----- Dense -----
# Support either a single dense index dir (DENSE_INDEX_DIR) or a sharded glob (DENSE_INDEX_GLOB).
# Exactly one of these should normally be set in the config; if both are set, DENSE_INDEX_GLOB wins.
if [ -n "${DENSE_INDEX_GLOB:-}" ]; then
  DENSE_ARGS=(
    --index_glob "$DENSE_INDEX_GLOB"
    --out_dir "$DENSE_OUT"
    --topk "$DENSE_TOP_K"
    --ks "$RECALL_KS"
  )
else
  DENSE_ARGS=(
    --index_dir "$DENSE_INDEX_DIR"
    --out_dir "$DENSE_OUT"
    --topk "$DENSE_TOP_K"
    --ks "$RECALL_KS"
  )
fi
[ -n "${INPUT_JSONL:-}" ] && DENSE_ARGS+=(--train-jsonl "$INPUT_JSONL")
[ -n "${INPUT_BATCH_JSONLS:-}" ] && DENSE_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
[ -n "${DENSE_EF_SEARCH:-}" ] && DENSE_ARGS+=(--ef_search "$DENSE_EF_SEARCH")
[ -n "${DENSE_EF_CAP:-}" ] && DENSE_ARGS+=(--ef_cap "$DENSE_EF_CAP")
[ -n "${DENSE_BATCH_SIZE:-}" ] && DENSE_ARGS+=(--batch_size "$DENSE_BATCH_SIZE")
[ -n "${DENSE_DEVICE:-}" ] && DENSE_ARGS+=(--device "$DENSE_DEVICE")
[ -n "${DENSE_MODEL_NAME:-}" ] && DENSE_ARGS+=(--model_name "$DENSE_MODEL_NAME")
[ "${DENSE_NO_EVAL:-0}" = "1" ] && DENSE_ARGS+=(--no_eval)
[ "${DENSE_SAVE_PER_QUERY:-0}" = "1" ] && DENSE_ARGS+=(--save_per_query)
[ -n "${DENSE_QUERY_FIELD:-}" ] && ! _query_field_has_comma "$DENSE_QUERY_FIELD" && DENSE_ARGS+=(--query-field "$DENSE_QUERY_FIELD")

if [ -f "$DENSE_OUT/metrics.csv" ] || [ -n "$(find "$DENSE_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
  echo "[2/$TOTAL_STEPS] Dense... (skip: output exists)"
  _log_run "step" "2" "Dense" "skip"
elif [ -n "${DENSE_QUERY_FIELD:-}" ] && _query_field_has_comma "$DENSE_QUERY_FIELD"; then
  echo "[2/$TOTAL_STEPS] Dense... (multi-query field fusion)"
  STEP_DENSE_START=$(date +%s)
  _parse_query_field_csv "$DENSE_QUERY_FIELD"
  if [ "${#_MQUERY_FIELDS[@]}" -lt 2 ]; then
    echo "Error: DENSE_QUERY_FIELD must list at least two fields when using commas." >&2
    exit 1
  fi
  _DENSE_FUSE_DIRS=()
  for _qf in "${_MQUERY_FIELDS[@]}"; do
    _dense_sub="$DENSE_OUT/$(_subdir_for_query_field "$_qf")"
    mkdir -p "$_dense_sub"
    if [ -n "$(find "$_dense_sub/runs" -maxdepth 1 -name 'dense_*.tsv' 2>/dev/null | head -1)" ]; then
      echo "  Dense sub-run skip (exists): $_dense_sub"
    else
      _dense_one=("${DENSE_ARGS[@]}")
      _dense_new=()
      _skip_next=0
      for _a in "${_dense_one[@]}"; do
        if [ "$_skip_next" = "1" ]; then
          _dense_new+=("$_dense_sub")
          _skip_next=0
          continue
        fi
        if [ "$_a" = "--out_dir" ]; then
          _dense_new+=("$_a")
          _skip_next=1
          continue
        fi
        _dense_new+=("$_a")
      done
      _dense_one=("${_dense_new[@]}")
      _dense_one+=(--query-field "$_qf" --no_eval --skip-empty-query-field)
      python "$SCRIPT_DIR/retrieval/eval_dense.py" "${_dense_one[@]}"
    fi
    _DENSE_FUSE_DIRS+=("$_dense_sub/runs")
  done
  mkdir -p "$DENSE_OUT/runs"
  _build_mquery_labels
  _FUSE_EXTRA_ARGS=(--plot recall --no-fused-all-plots)
  _run_multi_query_fuse "$DENSE_OUT/runs" "dense_*.tsv" "${DENSE_QUERY_FUSION_K_RRF:-60}" "${DENSE_QUERY_FUSION_WEIGHTS:-}" "$DENSE_TOP_K" "$_MQUERY_LABELS" "${DENSE_QUERY_BODY_WEIGHT:-}" -- "${_DENSE_FUSE_DIRS[@]}"
  _FUSE_EXTRA_ARGS=()
  STEP_DENSE_END=$(date +%s)
  echo "[timing] Dense step: $((STEP_DENSE_END-STEP_DENSE_START))s"
  _log_run "step" "2" "Dense" "$((STEP_DENSE_END-STEP_DENSE_START))s"
else
  echo "[2/$TOTAL_STEPS] Dense..."
  STEP_DENSE_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_dense.py" "${DENSE_ARGS[@]}"
  STEP_DENSE_END=$(date +%s)
  echo "[timing] Dense step: $((STEP_DENSE_END-STEP_DENSE_START))s"
  _log_run "step" "2" "Dense" "$((STEP_DENSE_END-STEP_DENSE_START))s"
fi

# ----- Hybrid -----
HYBRID_ARGS=(
  --bm25_runs_dir "$BM25_OUT/runs"
  --bm25_method "$BM25_METHOD_FOR_HYBRID"
  --bm25_topk "$BM25_TOP_K"
  --dense_root "$DENSE_OUT"
  --out_dir "$HYBRID_OUT"
  --k_max_eval "$HYBRID_K_MAX_EVAL"
  --cap "$HYBRID_CAP"
  --ks "$RECALL_KS"
)
[ -n "${INPUT_JSONL:-}" ] && HYBRID_ARGS+=(--train-jsonl "$INPUT_JSONL")
[ -n "${INPUT_BATCH_JSONLS:-}" ] && HYBRID_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
[ -n "${HYBRID_MODE:-}" ] && HYBRID_ARGS+=(--mode "$HYBRID_MODE")
[ -n "${HYBRID_K_RRF:-}" ] && HYBRID_ARGS+=(--k_rrf "$HYBRID_K_RRF")
[ -n "${HYBRID_W_BM25:-}" ] && HYBRID_ARGS+=(--w_bm25 "$HYBRID_W_BM25")
[ -n "${HYBRID_W_DENSE:-}" ] && HYBRID_ARGS+=(--w_dense "$HYBRID_W_DENSE")
[ -n "${HYBRID_WEIGHTS:-}" ] && HYBRID_ARGS+=(--weights "$HYBRID_WEIGHTS")
[ -n "${HYBRID_K_RRF_LIST:-}" ] && HYBRID_ARGS+=(--k_rrf_list "$HYBRID_K_RRF_LIST")
[ -n "${HYBRID_JOBS:-}" ] && HYBRID_ARGS+=(--jobs "$HYBRID_JOBS")
[ "${HYBRID_NO_EVAL:-0}" = "1" ] && HYBRID_ARGS+=(--no_eval)
[ "${HYBRID_NO_PLOTS:-0}" = "1" ] && HYBRID_ARGS+=(--no_plots)
[ "${HYBRID_SAVE_PLOTS:-0}" = "1" ] && HYBRID_ARGS+=(--save_plots)

if [ -f "$HYBRID_OUT/ranked_test_avg.csv" ] || [ -f "$HYBRID_OUT/metrics.csv" ] || [ -n "$(find "$HYBRID_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
  echo "[3/$TOTAL_STEPS] Hybrid... (skip: output exists)"
  _log_run "step" "3" "Hybrid" "skip"
else
  echo "[3/$TOTAL_STEPS] Hybrid..."
  STEP_HYBRID_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_hybrid.py" "${HYBRID_ARGS[@]}"
  STEP_HYBRID_END=$(date +%s)
  echo "[timing] Hybrid step: $((STEP_HYBRID_END-STEP_HYBRID_START))s"
  _log_run "step" "3" "Hybrid" "$((STEP_HYBRID_END-STEP_HYBRID_START))s"
fi

# ----- Reranker (optional: only if DOCS_JSONL set and not --no-rerank) -----
# Figures are named hybrid_reranker_recall_map10_{label}.png (one per dataset label)
if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "1" ]; then
  STEP_RERANK_START=$(date +%s)
  # Only consider rerank "complete" when metrics.csv exists; partial TSVs allow resume in rerank_stage2.py
  RERANK_RESULTS_EXIST=0
  if [ -f "$CROSS_ENCODER_OUT/metrics.csv" ] || [ -n "$(find "$CROSS_ENCODER_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
    RERANK_RESULTS_EXIST=1
  fi

  RERANK_FIGS_EXIST=0
  [ -n "$(find "$CROSS_ENCODER_OUT/figures" -maxdepth 1 -name 'hybrid_reranker_recall_map10_*.png' 2>/dev/null | head -1)" ] && RERANK_FIGS_EXIST=1

  # Check whether RRF fusion (retrieval fusion + cross-encoder -> post_rerank_fusion) already exists
  RRF_EXIST=0
  if [ -f "$POST_RERANK_FUSION_OUT/metrics.csv" ] || [ -n "$(find "$POST_RERANK_FUSION_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
    RRF_EXIST=1
  fi

  # ----- Step 4: Reranker (always report and run/skip before step 5) -----
  if [ "$RERANK_RESULTS_EXIST" = "1" ]; then
    if [ "$RERANK_FIGS_EXIST" = "1" ]; then
      echo "[4/$TOTAL_STEPS] Reranker... (skip: output exists)"
    elif [ -f "$CROSS_ENCODER_OUT/metrics.csv" ]; then
      echo "[4/$TOTAL_STEPS] Reranker... (generating eval plots from existing results)"
      PLOT_ARGS=(--output-dir "$CROSS_ENCODER_OUT" --runs-dir "$HYBRID_OUT/runs")
      [ -n "${INPUT_JSONL:-}" ] && PLOT_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && PLOT_ARGS+=(--test_batch_jsonls $INPUT_BATCH_JSONLS)
      python "$SCRIPT_DIR/rerank/plot_rerank_eval.py" "${PLOT_ARGS[@]}"
    else
      echo "[4/$TOTAL_STEPS] Reranker... (skip: run TSVs exist but no metrics.csv; plots require metrics, e.g. HAVE_GROUND_TRUTH=1)"
    fi
  else
    echo "[4/$TOTAL_STEPS] Reranker..."
    mkdir -p "$CROSS_ENCODER_OUT"
    # If DOCS_JSONL has fewer than 2000 docs, cap candidate-limit to that count
    if [ -f "$DOCS_JSONL" ]; then
      DOCS_JSONL_LINES=$(wc -l < "$DOCS_JSONL" 2>/dev/null || echo 0)
      if [ "$DOCS_JSONL_LINES" -gt 0 ] && [ "$DOCS_JSONL_LINES" -lt 2000 ] && [ "$DOCS_JSONL_LINES" -lt "$RERANK_EFFECTIVE" ]; then
        RERANK_EFFECTIVE=$DOCS_JSONL_LINES
        echo "Reranker candidate-limit capped to doc count in DOCS_JSONL ($DOCS_JSONL_LINES)"
      fi
    fi
    if [ -n "${RERANK_QUERY_FIELD:-}" ] && _query_field_has_comma "$RERANK_QUERY_FIELD"; then
      echo "[4/$TOTAL_STEPS] Reranker... (multi-query field fusion)"
      _parse_query_field_csv "$RERANK_QUERY_FIELD"
      if [ "${#_MQUERY_FIELDS[@]}" -lt 2 ]; then
        echo "Error: RERANK_QUERY_FIELD must list at least two fields when using commas." >&2
        exit 1
      fi
      _RERANK_FUSE_DIRS=()
      for _qf in "${_MQUERY_FIELDS[@]}"; do
        _rr_sub="$CROSS_ENCODER_OUT/$(_subdir_for_query_field "$_qf")"
        mkdir -p "$_rr_sub"
        if [ -n "$(find "$_rr_sub/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
          echo "  Rerank sub-run skip (exists): $_rr_sub"
        else
          RERANK_ARGS=(
            --runs-dir "$HYBRID_OUT/runs"
            --output-dir "$_rr_sub"
            --docs-jsonl "$DOCS_JSONL"
            --candidate-limit "$RERANK_EFFECTIVE"
            --ks-recall "${RERANK_KS_RECALL:-$RECALL_KS}"
            --query-field "$_qf"
            --disable-metrics
            --skip-empty-query-field
          )
          [ -n "${INPUT_JSONL:-}" ] && RERANK_ARGS+=(--train-jsonl "$INPUT_JSONL")
          [ -n "${INPUT_BATCH_JSONLS:-}" ] && RERANK_ARGS+=(--test_batch_jsonls $INPUT_BATCH_JSONLS)
          [ -n "${RERANK_MODEL:-}" ] && RERANK_ARGS+=(--model "$RERANK_MODEL")
          [ -n "${RERANK_MODEL_DEVICE:-}" ] && RERANK_ARGS+=(--model-device "$RERANK_MODEL_DEVICE")
          [ -n "${RERANK_MODEL_BATCH:-}" ] && RERANK_ARGS+=(--model-batch "$RERANK_MODEL_BATCH")
          [ -n "${RERANK_MODEL_MAX_LENGTH:-}" ] && RERANK_ARGS+=(--model-max-length "$RERANK_MODEL_MAX_LENGTH")
          [ "${RERANK_USE_MULTI_GPU:-0}" = "1" ] && RERANK_ARGS+=(--use-multi-gpu)
          [ -n "${RERANK_NUM_GPUS:-}" ] && RERANK_ARGS+=(--num-gpus "$RERANK_NUM_GPUS")
          if [ "${RERANK_RERANKER_TYPE:-}" = "llm" ] || [ "${RERANK_USE_LLM:-0}" = "1" ]; then
            RERANK_ARGS+=(--reranker-type llm)
          fi
          [ "${RERANK_LLM_USE_FP16:-1}" = "0" ] && RERANK_ARGS+=(--no-llm-use-fp16)
          [ "${RERANK_LLM_USE_BF16:-0}" = "1" ] && RERANK_ARGS+=(--llm-use-bf16)
          [ -n "${RERANK_PROGRESS_EVERY:-}" ] && RERANK_ARGS+=(--progress-every "$RERANK_PROGRESS_EVERY")
          python "$SCRIPT_DIR/rerank/rerank_stage2.py" "${RERANK_ARGS[@]}"
        fi
        _RERANK_FUSE_DIRS+=("$_rr_sub/runs")
      done
      mkdir -p "$CROSS_ENCODER_OUT/runs"
      _build_mquery_labels
      _FUSE_EXTRA_ARGS=(--no-fused-all-plots --recall-k-max "$RERANK_EFFECTIVE")
      _run_multi_query_fuse "$CROSS_ENCODER_OUT/runs" "*.tsv" "${RERANK_QUERY_FUSION_K_RRF:-60}" "${RERANK_QUERY_FUSION_WEIGHTS:-}" "$RERANK_EFFECTIVE" "$_MQUERY_LABELS" "${RERANK_QUERY_BODY_WEIGHT:-}" -- "${_RERANK_FUSE_DIRS[@]}"
      _FUSE_EXTRA_ARGS=()
    else
      RERANK_ARGS=(
        --runs-dir "$HYBRID_OUT/runs"
        --output-dir "$CROSS_ENCODER_OUT"
        --docs-jsonl "$DOCS_JSONL"
        --candidate-limit "$RERANK_EFFECTIVE"
        --ks-recall "${RERANK_KS_RECALL:-$RECALL_KS}"
      )
      [ -n "${INPUT_JSONL:-}" ] && RERANK_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && RERANK_ARGS+=(--test_batch_jsonls $INPUT_BATCH_JSONLS)
      [ -n "${RERANK_MODEL:-}" ] && RERANK_ARGS+=(--model "$RERANK_MODEL")
      [ -n "${RERANK_MODEL_DEVICE:-}" ] && RERANK_ARGS+=(--model-device "$RERANK_MODEL_DEVICE")
      [ -n "${RERANK_MODEL_BATCH:-}" ] && RERANK_ARGS+=(--model-batch "$RERANK_MODEL_BATCH")
      [ -n "${RERANK_MODEL_MAX_LENGTH:-}" ] && RERANK_ARGS+=(--model-max-length "$RERANK_MODEL_MAX_LENGTH")
      [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && RERANK_ARGS+=(--disable-metrics)
      [ "${RERANK_USE_MULTI_GPU:-0}" = "1" ] && RERANK_ARGS+=(--use-multi-gpu)
      [ -n "${RERANK_NUM_GPUS:-}" ] && RERANK_ARGS+=(--num-gpus "$RERANK_NUM_GPUS")
      [ -n "${RERANK_QUERY_FIELD:-}" ] && RERANK_ARGS+=(--query-field "$RERANK_QUERY_FIELD")
      if [ "${RERANK_RERANKER_TYPE:-}" = "llm" ] || [ "${RERANK_USE_LLM:-0}" = "1" ]; then
        RERANK_ARGS+=(--reranker-type llm)
      fi
      [ "${RERANK_LLM_USE_FP16:-1}" = "0" ] && RERANK_ARGS+=(--no-llm-use-fp16)
      [ "${RERANK_LLM_USE_BF16:-0}" = "1" ] && RERANK_ARGS+=(--llm-use-bf16)
      [ -n "${RERANK_PROGRESS_EVERY:-}" ] && RERANK_ARGS+=(--progress-every "$RERANK_PROGRESS_EVERY")
      python "$SCRIPT_DIR/rerank/rerank_stage2.py" "${RERANK_ARGS[@]}"
    fi
  fi
  STEP_RERANK_END=$(date +%s)
  echo "[timing] Reranker step: $((STEP_RERANK_END-STEP_RERANK_START))s"
  _log_run "step" "4" "Reranker" "$((STEP_RERANK_END-STEP_RERANK_START))s"

  RERANK_TSTAR_ENABLE="${RERANK_TSTAR_ENABLE:-0}"
  RERANK_TSTAR_FLOOR="${RERANK_TSTAR_FLOOR:-5}"

  _apply_rerank_tstar_cutoff() {
    local _in_runs="$1"
    local _out_base="$2"
    local _tag="$3"
    [ "${RERANK_TSTAR_ENABLE:-0}" = "1" ] || return 0
    if [ -z "${RERANK_TSTAR:-}" ]; then
      echo "Error: RERANK_TSTAR_ENABLE=1 but RERANK_TSTAR is not set." >&2
      exit 1
    fi
    if [ -n "${RERANK_TSTAR_CAP:-}" ] && [ "${RERANK_TSTAR_CAP}" -lt "${RERANK_TSTAR_FLOOR:-5}" ]; then
      echo "Error: RERANK_TSTAR_CAP (${RERANK_TSTAR_CAP}) < RERANK_TSTAR_FLOOR (${RERANK_TSTAR_FLOOR:-5})" >&2
      exit 1
    fi
    [ -d "$_in_runs" ] || {
      echo "[t* $_tag] skip: missing input runs dir: $_in_runs" >&2
      return 0
    }
    if [ -z "$(find "$_in_runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      echo "[t* $_tag] skip: no TSV in $_in_runs"
      return 0
    fi
    if [ -n "$(find "$_out_base/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      echo "[t* $_tag] skip: output exists in $_out_base/runs"
      return 0
    fi
    echo "[t* $_tag] threshold=${RERANK_TSTAR} floor=${RERANK_TSTAR_FLOOR:-5} -> $_out_base/runs"
    mkdir -p "$_out_base/runs"
    _TS_ARGS=(
      python "$SCRIPT_DIR/rerank/apply_tstar_cutoff.py"
      --input-runs-dir "$_in_runs"
      --rerank-runs-dir "$CROSS_ENCODER_OUT/runs"
      --output-runs-dir "$_out_base/runs"
      --tstar "$RERANK_TSTAR"
      --floor "${RERANK_TSTAR_FLOOR:-5}"
    )
    # Multi-query rerank: fused rerank/cross_encoder/runs scores are RRF-combined; t* uses max raw score per doc across _sub_* runs.
    if [ -n "${RERANK_QUERY_FIELD:-}" ] && _query_field_has_comma "$RERANK_QUERY_FIELD"; then
      _parse_query_field_csv "$RERANK_QUERY_FIELD"
      for _qf in "${_MQUERY_FIELDS[@]}"; do
        _TS_ARGS+=(--rerank-sub-runs-dir "$CROSS_ENCODER_OUT/$(_subdir_for_query_field "$_qf")/runs")
      done
      echo "[t* $_tag] score_source=sub_max (${#_MQUERY_FIELDS[@]} fields)"
    fi
    [ -n "${RERANK_TSTAR_CAP:-}" ] && _TS_ARGS+=(--cap "$RERANK_TSTAR_CAP")
    if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ] && [ "${RERANK_DISABLE_METRICS:-0}" != "1" ]; then
      [ -n "${INPUT_JSONL:-}" ] && _TS_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && _TS_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
    else
      _TS_ARGS+=(--disable-metrics)
    fi
    "${_TS_ARGS[@]}"
  }

  # ----- Step 5: RRF fusion (retrieval fusion + cross-encoder -> post_rerank_fusion); only when RUN_RRF_FUSION=1 -----
  # Snippet-only (no run-both): write pool=200 to post_rerank_fusion_snippet so snippet always gets 200-pool runs.
  # Use DO_SNIPPET_RRF (from config RUN_SNIPPET_RRF or flag --snippet-rrf) so config-only snippet route works.
  # Baseline or run-both: write pool=50 to post_rerank_fusion; step 5b (run-both only) writes pool=200 to post_rerank_fusion_snippet.
  if [ "$RUN_RRF_FUSION" = "1" ] && { [ "$TOTAL_STEPS" = "5" ] || [ "$TOTAL_STEPS" = "7" ]; }; then
    STEP_RRF_START=$(date +%s)
    # Resolve where step 5 writes and with which pool
    _RRF_DEFAULT=50
    [ "${DO_SNIPPET_RRF:-0}" = "1" ] && [ "${RUN_BOTH_ROUTES:-0}" != "1" ] && _RRF_DEFAULT=200
    _RRF_POOL_RERANK="${RRF_POOL_TOP_RERANK:-${RRF_POOL_TOP:-$_RRF_DEFAULT}}"
    _RRF_POOL_HYBRID="${RRF_POOL_TOP_HYBRID:-${RRF_POOL_TOP:-$_RRF_DEFAULT}}"
    if [ "${DO_SNIPPET_RRF:-0}" = "1" ] && [ "${RUN_BOTH_ROUTES:-0}" != "1" ]; then
      _RRF_STEP5_OUT="$POST_RERANK_FUSION_SNIPPET_OUT"
    else
      _RRF_STEP5_OUT="$POST_RERANK_FUSION_OUT"
    fi
    RRF_EXIST=0
    if [ -f "$_RRF_STEP5_OUT/metrics.csv" ] || [ -n "$(find "$_RRF_STEP5_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      RRF_EXIST=1
    fi
    if [ "$RRF_EXIST" = "1" ]; then
      echo "[5/$TOTAL_STEPS] RRF fusion (Hybrid + Rerank, top-10)... (skip: output exists in $_RRF_STEP5_OUT)"
    else
      echo "[5/$TOTAL_STEPS] RRF fusion (Hybrid + Rerank, top-10)... (pool R=$_RRF_POOL_RERANK H=$_RRF_POOL_HYBRID -> $_RRF_STEP5_OUT)"
      # Validate: pool sizes must not exceed upstream output sizes
      if [ "$_RRF_POOL_RERANK" -gt "$RERANK_EFFECTIVE" ]; then
        echo "WARNING: RRF_POOL_TOP_RERANK ($_RRF_POOL_RERANK) > RERANK_CANDIDATE_LIMIT ($RERANK_EFFECTIVE); reranker output will be silently truncated." >&2
      fi
      if [ "$_RRF_POOL_HYBRID" -gt "$HYBRID_CAP" ]; then
        echo "WARNING: RRF_POOL_TOP_HYBRID ($_RRF_POOL_HYBRID) > HYBRID_CAP ($HYBRID_CAP); hybrid output will be silently truncated." >&2
      fi

      RRF_ARGS=(
        --hybrid-runs-dir "$HYBRID_OUT/runs"
        --rerank-runs-dir "$CROSS_ENCODER_OUT/runs"
        --output-dir "$_RRF_STEP5_OUT"
        --pool-top-rerank "$_RRF_POOL_RERANK"
        --pool-top-hybrid "$_RRF_POOL_HYBRID"
      )
      [ -n "${RRF_K_RRF:-}" ] && RRF_ARGS+=(--k-rrf "$RRF_K_RRF")
      [ -n "${RRF_W_BGE:-}" ] && RRF_ARGS+=(--w-bge "$RRF_W_BGE")
      [ -n "${RRF_W_HYBRID:-}" ] && RRF_ARGS+=(--w-hybrid "$RRF_W_HYBRID")
      [ -n "${INPUT_JSONL:-}" ] && RRF_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && RRF_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
      [ -n "${RERANK_KS_RECALL:-}" ] && RRF_ARGS+=(--ks-recall "$RERANK_KS_RECALL")
      [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && RRF_ARGS+=(--disable-metrics)
      python "$SCRIPT_DIR/rerank/rerank_rrf_hybrid.py" "${RRF_ARGS[@]}"
    fi
    if [ "$_RRF_STEP5_OUT" = "$POST_RERANK_FUSION_OUT" ]; then
      _apply_rerank_tstar_cutoff "$_RRF_STEP5_OUT/runs" "$POST_RERANK_FUSION_TSTAR_OUT" "post_rerank_fusion"
    else
      _apply_rerank_tstar_cutoff "$_RRF_STEP5_OUT/runs" "$POST_RERANK_FUSION_SNIPPET_TSTAR_OUT" "post_rerank_fusion_snippet"
    fi
    STEP_RRF_END=$(date +%s)
    echo "[timing] Hybrid+Rerank RRF fusion step: $((STEP_RRF_END-STEP_RRF_START))s"
    _log_run "step" "5" "RRF" "$((STEP_RRF_END-STEP_RRF_START))s"
  fi

  # ----- Step 5b: RRF pool=200 for snippet route (only when --run-both-routes) -----
  if [ "${RUN_BOTH_ROUTES:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
    RRF_200_EXIST=0
    [ -f "$POST_RERANK_FUSION_SNIPPET_OUT/metrics.csv" ] || [ -n "$(find "$POST_RERANK_FUSION_SNIPPET_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && RRF_200_EXIST=1
    if [ "$RRF_200_EXIST" = "1" ]; then
      echo "[5b] RRF fusion pool=200 (for snippet)... (skip: output exists)"
    else
      echo "[5b] RRF fusion pool=200 (for snippet route)..."
      _RRF_POOL_RERANK_200="${RRF_POOL_TOP_RERANK:-${RRF_POOL_TOP:-200}}"
      _RRF_POOL_HYBRID_200="${RRF_POOL_TOP_HYBRID:-${RRF_POOL_TOP:-200}}"
      [ "$_RRF_POOL_RERANK_200" -lt 200 ] && _RRF_POOL_RERANK_200=200
      [ "$_RRF_POOL_HYBRID_200" -lt 200 ] && _RRF_POOL_HYBRID_200=200
      python "$SCRIPT_DIR/rerank/rerank_rrf_hybrid.py" \
        --hybrid-runs-dir "$HYBRID_OUT/runs" \
        --rerank-runs-dir "$CROSS_ENCODER_OUT/runs" \
        --output-dir "$POST_RERANK_FUSION_SNIPPET_OUT" \
        --pool-top-rerank "$_RRF_POOL_RERANK_200" \
        --pool-top-hybrid "$_RRF_POOL_HYBRID_200" \
        ${RRF_K_RRF:+--k-rrf "$RRF_K_RRF"} \
        ${RRF_W_BGE:+--w-bge "$RRF_W_BGE"} \
        ${RRF_W_HYBRID:+--w-hybrid "$RRF_W_HYBRID"} \
        ${INPUT_JSONL:+--train-jsonl "$INPUT_JSONL"} \
        ${INPUT_BATCH_JSONLS:+--test-batch-jsonls $INPUT_BATCH_JSONLS} \
        ${RERANK_KS_RECALL:+--ks-recall "$RERANK_KS_RECALL"} \
        $([ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && echo --disable-metrics)
    fi
    _apply_rerank_tstar_cutoff "$POST_RERANK_FUSION_SNIPPET_OUT/runs" "$POST_RERANK_FUSION_SNIPPET_TSTAR_OUT" "post_rerank_fusion_snippet_5b"
  fi

  # ----- Compare (independent checkpoints: run when inputs exist and figures missing; not tied to 5/5b) -----
  # (1) Rerank vs Hybrid+Rerank (pool 50): output in post_rerank_fusion/figures (always pre-t* fusion; t* summaries live under post_rerank_fusion_tstar/).
  COMPARE_KS="${COMPARE_KS:-10,20,30,50,100,200,300}"
  _COMPARE_RECALL_MAX=300
  if [ -n "${COMPARE_RECALL_K_MAX:-}" ]; then
    _COMPARE_RECALL_MAX="$COMPARE_RECALL_K_MAX"
  elif [ -n "${RRF_POOL_TOP_RERANK:-}" ] && [ -n "${RRF_POOL_TOP_HYBRID:-}" ]; then
    _COMPARE_RECALL_MAX="$((RRF_POOL_TOP_RERANK + RRF_POOL_TOP_HYBRID))"
  fi
  if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ] && [ -d "$CROSS_ENCODER_OUT/runs" ] && [ -d "$POST_RERANK_FUSION_OUT/runs" ]; then
    COMPARE_FIGS_EXIST=0
    [ -n "$(find "$POST_RERANK_FUSION_OUT/figures" -maxdepth 1 -name 'compare_*.png' 2>/dev/null | head -1)" ] && COMPARE_FIGS_EXIST=1
    if [ "$COMPARE_FIGS_EXIST" = "1" ]; then
      echo "[Compare] Rerank vs Hybrid+Rerank... (skip: figures exist)"
      _log_run "step" "Compare" "skip"
    else
      echo "[Compare] Rerank vs Hybrid+Rerank (recall & MAP @ $COMPARE_KS, recall-k-max $_COMPARE_RECALL_MAX)..."
      STEP_COMPARE_START=$(date +%s)
      COMPARE_ARGS=(
        --dirs "$CROSS_ENCODER_OUT" "$POST_RERANK_FUSION_OUT"
        --labels "Rerank" "Hybrid+Rerank"
        --plot both
        --map-ks "$COMPARE_KS"
        --ks-recall "$COMPARE_KS"
        --recall-k-max "$_COMPARE_RECALL_MAX"
        --output-dir "$POST_RERANK_FUSION_OUT"
        --force-from-runs
        --plots-by-split
      )
      [ -n "${INPUT_JSONL:-}" ] && COMPARE_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && COMPARE_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
      python "$SCRIPT_DIR/analysis/compare_result_dirs.py" "${COMPARE_ARGS[@]}"
      STEP_COMPARE_END=$(date +%s)
      echo "[timing] Compare step: $((STEP_COMPARE_END-STEP_COMPARE_START))s"
      _log_run "step" "Compare" "$((STEP_COMPARE_END-STEP_COMPARE_START))s"
    fi
  else
    echo "[Compare] Rerank vs Hybrid+Rerank... (skip: HAVE_GROUND_TRUTH=0 or rerank/cross_encoder/runs or rerank/post_rerank_fusion/runs missing)"
    _log_run "step" "Compare" "skip (condition)"
  fi

  # (2) Rerank vs Hybrid+Rerank pool=200: output in post_rerank_fusion_snippet/figures (always pre-t*; t* stats under post_rerank_fusion_snippet_tstar/).
  _COMPARE_RECALL_MAX_200=400
  [ -n "${COMPARE_RECALL_K_MAX:-}" ] && _COMPARE_RECALL_MAX_200="$COMPARE_RECALL_K_MAX"
  if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ] && [ -d "$CROSS_ENCODER_OUT/runs" ] && [ -d "$POST_RERANK_FUSION_SNIPPET_OUT/runs" ]; then
    COMPARE_200_FIGS_EXIST=0
    [ -n "$(find "$POST_RERANK_FUSION_SNIPPET_OUT/figures" -maxdepth 1 -name 'compare_*.png' 2>/dev/null | head -1)" ] && COMPARE_200_FIGS_EXIST=1
    if [ "$COMPARE_200_FIGS_EXIST" = "1" ]; then
      echo "[Compare] Rerank vs Hybrid+Rerank (pool=200)... (skip: figures exist)"
      _log_run "step" "Compare200" "skip"
    else
      echo "[Compare] Rerank vs Hybrid+Rerank (pool=200) (recall & MAP @ $COMPARE_KS, recall-k-max $_COMPARE_RECALL_MAX_200)..."
      STEP_COMPARE_200_START=$(date +%s)
      COMPARE_200_ARGS=(
        --dirs "$CROSS_ENCODER_OUT" "$POST_RERANK_FUSION_SNIPPET_OUT"
        --labels "Rerank" "Hybrid+Rerank (pool=200)"
        --plot both
        --map-ks "$COMPARE_KS"
        --ks-recall "$COMPARE_KS"
        --recall-k-max "$_COMPARE_RECALL_MAX_200"
        --output-dir "$POST_RERANK_FUSION_SNIPPET_OUT"
        --force-from-runs
        --plots-by-split
      )
      [ -n "${INPUT_JSONL:-}" ] && COMPARE_200_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && COMPARE_200_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
      python "$SCRIPT_DIR/analysis/compare_result_dirs.py" "${COMPARE_200_ARGS[@]}"
      STEP_COMPARE_200_END=$(date +%s)
      echo "[timing] Compare (pool=200) step: $((STEP_COMPARE_200_END-STEP_COMPARE_200_START))s"
      _log_run "step" "Compare200" "$((STEP_COMPARE_200_END-STEP_COMPARE_200_START))s"
    fi
  else
    echo "[Compare] Rerank vs Hybrid+Rerank (pool=200)... (skip: HAVE_GROUND_TRUTH=0 or rerank/cross_encoder/runs or rerank/post_rerank_fusion_snippet/runs missing)"
    _log_run "step" "Compare200" "skip (condition)"
  fi

  # ----- Step 6: Snippet extraction + CE reranking (only when --snippet-rrf) -----
  if [ "${DO_SNIPPET_RRF:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
    STEP_SNIPPET_START=$(date +%s)
    SNIPPET_EXIST=0
    if [ -f "$SNIPPET_RERANK_OUT/metrics.csv" ] || [ -n "$(find "$SNIPPET_RERANK_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      SNIPPET_EXIST=1
    fi
    if [ "$SNIPPET_EXIST" = "1" ]; then
      echo "[6/$TOTAL_STEPS] Snippet extraction + CE rerank... (skip: output exists)"
    else
      echo "[6/$TOTAL_STEPS] Snippet extraction + CE rerank..."
      mkdir -p "$SNIPPET_RERANK_OUT"
      # Snippet route always uses pool=200 runs (from post_rerank_fusion_snippet: step 5 snippet-only or step 5b run-both)
      if [ "${RERANK_TSTAR_ENABLE:-0}" = "1" ]; then
        _SNIPPET_RRF_INPUT="$POST_RERANK_FUSION_SNIPPET_TSTAR_OUT/runs"
      else
        _SNIPPET_RRF_INPUT="$POST_RERANK_FUSION_SNIPPET_OUT/runs"
      fi
      if [ -n "${RERANK_QUERY_FIELD:-}" ] && _query_field_has_comma "$RERANK_QUERY_FIELD"; then
        echo "[6/$TOTAL_STEPS] Snippet... (multi-query field fusion)"
        _parse_query_field_csv "$RERANK_QUERY_FIELD"
        if [ "${#_MQUERY_FIELDS[@]}" -lt 2 ]; then
          echo "Error: RERANK_QUERY_FIELD must list at least two fields when using commas (snippet uses same var)." >&2
          exit 1
        fi
        _SNIP_FUSE_CAP="${HYBRID_CAP:-$TOP_K}"
        _SNIP_FUSE_DIRS=()
        for _qf in "${_MQUERY_FIELDS[@]}"; do
          _sn_sub="$SNIPPET_RERANK_OUT/$(_subdir_for_query_field "$_qf")"
          mkdir -p "$_sn_sub"
          if [ -n "$(find "$_sn_sub/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
            echo "  Snippet sub-run skip (exists): $_sn_sub"
          else
            SNIPPET_ARGS=(
              --runs-dir "$_SNIPPET_RRF_INPUT"
              --docs-jsonl "$DOCS_JSONL"
              --output-dir "$_sn_sub"
              --n-docs "${SNIPPET_N_DOCS:-100}"
              --window-size "${SNIPPET_WINDOW_SIZE:-3}"
              --window-stride "${SNIPPET_WINDOW_STRIDE:-1}"
              --top-w "${SNIPPET_TOP_W:-8}"
              --dense-model "${SNIPPET_DENSE_MODEL:-abhinand/MedEmbed-small-v0.1}"
              --ce-model "${SNIPPET_CE_MODEL:-${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}}"
              --query-field "$_qf"
              --disable-metrics
              --skip-empty-query-field
            )
            [ -n "${INPUT_JSONL:-}" ] && SNIPPET_ARGS+=(--train-jsonl "$INPUT_JSONL")
            [ -n "${INPUT_BATCH_JSONLS:-}" ] && SNIPPET_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
            [ -n "${SNIPPET_DENSE_DEVICE:-}" ] && SNIPPET_ARGS+=(--dense-device "$SNIPPET_DENSE_DEVICE")
            [ -n "${SNIPPET_DENSE_BATCH:-}" ] && SNIPPET_ARGS+=(--dense-batch "$SNIPPET_DENSE_BATCH")
            [ -n "${SNIPPET_CE_DEVICE:-}" ] && SNIPPET_ARGS+=(--ce-device "$SNIPPET_CE_DEVICE")
            [ -n "${SNIPPET_CE_BATCH:-}" ] && SNIPPET_ARGS+=(--ce-batch "$SNIPPET_CE_BATCH")
            [ -n "${SNIPPET_CE_MAX_LENGTH:-}" ] && SNIPPET_ARGS+=(--ce-max-length "$SNIPPET_CE_MAX_LENGTH")
            [ "${SNIPPET_CE_USE_MULTI_GPU:-${RERANK_USE_MULTI_GPU:-0}}" = "1" ] && SNIPPET_ARGS+=(--ce-use-multi-gpu)
            [ -n "${SNIPPET_CE_NUM_GPUS:-${RERANK_NUM_GPUS:-}}" ] && SNIPPET_ARGS+=(--ce-num-gpus "${SNIPPET_CE_NUM_GPUS:-$RERANK_NUM_GPUS}")
            if [ "${SNIPPET_CE_RERANKER_TYPE:-${RERANK_RERANKER_TYPE:-}}" = "llm" ] || [ "${SNIPPET_CE_USE_LLM:-${RERANK_USE_LLM:-0}}" = "1" ]; then
              SNIPPET_ARGS+=(--ce-reranker-type llm)
            fi
            [ "${SNIPPET_CE_LLM_USE_FP16:-${RERANK_LLM_USE_FP16:-1}}" = "0" ] && SNIPPET_ARGS+=(--no-ce-llm-use-fp16)
            [ "${SNIPPET_CE_LLM_USE_BF16:-${RERANK_LLM_USE_BF16:-0}}" = "1" ] && SNIPPET_ARGS+=(--ce-llm-use-bf16)
            python "$SCRIPT_DIR/evidence/snippet_rerank.py" "${SNIPPET_ARGS[@]}"
          fi
          _SNIP_FUSE_DIRS+=("$_sn_sub/runs")
        done
        mkdir -p "$SNIPPET_RERANK_OUT/runs"
        _build_mquery_labels
        _FUSE_EXTRA_ARGS=(--no-fused-all-plots --recall-k-max "$_SNIP_FUSE_CAP")
        _run_multi_query_fuse "$SNIPPET_RERANK_OUT/runs" "*.tsv" "${RERANK_QUERY_FUSION_K_RRF:-60}" "${RERANK_QUERY_FUSION_WEIGHTS:-}" "$_SNIP_FUSE_CAP" "$_MQUERY_LABELS" "${RERANK_QUERY_BODY_WEIGHT:-}" -- "${_SNIP_FUSE_DIRS[@]}"
        _FUSE_EXTRA_ARGS=()
        mkdir -p "$SNIPPET_RERANK_OUT/windows"
        rm -f "$SNIPPET_RERANK_OUT"/windows/*.part 2>/dev/null || true
        for _sd in "$SNIPPET_RERANK_OUT"/_sub_*/; do
          [ -d "${_sd}windows" ] || continue
          for _wf in "${_sd}"windows/*.jsonl; do
            [ -f "$_wf" ] || continue
            _bn=$(basename "$_wf")
            cat "$_wf" >> "$SNIPPET_RERANK_OUT/windows/${_bn}.part"
          done
        done
        shopt -s nullglob
        for _part in "$SNIPPET_RERANK_OUT"/windows/*.part; do
          [ -f "$_part" ] || continue
          _final="${_part%.part}"
          mv -f "$_part" "$_final"
        done
        shopt -u nullglob
      else
        SNIPPET_ARGS=(
          --runs-dir "$_SNIPPET_RRF_INPUT"
          --docs-jsonl "$DOCS_JSONL"
          --output-dir "$SNIPPET_RERANK_OUT"
          --n-docs "${SNIPPET_N_DOCS:-100}"
          --window-size "${SNIPPET_WINDOW_SIZE:-3}"
          --window-stride "${SNIPPET_WINDOW_STRIDE:-1}"
          --top-w "${SNIPPET_TOP_W:-8}"
          --dense-model "${SNIPPET_DENSE_MODEL:-abhinand/MedEmbed-small-v0.1}"
          --ce-model "${SNIPPET_CE_MODEL:-${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}}"
        )
        [ -n "${INPUT_JSONL:-}" ] && SNIPPET_ARGS+=(--train-jsonl "$INPUT_JSONL")
        [ -n "${INPUT_BATCH_JSONLS:-}" ] && SNIPPET_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
        [ -n "${SNIPPET_DENSE_DEVICE:-}" ] && SNIPPET_ARGS+=(--dense-device "$SNIPPET_DENSE_DEVICE")
        [ -n "${SNIPPET_DENSE_BATCH:-}" ] && SNIPPET_ARGS+=(--dense-batch "$SNIPPET_DENSE_BATCH")
        [ -n "${SNIPPET_CE_DEVICE:-}" ] && SNIPPET_ARGS+=(--ce-device "$SNIPPET_CE_DEVICE")
        [ -n "${SNIPPET_CE_BATCH:-}" ] && SNIPPET_ARGS+=(--ce-batch "$SNIPPET_CE_BATCH")
        [ -n "${SNIPPET_CE_MAX_LENGTH:-}" ] && SNIPPET_ARGS+=(--ce-max-length "$SNIPPET_CE_MAX_LENGTH")
        # Snippet CE multi-GPU defaults to reranker settings when not set
        [ "${SNIPPET_CE_USE_MULTI_GPU:-${RERANK_USE_MULTI_GPU:-0}}" = "1" ] && SNIPPET_ARGS+=(--ce-use-multi-gpu)
        [ -n "${SNIPPET_CE_NUM_GPUS:-${RERANK_NUM_GPUS:-}}" ] && SNIPPET_ARGS+=(--ce-num-gpus "${SNIPPET_CE_NUM_GPUS:-$RERANK_NUM_GPUS}")
        [ -n "${RERANK_QUERY_FIELD:-}" ] && SNIPPET_ARGS+=(--query-field "$RERANK_QUERY_FIELD")
        [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && SNIPPET_ARGS+=(--disable-metrics)
        # LLM reranker for snippet Stage B (falls back to rerank-level settings when snippet-specific not set)
        if [ "${SNIPPET_CE_RERANKER_TYPE:-${RERANK_RERANKER_TYPE:-}}" = "llm" ] || [ "${SNIPPET_CE_USE_LLM:-${RERANK_USE_LLM:-0}}" = "1" ]; then
          SNIPPET_ARGS+=(--ce-reranker-type llm)
        fi
        [ "${SNIPPET_CE_LLM_USE_FP16:-${RERANK_LLM_USE_FP16:-1}}" = "0" ] && SNIPPET_ARGS+=(--no-ce-llm-use-fp16)
        [ "${SNIPPET_CE_LLM_USE_BF16:-${RERANK_LLM_USE_BF16:-0}}" = "1" ] && SNIPPET_ARGS+=(--ce-llm-use-bf16)
        python "$SCRIPT_DIR/evidence/snippet_rerank.py" "${SNIPPET_ARGS[@]}"
      fi
    fi
    STEP_SNIPPET_END=$(date +%s)
    echo "[timing] Snippet extraction + CE rerank step: $((STEP_SNIPPET_END-STEP_SNIPPET_START))s"
    if [ "$SNIPPET_EXIST" = "1" ]; then
      _log_run "step" "6" "Snippet" "skip"
    else
      _log_run "step" "6" "Snippet" "$((STEP_SNIPPET_END-STEP_SNIPPET_START))s"
    fi
  fi

  # ----- Step 7: Final RRF fusion (post_rerank_fusion doc side + snippet_rerank 0.2 -> snippet_doc_fusion); only when --snippet-rrf -----
  if [ "${DO_SNIPPET_RRF:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
    STEP_FINAL_RRF_START=$(date +%s)
    SNIPPET_RRF_EXIST=0
    if [ -f "$SNIPPET_DOC_FUSION_OUT/metrics.csv" ] || [ -n "$(find "$SNIPPET_DOC_FUSION_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      SNIPPET_RRF_EXIST=1
    fi
    if [ "$SNIPPET_RRF_EXIST" = "1" ]; then
      echo "[7/$TOTAL_STEPS] Final RRF (docs 0.8 + snippet 0.2)... (skip: output exists)"
    else
      echo "[7/$TOTAL_STEPS] Final RRF (docs 0.8 + snippet 0.2)..."
      # Doc side: post_rerank_fusion_snippet when snippet route (pool 200); else post_rerank_fusion
      _STEP7_DOCS_DIR="$POST_RERANK_FUSION_OUT"
      if [ "${DO_SNIPPET_RRF:-0}" = "1" ]; then
        _STEP7_DOCS_DIR="$POST_RERANK_FUSION_SNIPPET_OUT"
        [ "${RERANK_TSTAR_ENABLE:-0}" = "1" ] && _STEP7_DOCS_DIR="$POST_RERANK_FUSION_SNIPPET_TSTAR_OUT"
      elif [ "${RERANK_TSTAR_ENABLE:-0}" = "1" ]; then
        _STEP7_DOCS_DIR="$POST_RERANK_FUSION_TSTAR_OUT"
      fi
      # Pool size default: SNIPPET_FINAL_POOL, falling back to SNIPPET_N_DOCS (and then 100)
      _SNIP_POOL="${SNIPPET_FINAL_POOL:-${SNIPPET_N_DOCS:-100}}"
      python "$SCRIPT_DIR/rerank/rerank_rrf_hybrid.py" \
        --hybrid-runs-dir "$_STEP7_DOCS_DIR/runs" \
        --rerank-runs-dir "$SNIPPET_RERANK_OUT/runs" \
        --output-dir "$SNIPPET_DOC_FUSION_OUT" \
        --pool-top-rerank "$_SNIP_POOL" \
        --pool-top-hybrid "$_SNIP_POOL" \
        --k-rrf "${SNIPPET_RRF_K:-60}" \
        --w-bge "${SNIPPET_RRF_W_SNIPPET:-0.2}" \
        --w-hybrid "${SNIPPET_RRF_W_DOCS:-0.8}" \
        ${INPUT_JSONL:+--train-jsonl "$INPUT_JSONL"} \
        ${INPUT_BATCH_JSONLS:+--test-batch-jsonls $INPUT_BATCH_JSONLS} \
        ${RERANK_KS_RECALL:+--ks-recall "$RERANK_KS_RECALL"} \
        $([ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && echo --disable-metrics)
    fi
    STEP_FINAL_RRF_END=$(date +%s)
    echo "[timing] Final RRF (snippet_doc_fusion) step: $((STEP_FINAL_RRF_END-STEP_FINAL_RRF_START))s"
    if [ "$SNIPPET_RRF_EXIST" = "1" ]; then
      _log_run "step" "7" "FinalRRF" "skip"
    else
      _log_run "step" "7" "FinalRRF" "$((STEP_FINAL_RRF_END-STEP_FINAL_RRF_START))s"
    fi
  fi

  # ----- Compare (Snippet RRF vs Hybrid+Rerank): recall and MAP up to SNIPPET_N_DOCS -----
  # Compares snippet_doc_fusion run to the doc-side run (post_rerank_fusion_snippet); eval k capped at SNIPPET_N_DOCS.
  # Only run when snippet_doc_fusion actually has run files (step 7 may skip all if stems don't match).
  _SNIPPET_RRF_HAS_RUNS=0
  [ -n "$(find "$SNIPPET_DOC_FUSION_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _SNIPPET_RRF_HAS_RUNS=1
  # Doc-side compare uses pre-t* post_rerank_fusion_snippet (fusion diagnostic); production doc input for snippet remains *_tstar when enabled.
  if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ] && [ "${DO_SNIPPET_RRF:-0}" = "1" ] && [ "$_SNIPPET_RRF_HAS_RUNS" = "1" ] && [ -d "$POST_RERANK_FUSION_SNIPPET_OUT/runs" ]; then
    _SNIP_N="${SNIPPET_N_DOCS:-100}"
    COMPARE_KS_SNIPPET="${COMPARE_KS_SNIPPET:-10,20,30,50,100}"
    [ "$_SNIP_N" -gt 100 ] && COMPARE_KS_SNIPPET="${COMPARE_KS_SNIPPET},200"
    [ "$_SNIP_N" -gt 200 ] && COMPARE_KS_SNIPPET="${COMPARE_KS_SNIPPET},$_SNIP_N"
    SNIPPET_COMPARE_FIGS=0
    [ -n "$(find "$SNIPPET_DOC_FUSION_OUT/figures" -maxdepth 1 -name 'compare_*.png' 2>/dev/null | head -1)" ] && SNIPPET_COMPARE_FIGS=1
    if [ "$SNIPPET_COMPARE_FIGS" = "1" ]; then
      echo "[Compare] Snippet RRF vs Hybrid+Rerank... (skip: figures exist in snippet/snippet_doc_fusion/figures)"
      _log_run "step" "SnippetCompare" "skip"
    else
      echo "[Compare] Snippet RRF vs Hybrid+Rerank (recall & MAP up to k=$_SNIP_N)..."
      STEP_SNIPPET_COMPARE_START=$(date +%s)
      SNIPPET_COMPARE_ARGS=(
        --dirs "$POST_RERANK_FUSION_SNIPPET_OUT" "$SNIPPET_DOC_FUSION_OUT"
        --labels "Hybrid+Rerank (docs)" "Snippet RRF (docs+CE)"
        --plot both
        --map-ks "$COMPARE_KS_SNIPPET"
        --ks-recall "$COMPARE_KS_SNIPPET"
        --recall-k-max "$_SNIP_N"
        --output-dir "$SNIPPET_DOC_FUSION_OUT"
        --force-from-runs
        --plots-by-split
      )
      [ -n "${INPUT_JSONL:-}" ] && SNIPPET_COMPARE_ARGS+=(--train-jsonl "$INPUT_JSONL")
      [ -n "${INPUT_BATCH_JSONLS:-}" ] && SNIPPET_COMPARE_ARGS+=(--test-batch-jsonls $INPUT_BATCH_JSONLS)
      python "$SCRIPT_DIR/analysis/compare_result_dirs.py" "${SNIPPET_COMPARE_ARGS[@]}"
      STEP_SNIPPET_COMPARE_END=$(date +%s)
      echo "[timing] Snippet compare step: $((STEP_SNIPPET_COMPARE_END-STEP_SNIPPET_COMPARE_START))s"
      _log_run "step" "SnippetCompare" "$((STEP_SNIPPET_COMPARE_END-STEP_SNIPPET_COMPARE_START))s"
    fi
  fi

# ----- Evidence (post-rerank JSON + contexts): build all splits first -----
# Always use separate dirs: evidence/evidence_baseline + generation/generation_baseline (from post_rerank_fusion), evidence/evidence_snippet + generation/generation_snippet (from snippet_doc_fusion).
# Baseline first then snippet-rrf in same folder: both run without overwrite.
_DOCS_JSONL_OK=0
  if [ -n "${DOCS_JSONL:-}" ]; then
    if [ -f "$DOCS_JSONL" ]; then
      _DOCS_JSONL_OK=1
    elif compgen -G "$DOCS_JSONL" > /dev/null 2>&1; then
      _DOCS_JSONL_OK=1
    fi
  fi
  if [ "$_DOCS_JSONL_OK" = "1" ]; then
    STEP_EVIDENCE_GEN_START=$(date +%s)
    # Decide which routes to build evidence/generation for based on RUN_BASELINE / RUN_SNIPPET_RRF
    if [ "$RUN_BASELINE" = "1" ] && [ "$DO_SNIPPET_RRF" = "1" ]; then
      _ROUTES_LIST="baseline snippet"
    elif [ "$RUN_BASELINE" = "1" ]; then
      _ROUTES_LIST="baseline"
    elif [ "$DO_SNIPPET_RRF" = "1" ]; then
      _ROUTES_LIST="snippet"
    else
      # Fallback: if both disabled, keep baseline to avoid doing nothing silently
      _ROUTES_LIST="baseline"
    fi
    for _route in $_ROUTES_LIST; do
      if [ "$_route" = "baseline" ]; then
        # When RRF fusion is enabled, baseline evidence uses Hybrid+Rerank runs.
        # When RUN_RRF_FUSION=0, fall back to raw Rerank runs so downstream steps still work.
        if [ "$RUN_RRF_FUSION" = "1" ]; then
          if [ "${RERANK_TSTAR_ENABLE:-0}" = "1" ]; then
            _EVIDENCE_RUNS_DIR="$POST_RERANK_FUSION_TSTAR_OUT/runs"
            _EVIDENCE_POST_DIR="$POST_RERANK_FUSION_TSTAR_OUT"
          else
            _EVIDENCE_RUNS_DIR="$POST_RERANK_FUSION_OUT/runs"
            _EVIDENCE_POST_DIR="$POST_RERANK_FUSION_OUT"
          fi
        else
          _EVIDENCE_RUNS_DIR="$CROSS_ENCODER_OUT/runs"
          _EVIDENCE_POST_DIR="$CROSS_ENCODER_OUT"
        fi
        _EVIDENCE_SUBDIR="evidence/evidence_baseline"
        _GEN_SUBDIR="generation/generation_baseline"
        _USE_SNIPPET_CTX=0
      else
        _EVIDENCE_RUNS_DIR="$SNIPPET_DOC_FUSION_OUT/runs"
        _EVIDENCE_POST_DIR="$SNIPPET_DOC_FUSION_OUT"
        _EVIDENCE_SUBDIR="evidence/evidence_snippet"
        _GEN_SUBDIR="generation/generation_snippet"
        _USE_SNIPPET_CTX=1
      fi
      if [ "$_route" = "baseline" ]; then
        _EVIDENCE_TOP_K="${EVIDENCE_TOP_K_BASELINE:-${EVIDENCE_TOP_K:-10}}"
      else
        _EVIDENCE_TOP_K="${EVIDENCE_TOP_K_SNIPPET:-${EVIDENCE_TOP_K:-10}}"
      fi
      _POST_RERANK_POOL="${POST_RERANK_DOC_POOL:-30}"
      [ ! -d "$_EVIDENCE_RUNS_DIR" ] && continue
      mkdir -p "$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR"
      echo "[Evidence] Route: $_route -> $_EVIDENCE_SUBDIR, $_GEN_SUBDIR (post_rerank doc_pool=$_POST_RERANK_POOL evidence_top_k=$_EVIDENCE_TOP_K)"
    for _tsv in "$_EVIDENCE_RUNS_DIR/"*.tsv; do
      [ -f "$_tsv" ] || continue
      _stem=$(basename "$_tsv" .tsv)
      _split="${_stem#best_rrf_}"
      _split="${_split%%_top*}"
      [ -n "$_split" ] || continue

      # Snippet evidence: windows are named by split (snippet_rerank writes {split}.jsonl).
      # Resolve query JSONL for this split: match basename without .jsonl to split name.
      _query_jsonl=""
      if [ -f "${INPUT_JSONL:-}" ] && [ "$(basename "$INPUT_JSONL" .jsonl)" = "$_split" ]; then
        _query_jsonl="$INPUT_JSONL"
      fi
      if [ -z "$_query_jsonl" ]; then
        for _p in ${INPUT_BATCH_JSONLS:-}; do
          [ -f "$_p" ] || continue
          [ "$(basename "$_p" .jsonl)" = "$_split" ] || continue
          _query_jsonl="$_p"
          break
        done
      fi
      if [ -z "$_query_jsonl" ]; then
        echo "[Evidence] Skip $_split: no matching INPUT_JSONL or INPUT_BATCH_JSONLS (basename without .jsonl)"
        continue
      fi

      _post_json="$_EVIDENCE_POST_DIR/post_rerank_${_split}.jsonl"
      if [ ! -f "$_post_json" ]; then
        echo "[Evidence] Post-rerank JSONL ($_split)..."
        if [ "$_USE_SNIPPET_CTX" = "1" ] && [ -f "$SNIPPET_RERANK_OUT/windows/${_split}.jsonl" ]; then
          python "$SCRIPT_DIR/evidence/post_rerank_jsonl.py" \
            --run-path "$_tsv" \
            --query-jsonl "$_query_jsonl" \
            --output-path "$_post_json" \
            --top-k "$_POST_RERANK_POOL" \
            --windows-jsonl "$SNIPPET_RERANK_OUT/windows/${_split}.jsonl"
        else
          if [ "$_USE_SNIPPET_CTX" = "1" ]; then
            echo "[Evidence] Warning: no windows file $SNIPPET_RERANK_OUT/windows/${_split}.jsonl; post_rerank without CE merge" >&2
          fi
          python "$SCRIPT_DIR/evidence/post_rerank_jsonl.py" \
            --run-path "$_tsv" \
            --query-jsonl "$_query_jsonl" \
            --output-path "$_post_json" \
            --top-k "$_POST_RERANK_POOL"
        fi
      else
        echo "[Evidence] Post-rerank JSONL ($_split)... (skip: output exists)"
      fi

      _ctx_json="$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR/${_split}_contexts.jsonl"
      if [ ! -f "$_ctx_json" ]; then
        if [ "$_USE_SNIPPET_CTX" = "1" ]; then
          echo "[Evidence] Contexts from snippets ($_split)..."
          python "$SCRIPT_DIR/evidence/build_contexts_from_snippets.py" \
            --post-rerank-jsonl "$_post_json" \
            --snippet-windows-dir "$SNIPPET_RERANK_OUT/windows" \
            --split-name "$_split" \
            --corpus-path "$DOCS_JSONL" \
            --output-path "$_ctx_json" \
            --window-size "${SNIPPET_WINDOW_SIZE:-3}" \
            --top-windows "${SNIPPET_CONTEXT_TOP_WINDOWS:-2}" \
            --evidence-top-k "$_EVIDENCE_TOP_K"
        else
          echo "[Evidence] Contexts from documents ($_split)..."
          python "$SCRIPT_DIR/evidence/build_contexts_from_documents.py" \
            --post-rerank-jsonl "$_post_json" \
            --corpus-path "$DOCS_JSONL" \
            --output-path "$_ctx_json" \
            --evidence-top-k "$_EVIDENCE_TOP_K"
        fi
      else
        echo "[Evidence] Contexts ($_split)... (skip: output exists)"
      fi
    done

    # ----- Generation (LLM answers from contexts JSON): run after all evidence is built -----
    _RUN_GEN=0
    [ "$_route" = "baseline" ] && [ "${RUN_GENERATION_BASELINE:-1}" = "1" ] && _RUN_GEN=1
    [ "$_route" = "snippet" ] && [ "${RUN_GENERATION_SNIPPET:-1}" = "1" ] && _RUN_GEN=1
    if [ "$_RUN_GEN" = "1" ]; then
      mkdir -p "$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR"
      for _tsv in "$_EVIDENCE_RUNS_DIR/"*.tsv; do
        [ -f "$_tsv" ] || continue
        _stem=$(basename "$_tsv" .tsv)
        _split="${_stem#best_rrf_}"
        _split="${_split%%_top*}"
        [ -n "$_split" ] || continue

        _ctx_json="$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR/${_split}_contexts.jsonl"
        _gen_json="$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR/${_split}_answers.jsonl"
        if [ -f "$_gen_json" ]; then
          echo "[Generation] $_split... (skip: output exists)"
        elif [ ! -f "$_ctx_json" ]; then
          echo "[Generation] Skip $_split: evidence not found ($_ctx_json)"
        else
          echo "[Generation] $_split..."
          GENERATION_ARGS=(
            --input-path "$_ctx_json"
            --output-dir "$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR"
            --schemas-dir "$GENERATION_SCHEMAS_DIR"
          )
          [ -n "${GENERATION_CONCURRENCY:-}" ] && GENERATION_ARGS+=(--concurrency "$GENERATION_CONCURRENCY")
          if [ "$_route" = "baseline" ]; then
            _max_ctx="${GENERATION_MAX_CONTEXTS_BASELINE:-${GENERATION_MAX_CONTEXTS:-8}}"
          else
            _max_ctx="${GENERATION_MAX_CONTEXTS_SNIPPET:-${GENERATION_MAX_CONTEXTS:-10}}"
          fi
          [ -n "$_max_ctx" ] && GENERATION_ARGS+=(--max-contexts "$_max_ctx")
          if [ "$_route" = "baseline" ]; then
            _max_chars="${GENERATION_MAX_CHARS_PER_CONTEXT_BASELINE:-${GENERATION_MAX_CHARS_PER_CONTEXT:-1300}}"
          else
            _max_chars="${GENERATION_MAX_CHARS_PER_CONTEXT_SNIPPET:-${GENERATION_MAX_CHARS_PER_CONTEXT:-960}}"
          fi
          [ -n "$_max_chars" ] && GENERATION_ARGS+=(--max-chars-per-context "$_max_chars")
          [ -n "${GENERATION_SLEEP:-}" ] && GENERATION_ARGS+=(--sleep "$GENERATION_SLEEP")
          [ -n "${GENERATION_MODEL:-}" ] && GENERATION_ARGS+=(--model "$GENERATION_MODEL")
          [ "${GENERATION_NO_PROGRESS:-1}" = "1" ] && GENERATION_ARGS+=(--no-progress)
          python "$SCRIPT_DIR/generation/generate_answers.py" "${GENERATION_ARGS[@]}"
          # Rescue only when we just ran generation (avoid running rescue on reruns where gen was skipped)
          if [ -f "$_gen_json" ]; then
            echo "[Rescue] $_split..."
            RESCUE_ARGS=(--input "$_gen_json")
            [ -n "${GENERATION_RESCUE_TIMEOUT:-}" ] && RESCUE_ARGS+=(--timeout "$GENERATION_RESCUE_TIMEOUT")
            [ -n "${GENERATION_RESCUE_RETRY_SLEEP:-}" ] && RESCUE_ARGS+=(--retry-sleep "$GENERATION_RESCUE_RETRY_SLEEP")
            [ -n "$_max_ctx" ] && RESCUE_ARGS+=(--max-contexts "$_max_ctx")
            [ -n "${GENERATION_MAX_CHARS_PER_CONTEXT:-}" ] && RESCUE_ARGS+=(--max-chars-per-context "$GENERATION_MAX_CHARS_PER_CONTEXT")
            python "$SCRIPT_DIR/generation/rescue_failed_generation.py" "${RESCUE_ARGS[@]}"
          fi
        fi
      done
    else
      echo "[Generation] Skip $_route (generation disabled for this route)"
    fi
    done
    STEP_EVIDENCE_GEN_END=$(date +%s)
    echo "[timing] Evidence + Generation + Rescue step: $((STEP_EVIDENCE_GEN_END-STEP_EVIDENCE_GEN_START))s"
    _log_run "step" "EvidenceGen" "$((STEP_EVIDENCE_GEN_END-STEP_EVIDENCE_GEN_START))s"
  fi

  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (retrieval/{bm25,dense,fusion}/, rerank/{cross_encoder,post_rerank_fusion,...}/)"
  [ "${DO_SNIPPET_RRF:-0}" = "1" ] && echo "  Snippet route: snippet/{snippet_rerank,snippet_doc_fusion}/"
  [ "$_DOCS_JSONL_OK" = "1" ] && echo "  Evidence/Generation: evidence/evidence_baseline/, generation/generation_baseline/ (baseline); evidence/evidence_snippet/, generation/generation_snippet/ (when --snippet-rrf)"
else
  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (retrieval/{bm25,dense,fusion}/)"
  if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "0" ]; then
    echo "Reranker skipped (--no-rerank). Re-run without --no-rerank to run reranker."
  elif [ -z "${DOCS_JSONL:-}" ]; then
    echo "Optional: set DOCS_JSONL and re-run to add reranker step."
  fi
fi
_log_run "end"
