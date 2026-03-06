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
# Config file sets: WORKFLOW_OUTPUT_DIR, TRAIN_JSON, TEST_BATCH_JSONS (optional), TOP_K,
# RECALL_KS, BM25_INDEX_PATH, DENSE_INDEX_DIR, DOCS_JSONL (optional),
# BM25_QUERY_FIELD / DENSE_QUERY_FIELD (body | body_expansion_synonyms | body_expansion_long),
# and stage overrides (BM25_*, DENSE_*, HYBRID_*, RERANK_*). See workflow_config_full.env.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse -c / --config, --no-rerank, --no-rrf-fusion, --snippet-rrf, --run-both-routes, etc., -h / --help
CONFIG_FILE=""
RUN_RERANK=1
RUN_RRF_FUSION=1
SNIPPET_RRF=0
RUN_BOTH_ROUTES=0
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
    -h|--help)
      echo "Usage: $0 [--config|-c <config.env>] [--no-rerank] [--no-rrf-fusion] [--snippet-rrf] [--run-both-routes] [--bm25-query-field F] ..."
      echo "  -c, --config PATH       Source PATH as config (env vars) before running."
      echo "  --no-rerank             Run only BM25, Dense, Hybrid; skip reranker even if DOCS_JSONL is set."
      echo "  --no-rrf-fusion         Disable RRF fusion (Hybrid + Rerank) step after reranker."
      echo "  --snippet-rrf           Use snippet RRF route: upstream pool=200, snippet extraction, final RRF, snippet-based contexts (writes evidence/, generation/)."
      echo "  --run-both-routes        Run baseline and snippet routes; write evidence_baseline/, evidence_snippet/, generation_baseline/, generation_snippet/ (no overwrite)."
      echo "  --bm25-query-field F    Use F as query text for BM25 (overrides env; e.g. body, body_expansion_long)."
      echo "  --dense-query-field F   Use F as query text for Dense (overrides env)."
      echo "  --rerank-query-field F  Use F as query text for reranker (overrides env; e.g. body, body_expansion_long)."
      echo "  -h, --help              Show this help."
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

# Required env (set by config file or by sourcing before run)
: "${WORKFLOW_OUTPUT_DIR:?Set WORKFLOW_OUTPUT_DIR (e.g. output/workflow_run)}"
: "${TRAIN_JSON:?Set TRAIN_JSON (path to training questions JSON)}"
: "${BM25_INDEX_PATH:?Set BM25_INDEX_PATH (Terrier index directory)}"
if [ -z "${DENSE_INDEX_GLOB:-}" ]; then
  : "${DENSE_INDEX_DIR:?Set DENSE_INDEX_DIR or DENSE_INDEX_GLOB (Dense HNSW index directory or shard glob)}"
fi

TOP_K="${TOP_K:-5000}"
RECALL_KS="${RECALL_KS:-50,100,200,300,400,500,1000,2000,5000}"

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

BM25_OUT="$WORKFLOW_OUTPUT_DIR/bm25"
DENSE_OUT="$WORKFLOW_OUTPUT_DIR/dense"
HYBRID_OUT="$WORKFLOW_OUTPUT_DIR/hybrid"
RERANK_OUT="$WORKFLOW_OUTPUT_DIR/rerank"
RERANK_HYBRID_OUT="$WORKFLOW_OUTPUT_DIR/rerank_hybrid"
RERANK_HYBRID_200_OUT="$WORKFLOW_OUTPUT_DIR/rerank_hybrid_200"
SNIPPET_RERANK_OUT="$WORKFLOW_OUTPUT_DIR/snippet_rerank"
SNIPPET_RRF_OUT="$WORKFLOW_OUTPUT_DIR/snippet_rrf"

mkdir -p "$BM25_OUT" "$DENSE_OUT" "$HYBRID_OUT"

# Step count for progress (3 = retrieval only, 4 = + reranker, 5 = + RRF fusion; 7 = + snippet extraction + final RRF)
if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "1" ]; then
  if [ "$RUN_RRF_FUSION" = "1" ]; then
    if [ "${SNIPPET_RRF:-0}" = "1" ]; then
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

# ----- BM25 -----
BM25_ARGS=(
  --index_path "$BM25_INDEX_PATH"
  --train_json "$TRAIN_JSON"
  --out_dir "$BM25_OUT"
  --k_eval "$BM25_TOP_K"
  --ks "$RECALL_KS"
)
[ -n "${TEST_BATCH_JSONS:-}" ] && BM25_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
[ -n "${BM25_JAVA_MEM:-}" ] && BM25_ARGS+=(--java_mem "$BM25_JAVA_MEM")
[ -n "${BM25_THREADS:-}" ] && BM25_ARGS+=(--threads "$BM25_THREADS")
[ -n "${BM25_RM3_FEEDBACK_POOL:-}" ] && BM25_ARGS+=(--k_feedback "$BM25_RM3_FEEDBACK_POOL")
[ -n "${BM25_RM3_FB_DOCS:-}" ] && BM25_ARGS+=(--rm3_fb_docs "$BM25_RM3_FB_DOCS")
[ -n "${BM25_RM3_FB_TERMS:-}" ] && BM25_ARGS+=(--rm3_fb_terms "$BM25_RM3_FB_TERMS")
[ -n "${BM25_RM3_LAMBDA:-}" ] && BM25_ARGS+=(--rm3_lambda "$BM25_RM3_LAMBDA")
[ "${BM25_INCLUDE_BASELINE:-0}" = "1" ] && BM25_ARGS+=(--include_bm25)
[ "${BM25_NO_EVAL:-0}" = "1" ] && BM25_ARGS+=(--no_eval)
[ "${BM25_SAVE_RUNS:-1}" = "1" ] && BM25_ARGS+=(--save_runs)
[ "${BM25_SAVE_PER_QUERY:-0}" = "1" ] && BM25_ARGS+=(--save_per_query)
[ "${BM25_SAVE_ZERO_RECALL:-0}" = "1" ] && BM25_ARGS+=(--save_zero_recall)
[ "${BM25_NO_EXCLUDE_TEST_QIDS:-0}" = "1" ] && BM25_ARGS+=(--no_exclude_test_qids)
[ -n "${BM25_QUERY_FIELD:-}" ] && BM25_ARGS+=(--query-field "$BM25_QUERY_FIELD")

if [ -f "$BM25_OUT/metrics.csv" ] || [ -n "$(find "$BM25_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
  echo "[1/$TOTAL_STEPS] BM25... (skip: output exists)"
else
  echo "[1/$TOTAL_STEPS] BM25..."
  STEP_BM25_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_bm25_rm3.py" "${BM25_ARGS[@]}"
  STEP_BM25_END=$(date +%s)
  echo "[timing] BM25 step: $((STEP_BM25_END-STEP_BM25_START))s"
fi

# ----- Dense -----
# Support either a single dense index dir (DENSE_INDEX_DIR) or a sharded glob (DENSE_INDEX_GLOB).
# Exactly one of these should normally be set in the config; if both are set, DENSE_INDEX_GLOB wins.
if [ -n "${DENSE_INDEX_GLOB:-}" ]; then
  DENSE_ARGS=(
    --index_glob "$DENSE_INDEX_GLOB"
    --out_dir "$DENSE_OUT"
    --train-json "$TRAIN_JSON"
    --topk "$DENSE_TOP_K"
    --ks "$RECALL_KS"
  )
else
  DENSE_ARGS=(
    --index_dir "$DENSE_INDEX_DIR"
    --out_dir "$DENSE_OUT"
    --train-json "$TRAIN_JSON"
    --topk "$DENSE_TOP_K"
    --ks "$RECALL_KS"
  )
fi
[ -n "${TEST_BATCH_JSONS:-}" ] && DENSE_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
[ -n "${DENSE_EF_SEARCH:-}" ] && DENSE_ARGS+=(--ef_search "$DENSE_EF_SEARCH")
[ -n "${DENSE_EF_CAP:-}" ] && DENSE_ARGS+=(--ef_cap "$DENSE_EF_CAP")
[ -n "${DENSE_BATCH_SIZE:-}" ] && DENSE_ARGS+=(--batch_size "$DENSE_BATCH_SIZE")
[ -n "${DENSE_DEVICE:-}" ] && DENSE_ARGS+=(--device "$DENSE_DEVICE")
[ -n "${DENSE_MODEL_NAME:-}" ] && DENSE_ARGS+=(--model_name "$DENSE_MODEL_NAME")
[ "${DENSE_NO_EVAL:-0}" = "1" ] && DENSE_ARGS+=(--no_eval)
[ "${DENSE_SAVE_PER_QUERY:-0}" = "1" ] && DENSE_ARGS+=(--save_per_query)
[ -n "${DENSE_QUERY_FIELD:-}" ] && DENSE_ARGS+=(--query-field "$DENSE_QUERY_FIELD")

if [ -f "$DENSE_OUT/metrics.csv" ] || [ -n "$(find "$DENSE_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
  echo "[2/$TOTAL_STEPS] Dense... (skip: output exists)"
else
  echo "[2/$TOTAL_STEPS] Dense..."
  STEP_DENSE_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_dense.py" "${DENSE_ARGS[@]}"
  STEP_DENSE_END=$(date +%s)
  echo "[timing] Dense step: $((STEP_DENSE_END-STEP_DENSE_START))s"
fi

# ----- Hybrid -----
HYBRID_ARGS=(
  --bm25_runs_dir "$BM25_OUT/runs"
  --bm25_topk "$BM25_TOP_K"
  --dense_root "$DENSE_OUT"
  --train-json "$TRAIN_JSON"
  --out_dir "$HYBRID_OUT"
  --k_max_eval "$HYBRID_K_MAX_EVAL"
  --cap "$HYBRID_CAP"
  --ks "$RECALL_KS"
)
[ -n "${TEST_BATCH_JSONS:-}" ] && HYBRID_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
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

if [ -f "$HYBRID_OUT/ranked_test_avg.csv" ] || [ -f "$HYBRID_OUT/results_all.csv" ]; then
  echo "[3/$TOTAL_STEPS] Hybrid... (skip: output exists)"
else
  echo "[3/$TOTAL_STEPS] Hybrid..."
  STEP_HYBRID_START=$(date +%s)
  python "$SCRIPT_DIR/retrieval/eval_hybrid.py" "${HYBRID_ARGS[@]}"
  STEP_HYBRID_END=$(date +%s)
  echo "[timing] Hybrid step: $((STEP_HYBRID_END-STEP_HYBRID_START))s"
fi

# ----- Reranker (optional: only if DOCS_JSONL set and not --no-rerank) -----
# Figures are named hybrid_reranker_recall_map10_{label}.png (one per dataset label)
if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "1" ]; then
  STEP_RERANK_START=$(date +%s)
  RERANK_RESULTS_EXIST=0
  [ -f "$RERANK_OUT/metrics.csv" ] && RERANK_RESULTS_EXIST=1
  [ "$RERANK_RESULTS_EXIST" = "0" ] && [ -n "$(find "$RERANK_OUT" -maxdepth 2 -name '*.tsv' 2>/dev/null | head -1)" ] && RERANK_RESULTS_EXIST=1

  RERANK_FIGS_EXIST=0
  [ -n "$(find "$RERANK_OUT/figures" -maxdepth 1 -name 'hybrid_reranker_recall_map10_*.png' 2>/dev/null | head -1)" ] && RERANK_FIGS_EXIST=1

  # Check whether RRF fusion (Hybrid + Rerank -> rerank_hybrid) already exists
  RRF_EXIST=0
  if [ -f "$RERANK_HYBRID_OUT/metrics.csv" ] || [ -n "$(find "$RERANK_HYBRID_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
    RRF_EXIST=1
  fi

  # ----- Step 4: Reranker (always report and run/skip before step 5) -----
  if [ "$RERANK_RESULTS_EXIST" = "1" ]; then
    if [ "$RERANK_FIGS_EXIST" = "1" ]; then
      echo "[4/$TOTAL_STEPS] Reranker... (skip: output exists)"
    else
      echo "[4/$TOTAL_STEPS] Reranker... (generating eval plots from existing results)"
      PLOT_ARGS=(--output-dir "$RERANK_OUT" --runs-dir "$HYBRID_OUT/runs")
      [ -n "${TRAIN_JSON:-}" ] && PLOT_ARGS+=(--train-json "$TRAIN_JSON")
      [ -n "${TEST_BATCH_JSONS:-}" ] && PLOT_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
      python "$SCRIPT_DIR/rerank/plot_rerank_eval.py" "${PLOT_ARGS[@]}"
    fi
  else
    echo "[4/$TOTAL_STEPS] Reranker..."
    mkdir -p "$RERANK_OUT"
    # If DOCS_JSONL has fewer than 2000 docs, cap candidate-limit to that count
    if [ -f "$DOCS_JSONL" ]; then
      DOCS_JSONL_LINES=$(wc -l < "$DOCS_JSONL" 2>/dev/null || echo 0)
      if [ "$DOCS_JSONL_LINES" -gt 0 ] && [ "$DOCS_JSONL_LINES" -lt 2000 ] && [ "$DOCS_JSONL_LINES" -lt "$RERANK_EFFECTIVE" ]; then
        RERANK_EFFECTIVE=$DOCS_JSONL_LINES
        echo "Reranker candidate-limit capped to doc count in DOCS_JSONL ($DOCS_JSONL_LINES)"
      fi
    fi
    RERANK_ARGS=(
      --runs-dir "$HYBRID_OUT/runs"
      --output-dir "$RERANK_OUT"
      --docs-jsonl "$DOCS_JSONL"
      --candidate-limit "$RERANK_EFFECTIVE"
      --ks-recall "${RERANK_KS_RECALL:-$RECALL_KS}"
    )
    [ -n "${TRAIN_JSON:-}" ] && RERANK_ARGS+=(--train-json "$TRAIN_JSON")
    [ -n "${TEST_BATCH_JSONS:-}" ] && RERANK_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
    [ -n "${RERANK_MODEL:-}" ] && RERANK_ARGS+=(--model "$RERANK_MODEL")
    [ -n "${RERANK_MODEL_DEVICE:-}" ] && RERANK_ARGS+=(--model-device "$RERANK_MODEL_DEVICE")
    [ -n "${RERANK_MODEL_BATCH:-}" ] && RERANK_ARGS+=(--model-batch "$RERANK_MODEL_BATCH")
    [ -n "${RERANK_MODEL_MAX_LENGTH:-}" ] && RERANK_ARGS+=(--model-max-length "$RERANK_MODEL_MAX_LENGTH")
    [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && RERANK_ARGS+=(--disable-metrics)
    [ "${RERANK_USE_MULTI_GPU:-0}" = "1" ] && RERANK_ARGS+=(--use-multi-gpu)
    [ -n "${RERANK_NUM_GPUS:-}" ] && RERANK_ARGS+=(--num-gpus "$RERANK_NUM_GPUS")
    [ -n "${RERANK_QUERY_FIELD:-}" ] && RERANK_ARGS+=(--query-field "$RERANK_QUERY_FIELD")
    python "$SCRIPT_DIR/rerank/rerank_stage2.py" "${RERANK_ARGS[@]}"
  fi
  STEP_RERANK_END=$(date +%s)
  echo "[timing] Reranker step: $((STEP_RERANK_END-STEP_RERANK_START))s"

  # ----- Step 5: RRF fusion (Hybrid + Rerank -> rerank_hybrid); only when RUN_RRF_FUSION=1 -----
  if [ "$RUN_RRF_FUSION" = "1" ] && { [ "$TOTAL_STEPS" = "5" ] || [ "$TOTAL_STEPS" = "7" ]; }; then
    STEP_RRF_START=$(date +%s)
    # Re-check RRF output in case we just ran reranker in step 4
    RRF_EXIST=0
    if [ -f "$RERANK_HYBRID_OUT/metrics.csv" ] || [ -n "$(find "$RERANK_HYBRID_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      RRF_EXIST=1
    fi
    if [ "$RRF_EXIST" = "1" ]; then
      echo "[5/$TOTAL_STEPS] RRF fusion (Hybrid + Rerank, top-10)... (skip: output exists)"
    else
      echo "[5/$TOTAL_STEPS] RRF fusion (Hybrid + Rerank, top-10)..."
      # Resolve per-list pool sizes; snippet-rrf (only) defaults to 200; --run-both-routes keeps 50 for baseline
      _RRF_DEFAULT=50
      [ "${SNIPPET_RRF:-0}" = "1" ] && [ "${RUN_BOTH_ROUTES:-0}" != "1" ] && _RRF_DEFAULT=200
      _RRF_POOL_RERANK="${RRF_POOL_TOP_RERANK:-${RRF_POOL_TOP:-$_RRF_DEFAULT}}"
      _RRF_POOL_HYBRID="${RRF_POOL_TOP_HYBRID:-${RRF_POOL_TOP:-$_RRF_DEFAULT}}"

      # Validate: pool sizes must not exceed upstream output sizes
      if [ "$_RRF_POOL_RERANK" -gt "$RERANK_EFFECTIVE" ]; then
        echo "WARNING: RRF_POOL_TOP_RERANK ($_RRF_POOL_RERANK) > RERANK_CANDIDATE_LIMIT ($RERANK_EFFECTIVE); reranker output will be silently truncated." >&2
      fi
      if [ "$_RRF_POOL_HYBRID" -gt "$HYBRID_CAP" ]; then
        echo "WARNING: RRF_POOL_TOP_HYBRID ($_RRF_POOL_HYBRID) > HYBRID_CAP ($HYBRID_CAP); hybrid output will be silently truncated." >&2
      fi

      RRF_ARGS=(
        --hybrid-runs-dir "$HYBRID_OUT/runs"
        --rerank-runs-dir "$RERANK_OUT/runs"
        --output-dir "$RERANK_HYBRID_OUT"
        --pool-top-rerank "$_RRF_POOL_RERANK"
        --pool-top-hybrid "$_RRF_POOL_HYBRID"
      )
      [ -n "${RRF_K_RRF:-}" ] && RRF_ARGS+=(--k-rrf "$RRF_K_RRF")
      [ -n "${RRF_W_BGE:-}" ] && RRF_ARGS+=(--w-bge "$RRF_W_BGE")
      [ -n "${RRF_W_HYBRID:-}" ] && RRF_ARGS+=(--w-hybrid "$RRF_W_HYBRID")
      [ -n "${TRAIN_JSON:-}" ] && RRF_ARGS+=(--train-json "$TRAIN_JSON")
      [ -n "${TEST_BATCH_JSONS:-}" ] && RRF_ARGS+=(--test-batch-jsons $TEST_BATCH_JSONS)
      [ -n "${RERANK_KS_RECALL:-}" ] && RRF_ARGS+=(--ks-recall "$RERANK_KS_RECALL")
      python "$SCRIPT_DIR/rerank/rerank_rrf_hybrid.py" "${RRF_ARGS[@]}"
    fi
    STEP_RRF_END=$(date +%s)
    echo "[timing] Hybrid+Rerank RRF fusion step: $((STEP_RRF_END-STEP_RRF_START))s"
  fi

  # ----- Step 5b: RRF pool=200 for snippet route (only when --run-both-routes) -----
  if [ "${RUN_BOTH_ROUTES:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
    RRF_200_EXIST=0
    [ -f "$RERANK_HYBRID_200_OUT/metrics.csv" ] || [ -n "$(find "$RERANK_HYBRID_200_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && RRF_200_EXIST=1
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
        --rerank-runs-dir "$RERANK_OUT/runs" \
        --output-dir "$RERANK_HYBRID_200_OUT" \
        --pool-top-rerank "$_RRF_POOL_RERANK_200" \
        --pool-top-hybrid "$_RRF_POOL_HYBRID_200" \
        ${RRF_K_RRF:+--k-rrf "$RRF_K_RRF"} \
        ${RRF_W_BGE:+--w-bge "$RRF_W_BGE"} \
        ${RRF_W_HYBRID:+--w-hybrid "$RRF_W_HYBRID"} \
        ${TRAIN_JSON:+--train-json "$TRAIN_JSON"} \
        ${TEST_BATCH_JSONS:+--test-batch-jsons $TEST_BATCH_JSONS} \
        ${RERANK_KS_RECALL:+--ks-recall "$RERANK_KS_RECALL"}
    fi
  fi

  # ----- Step 6: Snippet extraction + CE reranking (only when --snippet-rrf) -----
  if [ "${SNIPPET_RRF:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
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
      _SNIPPET_RRF_INPUT="$RERANK_HYBRID_OUT/runs"
      [ "${RUN_BOTH_ROUTES:-0}" = "1" ] && _SNIPPET_RRF_INPUT="$RERANK_HYBRID_200_OUT/runs"
      SNIPPET_ARGS=(
        --runs-dir "$_SNIPPET_RRF_INPUT"
        --docs-jsonl "$DOCS_JSONL"
        --output-dir "$SNIPPET_RERANK_OUT"
        --n-docs "${SNIPPET_N_DOCS:-100}"
        --window-size "${SNIPPET_WINDOW_SIZE:-3}"
        --window-stride "${SNIPPET_WINDOW_STRIDE:-1}"
        --top-w "${SNIPPET_TOP_W:-8}"
        --dense-model "${SNIPPET_DENSE_MODEL:-BAAI/bge-base-en-v1.5}"
        --ce-model "${SNIPPET_CE_MODEL:-BAAI/bge-reranker-v2-m3}"
      )
      [ -n "${TRAIN_JSON:-}" ] && SNIPPET_ARGS+=(--train-json "$TRAIN_JSON")
      [ -n "${TEST_BATCH_JSONS:-}" ] && SNIPPET_ARGS+=(--test-batch-jsons $TEST_BATCH_JSONS)
      [ -n "${SNIPPET_DENSE_DEVICE:-}" ] && SNIPPET_ARGS+=(--dense-device "$SNIPPET_DENSE_DEVICE")
      [ -n "${SNIPPET_CE_DEVICE:-}" ] && SNIPPET_ARGS+=(--ce-device "$SNIPPET_CE_DEVICE")
      [ -n "${SNIPPET_CE_BATCH:-}" ] && SNIPPET_ARGS+=(--ce-batch "$SNIPPET_CE_BATCH")
      [ -n "${SNIPPET_CE_MAX_LENGTH:-}" ] && SNIPPET_ARGS+=(--ce-max-length "$SNIPPET_CE_MAX_LENGTH")
      # Snippet CE multi-GPU defaults to reranker settings when not set
      [ "${SNIPPET_CE_USE_MULTI_GPU:-${RERANK_USE_MULTI_GPU:-0}}" = "1" ] && SNIPPET_ARGS+=(--ce-use-multi-gpu)
      [ -n "${SNIPPET_CE_NUM_GPUS:-${RERANK_NUM_GPUS:-}}" ] && SNIPPET_ARGS+=(--ce-num-gpus "${SNIPPET_CE_NUM_GPUS:-$RERANK_NUM_GPUS}")
      [ -n "${RERANK_QUERY_FIELD:-}" ] && SNIPPET_ARGS+=(--query-field "$RERANK_QUERY_FIELD")
      [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && SNIPPET_ARGS+=(--disable-metrics)
      python "$SCRIPT_DIR/evidence/snippet_rerank.py" "${SNIPPET_ARGS[@]}"
    fi
    STEP_SNIPPET_END=$(date +%s)
    echo "[timing] Snippet extraction + CE rerank step: $((STEP_SNIPPET_END-STEP_SNIPPET_START))s"
  fi

  # ----- Step 7: Final RRF fusion (rerank_hybrid 0.8 + snippet_rerank 0.2 -> snippet_rrf); only when --snippet-rrf -----
  if [ "${SNIPPET_RRF:-0}" = "1" ] && [ "$TOTAL_STEPS" = "7" ]; then
    STEP_FINAL_RRF_START=$(date +%s)
    SNIPPET_RRF_EXIST=0
    if [ -f "$SNIPPET_RRF_OUT/metrics.csv" ] || [ -n "$(find "$SNIPPET_RRF_OUT/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ]; then
      SNIPPET_RRF_EXIST=1
    fi
    if [ "$SNIPPET_RRF_EXIST" = "1" ]; then
      echo "[7/$TOTAL_STEPS] Final RRF (docs 0.8 + snippet 0.2)... (skip: output exists)"
    else
      echo "[7/$TOTAL_STEPS] Final RRF (docs 0.8 + snippet 0.2)..."
      # Pool size default: SNIPPET_FINAL_POOL, falling back to SNIPPET_N_DOCS (and then 100)
      _SNIP_POOL="${SNIPPET_FINAL_POOL:-${SNIPPET_N_DOCS:-100}}"
      python "$SCRIPT_DIR/rerank/rerank_rrf_hybrid.py" \
        --hybrid-runs-dir "$RERANK_HYBRID_OUT/runs" \
        --rerank-runs-dir "$SNIPPET_RERANK_OUT/runs" \
        --output-dir "$SNIPPET_RRF_OUT" \
        --pool-top-rerank "$_SNIP_POOL" \
        --pool-top-hybrid "$_SNIP_POOL" \
        --k-rrf "${SNIPPET_RRF_K:-60}" \
        --w-bge "${SNIPPET_RRF_W_SNIPPET:-0.2}" \
        --w-hybrid "${SNIPPET_RRF_W_DOCS:-0.8}" \
        ${TRAIN_JSON:+--train-json "$TRAIN_JSON"} \
        ${TEST_BATCH_JSONS:+--test-batch-jsons $TEST_BATCH_JSONS} \
        ${RERANK_KS_RECALL:+--ks-recall "$RERANK_KS_RECALL"}
    fi
    STEP_FINAL_RRF_END=$(date +%s)
    echo "[timing] Final RRF (snippet_rrf) step: $((STEP_FINAL_RRF_END-STEP_FINAL_RRF_START))s"
  fi

  # ----- Compare (Rerank vs Hybrid+Rerank): recall and MAP curves -----
  COMPARE_KS="${COMPARE_KS:-10,20,30,50,100,200,300}"
  if [ "${HAVE_GROUND_TRUTH:-1}" != "0" ] && [ -d "$RERANK_OUT/runs" ] && [ -d "$RERANK_HYBRID_OUT/runs" ]; then
    COMPARE_FIGS_EXIST=0
    [ -n "$(find "$RERANK_HYBRID_OUT/figures" -maxdepth 1 -name 'compare_*.png' 2>/dev/null | head -1)" ] && COMPARE_FIGS_EXIST=1
    if [ "$COMPARE_FIGS_EXIST" = "1" ]; then
      echo "[Compare] Rerank vs Hybrid+Rerank... (skip: figures exist)"
    else
      echo "[Compare] Rerank vs Hybrid+Rerank (recall & MAP @ $COMPARE_KS)..."
      STEP_COMPARE_START=$(date +%s)
      COMPARE_ARGS=(
        --dirs "$RERANK_OUT" "$RERANK_HYBRID_OUT"
        --labels "Rerank" "Hybrid+Rerank"
        --plot both
        --map-ks "$COMPARE_KS"
        --ks-recall "$COMPARE_KS"
        --recall-k-max 300
        --output-dir "$RERANK_HYBRID_OUT"
        --force-from-runs
        --plots-by-split
      )
      [ -n "${TRAIN_JSON:-}" ] && COMPARE_ARGS+=(--train-json "$TRAIN_JSON")
      [ -n "${TEST_BATCH_JSONS:-}" ] && COMPARE_ARGS+=(--test-batch-jsons $TEST_BATCH_JSONS)
      python "$SCRIPT_DIR/compare_result_dirs.py" "${COMPARE_ARGS[@]}"
      STEP_COMPARE_END=$(date +%s)
      echo "[timing] Compare step: $((STEP_COMPARE_END-STEP_COMPARE_START))s"
    fi
  fi

  # ----- Evidence (post-rerank JSON + contexts): build all splits first -----
  # Always use separate dirs: evidence_baseline/generation_baseline (from rerank_hybrid), evidence_snippet/generation_snippet (from snippet_rrf).
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
    if [ "${RUN_BOTH_ROUTES:-0}" = "1" ]; then
      _ROUTES_LIST="baseline snippet"
    elif [ "${SNIPPET_RRF:-0}" = "1" ]; then
      _ROUTES_LIST="snippet"
    else
      _ROUTES_LIST="baseline"
    fi
    for _route in $_ROUTES_LIST; do
      if [ "$_route" = "baseline" ]; then
        _EVIDENCE_RUNS_DIR="$RERANK_HYBRID_OUT/runs"
        _EVIDENCE_POST_DIR="$RERANK_HYBRID_OUT"
        _EVIDENCE_SUBDIR="evidence_baseline"
        _GEN_SUBDIR="generation_baseline"
        _USE_SNIPPET_CTX=0
      else
        _EVIDENCE_RUNS_DIR="$SNIPPET_RRF_OUT/runs"
        _EVIDENCE_POST_DIR="$SNIPPET_RRF_OUT"
        _EVIDENCE_SUBDIR="evidence_snippet"
        _GEN_SUBDIR="generation_snippet"
        _USE_SNIPPET_CTX=1
      fi
      [ ! -d "$_EVIDENCE_RUNS_DIR" ] && continue
      mkdir -p "$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR"
      echo "[Evidence] Route: $_route -> $_EVIDENCE_SUBDIR, $_GEN_SUBDIR"
    for _tsv in "$_EVIDENCE_RUNS_DIR/"*.tsv; do
      [ -f "$_tsv" ] || continue
      _stem=$(basename "$_tsv" .tsv)
      _split="${_stem#best_rrf_}"
      _split="${_split%%_top*}"
      [ -n "$_split" ] || continue

      # Resolve query JSON for this split: match split to basename (no .json) of TRAIN_JSON or any TEST_BATCH_JSONS
      _query_json=""
      if [ -f "${TRAIN_JSON:-}" ] && [ "$(basename "$TRAIN_JSON" .json)" = "$_split" ]; then
        _query_json="$TRAIN_JSON"
      fi
      if [ -z "$_query_json" ]; then
        for _p in $TEST_BATCH_JSONS; do
          [ -f "$_p" ] || continue
          [ "$(basename "$_p" .json)" = "$_split" ] || continue
          _query_json="$_p"
          break
        done
      fi
      if [ -z "$_query_json" ]; then
        echo "[Evidence] Skip $_split: no matching TRAIN_JSON or TEST_BATCH_JSONS (basename without .json)"
        continue
      fi

      _post_json="$_EVIDENCE_POST_DIR/post_rerank_${_split}.json"
      if [ ! -f "$_post_json" ]; then
        echo "[Evidence] Post-rerank JSON ($_split)..."
        python "$SCRIPT_DIR/evidence/post_rerank_json.py" \
          --run-path "$_tsv" \
          --query-json "$_query_json" \
          --output-path "$_post_json" \
          --top-k 10
      else
        echo "[Evidence] Post-rerank JSON ($_split)... (skip: output exists)"
      fi

      _ctx_json="$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR/${_split}_contexts.json"
      if [ ! -f "$_ctx_json" ]; then
        if [ "$_USE_SNIPPET_CTX" = "1" ]; then
          echo "[Evidence] Contexts from snippets ($_split)..."
          python "$SCRIPT_DIR/evidence/build_contexts_from_snippets.py" \
            --post-rerank-json "$_post_json" \
            --snippet-windows-dir "$SNIPPET_RERANK_OUT/windows" \
            --split-name "$_stem" \
            --corpus-path "$DOCS_JSONL" \
            --output-path "$_ctx_json" \
            --window-size "${SNIPPET_WINDOW_SIZE:-3}" \
            --top-windows "${SNIPPET_CONTEXT_TOP_WINDOWS:-2}"
        else
          echo "[Evidence] Contexts from documents ($_split)..."
          python "$SCRIPT_DIR/evidence/build_contexts_from_documents.py" \
            --post-rerank-json "$_post_json" \
            --corpus-path "$DOCS_JSONL" \
            --output-path "$_ctx_json"
        fi
      else
        echo "[Evidence] Contexts ($_split)... (skip: output exists)"
      fi
    done

    # ----- Generation (LLM answers from contexts JSON): run after all evidence is built -----
    mkdir -p "$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR"
    for _tsv in "$_EVIDENCE_RUNS_DIR/"*.tsv; do
      [ -f "$_tsv" ] || continue
      _stem=$(basename "$_tsv" .tsv)
      _split="${_stem#best_rrf_}"
      _split="${_split%%_top*}"
      [ -n "$_split" ] || continue

      _ctx_json="$WORKFLOW_OUTPUT_DIR/$_EVIDENCE_SUBDIR/${_split}_contexts.json"
      _gen_json="$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR/${_split}_answers.json"
      if [ -f "$_gen_json" ]; then
        echo "[Generation] $_split... (skip: output exists)"
      elif [ ! -f "$_ctx_json" ]; then
        echo "[Generation] Skip $_split: evidence not found ($_ctx_json)"
      else
        echo "[Generation] $_split..."
        GENERATION_ARGS=(
          --input-path "$_ctx_json"
          --output-dir "$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR"
        )
        [ -n "${GENERATION_CONCURRENCY:-}" ] && GENERATION_ARGS+=(--concurrency "$GENERATION_CONCURRENCY")
        [ -n "${GENERATION_MAX_CONTEXTS:-}" ] && GENERATION_ARGS+=(--max-contexts "$GENERATION_MAX_CONTEXTS")
        [ -n "${GENERATION_MAX_CHARS_PER_CONTEXT:-}" ] && GENERATION_ARGS+=(--max-chars-per-context "$GENERATION_MAX_CHARS_PER_CONTEXT")
        [ -n "${GENERATION_SLEEP:-}" ] && GENERATION_ARGS+=(--sleep "$GENERATION_SLEEP")
        python "$SCRIPT_DIR/generation/generate_answers.py" "${GENERATION_ARGS[@]}"
      fi
    done

    # ----- Rescue: one pass over all generation outputs (in-place, save failed prompts to same folder) -----
    for _tsv in "$_EVIDENCE_RUNS_DIR/"*.tsv; do
      [ -f "$_tsv" ] || continue
      _stem=$(basename "$_tsv" .tsv)
      _split="${_stem#best_rrf_}"
      _split="${_split%%_top*}"
      [ -n "$_split" ] || continue
      _gen_json="$WORKFLOW_OUTPUT_DIR/$_GEN_SUBDIR/${_split}_answers.json"
      if [ -f "$_gen_json" ]; then
        echo "[Rescue] $_split..."
        RESCUE_ARGS=(--input "$_gen_json")
        [ -n "${GENERATION_RESCUE_TIMEOUT:-}" ] && RESCUE_ARGS+=(--timeout "$GENERATION_RESCUE_TIMEOUT")
        [ -n "${GENERATION_RESCUE_RETRY_SLEEP:-}" ] && RESCUE_ARGS+=(--retry-sleep "$GENERATION_RESCUE_RETRY_SLEEP")
        [ -n "${GENERATION_MAX_CHARS_PER_CONTEXT:-}" ] && RESCUE_ARGS+=(--max-chars-per-context "$GENERATION_MAX_CHARS_PER_CONTEXT")
        python "$SCRIPT_DIR/generation/rescue_failed_generation.py" "${RESCUE_ARGS[@]}"
      fi
    done
    done
    STEP_EVIDENCE_GEN_END=$(date +%s)
    echo "[timing] Evidence + Generation + Rescue step: $((STEP_EVIDENCE_GEN_END-STEP_EVIDENCE_GEN_START))s"
  fi

  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (bm25/, dense/, hybrid/, rerank/, rerank_hybrid/)"
  [ "${SNIPPET_RRF:-0}" = "1" ] && echo "  Snippet RRF: snippet_rerank/, snippet_rrf/"
  [ "$_DOCS_JSONL_OK" = "1" ] && echo "  Evidence/Generation: evidence_baseline/, generation_baseline/ (baseline); evidence_snippet/, generation_snippet/ (when --snippet-rrf)"
else
  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (bm25/, dense/, hybrid/)"
  if [ -n "${DOCS_JSONL:-}" ] && [ "$RUN_RERANK" = "0" ]; then
    echo "Reranker skipped (--no-rerank). Re-run without --no-rerank to run reranker."
  elif [ -z "${DOCS_JSONL:-}" ]; then
    echo "Optional: set DOCS_JSONL and re-run to add reranker step."
  fi
fi
