#!/usr/bin/env bash
#
# Listwise reranking (RankZephyr) – runs in its own container after the main pipeline.
#
# Reads snippet/snippet_doc_fusion/runs and snippet/snippet_rerank/windows produced by
# run_retrieval_rerank_pipeline.sh (with --snippet-rrf or --run-both-routes)
# and produces listwise_rerank/{single_window,sliding_window}/runs/*.tsv.
#
# Usage:
#   ./run_listwise_rerank.sh --config <path/to/config.env>
#   ./run_listwise_rerank.sh -c <path/to/config.env>
#
# Required config variables: WORKFLOW_OUTPUT_DIR
# Optional:  TRAIN_JSON, TEST_BATCH_JSONS, LISTWISE_* (see workflow_config_full.env)
#
set -e

# ---------------------------------------------------------------------------
# Resolve SCRIPT_DIR (handles vendored copies via SHARED_SCRIPTS_DIR env)
# ---------------------------------------------------------------------------
if [ -n "${SHARED_SCRIPTS_DIR:-}" ]; then
  SCRIPT_DIR="$SHARED_SCRIPTS_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
CONFIG_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--config)
      [ -z "${2:-}" ] && { echo "Error: --config requires a path." >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --config <config.env>"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1
      ;;
  esac
done

# Source config
if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  set -a; source "$CONFIG_FILE"; set +a
  echo "[listwise] Loaded config from $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------
if [ -z "${WORKFLOW_OUTPUT_DIR:-}" ]; then
  echo "Error: WORKFLOW_OUTPUT_DIR is required (set in config or env)." >&2
  exit 1
fi

SNIPPET_DOC_FUSION_RUNS="${WORKFLOW_OUTPUT_DIR}/snippet/snippet_doc_fusion/runs"
SNIPPET_WINDOWS="${WORKFLOW_OUTPUT_DIR}/snippet/snippet_rerank/windows"

if [ ! -d "$SNIPPET_DOC_FUSION_RUNS" ]; then
  echo "Error: snippet/snippet_doc_fusion/runs not found at $SNIPPET_DOC_FUSION_RUNS" >&2
  echo "  The main pipeline must run with --snippet-rrf or RUN_SNIPPET_RRF=1 first." >&2
  exit 1
fi

if [ ! -d "$SNIPPET_WINDOWS" ]; then
  echo "Error: snippet/snippet_rerank/windows not found at $SNIPPET_WINDOWS" >&2
  echo "  The main pipeline must run with --snippet-rrf or RUN_SNIPPET_RRF=1 first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Defaults from config (LISTWISE_* env vars) with fallbacks
# ---------------------------------------------------------------------------
LISTWISE_SINGLE_K="${LISTWISE_SINGLE_K:-15}"
LISTWISE_RUN_SLIDING="${LISTWISE_RUN_SLIDING:-0}"  # set to 1 to also run sliding-window
LISTWISE_POOL="${LISTWISE_POOL:-50}"
LISTWISE_WINDOW="${LISTWISE_WINDOW:-15}"
LISTWISE_STRIDE="${LISTWISE_STRIDE:-5}"
LISTWISE_MODEL="${LISTWISE_MODEL:-castorini/rank_zephyr_7b_v1_full}"
LISTWISE_CONTEXT_SIZE="${LISTWISE_CONTEXT_SIZE:-4096}"
LISTWISE_MAX_SNIPPET_TOKENS="${LISTWISE_MAX_SNIPPET_TOKENS:-250}"
LISTWISE_OUTPUT_DIR="${LISTWISE_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/listwise_rerank}"
LISTWISE_DISABLE_METRICS="${LISTWISE_DISABLE_METRICS:-${RERANK_DISABLE_METRICS:-0}}"
LISTWISE_QUERY_FIELD="${LISTWISE_QUERY_FIELD:-body}"

# Fusion parameters
LISTWISE_FUSION_W_SNIPPET="${LISTWISE_FUSION_W_SNIPPET:-0.4}"
LISTWISE_FUSION_W_LISTWISE="${LISTWISE_FUSION_W_LISTWISE:-0.6}"
LISTWISE_FUSION_K_RRF="${LISTWISE_FUSION_K_RRF:-20}"
LISTWISE_FUSION_POOL_TOP="${LISTWISE_FUSION_POOL_TOP:-15}"
LISTWISE_FUSE_SLIDING="${LISTWISE_FUSE_SLIDING:-0}"

echo "[listwise] Configuration:"
echo "  WORKFLOW_OUTPUT_DIR:  $WORKFLOW_OUTPUT_DIR"
echo "  Input runs:           $SNIPPET_DOC_FUSION_RUNS"
echo "  Input windows:        $SNIPPET_WINDOWS"
echo "  Output:               $LISTWISE_OUTPUT_DIR"
echo "  Model:                $LISTWISE_MODEL"
echo "  Context size:         $LISTWISE_CONTEXT_SIZE"
echo "  Single-window k:      $LISTWISE_SINGLE_K"
echo "  Run sliding window:   $LISTWISE_RUN_SLIDING"
echo "  Sliding pool:         $LISTWISE_POOL"
echo "  Sliding window:       $LISTWISE_WINDOW"
echo "  Sliding stride:       $LISTWISE_STRIDE"
echo "  Max snippet tokens:   $LISTWISE_MAX_SNIPPET_TOKENS"
echo "  Disable metrics:      $LISTWISE_DISABLE_METRICS"
echo "  Fusion w_snippet:     $LISTWISE_FUSION_W_SNIPPET"
echo "  Fusion w_listwise:    $LISTWISE_FUSION_W_LISTWISE"
echo "  Fusion k_rrf:         $LISTWISE_FUSION_K_RRF"
echo "  Fusion pool_top:      $LISTWISE_FUSION_POOL_TOP"
echo "  Fuse sliding:         $LISTWISE_FUSE_SLIDING"
echo "  Query field:          $LISTWISE_QUERY_FIELD"

# ---------------------------------------------------------------------------
# Save config.json for reproducibility
# ---------------------------------------------------------------------------
mkdir -p "$LISTWISE_OUTPUT_DIR"
cat > "$LISTWISE_OUTPUT_DIR/config.json" <<CONFIGEOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "config_file": "${CONFIG_FILE:-null}",
  "workflow_output_dir": "$WORKFLOW_OUTPUT_DIR",
  "input_runs": "$SNIPPET_DOC_FUSION_RUNS",
  "input_windows": "$SNIPPET_WINDOWS",
  "output_dir": "$LISTWISE_OUTPUT_DIR",
  "reranking": {
    "model": "$LISTWISE_MODEL",
    "context_size": $LISTWISE_CONTEXT_SIZE,
    "max_snippet_tokens": $LISTWISE_MAX_SNIPPET_TOKENS,
    "single_k": $LISTWISE_SINGLE_K,
    "run_sliding": $LISTWISE_RUN_SLIDING,
    "sliding_pool": $LISTWISE_POOL,
    "sliding_window": $LISTWISE_WINDOW,
    "sliding_stride": $LISTWISE_STRIDE,
    "disable_metrics": $LISTWISE_DISABLE_METRICS,
    "query_field": "$LISTWISE_QUERY_FIELD"
  },
  "fusion": {
    "w_snippet_rrf": $LISTWISE_FUSION_W_SNIPPET,
    "w_listwise": $LISTWISE_FUSION_W_LISTWISE,
    "k_rrf": $LISTWISE_FUSION_K_RRF,
    "pool_top_single": $LISTWISE_FUSION_POOL_TOP,
    "pool_top_sliding": $LISTWISE_POOL,
    "fuse_sliding": $LISTWISE_FUSE_SLIDING
  },
  "query_sources": {
    "train_json": "${TRAIN_JSON:-null}",
    "test_batch_jsons": "${TEST_BATCH_JSONS:-null}"
  }
}
CONFIGEOF
echo "[listwise] Saved config -> $LISTWISE_OUTPUT_DIR/config.json"

# ---------------------------------------------------------------------------
# Check if reranking output already exists
# ---------------------------------------------------------------------------
_SINGLE_RUNS_DIR="${LISTWISE_OUTPUT_DIR}/single_window/runs"
_SLIDING_RUNS_DIR="${LISTWISE_OUTPUT_DIR}/sliding_window/runs"

_HAS_SINGLE=0
_HAS_SLIDING=0
[ -d "$_SINGLE_RUNS_DIR" ] && [ -n "$(find "$_SINGLE_RUNS_DIR" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_SINGLE=1
[ -d "$_SLIDING_RUNS_DIR" ] && [ -n "$(find "$_SLIDING_RUNS_DIR" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_SLIDING=1

STEP_START=$(date +%s)

_SKIP_RERANK=0
if [ "$LISTWISE_RUN_SLIDING" != "1" ]; then
  [ "$_HAS_SINGLE" = "1" ] && _SKIP_RERANK=1
else
  [ "$_HAS_SINGLE" = "1" ] && [ "$_HAS_SLIDING" = "1" ] && _SKIP_RERANK=1
fi

if [ "$_SKIP_RERANK" = "1" ]; then
  echo "[listwise] Reranking output already exists – skipping reranking step."
  echo "  Delete single_window/runs/ (and sliding_window/runs/) to re-run."
else
  # ---------------------------------------------------------------------------
  # Build arguments & run reranking
  # ---------------------------------------------------------------------------
  LISTWISE_ARGS=(
    --runs-dir "$SNIPPET_DOC_FUSION_RUNS"
    --windows-dir "$SNIPPET_WINDOWS"
    --output-dir "$LISTWISE_OUTPUT_DIR"
    --single-k "$LISTWISE_SINGLE_K"
    --pool "$LISTWISE_POOL"
    --window-size "$LISTWISE_WINDOW"
    --stride "$LISTWISE_STRIDE"
    --model "$LISTWISE_MODEL"
    --context-size "$LISTWISE_CONTEXT_SIZE"
    --max-snippet-tokens "$LISTWISE_MAX_SNIPPET_TOKENS"
    --query-field "$LISTWISE_QUERY_FIELD"
  )

  [ -n "${TRAIN_JSON:-}" ] && [ -f "$TRAIN_JSON" ] && LISTWISE_ARGS+=(--train-json "$TRAIN_JSON")
  if [ -n "${TEST_BATCH_JSONS:-}" ]; then
    _TEST_PATHS=()
    for _p in $TEST_BATCH_JSONS; do
      [ -f "$_p" ] && _TEST_PATHS+=("$_p")
    done
    if [ ${#_TEST_PATHS[@]} -gt 0 ]; then
      LISTWISE_ARGS+=(--test-batch-jsons "${_TEST_PATHS[@]}")
    fi
  fi
  [ "$LISTWISE_DISABLE_METRICS" = "1" ] && LISTWISE_ARGS+=(--disable-metrics)
  [ "$LISTWISE_RUN_SLIDING" != "1" ] && LISTWISE_ARGS+=(--no-sliding)

  echo "[listwise] Running listwise reranking..."
  python3 "$SCRIPT_DIR/rerank/listwise_rerank.py" "${LISTWISE_ARGS[@]}"

  RERANK_END=$(date +%s)
  echo "[listwise] Reranking completed in $((RERANK_END - STEP_START))s"
fi

# ---------------------------------------------------------------------------
# RRF Fusion: listwise single_window + snippet_doc_fusion
# ---------------------------------------------------------------------------
_FUSED_DIR="${LISTWISE_OUTPUT_DIR}/listwise_fused"
_FUSED_SLIDING_DIR="${LISTWISE_OUTPUT_DIR}/listwise_fused_sliding"

_build_fusion_args() {
  local _lw_runs="$1"
  local _out="$2"
  local _pool_top="$3"

  local _FUSION_ARGS=(
    --listwise-runs-dir "$_lw_runs"
    --snippet-doc-fusion-runs-dir "$SNIPPET_DOC_FUSION_RUNS"
    --output-dir "$_out"
    --pool-top "$_pool_top"
    --k-rrf "$LISTWISE_FUSION_K_RRF"
    --w-snippet-rrf "$LISTWISE_FUSION_W_SNIPPET"
    --w-listwise "$LISTWISE_FUSION_W_LISTWISE"
  )
  [ -n "${TRAIN_JSON:-}" ] && [ -f "$TRAIN_JSON" ] && _FUSION_ARGS+=(--train-json "$TRAIN_JSON")
  if [ -n "${TEST_BATCH_JSONS:-}" ]; then
    local _TEST_PATHS=()
    for _p in $TEST_BATCH_JSONS; do
      [ -f "$_p" ] && _TEST_PATHS+=("$_p")
    done
    if [ ${#_TEST_PATHS[@]} -gt 0 ]; then
      _FUSION_ARGS+=(--test-batch-jsons "${_TEST_PATHS[@]}")
    fi
  fi
  [ "$LISTWISE_DISABLE_METRICS" = "1" ] && _FUSION_ARGS+=(--disable-metrics)

  echo "${_FUSION_ARGS[@]}"
}

# Single-window fusion (always)
_HAS_FUSED=0
[ -d "$_FUSED_DIR/runs" ] && [ -n "$(find "$_FUSED_DIR/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_FUSED=1

if [ "$_HAS_FUSED" = "1" ]; then
  echo "[listwise] Fused single-window output already exists in $_FUSED_DIR – skipping."
else
  echo "[listwise] RRF fusion: single_window + snippet_doc_fusion -> listwise_fused (pool_top=$LISTWISE_FUSION_POOL_TOP)..."
  FUSION_ARGS=($(_build_fusion_args "$_SINGLE_RUNS_DIR" "$_FUSED_DIR" "$LISTWISE_FUSION_POOL_TOP"))
  python3 "$SCRIPT_DIR/rerank/listwise_rrf_fusion.py" "${FUSION_ARGS[@]}"
fi

# Sliding-window fusion (optional)
if [ "$LISTWISE_RUN_SLIDING" = "1" ] && [ "$LISTWISE_FUSE_SLIDING" = "1" ]; then
  _HAS_FUSED_SLIDING=0
  [ -d "$_FUSED_SLIDING_DIR/runs" ] && [ -n "$(find "$_FUSED_SLIDING_DIR/runs" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_FUSED_SLIDING=1

  if [ "$_HAS_FUSED_SLIDING" = "1" ]; then
    echo "[listwise] Fused sliding-window output already exists in $_FUSED_SLIDING_DIR – skipping."
  else
    echo "[listwise] RRF fusion: sliding_window + snippet_doc_fusion -> listwise_fused_sliding (pool_top=$LISTWISE_POOL)..."
    FUSION_SLIDING_ARGS=($(_build_fusion_args "$_SLIDING_RUNS_DIR" "$_FUSED_SLIDING_DIR" "$LISTWISE_POOL"))
    python3 "$SCRIPT_DIR/rerank/listwise_rrf_fusion.py" "${FUSION_SLIDING_ARGS[@]}"
  fi
else
  echo "[listwise] Sliding-window fusion skipped (LISTWISE_RUN_SLIDING=$LISTWISE_RUN_SLIDING, LISTWISE_FUSE_SLIDING=$LISTWISE_FUSE_SLIDING)"
fi

STEP_END=$(date +%s)
echo "[listwise] Completed (reranking + fusion) in $((STEP_END - STEP_START))s"
echo "[listwise] Output: $LISTWISE_OUTPUT_DIR"
