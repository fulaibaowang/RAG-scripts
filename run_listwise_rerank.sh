#!/usr/bin/env bash
#
# Listwise reranking (RankZephyr) – runs in its own container after the main pipeline.
#
# Reads snippet_rrf/runs and snippet_rerank/windows produced by
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

SNIPPET_RRF_RUNS="${WORKFLOW_OUTPUT_DIR}/snippet_rrf/runs"
SNIPPET_WINDOWS="${WORKFLOW_OUTPUT_DIR}/snippet_rerank/windows"

if [ ! -d "$SNIPPET_RRF_RUNS" ]; then
  echo "Error: snippet_rrf/runs not found at $SNIPPET_RRF_RUNS" >&2
  echo "  The main pipeline must run with --snippet-rrf or RUN_SNIPPET_RRF=1 first." >&2
  exit 1
fi

if [ ! -d "$SNIPPET_WINDOWS" ]; then
  echo "Error: snippet_rerank/windows not found at $SNIPPET_WINDOWS" >&2
  echo "  The main pipeline must run with --snippet-rrf or RUN_SNIPPET_RRF=1 first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Defaults from config (LISTWISE_* env vars) with fallbacks
# ---------------------------------------------------------------------------
LISTWISE_SINGLE_K="${LISTWISE_SINGLE_K:-15}"
LISTWISE_RUN_SLIDING="${LISTWISE_RUN_SLIDING:-1}"  # set to 0 to skip sliding window
LISTWISE_POOL="${LISTWISE_POOL:-50}"
LISTWISE_WINDOW="${LISTWISE_WINDOW:-15}"
LISTWISE_STRIDE="${LISTWISE_STRIDE:-5}"
LISTWISE_MODEL="${LISTWISE_MODEL:-castorini/rank_zephyr_7b_v1_full}"
LISTWISE_CONTEXT_SIZE="${LISTWISE_CONTEXT_SIZE:-4096}"
LISTWISE_MAX_SNIPPET_TOKENS="${LISTWISE_MAX_SNIPPET_TOKENS:-250}"
LISTWISE_OUTPUT_DIR="${LISTWISE_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/listwise_rerank}"
LISTWISE_DISABLE_METRICS="${LISTWISE_DISABLE_METRICS:-${RERANK_DISABLE_METRICS:-0}}"

echo "[listwise] Configuration:"
echo "  WORKFLOW_OUTPUT_DIR:  $WORKFLOW_OUTPUT_DIR"
echo "  Input runs:           $SNIPPET_RRF_RUNS"
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

# ---------------------------------------------------------------------------
# Skip if output already exists
# ---------------------------------------------------------------------------
_SINGLE_RUNS_DIR="${LISTWISE_OUTPUT_DIR}/single_window/runs"
_SLIDING_RUNS_DIR="${LISTWISE_OUTPUT_DIR}/sliding_window/runs"

_HAS_SINGLE=0
_HAS_SLIDING=0
[ -d "$_SINGLE_RUNS_DIR" ] && [ -n "$(find "$_SINGLE_RUNS_DIR" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_SINGLE=1
[ -d "$_SLIDING_RUNS_DIR" ] && [ -n "$(find "$_SLIDING_RUNS_DIR" -maxdepth 1 -name '*.tsv' 2>/dev/null | head -1)" ] && _HAS_SLIDING=1

# If sliding is disabled, only check single exists
if [ "$LISTWISE_RUN_SLIDING" != "1" ]; then
  if [ "$_HAS_SINGLE" = "1" ]; then
    echo "[listwise] Output already exists in $LISTWISE_OUTPUT_DIR (single only) – skipping."
    echo "  Delete the runs/ directories to re-run."
    exit 0
  fi
else
  if [ "$_HAS_SINGLE" = "1" ] && [ "$_HAS_SLIDING" = "1" ]; then
    echo "[listwise] Output already exists in $LISTWISE_OUTPUT_DIR – skipping."
    echo "  Delete the runs/ directories to re-run."
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
STEP_START=$(date +%s)

LISTWISE_ARGS=(
  --runs-dir "$SNIPPET_RRF_RUNS"
  --windows-dir "$SNIPPET_WINDOWS"
  --output-dir "$LISTWISE_OUTPUT_DIR"
  --single-k "$LISTWISE_SINGLE_K"
  --pool "$LISTWISE_POOL"
  --window-size "$LISTWISE_WINDOW"
  --stride "$LISTWISE_STRIDE"
  --model "$LISTWISE_MODEL"
  --context-size "$LISTWISE_CONTEXT_SIZE"
  --max-snippet-tokens "$LISTWISE_MAX_SNIPPET_TOKENS"
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

STEP_END=$(date +%s)
echo "[listwise] Completed in $((STEP_END - STEP_START))s"
echo "[listwise] Output: $LISTWISE_OUTPUT_DIR"
