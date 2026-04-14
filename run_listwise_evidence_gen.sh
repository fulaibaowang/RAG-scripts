#!/usr/bin/env bash
#
# Evidence + Generation for listwise-reranked runs.
#
# Runs in the MAIN container (not the listwise/vLLM container) because it
# reuses the same evidence/generation Python scripts as the main pipeline.
#
# Default route: listwise_fused (RRF fusion of listwise single-window + snippet_rrf).
# Optional: listwise_fused_sliding (if LISTWISE_FUSE_SLIDING=1).
#
# Usage:
#   ./run_listwise_evidence_gen.sh --config <path/to/config.env>
#
# Required config variables: WORKFLOW_OUTPUT_DIR, DOCS_JSONL
# Optional: TRAIN_JSON, TEST_BATCH_JSONS, EVIDENCE_TOP_K (post_rerank top-k; default 10), GENERATION_*
#
set -e

# ---------------------------------------------------------------------------
# Resolve SCRIPT_DIR
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

if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  set -a; source "$CONFIG_FILE"; set +a
  echo "[listwise-evgen] Loaded config from $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ -z "${WORKFLOW_OUTPUT_DIR:-}" ]; then
  echo "Error: WORKFLOW_OUTPUT_DIR is required." >&2
  exit 1
fi

_DOCS_JSONL_OK=0
if [ -n "${DOCS_JSONL:-}" ]; then
  if [ -f "$DOCS_JSONL" ]; then
    _DOCS_JSONL_OK=1
  elif compgen -G "$DOCS_JSONL" > /dev/null 2>&1; then
    _DOCS_JSONL_OK=1
  fi
fi
if [ "$_DOCS_JSONL_OK" = "0" ]; then
  echo "Error: DOCS_JSONL not set or not found. Required for evidence building." >&2
  exit 1
fi

LISTWISE_OUTPUT_DIR="${LISTWISE_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/listwise_rerank}"
SNIPPET_WINDOWS="${WORKFLOW_OUTPUT_DIR}/snippet_rerank/windows"
RUN_GENERATION_LISTWISE="${RUN_GENERATION_LISTWISE:-1}"

if [ ! -d "$LISTWISE_OUTPUT_DIR" ]; then
  echo "Error: listwise output not found at $LISTWISE_OUTPUT_DIR" >&2
  echo "  Run run_listwise_rerank.sh first." >&2
  exit 1
fi

echo "[listwise-evgen] Configuration:"
echo "  WORKFLOW_OUTPUT_DIR:  $WORKFLOW_OUTPUT_DIR"
echo "  LISTWISE_OUTPUT_DIR: $LISTWISE_OUTPUT_DIR"
echo "  DOCS_JSONL:          $DOCS_JSONL"
echo "  SNIPPET_WINDOWS:     $SNIPPET_WINDOWS"

STEP_START=$(date +%s)

# ---------------------------------------------------------------------------
# Helper: resolve query JSON for a split
# ---------------------------------------------------------------------------
_resolve_query_json() {
  local _split="$1"
  if [ -f "${TRAIN_JSON:-}" ] && [ "$(basename "$TRAIN_JSON" .json)" = "$_split" ]; then
    echo "$TRAIN_JSON"
    return
  fi
  for _p in ${TEST_BATCH_JSONS:-}; do
    [ -f "$_p" ] || continue
    [ "$(basename "$_p" .json)" = "$_split" ] || continue
    echo "$_p"
    return
  done
}

# ---------------------------------------------------------------------------
# Process a single listwise route (single_window or sliding_window)
# ---------------------------------------------------------------------------
_process_route() {
  local _route_label="$1"     # e.g. "listwise_single" or "listwise_sliding"
  local _runs_dir="$2"        # path to runs/ dir with TSVs
  local _evidence_subdir="evidence_${_route_label}"
  local _gen_subdir="generation_${_route_label}"
  local _post_dir="${LISTWISE_OUTPUT_DIR}/${_route_label}"

  if [ ! -d "$_runs_dir" ]; then
    echo "[listwise-evgen] No runs directory: $_runs_dir – skipping $_route_label"
    return
  fi
  # Check for any TSV files
  local _tsv_count
  _tsv_count=$(find "$_runs_dir" -maxdepth 1 -name '*.tsv' 2>/dev/null | wc -l)
  if [ "$_tsv_count" -eq 0 ]; then
    echo "[listwise-evgen] No TSV files in $_runs_dir – skipping $_route_label"
    return
  fi

  mkdir -p "$WORKFLOW_OUTPUT_DIR/$_evidence_subdir"
  mkdir -p "$_post_dir"
  echo "[listwise-evgen] Route: $_route_label -> $_evidence_subdir / $_gen_subdir"

  for _tsv in "$_runs_dir/"*.tsv; do
    [ -f "$_tsv" ] || continue
    _split=$(basename "$_tsv" .tsv)
    [ -n "$_split" ] || continue

    _query_json=$(_resolve_query_json "$_split")
    if [ -z "$_query_json" ]; then
      echo "[listwise-evgen] Skip $_split: no matching query JSON"
      continue
    fi

    # -- Post-rerank JSON --
    _post_json="${_post_dir}/post_rerank_${_split}.json"
    if [ ! -f "$_post_json" ]; then
      echo "[listwise-evgen] Post-rerank JSON ($_split)..."
      python "$SCRIPT_DIR/evidence/post_rerank_json.py" \
        --run-path "$_tsv" \
        --query-json "$_query_json" \
        --output-path "$_post_json" \
        --top-k "${EVIDENCE_TOP_K:-10}"
    else
      echo "[listwise-evgen] Post-rerank JSON ($_split)... (skip: exists)"
    fi

    # -- Contexts from snippets --
    _ctx_json="$WORKFLOW_OUTPUT_DIR/$_evidence_subdir/${_split}_contexts.json"
    if [ ! -f "$_ctx_json" ]; then
      if [ -d "$SNIPPET_WINDOWS" ]; then
        echo "[listwise-evgen] Contexts from snippets ($_split)..."
        python "$SCRIPT_DIR/evidence/build_contexts_from_snippets.py" \
          --post-rerank-json "$_post_json" \
          --snippet-windows-dir "$SNIPPET_WINDOWS" \
          --split-name "$_split" \
          --corpus-path "$DOCS_JSONL" \
          --output-path "$_ctx_json" \
          --window-size "${SNIPPET_WINDOW_SIZE:-3}" \
          --top-windows "${SNIPPET_CONTEXT_TOP_WINDOWS:-2}"
      else
        echo "[listwise-evgen] Contexts from documents ($_split)..."
        python "$SCRIPT_DIR/evidence/build_contexts_from_documents.py" \
          --post-rerank-json "$_post_json" \
          --corpus-path "$DOCS_JSONL" \
          --output-path "$_ctx_json"
      fi
    else
      echo "[listwise-evgen] Contexts ($_split)... (skip: exists)"
    fi
  done

  # -- Generation --
  if [ "${RUN_GENERATION_LISTWISE:-1}" != "1" ]; then
    echo "[listwise-evgen] Generation disabled for $_route_label"
    return
  fi

  mkdir -p "$WORKFLOW_OUTPUT_DIR/$_gen_subdir"
  for _tsv in "$_runs_dir/"*.tsv; do
    [ -f "$_tsv" ] || continue
    _split=$(basename "$_tsv" .tsv)
    [ -n "$_split" ] || continue

    _ctx_json="$WORKFLOW_OUTPUT_DIR/$_evidence_subdir/${_split}_contexts.json"
    _gen_json="$WORKFLOW_OUTPUT_DIR/$_gen_subdir/${_split}_answers.json"

    if [ -f "$_gen_json" ]; then
      echo "[listwise-evgen] Generation $_split... (skip: exists)"
    elif [ ! -f "$_ctx_json" ]; then
      echo "[listwise-evgen] Generation skip $_split: no evidence"
    else
      echo "[listwise-evgen] Generation $_split..."
      GENERATION_ARGS=(
        --input-path "$_ctx_json"
        --output-dir "$WORKFLOW_OUTPUT_DIR/$_gen_subdir"
      )
      [ -n "${GENERATION_CONCURRENCY:-}" ] && GENERATION_ARGS+=(--concurrency "$GENERATION_CONCURRENCY")
      _max_ctx="${GENERATION_MAX_CONTEXTS_SNIPPET:-${GENERATION_MAX_CONTEXTS:-10}}"
      [ -n "$_max_ctx" ] && GENERATION_ARGS+=(--max-contexts "$_max_ctx")
      _max_chars="${GENERATION_MAX_CHARS_PER_CONTEXT_SNIPPET:-${GENERATION_MAX_CHARS_PER_CONTEXT:-960}}"
      [ -n "$_max_chars" ] && GENERATION_ARGS+=(--max-chars-per-context "$_max_chars")
      [ -n "${GENERATION_SLEEP:-}" ] && GENERATION_ARGS+=(--sleep "$GENERATION_SLEEP")
      [ -n "${GENERATION_MODEL:-}" ] && GENERATION_ARGS+=(--model "$GENERATION_MODEL")
      [ "${GENERATION_NO_PROGRESS:-1}" = "1" ] && GENERATION_ARGS+=(--no-progress)
      python "$SCRIPT_DIR/generation/generate_answers.py" "${GENERATION_ARGS[@]}"

      # Rescue
      if [ -f "$_gen_json" ]; then
        echo "[listwise-evgen] Rescue $_split..."
        RESCUE_ARGS=(--input "$_gen_json")
        [ -n "${GENERATION_RESCUE_TIMEOUT:-}" ] && RESCUE_ARGS+=(--timeout "$GENERATION_RESCUE_TIMEOUT")
        [ -n "${GENERATION_RESCUE_RETRY_SLEEP:-}" ] && RESCUE_ARGS+=(--retry-sleep "$GENERATION_RESCUE_RETRY_SLEEP")
        [ -n "${GENERATION_MAX_CHARS_PER_CONTEXT:-}" ] && RESCUE_ARGS+=(--max-chars-per-context "$GENERATION_MAX_CHARS_PER_CONTEXT")
        python "$SCRIPT_DIR/generation/rescue_failed_generation.py" "${RESCUE_ARGS[@]}"
      fi
    fi
  done
}

# ---------------------------------------------------------------------------
# Run routes: fused (default); optional fused-sliding
# ---------------------------------------------------------------------------
LISTWISE_FUSE_SLIDING="${LISTWISE_FUSE_SLIDING:-0}"

_process_route "listwise" "${LISTWISE_OUTPUT_DIR}/listwise_fused/runs"
if [ "$LISTWISE_FUSE_SLIDING" = "1" ]; then
  _process_route "listwise_sliding" "${LISTWISE_OUTPUT_DIR}/listwise_fused_sliding/runs"
else
  echo "[listwise-evgen] Skipping sliding fused route (LISTWISE_FUSE_SLIDING=$LISTWISE_FUSE_SLIDING)"
fi

STEP_END=$(date +%s)
echo "[listwise-evgen] Completed in $((STEP_END - STEP_START))s"
echo "[listwise-evgen] Evidence/generation written under $WORKFLOW_OUTPUT_DIR"
