#!/usr/bin/env python3
"""
Rescue failed questions from a generation output JSONL.

Loads an answers .jsonl file, finds records with an "error" field (optionally
only 504 timeouts), re-runs generation with a longer client timeout, optional --max-contexts
(default 10), and a lower max-chars-per-context (default 1100) to reduce JSON parse errors.
Note: 504 Gateway Time-out is set by the server/proxy, not by --timeout;
retrying with concurrency 1 or during off-peak may help.
By default overwrites the input file; use --output to write to a new file instead.

Usage:
  python rescue_failed_generation.py --input output/.../generation/13B3_golden_answers.jsonl
  python rescue_failed_generation.py --input 13B3_golden_answers.jsonl --output rescued.jsonl
  python rescue_failed_generation.py --input 13B3_golden_answers.jsonl --only-504 --timeout 300
  # With openai_compat: export GENERATION_MODEL (or pass --model); GEN_API_KEY may come from repo .env
  # (generate_answers loads only that key from .env). Backend/base/model otherwise come from the shell.
  # Schema snippets: export GENERATION_SCHEMAS_DIR (e.g. from pipeline config) or pass --schemas-dir;
  # otherwise generate_answers uses shared_scripts/prompts/schemas.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from sibling module for building prompts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_answers import build_full_prompt_for_record  # noqa: E402

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_ANSWERS = SCRIPT_DIR / "generate_answers.py"


def answers_jsonl_stem_for_input(input_jsonl: Path) -> str:
    """Match generate_answers.py: <stem>_answers.jsonl with optional *_contexts stem strip."""
    stem = input_jsonl.stem
    if stem.endswith("_contexts"):
        stem = stem[: -len("_contexts")]
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run generation for failed questions (e.g. 504) with longer timeout."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to generation output .jsonl (e.g. .../generation/13B3_golden_answers.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write merged result here; otherwise overwrite --input in place.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Client-side seconds to wait per LLM response (default: 300). 504 is usually from the server/gateway timeout; increasing this only avoids us giving up before the server responds.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=10,
        help="Cap on contexts in evidence block for rescue run (default: 10, same as generate_answers.py).",
    )
    parser.add_argument(
        "--max-chars-per-context",
        type=int,
        default=1100,
        help="Shorter context truncation for rescue run to reduce JSON parse errors (default: 1100).",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=60,
        help="Seconds to sleep between retries after a failed LLM call (default: 60 = 1 min).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model id passed to generate_answers.py (OpenRouter / openai_compat). "
            "Default: GENERATION_MODEL env if set; otherwise generate_answers uses its own default."
        ),
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=None,
        help=(
            "Directory of schema *.txt forwarded to generate_answers.py. "
            "Default: GENERATION_SCHEMAS_DIR if set; else generate_answers resolves under --prompts-dir."
        ),
    )
    parser.add_argument(
        "--only-504",
        action="store_true",
        dest="only_504",
        help="Only retry records whose error contains '504' (e.g. Gateway Time-out).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print how many records would be retried; do not run.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main() -> int:
    import sys
    _shared = Path(__file__).resolve().parents[1]  # generation/ -> shared_scripts/
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    try:
        from logging_config import configure_logging_from_env
        configure_logging_from_env()
    except ImportError:
        pass
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    schemas_path: Optional[Path] = None
    if args.schemas_dir is not None:
        schemas_path = args.schemas_dir.expanduser().resolve()
    else:
        _env_schemas = (os.getenv("GENERATION_SCHEMAS_DIR") or "").strip()
        if _env_schemas:
            schemas_path = Path(_env_schemas).expanduser().resolve()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    if args.input.suffix.lower() != ".jsonl":
        logger.error("Input must be .jsonl, got %s", args.input)
        return 1
    if not GENERATE_ANSWERS.exists():
        logger.error("generate_answers.py not found: %s", GENERATE_ANSWERS)
        return 1

    from retrieval_eval.common import iter_questions_jsonl, question_body, question_qid, question_type, write_questions_jsonl

    records = list(iter_questions_jsonl(args.input))

    # Find records with error; optionally only 504
    failed = [
        r for r in records
        if r.get("error")
        and (not args.only_504 or "504" in str(r.get("error", "")))
    ]
    if not failed:
        logger.info("No failed records to rescue (only-504=%s).", args.only_504)
        return 0

    logger.info("Found %d failed record(s) to rescue (only-504=%s).", len(failed), args.only_504)
    for r in failed[:5]:
        logger.info("  query_id=%s error=%s", question_qid(r), (r.get("error") or "")[:80])
    if len(failed) > 5:
        logger.info("  ... and %d more.", len(failed) - 5)

    if args.dry_run:
        logger.info("Dry run: would retry %d question(s).", len(failed))
        return 0

    # Build contexts-style input for failed questions (doc_ids or legacy documents + contexts)
    failed_questions = []
    for r in failed:
        q: Dict[str, Any] = {
            "query_id": question_qid(r),
            "query_text": question_body(r),
            "query_type": question_type(r) or "",
            "contexts": r.get("contexts", []),
        }
        _ids = r.get("doc_ids") or r.get("docnos")
        if isinstance(_ids, list) and _ids:
            q["doc_ids"] = list(_ids)
        else:
            q["documents"] = r.get("documents", [])
        failed_questions.append(q)

    with tempfile.TemporaryDirectory(prefix="rescue_gen_") as tmpdir:
        tmpdir = Path(tmpdir)
        contexts_path = tmpdir / "rescue_contexts.jsonl"
        write_questions_jsonl(contexts_path, failed_questions)

        out_dir = tmpdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(GENERATE_ANSWERS),
            "--input-path", str(contexts_path),
            "--output-dir", str(out_dir),
            "--timeout", str(args.timeout),
            "--retry-sleep", str(args.retry_sleep),
            "--max-contexts", str(args.max_contexts),
            "--max-chars-per-context", str(args.max_chars_per_context),
            "--concurrency", "1",
        ]
        if args.verbose:
            cmd.append("--verbose")
        # Match pipeline: forward provider model id when using openai_compat (generate_answers default is Ollama).
        _gen_model = (args.model or os.getenv("GENERATION_MODEL") or "").strip()
        if _gen_model:
            cmd.extend(["--model", _gen_model])
        if schemas_path is not None:
            cmd.extend(["--schemas-dir", str(schemas_path)])
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("generate_answers.py exited with code %s", result.returncode)
            return result.returncode

        rescued_path = out_dir / f"{answers_jsonl_stem_for_input(contexts_path)}_answers.jsonl"
        if not rescued_path.exists():
            logger.error("Expected output not found: %s", rescued_path)
            return 1
        rescued_list = list(iter_questions_jsonl(rescued_path))

    # Merge: build id -> rescued record, then replace in original list
    rescued_by_id = {str(question_qid(r)): r for r in rescued_list if question_qid(r) is not None}
    merged = []
    replaced = 0
    for r in records:
        qid = question_qid(r)
        if qid is not None and str(qid) in rescued_by_id:
            merged.append(rescued_by_id[str(qid)])
            replaced += 1
        else:
            merged.append(r)
    logger.info("Replaced %d record(s) with rescued results.", replaced)

    out_path = args.output if args.output is not None else args.input
    if out_path.suffix.lower() != ".jsonl":
        logger.error("Output path must end with .jsonl, got %s", out_path)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_questions_jsonl(out_path, merged)
    logger.info("Wrote %d records to %s", len(merged), out_path)

    # For records that still have an error after rescue, save full prompt to {id}.txt in same folder
    for rec in merged:
        if not rec.get("error"):
            continue
        qid = question_qid(rec)
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(qid)) if qid else "unknown"
        # Use same default prompts layout as generate_answers.py (prompts/ sibling to generation/)
        prompts_dir = SCRIPT_DIR.parent / "prompts"
        full_prompt = build_full_prompt_for_record(
            rec,
            prompts_dir,
            max_contexts=args.max_contexts,
            max_chars_per_context=args.max_chars_per_context,
            schemas_dir=schemas_path,
        )
        if full_prompt:
            prompt_path = out_path.parent / f"{safe_id}.txt"
            try:
                prompt_path.write_text(full_prompt, encoding="utf-8")
                logger.info("Saved failed prompt to %s", prompt_path)
            except Exception as e:
                logger.warning("Could not write failed prompt to %s: %s", prompt_path, e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
