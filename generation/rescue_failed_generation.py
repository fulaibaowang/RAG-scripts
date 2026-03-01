#!/usr/bin/env python3
"""
Rescue failed questions from a generation output JSON.

Loads an answers JSON file, finds records with an "error" field (optionally
only 504 timeouts), re-runs generation with a longer client timeout and a
lower max-chars-per-context (default 1100) to reduce JSON parse errors.
Note: 504 Gateway Time-out is set by the server/proxy, not by --timeout;
retrying with concurrency 1 or during off-peak may help.
By default overwrites the input file; use --output to write to a new file instead.

Usage:
  python rescue_failed_generation.py --input output/.../generation/13B3_golden_answers.json
  python rescue_failed_generation.py --input 13B3_golden_answers.json --output rescued.json
  python rescue_failed_generation.py --input 13B3_golden_answers.json --only-504 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Import from sibling module for building prompts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_answers import build_full_prompt_for_record  # noqa: E402

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_ANSWERS = SCRIPT_DIR / "generate_answers.py"
REPO_ROOT = SCRIPT_DIR.resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "scripts" / "public" / "prompts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run generation for failed questions (e.g. 504) with longer timeout."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to generation output JSON (e.g. .../generation/13B3_golden_answers.json).",
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
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    if not GENERATE_ANSWERS.exists():
        logger.error("generate_answers.py not found: %s", GENERATE_ANSWERS)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if not isinstance(records, list):
        logger.error("Input JSON must be a list of records")
        return 1

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
        logger.info("  id=%s error=%s", r.get("id"), (r.get("error") or "")[:80])
    if len(failed) > 5:
        logger.info("  ... and %d more.", len(failed) - 5)

    if args.dry_run:
        logger.info("Dry run: would retry %d question(s).", len(failed))
        return 0

    # Build contexts-style input for failed questions (need id, body, type, documents, contexts)
    failed_questions = []
    for r in failed:
        q = {
            "id": r.get("id"),
            "body": r.get("body"),
            "type": r.get("type", "summary"),
            "documents": r.get("documents", []),
            "contexts": r.get("contexts", []),
        }
        failed_questions.append(q)

    with tempfile.TemporaryDirectory(prefix="rescue_gen_") as tmpdir:
        tmpdir = Path(tmpdir)
        contexts_path = tmpdir / "rescue_contexts.json"
        with open(contexts_path, "w", encoding="utf-8") as f:
            json.dump({"questions": failed_questions}, f, ensure_ascii=False, indent=2)

        out_dir = tmpdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(GENERATE_ANSWERS),
            "--input-path", str(contexts_path),
            "--output-dir", str(out_dir),
            "--timeout", str(args.timeout),
            "--retry-sleep", str(args.retry_sleep),
            "--max-chars-per-context", str(args.max_chars_per_context),
            "--concurrency", "1",
        ]
        if args.verbose:
            cmd.append("--verbose")
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("generate_answers.py exited with code %s", result.returncode)
            return result.returncode

        rescued_path = out_dir / "rescue_answers.json"
        if not rescued_path.exists():
            logger.error("Expected output not found: %s", rescued_path)
            return 1
        with open(rescued_path, "r", encoding="utf-8") as f:
            rescued_list: List[Dict[str, Any]] = json.load(f)

    # Merge: build id -> rescued record, then replace in original list
    rescued_by_id = {r.get("id"): r for r in rescued_list if r.get("id") is not None}
    merged = []
    replaced = 0
    for r in records:
        qid = r.get("id")
        if qid in rescued_by_id:
            merged.append(rescued_by_id[qid])
            replaced += 1
        else:
            merged.append(r)
    logger.info("Replaced %d record(s) with rescued results.", replaced)

    out_path = args.output if args.output is not None else args.input
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d records to %s", len(merged), out_path)

    # For records that still have an error after rescue, save full prompt to {id}.txt in same folder
    for rec in merged:
        if not rec.get("error"):
            continue
        qid = rec.get("id")
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(qid)) if qid else "unknown"
        full_prompt = build_full_prompt_for_record(
            rec,
            PROMPTS_DIR,
            max_contexts=8,
            max_chars_per_context=args.max_chars_per_context,
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
