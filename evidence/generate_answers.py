#!/usr/bin/env python3
"""
Generate BioASQ answers from contexts JSON using an LLM.

Reads the JSON produced by build_contexts_from_documents.py (id, body, type,
documents, contexts), calls an LLM per question, parses ideal_answer and
evidence_ids (and exact_answer for yesno/factoid/list), and writes a single
JSON file to output_dir (e.g. output_dir/<stem>_answers.json).

Requires: LLAMA_API_KEY in env or .env at repo root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[4]

OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
OLLAMA_MODEL = "llama3.3:latest"

MAX_LLM_RETRIES = 3
RETRY_SLEEP_SECONDS = 5

logger = logging.getLogger(__name__)


def _is_retryable_request_error(exc: BaseException) -> bool:
    """True if the exception is a transient error worth retrying (timeout, 5xx, connection)."""
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 502, 503, 504)
    return False


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv as _load
        _load(env_path)
    except ImportError:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BioASQ answers from contexts JSON using an LLM."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to contexts JSON (output of build_contexts_from_documents.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; writes <stem>_answers.json here.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max parallel LLM calls (default: 2).",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="Cap on number of contexts in evidence block (default: 8).",
    )
    parser.add_argument(
        "--max-chars-per-context",
        type=int,
        default=2000,
        help="Truncation length per context (default: 2000).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep after each LLM call (default: 0.5).",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Prompts directory (default: REPO_ROOT/scripts/public/prompts).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def get_api_key() -> str:
    key = (os.getenv("LLAMA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing LLAMA_API_KEY in environment or .env"
        )
    return key


def call_llm(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 120,
) -> str:
    prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    r = requests.post(
        OLLAMA_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": OLLAMA_MODEL, "stream": False, "prompt": prompt},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def format_evidence_block(
    contexts: List[Dict[str, Any]],
    max_contexts: int,
    max_chars_per_context: int,
) -> str:
    lines: List[str] = []
    for ctx in contexts[:max_contexts]:
        cid = str(ctx.get("id", ""))
        text = str(ctx.get("text", "")).strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + "..."
        doc = str(ctx.get("doc", "")).strip()
        header_parts = [cid]
        if doc:
            header_parts.append(doc)
        header = " | ".join(header_parts) if header_parts else "(no id)"
        block = f"[{header}]\n{text}" if text else f"[{header}]"
        lines.append(block)
    return "\n\n".join(lines)


def extract_first_json_object(raw: str) -> str:
    raw = raw.strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found in response; cannot parse JSON")
    depth = 0
    in_string = False
    escape = False
    quote_char = '"'
    i = start
    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if c == quote_char and not escape:
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
        i += 1
    raise ValueError("No matching '}' for first '{'; incomplete JSON object")


def parse_answer_json_for_type(raw: str, qtype: str, q_id: Optional[str] = None) -> Dict[str, Any]:
    raw_stripped = raw.strip()
    if not raw_stripped:
        raise ValueError("Empty response; cannot parse JSON")
    json_str = extract_first_json_object(raw_stripped)
    obj = json.loads(json_str)
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object")

    if "ideal_answer" not in obj or "evidence_ids" not in obj:
        raise ValueError("Model output must contain 'ideal_answer' and 'evidence_ids' keys")

    ideal = obj["ideal_answer"]
    ev_ids = obj["evidence_ids"]

    if not isinstance(ideal, str):
        raise ValueError("'ideal_answer' must be a string")
    if not isinstance(ev_ids, list) or not all(isinstance(x, str) for x in ev_ids):
        raise ValueError("'evidence_ids' must be a list of strings")

    qtype = (qtype or "summary").strip().lower()
    out: Dict[str, Any] = {"ideal_answer": ideal, "evidence_ids": ev_ids}

    if qtype == "yesno":
        if "exact_answer" not in obj:
            raise ValueError("yesno type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, str):
            raise ValueError("yesno exact_answer must be a string")
        if ea.strip().lower() not in ("yes", "no"):
            raise ValueError(f"yesno exact_answer must be 'yes' or 'no', got: {ea!r}")
        out["exact_answer"] = ea.strip().lower()
    elif qtype in ("factoid", "list"):
        if "exact_answer" not in obj:
            raise ValueError(f"{qtype} type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, list):
            raise ValueError(f"{qtype} exact_answer must be a list of strings")
        if not all(isinstance(x, str) for x in ea):
            raise ValueError(f"{qtype} exact_answer list must contain only strings")
        if qtype == "factoid" and len(ea) > 5:
            raise ValueError("factoid exact_answer must have 0-5 items")
        out["exact_answer"] = ea

    return out


def load_contexts_json(path: Path) -> List[Dict[str, Any]]:
    """Load contexts from JSON: expects {"questions": [...]} or a top-level list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError("Input JSON must be a list or an object with 'questions' key")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    _load_dotenv()
    api_key = get_api_key()

    prompts_dir = args.prompts_dir or (REPO_ROOT / "scripts" / "public" / "prompts")
    system_path = prompts_dir / "system.txt"
    user_base_path = prompts_dir / "user_base.txt"
    schemas_dir = prompts_dir / "schemas"

    if not system_path.exists() or not user_base_path.exists():
        logger.error("Prompts not found under %s", prompts_dir)
        return 1
    if not args.input_path.exists():
        logger.error("Input file not found: %s", args.input_path)
        return 1

    with open(system_path, "r", encoding="utf-8") as f:
        system_text = f.read().strip()
    with open(user_base_path, "r", encoding="utf-8") as f:
        user_base_text = f.read().strip()

    SCHEMA_BLOCKS: Dict[str, str] = {}

    def get_schema_block(qtype: str) -> str:
        qtype = (qtype or "summary").strip().lower()
        if qtype in SCHEMA_BLOCKS:
            return SCHEMA_BLOCKS[qtype]
        path = schemas_dir / f"{qtype}.txt"
        if not path.exists():
            path = schemas_dir / "summary.txt"
        with open(path, "r", encoding="utf-8") as f:
            block = f.read().strip()
        SCHEMA_BLOCKS[qtype] = block
        return block

    for _q in ("summary", "yesno", "factoid", "list"):
        get_schema_block(_q)

    def fill_user_prompt(question: str, evidence_block: str, qtype: str, schema_block: str) -> str:
        return (
            user_base_text
            .replace("{SCHEMA_BLOCK}", schema_block)
            .replace("{QTYPE}", qtype)
            .replace("{QUESTION}", question)
            .replace("{EVIDENCE_BLOCK}", evidence_block)
        )

    all_objs = load_contexts_json(args.input_path)
    total = len(all_objs)
    if total == 0:
        logger.warning("No questions in input; nothing to write.")
        return 0

    stem = args.input_path.stem
    if stem.endswith("_contexts"):
        stem = stem[: -len("_contexts")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{stem}_answers.json"

    def process_one(idx: int, obj: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        q_id = obj.get("id")
        qtype = obj.get("type", "summary")
        question = obj.get("body", "") or ""
        contexts = obj.get("contexts", []) or []
        documents = obj.get("documents", [])

        out = dict(obj)
        out.setdefault("documents", documents)
        out.setdefault("contexts", contexts)

        if not question or not contexts:
            out["ideal_answer"] = None
            out["evidence_ids"] = []
            out["error"] = "missing_question_or_contexts"
            if qtype in ("yesno", "factoid", "list"):
                out["exact_answer"] = None
            return idx, out

        schema_block = get_schema_block(qtype)
        evidence_block = format_evidence_block(
            contexts, args.max_contexts, args.max_chars_per_context
        )
        user_prompt = fill_user_prompt(question, evidence_block, qtype, schema_block)

        raw = None
        last_error: Optional[Exception] = None
        for attempt in range(MAX_LLM_RETRIES):
            try:
                raw = call_llm(api_key, system_text, user_prompt)
                if args.sleep > 0:
                    time.sleep(args.sleep)
                parsed = parse_answer_json_for_type(raw, qtype, q_id=q_id)
                out["ideal_answer"] = parsed["ideal_answer"]
                out["evidence_ids"] = parsed["evidence_ids"]
                if qtype in ("yesno", "factoid", "list"):
                    out["exact_answer"] = parsed.get("exact_answer")
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_LLM_RETRIES - 1 and _is_retryable_request_error(e):
                    logger.warning(
                        "LLM call failed (attempt %s/%s) for id=%s: %s; retrying in %ss...",
                        attempt + 1,
                        MAX_LLM_RETRIES,
                        q_id,
                        e,
                        RETRY_SLEEP_SECONDS,
                    )
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    break

        if last_error is not None:
            logger.warning(
                "LLM call failed after %s attempts for id=%s: %s",
                MAX_LLM_RETRIES,
                q_id,
                last_error,
            )
            logger.debug("Parse failed for id=%s type=%s: %s", q_id, qtype, last_error)
            if args.verbose and raw:
                logger.debug("Raw response (first 600 chars): %s", repr(raw[:600]))
            out["ideal_answer"] = None
            out["evidence_ids"] = []
            out["error"] = str(last_error)
            if qtype in ("yesno", "factoid", "list"):
                out["exact_answer"] = None
        return idx, out

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(process_one, idx, obj): idx
            for idx, obj in enumerate(all_objs, start=1)
        }
        completed = as_completed(futs)
        if tqdm is not None:
            completed = tqdm(completed, total=total, desc="Generation")
        for fut in completed:
            idx, rec = fut.result()
            results_by_idx[idx] = rec
            if not args.verbose and tqdm is None:
                logger.info("Completed %d/%d (id=%s)", idx, total, rec.get("id"))

    records_out = [results_by_idx[i] for i in range(1, total + 1)]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records_out, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d records to %s", len(records_out), json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
