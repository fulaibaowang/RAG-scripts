#!/usr/bin/env python3
"""
Generate BioASQ answers from contexts JSON using an LLM.

Reads contexts JSONL from build_contexts_from_documents.py / build_contexts_from_snippets.py
(id, body, type, contexts with ``doc_id`` per row; optional ``doc_ids`` on the question),
calls an LLM per question, parses ideal_answer and
evidence_ids (and exact_answer for yesno/factoid/list), and writes a single
JSONL file to output_dir (e.g. output_dir/<stem>_answers.jsonl).

Backend and model come **only** from the process environment and CLI (sourced run
``config.env``, ``export``, scheduler, etc.). Repo-root ``.env`` is read **only**
for ``GEN_API_KEY`` and ``LLAMA_API_KEY`` (``setdefault`` — never overrides an
already-exported key). It does **not** load ``GENERATION_BACKEND``,
``GENERATION_MODEL``, or ``GEN_API_BASE`` from that file.

- ``GENERATION_BACKEND`` (default ``ollama`` when unset): ``ollama`` — HTTP to
  ``OLLAMA_URL`` with ``LLAMA_API_KEY`` (export, scheduler, or ``LLAMA_API_KEY`` in repo-root ``.env``).
  ``openai_compat`` (aliases: ``openrouter``, ``openai``) — POST
  ``{GEN_API_BASE}/chat/completions`` with ``GEN_API_KEY``. Requires ``GEN_API_BASE``.

- **Model id**: ``--model`` (pipeline may set ``GENERATION_MODEL`` for the same value).
  Default when ``--model`` is omitted: ``llama3.3:latest`` (Ollama tag).

OpenAI-compatible path is equivalent to the OpenAI client's ``base_url`` +
``chat.completions``; this script uses ``requests`` only.

Schema snippets for ``{SCHEMA_BLOCK}`` in ``user_base.txt``: directory of ``*.txt``
(typed names + optional ``default.txt``). Resolution: ``--schemas-dir``, then
``GENERATION_SCHEMAS_DIR``, then ``<prompts-dir>/schemas`` (default prompts dir is
next to this script under ``shared_scripts/prompts``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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

def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (.git marker)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent

REPO_ROOT = _find_repo_root()

OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
OLLAMA_MODEL = "llama3.3:latest"

MAX_LLM_RETRIES = 3

logger = logging.getLogger(__name__)


def _is_retryable_request_error(exc: BaseException) -> bool:
    """True if the exception is a transient error worth retrying (timeout, 5xx, connection)."""
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 502, 503, 504)
    return False


# Only these keys are read from repo-root .env (never override existing exports).
_DOTENV_ALLOWED_KEYS = frozenset({"GEN_API_KEY", "LLAMA_API_KEY"})


def _load_gen_api_key_from_dotenv() -> None:
    """Set GEN_API_KEY / LLAMA_API_KEY from repo-root .env if unset. Ignores all other keys in that file."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, rest = line.partition("=")
                key = key.strip()
                if key not in _DOTENV_ALLOWED_KEYS:
                    continue
                val = rest.split("#", 1)[0].strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if val:
                    os.environ.setdefault(key, val)
    except OSError as e:
        logger.warning("Could not read .env for generation API keys: %s", e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BioASQ answers from contexts JSONL using an LLM."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to contexts .jsonl (output of build_contexts_*.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; writes <stem>_answers.jsonl here.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max parallel LLM calls (default: 1).",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=10,
        help="Cap on number of contexts in evidence block (default: 10).",
    )
    parser.add_argument(
        "--max-chars-per-context",
        type=int,
        default=1300,
        help="Truncation length per context (default: 1300).",
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
        help="Prompts directory (default: REPO_ROOT/scripts/public/shared_scripts/prompts).",
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=None,
        help=(
            "Directory of schema *.txt files (per question type + optional default.txt). "
            "Overrides GENERATION_SCHEMAS_DIR; if both unset, uses <prompts-dir>/schemas."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for each LLM response (default: 120).",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=5,
        help="Seconds to sleep between retries after a failed LLM call (default: 5).",
    )
    parser.add_argument(
        "--evidence-source",
        choices=["contexts", "snippets"],
        default="contexts",
        help="Which evidence field to use for prompts: contexts or snippets (default: contexts).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=(
            f"Model id: Ollama tag when GENERATION_BACKEND is ollama (default: {OLLAMA_MODEL}); "
            f"OpenAI-compatible model id when GENERATION_BACKEND is openai_compat."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (Ollama options or chat completions body; default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (Ollama options or chat completions when < 1.0; default: 1.0).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar and per-question progress logs (use for batch/sbatch).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log generation progress every N questions (e.g. 10 -> 1/n, 10/n, 20/n, ...). 0 disables (default: 10).",
    )
    return parser.parse_args()


def openai_compat_enabled() -> bool:
    """True when GENERATION_BACKEND requests OpenAI-compatible chat completions."""
    raw = (os.getenv("GENERATION_BACKEND") or "").strip().lower()
    if not raw or raw == "ollama":
        return False
    if raw in ("openai_compat", "openrouter", "openai"):
        return True
    raise RuntimeError(
        f"Unknown GENERATION_BACKEND={raw!r}; use ollama (default) or openai_compat"
    )


def chat_completions_endpoint(gen_api_base: str) -> str:
    """Return full URL for POST .../chat/completions given GEN_API_BASE (e.g. https://openrouter.ai/api/v1)."""
    base = gen_api_base.strip().rstrip("/")
    if base.lower().endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def get_ollama_api_key() -> str:
    key = (os.getenv("LLAMA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing LLAMA_API_KEY (Ollama backend). "
            "Export it, inject via scheduler, or set LLAMA_API_KEY in repo-root .env "
            "(read if unset, same rule as GEN_API_KEY)."
        )
    return key


def get_openai_compat_api_key() -> str:
    key = (os.getenv("GEN_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing GEN_API_KEY (required when GENERATION_BACKEND=openai_compat). "
            "Export it, inject via scheduler, or set only GEN_API_KEY in repo-root .env "
            "(generate_answers reads that key from .env if unset)."
        )
    return key


def get_api_key() -> str:
    """Return the active backend API key (GEN_API_KEY or LLAMA_API_KEY) based on GENERATION_BACKEND."""
    if openai_compat_enabled():
        return get_openai_compat_api_key()
    return get_ollama_api_key()


def call_llm_ollama(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = 120,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    r = requests.post(
        OLLAMA_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "stream": False,
            "prompt": prompt,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def call_llm_openai_compat(
    api_key: str,
    gen_api_base: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout: int = 120,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """OpenAI-compatible chat completions (OpenRouter, vLLM, OpenAI, etc.)."""
    url = chat_completions_endpoint(gen_api_base)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
    }
    if float(top_p) < 1.0:
        payload["top_p"] = float(top_p)
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            msg = str(err.get("message", err))
        else:
            msg = str(err)
        raise RuntimeError(f"Chat API error: {msg}")
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices, list):
        raise ValueError("No choices in chat completions response")
    msg0 = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg0, dict):
        raise ValueError("Invalid message object in chat completions response")
    content = msg0.get("content")
    if content is None:
        raise ValueError("Empty message content in chat completions response")
    return str(content)


def call_llm(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = 120,
    temperature: float = 0.0,
    top_p: float = 1.0,
    *,
    openai_compat: bool = False,
    gen_api_base: str = "",
) -> str:
    if openai_compat:
        return call_llm_openai_compat(
            api_key,
            gen_api_base,
            system_prompt,
            user_prompt,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
        )
    return call_llm_ollama(
        api_key,
        system_prompt,
        user_prompt,
        model=model,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
    )


def format_evidence_block(
    contexts: List[Dict[str, Any]],
    max_contexts: int,
    max_chars_per_context: int,
) -> str:
    lines: List[str] = []
    for ctx in contexts[:max_contexts]:
        cid = str(ctx.get("id", "")) or "(no id)"
        text = str(ctx.get("text", "")).strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + "..."
        block = f"[{cid}],\n{text}" if text else f"[{cid}],"
        lines.append(block)
    return "\n\n".join(lines)


def snippets_to_contexts(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BioASQ golden `snippets` into `contexts` compatible with generation prompts.

    Evidence ID scheme:
    - If snippet["id"] exists and is non-empty, use it as the context id.
    - Otherwise, extract PMID from snippet["document"] (PubMed URL); if offsetInBeginSection
      is present and can be parsed as int, use "{pmid}-{offset}", else just "{pmid}".
    - If no PMID can be extracted, fall back to a stable per-snippet index-based id.
    """
    pmid_re = re.compile(r"/pubmed/(\d+)")
    contexts: List[Dict[str, Any]] = []
    for idx, snip in enumerate(snippets):
        raw_id = snip.get("id")
        if raw_id:
            cid = str(raw_id)
        else:
            doc_ref = str(snip.get("document") or snip.get("doc") or "").strip()
            m = pmid_re.search(doc_ref)
            if m:
                doc_id = m.group(1)
            elif doc_ref.isdigit():
                doc_id = doc_ref
            else:
                doc_id = ""
            offset_raw = snip.get("offsetInBeginSection")
            cid = doc_id if doc_id else f"snippet-{idx}"
            if doc_id and offset_raw is not None:
                try:
                    offset_int = int(offset_raw)
                    cid = f"{doc_id}-{offset_int}"
                except (TypeError, ValueError):
                    cid = doc_id
        doc_ref = str(snip.get("document") or snip.get("doc") or "").strip()
        text = str(snip.get("text", "")).strip()
        ctx: Dict[str, Any] = {"id": cid, "text": text}
        if doc_ref:
            if doc_ref.isdigit():
                ctx["doc_id"] = doc_ref
            else:
                ctx["doc"] = doc_ref
                m2 = pmid_re.search(doc_ref)
                if m2:
                    ctx["doc_id"] = m2.group(1)
        contexts.append(ctx)
    return contexts


def resolve_schema_block(schemas_dir: Path, qtype: str) -> str:
    """Resolve schema snippet: typed ``{qtype}.txt`` if present; untyped uses ``default.txt`` if present; else ``""``."""
    raw = (qtype or "").strip().lower()
    if raw:
        path = schemas_dir / f"{raw}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
    default_path = schemas_dir / "default.txt"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8").strip()
    return ""


def format_user_prompt(
    user_base_text: str,
    *,
    schema_block: str,
    qtype: str,
    question: str,
    evidence_block: str,
) -> str:
    raw = (qtype or "").strip().lower()
    schema_prefix = (schema_block.strip() + "\n\n") if schema_block.strip() else ""
    question_header = (f"Question (type={raw}):\n" if raw else "Question:\n")
    return (
        user_base_text.replace("{SCHEMA_BLOCK}", schema_prefix)
        .replace("{QUESTION_HEADER}", question_header)
        .replace("{QUESTION}", question)
        .replace("{EVIDENCE_BLOCK}", evidence_block)
    )


def build_full_prompt_for_record(
    record: Dict[str, Any],
    prompts_dir: Path,
    max_contexts: int = 8,
    max_chars_per_context: int = 1200,
    schemas_dir: Optional[Path] = None,
) -> str:
    """Build the exact prompt that would be sent for this record. Used by rescue script.

    Schema snippets: explicit ``schemas_dir`` if passed; else ``GENERATION_SCHEMAS_DIR``;
    else ``prompts_dir / "schemas"`` (same order as ``main()`` without ``--schemas-dir``).
    """
    qtype = (record.get("type") or "").strip().lower()
    question = (record.get("body") or "").strip()
    contexts = record.get("contexts") or []
    if not question or not contexts:
        return ""
    system_path = prompts_dir / "system.txt"
    user_path = prompts_dir / "user_base.txt"
    if schemas_dir is not None:
        _schemas = schemas_dir.expanduser().resolve()
    else:
        env_schemas = (os.getenv("GENERATION_SCHEMAS_DIR") or "").strip()
        if env_schemas:
            _schemas = Path(env_schemas).expanduser().resolve()
        else:
            _schemas = (prompts_dir / "schemas").resolve()
    if not system_path.exists() or not user_path.exists():
        return ""
    system_text = system_path.read_text(encoding="utf-8").strip()
    user_base_text = user_path.read_text(encoding="utf-8").strip()
    schema_block = resolve_schema_block(_schemas, record.get("type") or "")
    evidence_block = format_evidence_block(contexts, max_contexts, max_chars_per_context)
    user_prompt = format_user_prompt(
        user_base_text,
        schema_block=schema_block,
        qtype=qtype,
        question=question,
        evidence_block=evidence_block,
    )
    return f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_prompt}"


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

    qt = (qtype or "").strip().lower()
    out: Dict[str, Any] = {"ideal_answer": ideal, "evidence_ids": ev_ids}

    if qt == "yesno":
        if "exact_answer" not in obj:
            raise ValueError("yesno type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, str):
            raise ValueError("yesno exact_answer must be a string")
        if ea.strip().lower() not in ("yes", "no"):
            raise ValueError(f"yesno exact_answer must be 'yes' or 'no', got: {ea!r}")
        out["exact_answer"] = ea.strip().lower()
    elif qt in ("factoid", "list"):
        if "exact_answer" not in obj:
            raise ValueError(f"{qt} type requires 'exact_answer'")
        ea = obj["exact_answer"]
        if not isinstance(ea, list):
            raise ValueError(f"{qt} exact_answer must be a list (or list of lists)")
        if len(ea) == 0:
            out["exact_answer"] = []
        elif isinstance(ea[0], str):
            # Flat list of strings -> array-of-arrays (one inner array per answer)
            if not all(isinstance(x, str) for x in ea):
                raise ValueError(f"{qt} exact_answer list must contain only strings")
            out["exact_answer"] = [[s] for s in ea]
        elif isinstance(ea[0], list):
            # Already array-of-arrays
            if not all(isinstance(inner, list) and all(isinstance(x, str) for x in inner) for inner in ea):
                raise ValueError(f"{qt} exact_answer must be list of lists of strings")
            out["exact_answer"] = ea
        else:
            raise ValueError(f"{qt} exact_answer must be list of strings or list of lists of strings")

    return out


def load_contexts_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load one question record per line from contexts .jsonl."""
    _shared = Path(__file__).resolve().parents[1]
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    from retrieval_eval.common import iter_questions_jsonl

    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Contexts input must be .jsonl, got: {path}")
    return list(iter_questions_jsonl(path))


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
    _load_gen_api_key_from_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        openai_compat = openai_compat_enabled()
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    if openai_compat:
        gen_base = (os.getenv("GEN_API_BASE") or "").strip()
        if not gen_base:
            logger.error(
                "GENERATION_BACKEND=openai_compat requires GEN_API_BASE (e.g. https://.../v1 in run env)."
            )
            return 1
        api_key = get_openai_compat_api_key()
        effective_model = args.model
        logger.info(
            "Generation backend: OpenAI-compatible chat (%s) model=%s",
            chat_completions_endpoint(gen_base),
            effective_model,
        )
    else:
        gen_base = ""
        api_key = get_ollama_api_key()
        effective_model = args.model
        logger.info("Generation backend: Ollama model=%s", effective_model)

    # Default prompts directory: resolve relative to this script so layout is portable.
    # This works for both:
    # - REPO_ROOT/scripts/public/shared_scripts/generation/generate_answers.py
    # - REPO_ROOT/shared_scripts/generation/generate_answers.py
    default_prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    prompts_dir = args.prompts_dir or default_prompts_dir
    system_path = prompts_dir / "system.txt"
    user_base_path = prompts_dir / "user_base.txt"

    # Schema snippets: --schemas-dir > GENERATION_SCHEMAS_DIR > <prompts-dir>/schemas
    if args.schemas_dir is not None:
        schemas_dir = args.schemas_dir.expanduser().resolve()
    else:
        schemas_override = (os.getenv("GENERATION_SCHEMAS_DIR") or "").strip()
        if schemas_override:
            schemas_dir = Path(schemas_override).expanduser().resolve()
        else:
            schemas_dir = (prompts_dir / "schemas").resolve()

    # Optional override from environment/config:
    # - GENERATION_SYSTEM_PATH: path to system.txt replacement
    system_override = (os.getenv("GENERATION_SYSTEM_PATH") or "").strip()
    if system_override:
        system_path = Path(system_override)

    if not system_path.exists() or not user_base_path.exists():
        logger.error("Prompts not found under %s", prompts_dir)
        return 1
    if not args.input_path.exists():
        logger.error("Input file not found: %s", args.input_path)
        return 1
    if args.input_path.suffix.lower() != ".jsonl":
        logger.error("Input must be .jsonl, got %s", args.input_path)
        return 1

    with open(system_path, "r", encoding="utf-8") as f:
        system_text = f.read().strip()
    with open(user_base_path, "r", encoding="utf-8") as f:
        user_base_text = f.read().strip()

    schema_cache: Dict[str, str] = {}

    def get_schema_block(qtype: str) -> str:
        key = (qtype or "").strip().lower()
        if key not in schema_cache:
            schema_cache[key] = resolve_schema_block(schemas_dir, qtype)
        return schema_cache[key]

    # Warm cache for common typed labels and untyped ("") default.txt resolution.
    for _q in ("", "summary", "yesno", "factoid", "list"):
        get_schema_block(_q)

    from retrieval_eval.common import write_questions_jsonl

    all_objs = load_contexts_jsonl(args.input_path)
    total = len(all_objs)
    if total == 0:
        logger.warning("No questions in input; nothing to write.")
        return 0

    stem = args.input_path.stem
    if stem.endswith("_contexts"):
        stem = stem[: -len("_contexts")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{stem}_answers.jsonl"

    def process_one(idx: int, obj: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        q_id = obj.get("id")
        qtype = (obj.get("type") or "").strip().lower()
        question = obj.get("body", "") or ""
        if args.evidence_source == "contexts":
            contexts = obj.get("contexts") or []
        else:
            snippets = obj.get("snippets") or []
            contexts = snippets_to_contexts(snippets) if snippets else []
        out = dict(obj)
        doc_ids = obj.get("doc_ids") or obj.get("docnos")
        if isinstance(doc_ids, list) and doc_ids:
            out["doc_ids"] = list(doc_ids)
            out.pop("documents", None)
        else:
            out.setdefault("documents", obj.get("documents", []))
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
        user_prompt = format_user_prompt(
            user_base_text,
            schema_block=schema_block,
            qtype=qtype,
            question=question,
            evidence_block=evidence_block,
        )

        raw = None
        last_error: Optional[Exception] = None
        for attempt in range(MAX_LLM_RETRIES):
            try:
                raw = call_llm(
                    api_key,
                    system_text,
                    user_prompt,
                    model=effective_model,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    openai_compat=openai_compat,
                    gen_api_base=gen_base,
                )
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
                        args.retry_sleep,
                    )
                    time.sleep(args.retry_sleep)
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
    progress_every = 0 if args.no_progress else (args.progress_every if args.progress_every > 0 else 0)
    completed_count = 0
    use_tqdm = not args.no_progress and tqdm is not None and sys.stderr.isatty()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(process_one, idx, obj): (idx, obj)
            for idx, obj in enumerate(all_objs, start=1)
        }
        completed = as_completed(futs)
        if use_tqdm:
            completed = tqdm(completed, total=total, desc="Generation")
        for fut in completed:
            idx, obj = futs[fut]
            try:
                _, rec = fut.result()
                results_by_idx[idx] = rec
            except Exception as e:
                logger.warning("Task failed for id=%s: %s; recording as error", obj.get("id"), e)
                rec = dict(obj)
                _ids = obj.get("doc_ids") or obj.get("docnos")
                if isinstance(_ids, list) and _ids:
                    rec["doc_ids"] = list(_ids)
                    rec.pop("documents", None)
                else:
                    rec.setdefault("documents", obj.get("documents", []))
                rec.setdefault("contexts", obj.get("contexts", []))
                rec["ideal_answer"] = None
                rec["evidence_ids"] = []
                rec["error"] = str(e)
                qtype = (obj.get("type") or "").strip().lower()
                if qtype in ("yesno", "factoid", "list"):
                    rec["exact_answer"] = None
                results_by_idx[idx] = rec
            completed_count += 1
            if progress_every and (completed_count == 1 or completed_count % progress_every == 0 or completed_count == total):
                logger.info("Generation progress: %d/%d", completed_count, total)

    # Ensure every input question has a record (fallback for any missing index)
    records_out: List[Dict[str, Any]] = []
    for i in range(1, total + 1):
        if i in results_by_idx:
            records_out.append(results_by_idx[i])
        else:
            obj = all_objs[i - 1]
            rec = dict(obj)
            _ids2 = obj.get("doc_ids") or obj.get("docnos")
            if isinstance(_ids2, list) and _ids2:
                rec["doc_ids"] = list(_ids2)
                rec.pop("documents", None)
            else:
                rec.setdefault("documents", obj.get("documents", []))
            rec.setdefault("contexts", obj.get("contexts", []))
            rec["ideal_answer"] = None
            rec["evidence_ids"] = []
            rec["error"] = "missing_from_results"
            if (obj.get("type") or "").strip().lower() in ("yesno", "factoid", "list"):
                rec["exact_answer"] = None
            records_out.append(rec)
            logger.warning("No result for index %d (id=%s); added record with error", i, obj.get("id"))

    write_questions_jsonl(json_path, records_out)

    logger.info("Wrote %d records to %s", len(records_out), json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
