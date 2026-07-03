#!/usr/bin/env python3
"""Extract atomic, query-relevant claims from generation contexts (distillation stage 1).

For each query, an LLM extracts short self-contained factual claims from each of the top-N
contexts in the generation contexts JSONL (N deliberately larger than what fits a raw prompt).
Results are appended to a resumable cache keyed by (qid, context text), so no context is ever
extracted twice — reruns and interrupted runs are cheap. Downstream, either distil_claims.py
(claims mode) or summarize_facets.py + select_contexts.py (facets mode) turn the cached claims
into a distilled contexts JSONL for generate_answers.py.

Runs with a bare Python (stdlib only). Enabled via GENERATION_MODE=claims|facets in the
pipeline; byte-identical no-op for the default direct mode.

Env defaults (flags win): GENERATION_EXTRACT_MODEL (falls back to GENERATION_MODEL),
GENERATION_EXTRACT_URL (falls back to OLLAMA_URL), GENERATION_EXTRACT_TOP_N,
GENERATION_EXTRACT_MAX_CHARS, GENERATION_EXTRACT_TIMEOUT, GENERATION_EXTRACT_RETRIES,
GENERATION_EXTRACT_RETRY_SLEEP, GENERATION_EXTRACT_THINK, LLAMA_API_KEY (bearer, optional).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_common import ce_of, call_ollama, env_first, sha_key  # noqa: E402

DEFAULT_OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"  # same default as generate_answers.py

EXTRACT_SYSTEM = (
    "You extract atomic factual claims from a passage to answer a question. "
    "A claim is ONE short, self-contained sentence stating a single fact, definition, "
    "cause, effect, number, recommendation, condition, or caveat. Resolve pronouns and "
    "references so each claim stands alone without the passage. Use ONLY information present "
    "in the passage; do not add outside knowledge. Keep ONLY claims relevant to the question; "
    "drop navigation text, menus, ads, and boilerplate. If the passage contains nothing "
    "relevant, return an empty list."
)
EXTRACT_USER_TMPL = (
    "Question: {query}\n\n"
    "Passage:\n{passage}\n\n"
    'Return valid JSON exactly in this format: {{"claims": ["claim one.", "claim two."]}}'
)

_CLAIMS_RE = re.compile(r'"claims"\s*:\s*(\[.*?\])', re.DOTALL)


def parse_claims(raw: str) -> Optional[List[str]]:
    """Tolerant parse of {"claims":[...]}. Returns None on unparseable (-> retry/skip)."""
    raw = raw.strip()
    # strip a ```json fence if present
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw
        raw = re.sub(r"^json", "", raw.strip(), flags=re.IGNORECASE).strip()
    for candidate in (raw, None):
        if candidate is None:
            m = _CLAIMS_RE.search(raw)
            if not m:
                return None
            blob = m.group(1)
        else:
            blob = candidate
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if isinstance(obj, dict) and "claims" in obj:
            obj = obj["claims"]
        if isinstance(obj, list):
            out, seen = [], set()
            for c in obj:
                if not isinstance(c, str):
                    continue
                s = c.strip()
                k = re.sub(r"\W+", " ", s.lower()).strip()
                if len(s.split()) >= 3 and k and k not in seen:
                    seen.add(k)
                    out.append(s)
            return out
    return None


def _parse_think(v: Optional[str]) -> Optional[bool]:
    if v is None or v == "":
        return None
    return v.strip().lower() not in ("0", "false", "no", "off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True,
                    help="generation contexts JSONL (the generate_answers.py input schema)")
    ap.add_argument("--cache", type=Path, required=True,
                    help="claim cache JSONL, appended to (resumable)")
    ap.add_argument("--top-n", type=int,
                    default=int(env_first("GENERATION_EXTRACT_TOP_N", default="30")))
    ap.add_argument("--model",
                    default=env_first("GENERATION_EXTRACT_MODEL", "GENERATION_MODEL",
                                      default="llama3.3:latest"))
    ap.add_argument("--ollama-url",
                    default=env_first("GENERATION_EXTRACT_URL", "OLLAMA_URL",
                                      default=DEFAULT_OLLAMA_URL))
    ap.add_argument("--timeout", type=int,
                    default=int(env_first("GENERATION_EXTRACT_TIMEOUT", default="120")))
    ap.add_argument("--retries", type=int,
                    default=int(env_first("GENERATION_EXTRACT_RETRIES", default="3")))
    ap.add_argument("--retry-sleep", type=int,
                    default=int(env_first("GENERATION_EXTRACT_RETRY_SLEEP", default="10")))
    ap.add_argument("--max-chars", type=int,
                    default=int(env_first("GENERATION_EXTRACT_MAX_CHARS", default="4000")))
    ap.add_argument("--qids", type=Path, default=None, help="optional qid allowlist file")
    args = ap.parse_args()

    api_key = os.environ.get("LLAMA_API_KEY", "")
    if not api_key:
        print("[extract] LLAMA_API_KEY unset -> no Authorization header (fine for self-hosted ollama)")
    think = _parse_think(os.environ.get("GENERATION_EXTRACT_THINK"))

    only = None
    if args.qids:
        only = {q.strip() for q in Path(args.qids).read_text().split() if q.strip()}

    done: set[str] = set()
    if args.cache.exists():
        for line in args.cache.open():
            line = line.strip()
            if line:
                done.add(json.loads(line)["key"])
    print(f"[extract] cache {args.cache} has {len(done)} records")

    # build worklist: (qid, qtext, rank, ctx) for top-N, skipping cached keys
    work: List[tuple] = []
    for line in args.evidence.open():
        rec = json.loads(line)
        qid = str(rec["query_id"])
        if only is not None and qid not in only:
            continue
        qtext = rec.get("query_text", "")
        for rank, ctx in enumerate(rec["contexts"][: args.top_n]):
            key = sha_key(qid, ctx.get("text", ""))
            if key in done:
                continue
            work.append((qid, qtext, rank, ctx, key))
            done.add(key)  # avoid duplicate (qid,text) within this run too

    print(f"[extract] {len(work)} contexts to extract (top-{args.top_n}, model={args.model})")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    with args.cache.open("a") as fh:
        for i, (qid, qtext, rank, ctx, key) in enumerate(work):
            text = ctx.get("text", "")
            user = EXTRACT_USER_TMPL.format(query=qtext, passage=text[: args.max_chars])
            prompt = f"[SYSTEM]\n{EXTRACT_SYSTEM}\n\n[USER]\n{user}"
            claims = None
            for attempt in range(args.retries):
                try:
                    raw = call_ollama(args.ollama_url, args.model, prompt,
                                      options={"temperature": 0.0, "top_p": 1.0},
                                      timeout=args.timeout, think=think, api_key=api_key)
                    claims = parse_claims(raw)
                    if claims is not None:
                        break
                except Exception as e:
                    if attempt == args.retries - 1:
                        print(f"  [fail] {qid} rank{rank}: {type(e).__name__}: {e}")
                    time.sleep(args.retry_sleep)
            if claims is None:
                n_fail += 1
                claims = []  # record the failure so we don't loop forever; distil ignores empties
            elif claims:
                n_ok += 1
            else:
                n_empty += 1
            fh.write(json.dumps({
                "key": key, "qid": qid, "context_id": ctx.get("id"),
                "doc_id": ctx.get("doc_id"), "rank": rank,
                "ce_score": ce_of(ctx), "n_claims": len(claims), "claims": claims,
            }) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(work):
                el = time.time() - t0
                print(f"  {i+1}/{len(work)} | ok={n_ok} empty={n_empty} fail={n_fail} "
                      f"| {el/(i+1):.1f}s/call | elapsed {el/60:.1f}m", flush=True)
    print(f"[extract] done: ok={n_ok} empty={n_empty} fail={n_fail}")


if __name__ == "__main__":
    main()
