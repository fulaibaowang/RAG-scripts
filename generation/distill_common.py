#!/usr/bin/env python3
"""Shared helpers for the optional context-distillation stages.

The distillation stages (extract_claims.py, distil_claims.py, summarize_facets.py,
select_contexts.py) are pre-processors over the generation contexts JSONL: they read the same
``{query_id, query_text, contexts: [{id, doc_id, text, ...}]}`` records that
generate_answers.py consumes and emit a rewritten contexts JSONL whose contexts are distilled
claim/summary slots. A slot's ``id`` is always a source context id, so citation lineage stays
corpus-resolvable — no minted ids.

Cache contracts (STABLE — existing caches must remain readable byte-for-byte):
  claim cache    one JSON per line
                 {key, qid, context_id, doc_id, rank, ce_score, n_claims, claims}
                 key = sha1(qid + "\\0" + context_text) hexdigest  (sha_key)
  summary cache  one JSON per line {key, qid, n_members, summary}
                 key = f"{qid}:{member_hash(member_texts)}"
  emb cache      npz {"emb": ...}, validated by row count against the flat claim list

Heavy deps (sentence-transformers, scikit-learn, numpy) are imported lazily inside functions,
and LLM calls use stdlib urllib, so the LLM-only stages run in a bare Python environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bi-encoder used for claim/facet clustering and slot selection. Frozen: all banked embedding
# caches and facet assignments were produced with it.
EMB_MODEL = "all-MiniLM-L6-v2"


def env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """First non-empty value among the named environment variables, else default."""
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def sha_key(qid: str, text: str) -> str:
    """Claim-cache key: one entry per (query, context text)."""
    return hashlib.sha1((qid + "\0" + text).encode("utf-8")).hexdigest()


def member_hash(texts: List[str]) -> str:
    """Summary-cache key component: order-independent hash of the facet's member claims."""
    h = hashlib.sha1("\n".join(sorted(norm(t) for t in texts)).encode()).hexdigest()
    return h[:16]


def ce_of(ctx: Dict[str, Any]) -> float:
    """Max cross-encoder score over a context's selected windows (NaN when absent —
    e.g. document-route contexts, which carry no snippet windows)."""
    wins = ctx.get("selected_windows") or []
    scores = [w.get("ce_score") for w in wins if isinstance(w.get("ce_score"), (int, float))]
    return max(scores) if scores else float("nan")


_PREAMBLE = re.compile(
    r"(?is)^\s*(here(?:'s| is| are)|the combined|combined statement|sure[,:]?|below is)\b[^:]*:\s*")


def strip_preamble(s: str) -> str:
    """Drop a single leading instruction-echo clause ('Here is the combined statement: ...')
    some models prepend to summaries. Pure hygiene on the slot text."""
    return _PREAMBLE.sub("", s, count=1).strip()


def load_qtext(evidence: Path) -> Dict[str, str]:
    qt: Dict[str, str] = {}
    for line in evidence.open():
        rec = json.loads(line)
        qt[str(rec["query_id"])] = rec.get("query_text", "")
    return qt


def deduped_claims(cache: Path, top_n: int) -> Dict[str, List[dict]]:
    """Per qid: sources CE-desc (NaN last; ties keep cache order = source rank), then
    exact-normalized dedup across the whole query. Claim order is deterministic given the
    same cache and top_n."""
    by_qid: Dict[str, List[dict]] = {}
    for line in cache.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["rank"] >= top_n:
            continue
        by_qid.setdefault(rec["qid"], []).append(rec)

    out: Dict[str, List[dict]] = {}
    for qid, ctxs in by_qid.items():
        def ce(r):
            v = r.get("ce_score")
            return v if isinstance(v, (int, float)) and v == v else -1e9
        ctxs.sort(key=ce, reverse=True)
        seen: set[str] = set()
        claims: List[dict] = []
        for r in ctxs:
            for c in r.get("claims") or []:
                k = norm(c)
                if k and k not in seen:
                    seen.add(k)
                    claims.append({
                        "id": r["context_id"], "doc_id": r["doc_id"], "text": c,
                        "source_rank": r["rank"], "ce_score": r.get("ce_score"),
                    })
        out[qid] = claims
    return out


def cluster_labels(emb, dist_thr: float):
    """Agglomerative facet clustering (cosine/average) at a distance threshold."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=dist_thr).fit_predict(emb)


def embed_texts(texts: List[str], emb_cache: Optional[Path] = None,
                model_name: str = EMB_MODEL, st_model=None, batch_size: int = 64):
    """Bi-encoder embed (normalized), with an optional npz cache validated by row count."""
    import numpy as np
    if emb_cache and emb_cache.exists():
        cached = np.load(emb_cache, allow_pickle=True)["emb"]
        if len(cached) == len(texts):
            print(f"  loaded cached embeddings {cached.shape}", flush=True)
            return cached
    if st_model is None:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name)
    emb = st_model.encode(texts, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=True)
    if emb_cache:
        emb_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(emb_cache, emb=emb)
        print(f"  embedded {emb.shape} -> {emb_cache}", flush=True)
    return emb


def call_ollama(url: str, model: str, prompt: str, *, options: Optional[dict] = None,
                timeout: int = 120, think: Optional[bool] = None,
                api_key: Optional[str] = None) -> str:
    """POST to an ollama /api/generate endpoint, return the raw response text.

    think: True/False forces the ollama ``think`` flag; None = auto — send ``think: false``
    for known thinking models (currently gemma), which otherwise spend the whole num_predict
    budget on reasoning and return an empty response; omitted entirely for other models so
    their payloads are unchanged.
    The Authorization bearer header is sent only when api_key is non-empty (hosted endpoints
    need it; self-hosted ollama ignores it).
    Pure stdlib (urllib) so every stage runs in a bare venv.
    """
    payload: Dict[str, Any] = {"model": model, "stream": False, "prompt": prompt}
    if options:
        payload["options"] = options
    if think is None and "gemma" in model.lower():
        think = False
    if think is not None:
        payload["think"] = think
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")
