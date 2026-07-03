#!/usr/bin/env python3
"""Cluster cached claims into facets and summarize each multi-claim facet (facets mode, stage 2).

Reads the claim cache written by extract_claims.py, dedups per query, clusters the claims into
facets (bi-encoder embeddings + agglomerative clustering at --dist-thr), then has an LLM
combine each facet with >= --min-cluster claims into ONE dense statement that preserves every
distinct member fact. Singleton facets are emitted verbatim (a summary of one claim is a wasted
call). Emits ALL facet slots per query, ordered by the facet's max cross-encoder score — feed
this to select_contexts.py to pick the generation slot budget.

Slot ``id`` = the highest-CE member's source context id (corpus-resolvable); ``member_ids``
carries the full citation lineage. Summaries are cached (resume-safe) keyed by qid + an
order-independent hash of the member claims, so reruns and threshold sweeps only pay for new
facets. Requires sentence-transformers + scikit-learn.

Env defaults (flags win): GENERATION_EXTRACT_TOP_N, GENERATION_FACET_DIST_THR,
GENERATION_FACET_MIN_CLUSTER, GENERATION_FACET_EMB_MODEL, GENERATION_SUMMARY_MODEL (falls back
to GENERATION_EXTRACT_MODEL then GENERATION_MODEL), GENERATION_SUMMARY_URL (falls back to
GENERATION_EXTRACT_URL then OLLAMA_URL), GENERATION_SUMMARY_NUM_CTX, GENERATION_SUMMARY_TIMEOUT,
GENERATION_SUMMARY_CONCURRENCY, GENERATION_SUMMARY_THINK, LLAMA_API_KEY (bearer, optional).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_common import (  # noqa: E402
    EMB_MODEL, call_ollama, cluster_labels, deduped_claims, embed_texts, env_first,
    load_qtext, member_hash, strip_preamble,
)

DEFAULT_OLLAMA_URL = "https://chat.fri.uni-lj.si/ollama/api/generate"  # same default as generate_answers.py

SUMMARY_PROMPT = (
    "Combine the following claims into ONE concise statement that preserves every distinct fact. "
    "Do not add any information that is not in the claims, and do not omit any distinct fact.\n\n"
    "Claims:\n{claims}\n\nCombined statement:"
)


def _parse_think(v: Optional[str]) -> Optional[bool]:
    if v is None or v == "":
        return None
    return v.strip().lower() not in ("0", "false", "no", "off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True,
                    help="generation contexts JSONL (query text source)")
    ap.add_argument("--cache", type=Path, required=True,
                    help="claim cache JSONL from extract_claims.py")
    ap.add_argument("--out", type=Path, required=True,
                    help="all-facet slot file (input to select_contexts.py)")
    ap.add_argument("--summary-cache", type=Path, required=True,
                    help="jsonl cache of facet summaries (key = qid + member hash)")
    ap.add_argument("--top-n", type=int,
                    default=int(env_first("GENERATION_EXTRACT_TOP_N", default="30")))
    ap.add_argument("--dist-thr", type=float,
                    default=float(env_first("GENERATION_FACET_DIST_THR", default="0.4")))
    ap.add_argument("--min-cluster", type=int,
                    default=int(env_first("GENERATION_FACET_MIN_CLUSTER", default="2")),
                    help="summarize facets with >= this many claims; emit smaller verbatim")
    ap.add_argument("--emb-model",
                    default=env_first("GENERATION_FACET_EMB_MODEL", default=EMB_MODEL))
    ap.add_argument("--emb-cache", type=Path, default=None)
    ap.add_argument("--ollama-url",
                    default=env_first("GENERATION_SUMMARY_URL", "GENERATION_EXTRACT_URL",
                                      "OLLAMA_URL", default=DEFAULT_OLLAMA_URL))
    ap.add_argument("--model",
                    default=env_first("GENERATION_SUMMARY_MODEL", "GENERATION_EXTRACT_MODEL",
                                      "GENERATION_MODEL", default="llama3.3:latest"))
    ap.add_argument("--num-ctx", type=int,
                    default=int(env_first("GENERATION_SUMMARY_NUM_CTX", default="8192")))
    ap.add_argument("--timeout", type=int,
                    default=int(env_first("GENERATION_SUMMARY_TIMEOUT", default="600")))
    ap.add_argument("--concurrency", type=int,
                    default=int(env_first("GENERATION_SUMMARY_CONCURRENCY", default="3")))
    args = ap.parse_args()

    api_key = os.environ.get("LLAMA_API_KEY", "")
    think = _parse_think(os.environ.get("GENERATION_SUMMARY_THINK"))

    qtext = load_qtext(args.evidence)
    claims_by_qid = deduped_claims(args.cache, args.top_n)
    qids = sorted(claims_by_qid)

    # embed all claims once (flat); npz cache is validated by row count
    flat = [c["text"] for q in qids for c in claims_by_qid[q]]
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.emb_model)
    emb = embed_texts(flat, emb_cache=args.emb_cache, st_model=st_model)

    # ---- build per-query facets; collect multi-claim facets to summarize ----
    off = 0
    facets_by_qid: Dict[str, List[List[dict]]] = {}   # qid -> list of facets (list of claim dicts)
    for qid in qids:
        claims = claims_by_qid[qid]
        n = len(claims)
        qemb = emb[off:off + n]
        off += n
        labels = cluster_labels(qemb, args.dist_thr) if n else np.array([], dtype=int)
        groups: Dict[int, List[dict]] = {}
        for c, lab in zip(claims, labels):
            groups.setdefault(int(lab), []).append(c)
        facets_by_qid[qid] = list(groups.values())

    # ---- summary cache (resume-safe) ----
    args.summary_cache.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, str] = {}
    if args.summary_cache.exists():
        for line in args.summary_cache.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[r["key"]] = r["summary"]
    print(f"  summary cache: {len(cache)} preexisting", flush=True)

    # tasks = (qid, facet_idx, member_texts, key) for every multi-claim facet missing from cache
    tasks = []
    for qid in qids:
        for fi, fac in enumerate(facets_by_qid[qid]):
            if len(fac) >= args.min_cluster:
                key = f"{qid}:{member_hash([c['text'] for c in fac])}"
                if key not in cache:
                    tasks.append((qid, fi, [c["text"] for c in fac], key))
    print(f"  {len(tasks)} multi-claim facets to summarize (>= {args.min_cluster} claims)", flush=True)

    lock = threading.Lock()
    done = [0]
    cache_fh = args.summary_cache.open("a")

    def work(task):
        qid, fi, texts, key = task
        prompt = SUMMARY_PROMPT.format(claims="\n".join(f"- {c}" for c in texts))
        for attempt in range(3):
            try:
                s = call_ollama(
                    args.ollama_url, args.model, prompt,
                    options={"temperature": 0, "num_ctx": args.num_ctx, "num_predict": 256},
                    timeout=args.timeout, think=think, api_key=api_key).strip()
                if s:
                    with lock:
                        cache[key] = s
                        cache_fh.write(json.dumps({"key": key, "qid": qid, "n_members": len(texts),
                                                   "summary": s}) + "\n")
                        cache_fh.flush()
                        done[0] += 1
                        print(f"    [{done[0]}/{len(tasks)}] {qid} facet{fi} "
                              f"{len(texts)}claims -> {len(s.split())}w", flush=True)
                    return
            except Exception as e:
                if attempt == 2:
                    with lock:
                        print(f"    FAIL {qid} facet{fi}: {e}", flush=True)
                time.sleep(15)

    if tasks:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(work, tasks))
    cache_fh.close()

    # ---- emit slots (singleton -> claim, multi -> summary) in facet order (max-CE desc) ----
    def ce(c):
        v = c.get("ce_score")
        return v if isinstance(v, (int, float)) and v == v else -1e9

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summ_lens: List[int] = []
    claim_lens: List[int] = []
    member_counts: List[int] = []
    fidelity_rows = []   # (qid, n_members, mean_cov, min_cov)
    missing = 0
    with args.out.open("w") as fh:
        for qid in qids:
            facs = facets_by_qid[qid]
            facs_sorted = sorted(facs, key=lambda f: max(ce(c) for c in f), reverse=True)
            slots = []
            for fac in facs_sorted:
                fac_sorted = sorted(fac, key=ce, reverse=True)
                lead = fac_sorted[0]
                if len(fac) < args.min_cluster:
                    slots.append({
                        "id": lead["id"], "doc_id": lead["doc_id"], "text": lead["text"],
                        "kind": "claim", "n_members": 1, "ce_score": lead.get("ce_score"),
                        "member_ids": [lead["id"]],
                    })
                    claim_lens.append(len(lead["text"].split()))
                else:
                    key = f"{qid}:{member_hash([c['text'] for c in fac])}"
                    summary = cache.get(key)
                    if not summary:
                        missing += 1
                        summary = " ".join(c["text"] for c in fac_sorted)  # fallback: concat
                    summary = strip_preamble(summary)
                    slots.append({
                        "id": lead["id"], "doc_id": lead["doc_id"], "text": summary,
                        "kind": "summary", "n_members": len(fac), "ce_score": lead.get("ce_score"),
                        "member_ids": [c["id"] for c in fac_sorted],
                        "member_doc_ids": sorted({c["doc_id"] for c in fac_sorted}),
                    })
                    summ_lens.append(len(summary.split()))
                    member_counts.append(len(fac))
                    # fidelity: cosine(summary, each member claim) via the bi-encoder
                    semb = st_model.encode([summary], normalize_embeddings=True)[0]
                    memb = st_model.encode([c["text"] for c in fac_sorted], normalize_embeddings=True)
                    cov = memb @ semb
                    fidelity_rows.append((qid, len(fac), float(cov.mean()), float(cov.min())))
            fh.write(json.dumps({
                "query_id": qid, "query_text": qtext.get(qid, ""),
                "context_mode": "claim_facet_summary",
                "doc_ids": sorted({s["doc_id"] for s in slots}),
                "contexts": slots,
            }) + "\n")

    # ---- report ----
    import statistics as stx
    def pct(xs, p):
        xs = sorted(xs)
        return xs[int(p / 100 * (len(xs) - 1))] if xs else 0
    print(f"\n[summarize_facets] {len(qids)} queries -> {args.out}", flush=True)
    if missing:
        print(f"  WARN {missing} multi-claim facets had no cached summary (concat fallback)", flush=True)
    print(f"  singletons (verbatim claim slots): {len(claim_lens)}  "
          f"len words median {int(stx.median(claim_lens)) if claim_lens else 0}", flush=True)
    print(f"  multi-claim facets (summarized): {len(summ_lens)}  "
          f"member-claims median {int(stx.median(member_counts)) if member_counts else 0} "
          f"(max {max(member_counts) if member_counts else 0})", flush=True)
    if summ_lens:
        print(f"  SUMMARY LENGTH (words): median {int(stx.median(summ_lens))} "
              f"mean {stx.mean(summ_lens):.1f} p10 {pct(summ_lens,10)} p90 {pct(summ_lens,90)} "
              f"max {max(summ_lens)}", flush=True)
        print(f"    vs claim length: claim median {int(stx.median(claim_lens)) if claim_lens else 0}w "
              f"-> summary median {int(stx.median(summ_lens))}w "
              f"(ratio {stx.median(summ_lens)/max(1,stx.median(claim_lens)):.2f}x)", flush=True)
    if fidelity_rows:
        mean_covs = [r[2] for r in fidelity_rows]
        min_covs = [r[3] for r in fidelity_rows]
        print(f"  FIDELITY (cosine summary<->member claim, bi-encoder):", flush=True)
        print(f"    mean coverage per facet: median {stx.median(mean_covs):.3f} "
              f"p10 {pct(mean_covs,10):.3f}  (1.0 = summary captures every member)", flush=True)
        print(f"    WORST member coverage per facet: median {stx.median(min_covs):.3f} "
              f"p10 {pct(min_covs,10):.3f}  (low p10 = some distinct claim dropped)", flush=True)
        total_summ_words = sum(summ_lens)
        total_singleton_words = sum(claim_lens)
        print(f"  INPUT BUDGET: summaries {total_summ_words}w + "
              f"singletons {total_singleton_words}w over {len(qids)}q "
              f"= {(total_summ_words+total_singleton_words)/len(qids):.0f}w/q input", flush=True)


if __name__ == "__main__":
    main()
