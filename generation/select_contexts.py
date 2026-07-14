#!/usr/bin/env python3
"""Select the generation slot budget from the all-facet slot file (facets mode, stage 3).

Reads summarize_facets.py's output (ALL facet slots per query, each a summary or a singleton
claim) and keeps the top-N slots ranked by **slot-text hybrid RRF**: reciprocal-rank fusion
(equal weight, k=--rrf-k) of a semantic leg — bi-encoder cosine between the slot text and the
query — and a lexical leg — BM25 over the slot texts. Both legs read the actual slot text
(the summary), not scores inherited from the source snippets; BM25Okapi's length normalization
handles the summary-length spread. This was the winning selection signal; alternatives ranked
on inherited cross-encoder scores are not carried (see git history).

Output schema is unchanged (consumable by generate_answers.py). Requires sentence-transformers
and rank_bm25.

Env defaults (flags win): GENERATION_SELECT_N, GENERATION_SELECT_RRF_K,
GENERATION_FACET_EMB_MODEL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_common import EMB_MODEL, env_first  # noqa: E402


def toks(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _ranks(order):
    return {i: r for r, i in enumerate(order)}


def hybrid_rrf_order(slots, qtext, k, st_model):
    """RRF of bi-encoder cosine(slot text, query) and BM25(slot texts) ranks, 1:1."""
    n = len(slots)
    if n == 0:  # query yielded no facet slots (e.g. extraction found no claims)
        return []
    emb = st_model.encode([s["text"] for s in slots] + [qtext], normalize_embeddings=True)
    cos = emb[:-1] @ emb[-1]
    sem_rank = _ranks(sorted(range(n), key=lambda i: float(cos[i]), reverse=True))
    bm = BM25Okapi([toks(s["text"]) for s in slots])
    scores = bm.get_scores(toks(qtext))
    bm_rank = _ranks(sorted(range(n), key=lambda i: scores[i], reverse=True))
    rrf = {i: 1.0 / (k + sem_rank[i]) + 1.0 / (k + bm_rank[i]) for i in range(n)}
    return sorted(range(n), key=lambda i: rrf[i], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", type=Path, required=True,
                    help="all-facet slot file from summarize_facets.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int,
                    default=int(env_first("GENERATION_SELECT_N", default="16")))
    ap.add_argument("--rrf-k", "--k", dest="k", type=int,
                    default=int(env_first("GENERATION_SELECT_RRF_K", default="60")))
    ap.add_argument("--emb-model",
                    default=env_first("GENERATION_FACET_EMB_MODEL", default=EMB_MODEL))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.emb_model)

    import statistics as st
    W = []
    overlap = []
    empty_qids = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for l in args.full.open():
            r = json.loads(l)
            slots = r.get("contexts") or []
            if not slots:
                # no facet slots (extraction found nothing) -> keep the query with an
                # empty context set rather than crashing BM25; surface it so it is not
                # silently dropped.
                empty_qids.append(str(r.get("query_id", "?")))
            ce_top = list(range(min(args.n, len(slots))))   # full file is already CE-desc
            keep_idx = hybrid_rrf_order(slots, r.get("query_text", ""), args.k, st_model)[:args.n]
            overlap.append(len(set(keep_idx) & set(ce_top)))
            keep = [slots[i] for i in keep_idx]
            W.append(sum(len((s.get("text") or "").split()) for s in keep))
            r["contexts"] = keep
            r["doc_ids"] = sorted({s["doc_id"] for s in keep})
            fh.write(json.dumps(r) + "\n")
    print(f"[select_contexts] N={args.n} -> {args.out}")
    if empty_qids:
        import sys
        print(f"[select_contexts] WARNING: {len(empty_qids)} query(ies) had 0 facet "
              f"slots (no claims extracted) -> emitted with empty contexts: "
              f"{', '.join(empty_qids[:10])}{' ...' if len(empty_qids) > 10 else ''}",
              file=sys.stderr)
    print(f"  words/q mean {st.mean(W):.0f} median {st.median(W):.0f} max {max(W)}")
    if overlap:
        print(f"  hybrid-RRF vs CE-order selection overlap: mean {st.mean(overlap):.1f}/{args.n} "
              f"(swaps {args.n-st.mean(overlap):.1f} slots/q vs CE-only)")


if __name__ == "__main__":
    main()
