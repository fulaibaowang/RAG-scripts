#!/usr/bin/env python3
"""Distil cached claims into a generation-budget contexts JSONL (claims mode, stage 2).

Reads the claim cache written by extract_claims.py, exact-dedups claims per query, then fills
the slot budget by CE-ordered round-robin across source contexts (highest-scoring source
contributes first; every source is represented before any source repeats). Emits a contexts
JSONL that generate_answers.py consumes exactly like an undistilled one — each context is now
a single claim whose ``id`` is its source context id (citation lineage preserved).

Stops at a FIXED slot count when --slots is set (deterministic, budget-matched); otherwise at
the word budget. No LLM calls; runs with the standard library only.

Env defaults (flags win): GENERATION_EXTRACT_TOP_N, GENERATION_DISTIL_SLOTS,
GENERATION_DISTIL_BUDGET_WORDS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_common import env_first, norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True,
                    help="generation contexts JSONL (query text source; cache has no qtext)")
    ap.add_argument("--cache", type=Path, required=True,
                    help="claim cache JSONL from extract_claims.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-n", type=int,
                    default=int(env_first("GENERATION_EXTRACT_TOP_N", default="30")))
    _slots_env = env_first("GENERATION_DISTIL_SLOTS")
    ap.add_argument("--slots", type=int, default=int(_slots_env) if _slots_env else None,
                    help="FIXED claim-slot count (deterministic top-N, importance-ranked via "
                    "CE-ordered round-robin across sources). When set, overrides --budget-words.")
    ap.add_argument("--budget-words", type=int,
                    default=int(env_first("GENERATION_DISTIL_BUDGET_WORDS", default="944")),
                    help="word budget (used only when --slots is unset)")
    args = ap.parse_args()

    # query text from the evidence file (cache has no qtext)
    qtext: Dict[str, str] = {}
    for line in args.evidence.open():
        rec = json.loads(line)
        qtext[str(rec["query_id"])] = rec.get("query_text", "")

    by_qid: Dict[str, List[dict]] = {}
    for line in args.cache.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["rank"] >= args.top_n:
            continue
        by_qid.setdefault(rec["qid"], []).append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with args.out.open("w") as fh:
        for qid in sorted(by_qid):
            ctxs = by_qid[qid]
            # one "source" per cached context, CE desc (NaN last); claims keep listed order
            def ce(r):
                v = r.get("ce_score")
                return v if isinstance(v, (int, float)) and v == v else -1e9
            ctxs.sort(key=ce, reverse=True)
            queues = []  # list of (source_ctx, claim-list)
            seen: set[str] = set()  # exact-normalized dedup across the whole query
            for r in ctxs:
                cl = []
                for c in r.get("claims") or []:
                    k = norm(c)
                    if k and k not in seen:
                        seen.add(k)
                        cl.append(c)
                if cl:
                    queues.append([r, cl])
            # CE-ordered round-robin across sources, capped by slot count or word budget.
            picked: List[dict] = []
            words = 0
            cap_slots = args.slots if args.slots else 10 ** 9
            cap_words = args.budget_words if not args.slots else 10 ** 9
            active = list(range(len(queues)))
            while active and len(picked) < cap_slots and words < cap_words:
                nxt = []
                for idx in active:
                    if len(picked) >= cap_slots or words >= cap_words:
                        break
                    r, cl = queues[idx]
                    claim = cl.pop(0)
                    picked.append({
                        "id": r["context_id"], "doc_id": r["doc_id"],
                        "text": claim, "source_rank": r["rank"],
                        "ce_score": r.get("ce_score"),
                    })
                    words += len(claim.split())
                    if cl:
                        nxt.append(idx)
                active = nxt
            fh.write(json.dumps({
                "query_id": qid, "query_text": qtext.get(qid, ""),
                "context_mode": "claim",
                "doc_ids": sorted({p["doc_id"] for p in picked}),
                "contexts": picked,
            }) + "\n")
            stats.append((qid, len(ctxs), len(seen), len(picked), words,
                          len({p["id"] for p in picked})))
    import statistics as st
    print(f"[distil] {len(stats)} queries -> {args.out}")
    print(f"  sources/q (top-{args.top_n}): median {int(st.median(s[1] for s in stats))}")
    print(f"  distinct claims/q after dedup: median {int(st.median(s[2] for s in stats))}")
    cap_desc = f"fixed {args.slots} slots" if args.slots else f"within {args.budget_words}w"
    print(f"  claims emitted/q ({cap_desc}): median {int(st.median(s[3] for s in stats))} "
          f"(min {min(s[3] for s in stats)}, max {max(s[3] for s in stats)})")
    print(f"  prompt words/q: median {int(st.median(s[4] for s in stats))}")
    print(f"  distinct source contexts represented/q: median {int(st.median(s[5] for s in stats))}")


if __name__ == "__main__":
    main()
