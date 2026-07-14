#!/usr/bin/env python3
"""Post-hoc sentence-level citation attribution (separate stage; generation stays frozen).

Input is generate_answers' queries_answers.jsonl (rows unchanged, one per question:
ideal_answer blob + answer-level evidence_ids + the contexts/slots the generator read).
Output is the same rows with a neutral ``answer_sentences`` block appended:

    answer_sentences: [{"text": <sentence>, "doc_ids": [<real corpus docid>, ...]}]

doc_ids are REAL corpus ids (slot ids are resolved through each row's contexts[]), in
best-match-first order per sentence. Index remapping / cited-only reference lists are a
wire-format concern and belong in each repo's adapt-out, not here. Everything else in the
row (ideal_answer, evidence_ids, contexts) is passed through untouched.

Attribution requires slot-based contexts (a sentence is matched against the claim/facet
slot text it restates — provenance propagation, not re-retrieval). The stage keys on each
row's stamped ``context_mode``:
    claim               -> lineage  (sentence -> claim slot -> its doc_id)
    claim_facet_summary -> descent  (sentence -> facet summary -> member claims -> doc_ids;
                                     needs --member-claims, the extract_claims cache)
    anything else       -> HARD ERROR (direct/snippet/document contexts carry no slots;
                                     refusing beats silently degraded attribution)

Matching (--rerank):
    rrf (default) — candidates ranked by RRF fusion of the lexical-claim-recall order and
        the bge-m3 cosine order, top-k cited (k = --lineage-max-cites, default 1). For
        descent the fusion applies to the stage-1 FACET pick only (where it is a measured
        significant win); member selection inside a matched facet stays lexical (cosine
        cannot separate members inside a cluster — similarity is not entailment).
    blend — same selection but ranked by (1-alpha)*lex + alpha*cos.
    lex — pure lexical, byte-identical to the banked serializer behavior (threshold gate
        with min-score/min-overlap, argmax floor, --descent-fallback policy). No embedding
        import ever happens on this path; --mock forces it (CI stays dependency-free).

The embedding backend (sentence_transformers, already a subtree dependency) is imported
lazily and only when a rrf/blend run actually has sentences to match, so answer-level /
mock-LLM / CI paths never load torch.

Usage:
  python3 generation/attribute_sentences.py \
      --answers .../queries_answers.jsonl --out .../queries_answers_attributed.jsonl \
      [--attribution auto|trivial|lineage|descent] [--rerank rrf|blend|lex] [--mock] \
      [--member-claims .../claims_cache.jsonl] [--alpha 0.6] \
      [--lineage-min-score 0.5] [--lineage-min-overlap 2] [--lineage-max-cites 0] \
      [--descent-fallback argmax|all|none]
"""
import argparse
import json
import re
import sys
from pathlib import Path

# --- sentence segmentation (simple, deterministic; deliberately not model-based) ---

# Don't split after common abbreviations or single initials.
_ABBREV = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|cf|approx|no|fig|dr|mr|mrs|ms|st|jr|sr|inc|ltd|co|dept"
    r"|est|min|max|u\.s|u\.k|a\.m|p\.m)\.|\b[A-Za-z]\.)$", re.IGNORECASE)
# Sentence boundary: terminal punctuation, whitespace, then an upper/digit/quote opener.
_BOUNDARY = re.compile(r"(?<=[.!?])[\)\"”’]?\s+(?=[\"“‘(]?[A-Z0-9])")
_MIN_SENT_CHARS = 25  # fragments shorter than this merge into the previous sentence


def split_sentences(text):
    """Paragraph-aware regex splitter with abbreviation guard + short-fragment merge."""
    sents = []
    for para in re.split(r"\n+", text):
        para = " ".join(para.split())
        if not para:
            continue
        start = 0
        parts = []
        for m in _BOUNDARY.finditer(para):
            cand = para[start:m.start() + 1].strip()
            if _ABBREV.search(para[:m.start() + 1].rstrip()):
                continue  # abbreviation, not a boundary
            if cand:
                parts.append(cand)
                start = m.end() - 0
        tail = para[start:].strip()
        if tail:
            parts.append(tail)
        for p in parts:
            if sents and len(p) < _MIN_SENT_CHARS:
                sents[-1] = f"{sents[-1]} {p}"
            else:
                sents.append(p)
    return sents


# --- lexical matching primitives ---

_STOP = frozenset("""a an the this that these those and or but nor so yet for of to in on at
by with from as is are was were be been being it its it's they them their there here he she
his her you your we our i me my do does did done have has had having not no can could may
might must shall should will would about into over under out up down off than then when
where which who whom whose what why how also such more most many much some any each both few
per if because while during between within without across after before above below only very
just both either neither via using use used one two per e g i e etc vs""".split())


def _content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 1 and t not in _STOP}


def _claim_recall(stoks, ttoks):
    """(ratio, overlap): how fully the sentence restates a target claim/summary."""
    if not ttoks:
        return (0.0, 0)
    ov = len(stoks & ttoks)
    return (ov / len(ttoks), ov)


def _rrf_order(n, lex_key, cos_key, k_rrf):
    """Indices 0..n-1 ranked by RRF fusion of the lex-key order and the cos-key order."""
    lex_order = sorted(range(n), key=lex_key, reverse=True)
    cos_order = sorted(range(n), key=cos_key, reverse=True)
    score = {}
    for rank, i in enumerate(lex_order):
        score[i] = score.get(i, 0.0) + 1.0 / (k_rrf + rank)
    for rank, i in enumerate(cos_order):
        score[i] = score.get(i, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(range(n), key=lambda i: score[i], reverse=True)


class Embedder:
    """Lazy bge-m3 embedder: torch/sentence_transformers load only on first use."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
        self._cache = {}

    def add(self, texts):
        """Embed and cache any not-yet-seen texts (batched)."""
        todo = sorted({t for t in texts if t and t not in self._cache})
        if not todo:
            return
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vecs = self._model.encode(todo, normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=False)
        self._cache.update(zip(todo, vecs))

    def cos(self, a, b):
        """Cosine of two cached texts; -1.0 when either is empty/unseen."""
        va, vb = self._cache.get(a), self._cache.get(b)
        if va is None or vb is None:
            return -1.0
        return float(va @ vb)


# --- slot -> doc_id resolution (trivial mode / reference expansion) ---

def resolve_references(row):
    """evidence_ids (slot ids) -> ordered, deduped real corpus doc_ids.

    A-trivial cites every source doc behind each evidence slot: a claim slot
    contributes its single doc_id; a facet-summary slot (kind=="summary" /
    n_members>1) contributes its FULL member_doc_ids — the cluster's whole
    provenance. On pure claim-mode input no slot is a facet, so this is identical
    to one-doc-per-evidence-id.
    """
    slots = {c["id"]: c for c in row.get("contexts", [])}
    refs, seen, unresolved = [], set(), []
    for eid in row.get("evidence_ids", []):
        c = slots.get(eid)
        if c is None:
            unresolved.append(eid)
            continue
        if c.get("kind") == "summary" or c.get("n_members", 1) > 1:
            member_docs = c.get("member_doc_ids") or [c["doc_id"]]
        else:
            member_docs = [c["doc_id"]]
        for doc_id in member_docs:
            if doc_id not in seen:
                seen.add(doc_id)
                refs.append(doc_id)
    return refs, unresolved


# --- attribution modes: each returns (per_sentence_doc_lists, unresolved_evidence_ids) ---

def attribute_trivial(sentences, row):
    """A-trivial: every sentence cites all resolved reference docs (the baseline floor)."""
    references, unresolved = resolve_references(row)
    return [list(references) for _ in sentences], unresolved


def attribute_lineage(sentences, row, opts):
    """Claim mode: cite the evidence slot(s) each sentence restates.

    Candidates are the answer's evidence slots only, so cited docids stay a subset of
    A-trivial's (free judging). rerank=lex reproduces the banked threshold+argmax
    behavior exactly; rerank=rrf/blend is the measured selection re-rank (top-k by
    fused lexical+cosine score — fixes spurious word-overlap top picks without
    admitting extra docs).
    """
    slot_doc = {c["id"]: c["doc_id"] for c in row.get("contexts", [])}
    slot_txt = {c["id"]: c.get("text", "") for c in row.get("contexts", [])}
    # candidate docs in evidence order (dedup), each carrying its slots' texts
    cand_docs, doc_slots, unresolved = [], {}, []
    for eid in row.get("evidence_ids", []):
        doc_id = slot_doc.get(eid)
        if doc_id is None:
            unresolved.append(eid)
            continue
        if doc_id not in doc_slots:
            doc_slots[doc_id] = []
            cand_docs.append(doc_id)
        doc_slots[doc_id].append(slot_txt.get(eid, ""))
    doc_toks = {d: [_content_tokens(t) for t in txts] for d, txts in doc_slots.items()}

    if opts.emb is not None:
        opts.emb.add(sentences)
        opts.emb.add(t for txts in doc_slots.values() for t in txts)

    def lex_score(stoks, doc_id):
        best = (0.0, 0)  # (ratio, overlap)
        for toks in doc_toks[doc_id]:
            if toks:
                best = max(best, _claim_recall(stoks, toks))
        return best

    def cos_score(s, doc_id):
        return max((opts.emb.cos(s, t) for t in doc_slots[doc_id]), default=-1.0)

    per_sentence = []
    for s in sentences:
        stoks = _content_tokens(s)
        scored = [(lex_score(stoks, d), d) for d in cand_docs]  # ((ratio,ov), doc)
        if opts.rerank == "lex":
            chosen = [(sc, d) for sc, d in scored
                      if sc[0] >= opts.min_score and sc[1] >= opts.min_overlap]
            if not chosen and scored:
                chosen = [max(scored, key=lambda x: x[0])]  # argmax fallback (top-1)
            chosen.sort(key=lambda x: x[0], reverse=True)
            order = [d for _, d in chosen]
            cap = opts.max_cites  # 0 = unlimited (banked behavior)
        else:
            cos = [cos_score(s, d) for d in cand_docs]
            if opts.rerank == "rrf":
                idx = _rrf_order(len(cand_docs), lambda i: scored[i][0],
                                 lambda i: cos[i], opts.k_rrf)
            else:  # blend
                idx = sorted(range(len(cand_docs)),
                             key=lambda i: (1 - opts.alpha) * scored[i][0][0]
                             + opts.alpha * cos[i], reverse=True)
            order = [cand_docs[i] for i in idx]
            cap = opts.max_cites or 1  # re-rank cites exactly top-k (default 1)
        docs = []
        for d in order:
            if d not in docs:
                docs.append(d)
            if cap and len(docs) >= cap:
                break
        per_sentence.append(docs)
    return per_sentence, unresolved


def load_member_claims(path):
    """extract_claims cache -> {(qid, context_id): {"doc_id", "claims"}} for descent."""
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cache[(str(r["qid"]), r["context_id"])] = {
                "doc_id": r["doc_id"], "claims": r.get("claims", [])}
    return cache


def attribute_descent(sentences, row, member_cache, opts):
    """Facet mode: two-stage member descent.

    Per sentence: (1) select the evidence facet(s) it restates — lexical gate on the
    facet's summary text; when no facet clears the gate, the floor pick uses the
    --rerank matcher (rrf/blend fuse cosine into the facet pick, the measured win;
    lex is the banked argmax). (2) within each matched facet, score the sentence
    against each member context's claim list and cite the member doc(s) that clear
    the gate — member selection is ALWAYS lexical. rerank=lex keeps the banked
    --descent-fallback policy; rrf/blend pool matched facets' members and cite the
    lexical top-k. Cited docids are a subset of the facets' member docids (free join).
    """
    qid = str(row.get("query_id"))
    unresolved = []
    slots = {c["id"]: c for c in row.get("contexts", [])}
    facets = []  # [{"summary", "summary_tok", "members": [(doc_id, claim_toks_list)]}]
    for eid in row.get("evidence_ids", []):
        c = slots.get(eid)
        if c is None:
            unresolved.append(eid)
            continue
        if c.get("kind") == "summary" or c.get("n_members", 1) > 1:
            member_ids = c.get("member_ids") or [c["id"]]
            member_docs = c.get("member_doc_ids") or [c["doc_id"]]
        else:  # singleton claim slot: it is its own single member
            member_ids, member_docs = [c["id"]], [c["doc_id"]]
        members = []
        for mid, mdoc in zip(member_ids, member_docs):
            ent = member_cache.get((qid, mid))
            claim_toks = [_content_tokens(cl) for cl in ent["claims"]] if ent else []
            # fall back to the summary text if a member's claims are unavailable
            if not claim_toks:
                claim_toks = [_content_tokens(c.get("text", ""))]
            members.append((mdoc, claim_toks))
        facets.append({"summary": c.get("text", ""),
                       "summary_tok": _content_tokens(c.get("text", "")),
                       "members": members})

    if opts.emb is not None:
        opts.emb.add(sentences)
        opts.emb.add(f["summary"] for f in facets)

    per_sentence = []
    for s in sentences:
        stoks = _content_tokens(s)
        # stage 1: facets the sentence restates (lexical gate; matcher-ranked floor)
        f_lex = [_claim_recall(stoks, f["summary_tok"]) for f in facets]
        matched = [f for f, sc in zip(facets, f_lex)
                   if sc[0] >= opts.min_score and sc[1] >= opts.min_overlap]
        if not matched and facets:
            if opts.rerank == "lex":
                best = max(range(len(facets)), key=lambda i: f_lex[i])
            else:
                f_cos = [opts.emb.cos(s, f["summary"]) for f in facets]
                if opts.rerank == "rrf":
                    best = _rrf_order(len(facets), lambda i: f_lex[i],
                                      lambda i: f_cos[i], opts.k_rrf)[0]
                else:  # blend
                    best = max(range(len(facets)),
                               key=lambda i: (1 - opts.alpha) * f_lex[i][0]
                               + opts.alpha * f_cos[i])
            matched = [facets[best]]
        # stage 2: within each matched facet, cite the member doc(s) the sentence
        # restates (lexical always — cosine cannot separate members inside a cluster)
        m_scored = []  # ((ratio, overlap), doc_id) pooled over matched facets
        chosen = []
        for f in matched:
            f_members = []
            for mdoc, claim_toks in f["members"]:
                best = max((_claim_recall(stoks, t) for t in claim_toks),
                           default=(0.0, 0))
                f_members.append((best, mdoc))
            m_scored.extend(f_members)
            if opts.rerank == "lex":
                hits = [(sc, d) for sc, d in f_members
                        if sc[0] >= opts.min_score and sc[1] >= opts.min_overlap]
                if not hits and f_members:
                    if opts.fallback == "argmax":
                        hits = [max(f_members, key=lambda x: x[0])]  # best member floor
                    elif opts.fallback == "all":
                        hits = f_members                             # whole cluster floor
                    # fallback == "none": leave hits empty (sentence uncited here)
                chosen.extend(hits)
        if opts.rerank == "lex":
            chosen.sort(key=lambda x: x[0], reverse=True)
            cap = opts.max_cites  # 0 = unlimited (banked behavior)
        else:
            chosen = sorted(m_scored, key=lambda x: x[0], reverse=True)
            cap = opts.max_cites or 1  # re-rank cites exactly top-k (default 1)
        docs = []
        for _, d in chosen:
            if d not in docs:
                docs.append(d)
            if cap and len(docs) >= cap:
                break
        per_sentence.append(docs)
    return per_sentence, unresolved


# --- stage driver ---

_MODE_TO_ATTRIBUTION = {"claim": "lineage", "claim_facet_summary": "descent"}


class Opts:
    def __init__(self, args, emb):
        self.rerank = args.rerank
        self.alpha = args.alpha
        self.k_rrf = args.k_rrf
        self.min_score = args.lineage_min_score
        self.min_overlap = args.lineage_min_overlap
        self.max_cites = args.lineage_max_cites
        self.fallback = args.descent_fallback
        self.emb = emb


def attribute_row(row, args, member_cache, emb):
    """-> (answer_sentences, attribution_mode, unresolved). Raises on non-slot rows."""
    mode = row.get("context_mode")
    if mode not in _MODE_TO_ATTRIBUTION:
        raise SystemExit(
            f"ERROR {row.get('query_id')}: context_mode={mode!r} carries no claim/facet "
            f"slots — sentence attribution requires GENERATION_MODE=claims|facets "
            f"(direct/snippet/document contexts are refused, not degraded)")
    attribution = args.attribution
    if attribution == "auto":
        attribution = _MODE_TO_ATTRIBUTION[mode]
    sentences = split_sentences(row["ideal_answer"])
    opts = Opts(args, emb)
    if attribution == "trivial":
        per_sentence, unresolved = attribute_trivial(sentences, row)
    elif attribution == "lineage":
        per_sentence, unresolved = attribute_lineage(sentences, row, opts)
    else:  # descent
        if member_cache is None:
            raise SystemExit(
                f"ERROR {row.get('query_id')}: descent attribution (context_mode="
                f"claim_facet_summary) requires --member-claims (extract_claims cache)")
        per_sentence, unresolved = attribute_descent(sentences, row, member_cache, opts)
    answer_sentences = [{"text": s, "doc_ids": docs}
                        for s, docs in zip(sentences, per_sentence)]
    return answer_sentences, attribution, unresolved


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", type=Path, required=True,
                    help="queries_answers.jsonl (generation output)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSONL: same rows + answer_sentences")
    ap.add_argument("--attribution", default="auto",
                    choices=["auto", "trivial", "lineage", "descent"],
                    help="auto (default) keys on each row's context_mode: "
                         "claim->lineage, claim_facet_summary->descent")
    ap.add_argument("--rerank", default="rrf", choices=["rrf", "blend", "lex"],
                    help="candidate matcher: rrf (default) / blend fuse bge-m3 cosine "
                         "into the pick (top-k); lex = banked lexical threshold behavior")
    ap.add_argument("--mock", action="store_true",
                    help="lexical-only (forces --rerank lex; never imports the "
                         "embedding backend — for CI / dependency-free runs)")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="blend weight: (1-alpha)*lex + alpha*cos (validated band .5-.7)")
    ap.add_argument("--k-rrf", type=int, default=60, help="RRF constant")
    ap.add_argument("--emb-model", default="BAAI/bge-m3",
                    help="embedding model for rrf/blend matching")
    ap.add_argument("--member-claims", type=Path, default=None,
                    help="descent: extract_claims cache (qid+context_id -> claims) for "
                         "the sentence->member matching hop")
    ap.add_argument("--descent-fallback", default="argmax",
                    choices=["argmax", "all", "none"],
                    help="lex descent: within-facet policy when no member clears "
                         "threshold (argmax=best member, all=whole cluster, none=uncited)")
    ap.add_argument("--lineage-min-score", type=float, default=0.5,
                    help="lexical gate: min claim-recall ratio")
    ap.add_argument("--lineage-min-overlap", type=int, default=2,
                    help="lexical gate: min shared content words")
    ap.add_argument("--lineage-max-cites", type=int, default=0,
                    help="citations per sentence: cap for lex (0 = unlimited), "
                         "exact top-k for rrf/blend (0 -> 1)")
    args = ap.parse_args()
    if args.mock:
        args.rerank = "lex"

    member_cache = load_member_claims(args.member_claims) if args.member_claims else None
    emb = None if args.rerank == "lex" else Embedder(args.emb_model)

    rows_out, skipped, n_sent, n_cites, n_unresolved = [], [], 0, 0, 0
    modes_used = set()
    with open(args.answers, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not (row.get("ideal_answer") or "").strip():
                skipped.append(str(row.get("query_id")))
                rows_out.append(row)  # pass through unattributed
                continue
            answer_sentences, attribution, unresolved = attribute_row(
                row, args, member_cache, emb)
            modes_used.add(attribution)
            n_unresolved += len(unresolved)
            if unresolved:
                print(f"WARN {row['query_id']}: {len(unresolved)} unresolved evidence "
                      f"ids (first: {unresolved[:2]})", file=sys.stderr)
            row["answer_sentences"] = answer_sentences
            n_sent += len(answer_sentences)
            n_cites += sum(len(s["doc_ids"]) for s in answer_sentences)
            rows_out.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows_out)} rows -> {args.out}")
    print(f"  sentences={n_sent} citations={n_cites} "
          f"cites/sent={n_cites / n_sent if n_sent else 0:.2f} "
          f"attribution={'+'.join(sorted(modes_used)) or 'none'} rerank={args.rerank} "
          f"unresolved_evidence_ids={n_unresolved}")
    if skipped:
        print(f"  SKIPPED (passed through, empty ideal_answer): {skipped}")


if __name__ == "__main__":
    main()
