"""Chunk-level → PMID-level aggregation for retrieval evaluation.

When the corpus contains multiple chunks per paper (e.g. abstract + body chunks),
docnos look like ``<pmid>#abstract`` or ``<pmid>#body_007``. Gold relevance is
keyed at the paper level (PMID), so eval needs to collapse chunks of the same
PMID to a single per-paper score (max-pool) before computing recall / MAP / MRR.

This module is the standalone helper for that aggregation, used by
``retrieval_eval.common`` and any caller that wants to evaluate at PMID level
without changing the retrieval pipeline itself.

Backward compatibility: when docnos are already bare PMIDs (no ``#``),
:func:`docno_to_pmid` returns them unchanged and :func:`max_pool_by_pmid` is
the identity function (each PMID has one row already).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def docno_to_pmid(docno: str) -> str:
    """Strip the chunk suffix from a docno.

    ``"12345#body_007"`` → ``"12345"``
    ``"12345#abstract"`` → ``"12345"``
    ``"12345"``          → ``"12345"``  (legacy abstracts-as-pmid case)
    """
    s = str(docno)
    hash_pos = s.find("#")
    return s if hash_pos < 0 else s[:hash_pos]


def max_pool_by_pmid(
    res_df: pd.DataFrame,
    qid_col: str = "qid",
    docno_col: str = "docno",
    score_col: str = "score",
) -> pd.DataFrame:
    """Collapse chunk-level results to PMID level by max score.

    For each (qid, pmid) group, keep the highest-scoring chunk's score.
    Output columns: qid, docno (= pmid), score, rank. Sorted by (qid, score desc);
    rank is recomputed within each qid starting at 0.

    The output uses ``docno`` as the column name for the PMID, so it can be
    consumed by existing eval helpers (``run_df_to_run_map``) without further
    column renaming.
    """
    if res_df.empty:
        return res_df.assign(**{docno_col: pd.Series(dtype=str)}).iloc[0:0]

    df = res_df.copy()
    df["__pmid"] = df[docno_col].map(docno_to_pmid)

    if score_col in df.columns:
        idx = df.groupby([qid_col, "__pmid"])[score_col].idxmax()
        agg = df.loc[idx, [qid_col, "__pmid", score_col]].copy()
        agg = agg.sort_values([qid_col, score_col], ascending=[True, False])
    else:
        # No score column: dedupe by (qid, pmid) preserving input order
        agg = df.drop_duplicates(subset=[qid_col, "__pmid"], keep="first")[[qid_col, "__pmid"]].copy()

    agg = agg.rename(columns={"__pmid": docno_col})
    agg["rank"] = agg.groupby(qid_col).cumcount()
    return agg.reset_index(drop=True)


def dedupe_run_map_by_pmid(run_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Convert a chunk-level run_map to a PMID-level run_map.

    Preserves order: the first occurrence of each PMID wins. Combined with
    score-sorted input (the convention everywhere in this codebase), this is
    equivalent to picking the highest-scoring chunk per PMID.
    """
    out: Dict[str, List[str]] = {}
    for qid, ranked in run_map.items():
        seen: set[str] = set()
        unique: list[str] = []
        for d in ranked:
            p = docno_to_pmid(d)
            if p not in seen:
                seen.add(p)
                unique.append(p)
        out[qid] = unique
    return out


def take_top_k_distinct_pmids(
    ranked_docnos: List[str],
    k: int,
    max_chunks_per_pmid: int = 2,
) -> List[str]:
    """Trim a chunk-level ranked list to the top *k* distinct PMIDs, with at
    most *max_chunks_per_pmid* chunks kept per PMID.

    Iterates ranked_docnos in their input order (assumed: best score first)
    and keeps the highest-scoring chunks for each PMID. Stops when *k*
    distinct PMIDs have been seen.

    Returns chunks in their original rank order (interleaved across PMIDs).
    The result has at most ``k * max_chunks_per_pmid`` entries.

    Used by the evidence stage to enforce "top K papers, up to M chunks each"
    semantics on top of chunk-level rerank output. ``max_chunks_per_pmid <= 0``
    means unlimited (keep all chunks of each kept PMID).
    """
    unlimited = max_chunks_per_pmid <= 0
    seen_count: Dict[str, int] = {}
    distinct_pmids: List[str] = []
    out: List[str] = []
    for d in ranked_docnos:
        p = docno_to_pmid(d)
        if p not in seen_count:
            if len(distinct_pmids) >= k:
                # k distinct PMIDs already found; drop further new PMIDs.
                continue
            distinct_pmids.append(p)
            seen_count[p] = 0
        if unlimited or seen_count[p] < max_chunks_per_pmid:
            out.append(d)
            seen_count[p] += 1
    return out
