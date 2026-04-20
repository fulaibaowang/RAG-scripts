"""Max-pool CE snippet windows and select top disjoint windows (shared by post_rerank + build_contexts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Post-rerank JSONL field: doc_id -> list of {window_idx, ce_score, optional query_field}
DOC_SNIPPET_WINDOWS_KEY = "doc_snippet_windows"

# When ce_score ties, prefer lexicographically smaller query_field; missing sorts last.
_QF_TIE_SENTINEL = "\uffff"


def _better_max_candidate(
    score: float,
    qf: Optional[str],
    best_score: float,
    best_qf: Optional[str],
) -> bool:
    if score > best_score:
        return True
    if score < best_score:
        return False
    a = qf if qf is not None else _QF_TIE_SENTINEL
    b = best_qf if best_qf is not None else _QF_TIE_SENTINEL
    return a < b


def sent_ids_for_window(window_idx: int, window_size: int) -> List[int]:
    return [window_idx + j for j in range(window_size)]


def load_by_pair_from_windows_jsonl(
    windows_path: Path,
) -> Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]]:
    """Stream window JSONL, max-pool CE per (qid, docno, window_idx), return sorted scored lists per (qid, docno)."""
    max_by_triple: Dict[Tuple[str, str, int], Tuple[float, Optional[str]]] = {}
    with open(windows_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("qid")
            docno = obj.get("docno")
            if qid is None or docno is None:
                continue
            try:
                wi = int(obj.get("window_idx", 0))
            except (TypeError, ValueError):
                continue
            try:
                ce_score = float(obj.get("ce_score", 0.0))
            except (TypeError, ValueError):
                ce_score = 0.0
            qf_raw = obj.get("query_field", None)
            qf: Optional[str] = str(qf_raw) if qf_raw is not None and qf_raw != "" else None

            key3 = (str(qid), str(docno), wi)
            if key3 not in max_by_triple:
                max_by_triple[key3] = (ce_score, qf)
            else:
                bs, bqf = max_by_triple[key3]
                if _better_max_candidate(ce_score, qf, bs, bqf):
                    max_by_triple[key3] = (ce_score, qf)

    by_pair: Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]] = {}
    for (qid, docno, wi), (sc, qf) in max_by_triple.items():
        k2 = (qid, docno)
        by_pair.setdefault(k2, []).append((sc, wi, qf))
    for k2 in by_pair:
        by_pair[k2].sort(key=lambda x: (-x[0], x[1]))
    return by_pair


def by_pair_from_question_embedded(q: dict) -> Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]]:
    """Build the same by_pair slice as load_by_pair_from_windows_jsonl, from one question's embedded windows."""
    qid = q.get("id")
    if qid is None:
        return {}
    qid_s = str(qid)
    raw = q.get(DOC_SNIPPET_WINDOWS_KEY)
    if not isinstance(raw, dict):
        return {}
    by_pair: Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]] = {}
    for doc_id, lst in raw.items():
        if not isinstance(lst, list):
            continue
        scored: List[Tuple[float, int, Optional[str]]] = []
        for rec in lst:
            if not isinstance(rec, dict):
                continue
            try:
                wi = int(rec.get("window_idx", 0))
            except (TypeError, ValueError):
                continue
            try:
                ce_score = float(rec.get("ce_score", 0.0))
            except (TypeError, ValueError):
                ce_score = 0.0
            qf_raw = rec.get("query_field", None)
            qf: Optional[str] = str(qf_raw) if qf_raw is not None and qf_raw != "" else None
            scored.append((ce_score, wi, qf))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if scored:
            by_pair[(qid_s, str(doc_id))] = scored
    return by_pair


def embed_doc_snippet_windows_for_question(
    qid: str,
    doc_ids: List[str],
    by_pair_global: Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]],
) -> Optional[Dict[str, List[dict]]]:
    """Pick max-pooled window rows for this question's doc_ids; return JSON-serializable dict or None if empty."""
    qid_s = str(qid)
    out: Dict[str, List[dict]] = {}
    for doc in doc_ids:
        k = (qid_s, str(doc))
        rows = by_pair_global.get(k)
        if not rows:
            continue
        out[str(doc)] = [
            {"window_idx": wi, "ce_score": sc, **({"query_field": qf} if qf is not None else {})}
            for sc, wi, qf in rows
        ]
    return out if out else None


def questions_use_embedded_windows(questions: List[dict]) -> bool:
    """True if any question carries a non-empty doc_snippet_windows map."""
    for q in questions:
        raw = q.get(DOC_SNIPPET_WINDOWS_KEY)
        if isinstance(raw, dict) and len(raw) > 0:
            return True
    return False


def select_windows_from_by_pair(
    by_pair: Dict[Tuple[str, str], List[Tuple[float, int, Optional[str]]]],
    window_size: int,
    top_windows: int,
) -> Tuple[
    Dict[Tuple[str, str], List[int]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], dict],
]:
    """
    top_windows=1: keep best only.
    top_windows=2: keep best; add second-ranked only if sentence spans are disjoint.
    """
    def _window_record(sc: float, wi: int, qf: Optional[str], **extra: str) -> dict:
        sids = sent_ids_for_window(wi, window_size)
        rec: dict = {
            "window_idx": wi,
            "ce_score": sc,
            "sent_ids": sids,
        }
        if qf is not None:
            rec["query_field"] = qf
        rec.update(extra)
        return rec

    merged_by_pair: Dict[Tuple[str, str], List[int]] = {}
    selected_by_pair: Dict[Tuple[str, str], List[dict]] = {}
    rejected_by_pair: Dict[Tuple[str, str], List[dict]] = {}
    pair_aux: Dict[Tuple[str, str], dict] = {}
    for k2, scored in by_pair.items():
        scored = list(scored)
        scored.sort(key=lambda x: (-x[0], x[1]))
        ranked = scored[:top_windows]
        aux: dict = {
            "had_two_ranked": len(ranked) >= 2 and top_windows >= 2,
            "dropped_second_overlap": False,
            "candidate_overlap_sent_count": 0,
        }
        picked: List[Tuple[float, int, Optional[str]]] = []
        rejected: List[dict] = []
        if ranked:
            picked.append(ranked[0])
        if top_windows >= 2 and len(ranked) >= 2:
            s0 = set(sent_ids_for_window(ranked[0][1], window_size))
            s1 = set(sent_ids_for_window(ranked[1][1], window_size))
            inter = s0 & s1
            n_inter = len(inter)
            aux["candidate_overlap_sent_count"] = n_inter
            if not inter:
                picked.append(ranked[1])
            else:
                aux["dropped_second_overlap"] = True
                sc, wi, qf = ranked[1]
                rejected.append(
                    _window_record(sc, wi, qf, reason="overlap_with_top1"),
                )

        selected: List[dict] = []
        indices: set[int] = set()
        for sc, wi, qf in picked:
            sids = sent_ids_for_window(wi, window_size)
            for j in sids:
                indices.add(j)
            selected.append(_window_record(sc, wi, qf))
        merged_by_pair[k2] = sorted(indices)
        selected_by_pair[k2] = selected
        if rejected:
            rejected_by_pair[k2] = rejected
        pair_aux[k2] = aux
    return merged_by_pair, selected_by_pair, rejected_by_pair, pair_aux


def select_windows_max_pool_from_path(
    windows_path: Path,
    window_size: int,
    top_windows: int,
) -> Tuple[
    Dict[Tuple[str, str], List[int]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], dict],
]:
    """Load windows JSONL from disk and run selection (legacy path)."""
    by_pair = load_by_pair_from_windows_jsonl(windows_path)
    return select_windows_from_by_pair(by_pair, window_size, top_windows)


def merge_window_selection_from_embedded_questions(
    questions: List[dict],
    window_size: int,
    top_windows: int,
) -> Tuple[
    Dict[Tuple[str, str], List[int]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], dict],
]:
    """For each question, run selection on its embedded by_pair; merge all pair dicts."""
    merged_all: Dict[Tuple[str, str], List[int]] = {}
    selected_all: Dict[Tuple[str, str], List[dict]] = {}
    rejected_all: Dict[Tuple[str, str], List[dict]] = {}
    pair_aux_all: Dict[Tuple[str, str], dict] = {}
    for q in questions:
        bp = by_pair_from_question_embedded(q)
        if not bp:
            continue
        m, s, r, p = select_windows_from_by_pair(bp, window_size, top_windows)
        merged_all.update(m)
        selected_all.update(s)
        rejected_all.update(r)
        pair_aux_all.update(p)
    return merged_all, selected_all, rejected_all, pair_aux_all
