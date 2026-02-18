from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# BioASQ parsing
# ---------------------------
_PUBMED_URL_RE = re.compile(r"/pubmed/(\d+)", re.IGNORECASE)
_PUBMED_NCBI_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"^\d+$")


def normalize_pmid(x) -> str:
    """Normalize BioASQ document entry to PMID (best-effort)."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = _PUBMED_URL_RE.search(s)
    if m:
        return m.group(1)
    m = _PUBMED_NCBI_URL_RE.search(s)
    if m:
        return m.group(1)
    if _DIGITS_RE.fullmatch(s):
        return s
    return s  # fallback


def load_questions(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "questions" not in data:
        raise KeyError(f"{json_path} missing top-level 'questions'")
    return data["questions"]


def build_topics_and_gold(questions: List[dict]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Return topics_df(qid, query) and gold_map[qid]=[pmids]."""
    rows = []
    gold: Dict[str, List[str]] = {}
    for i, q in enumerate(questions):
        qid = str(q.get("id") or q.get("qid") or i)
        query = str(q.get("body") or q.get("query") or q.get("question") or "").strip()

        docs = q.get("documents") or []
        pmids = [normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]
        gold[qid] = pmids

        rows.append({"qid": qid, "query": query})

    return pd.DataFrame(rows), gold


def collect_qids_from_questions(questions: List[dict]) -> set[str]:
    out = set()
    for i, q in enumerate(questions):
        out.add(str(q.get("id") or q.get("qid") or i))
    return out


# ---------------------------
# Evaluation (BioASQ-style)
# ---------------------------
def ap_at_k(gold: set[str], ranked: List[str], k: int = 10) -> float:
    """
    BioASQ-style AP@k:
      AP@k = (1 / min(|gold|, k)) * sum_{i=1..k} P@i * rel(i)
    """
    if not gold:
        return 0.0

    denom = min(len(gold), k)
    if denom == 0:
        return 0.0

    hit = 0
    s = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in gold:
            hit += 1
            s += hit / i
    return s / denom


def rr_at_k(gold: set[str], ranked: List[str], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def success_at_k(gold: set[str], ranked: List[str], k: int = 10) -> float:
    return 1.0 if any(doc in gold for doc in ranked[:k]) else 0.0


def recall_at_k(gold: set[str], ranked: List[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)


# Canonical recall@K grid (same across BM25, Dense, Hybrid, Reranker for comparable curves)
RECALL_KS: Tuple[int, ...] = (50, 100, 200, 300, 400, 500, 1000, 2000, 5000)


def evaluate_run(
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
    ks_recall: Optional[Sequence[int]] = None,
    eps: float = 1e-5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if ks_recall is None:
        ks_recall = RECALL_KS
    qids = list(gold_map.keys())

    ap10s, rr10s, succ10s = [], [], []
    recalls = {k: [] for k in ks_recall}
    perq_rows = []

    for qid in qids:
        gold = set(map(str, gold_map.get(qid, [])))
        ranked = list(map(str, run_map.get(qid, [])))

        ap10 = ap_at_k(gold, ranked, k=10)
        rr10 = rr_at_k(gold, ranked, k=10)
        succ10 = success_at_k(gold, ranked, k=10)

        ap10s.append(ap10)
        rr10s.append(rr10)
        succ10s.append(succ10)

        row = {"qid": qid, "AP@10": ap10, "RR@10": rr10, "Success@10": succ10}
        for k in ks_recall:
            r = recall_at_k(gold, ranked, k=k)
            recalls[k].append(r)
            row[f"R@{k}"] = r
        perq_rows.append(row)

    gmap10 = math.exp(sum(math.log(max(eps, x)) for x in ap10s) / max(1, len(ap10s))) if ap10s else 0.0

    summary: Dict[str, float] = {
        "MAP@10": float(np.mean(ap10s)) if ap10s else 0.0,
        "GMAP@10": float(gmap10),
        "MRR@10": float(np.mean(rr10s)) if rr10s else 0.0,
        "Success@10": float(np.mean(succ10s)) if succ10s else 0.0,
    }
    for k in ks_recall:
        summary[f"MeanR@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0

    return summary, pd.DataFrame(perq_rows)


def run_df_to_run_map(res_df: pd.DataFrame, qid_col: str = "qid", docno_col: str = "docno") -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid, g in res_df.groupby(qid_col, sort=False):
        run[str(qid)] = [str(x) for x in g[docno_col].tolist()]
    return run


def zero_recall_qids(gold_map: Dict[str, List[str]], run_map: Dict[str, List[str]], k: int) -> List[str]:
    out = []
    for qid, gold_list in gold_map.items():
        gold = set(map(str, gold_list))
        ranked = list(map(str, run_map.get(qid, [])))[:k]
        if gold and len(gold.intersection(ranked)) == 0:
            out.append(str(qid))
    return out


@dataclass
class BatchResult:
    method: str
    batch: str
    n_queries: int
    metrics: Dict[str, float]

    def to_row(self) -> Dict[str, object]:
        row = {"method": self.method, "batch": self.batch, "n_queries": self.n_queries}
        row.update(self.metrics)
        return row
