#!/usr/bin/env python3
"""
RRF fusion between listwise reranking output and snippet_doc_fusion output.

Takes two sets of runs:
  - Listwise runs (simple names: ``{split}.tsv``)
  - Snippet doc fusion runs (long names: ``best_rrf_{split}_top5000_...tsv``)

Matches them by parsed split name, applies weighted RRF fusion,
and outputs fused TSVs with simple ``{split}.tsv`` names for
downstream evidence/generation compatibility.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SHARED_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SHARED_SCRIPTS))

from retrieval_eval.common import (  # type: ignore
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    question_qid_str,
    run_df_to_run_map,
)


# ---------------------------------------------------------------------------
# Split-name parsing
# ---------------------------------------------------------------------------

def _parse_split_from_long_stem(run_stem: str) -> Optional[str]:
    """Extract split from a ``best_rrf_*`` run stem (snippet doc fusion naming)."""
    m = re.fullmatch(
        r"best_rrf_(.+?)_top\d+(?:_rrf_pool[^\s]+)?",
        run_stem,
    )
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Run I/O
# ---------------------------------------------------------------------------

def _normalize_pmid(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"/pubmed/(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+", s):
        return s
    return s


def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")

    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(_normalize_pmid),
    })
    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1
    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def _rrf_fuse(
    listwise_docs: List[str],
    snippet_docs: List[str],
    pool_top: int,
    k_rrf: int,
    w_listwise: float,
    w_snippet: float,
) -> List[str]:
    lw_top = listwise_docs[:pool_top]
    sn_top = snippet_docs[:pool_top]
    rank_lw = {d: i + 1 for i, d in enumerate(lw_top)}
    rank_sn = {d: i + 1 for i, d in enumerate(sn_top)}

    union: List[str] = list(dict.fromkeys(lw_top + sn_top))
    scores = []
    for d in union:
        s = 0.0
        rl = rank_lw.get(d)
        rs = rank_sn.get(d)
        if rl is not None:
            s += w_listwise / (k_rrf + rl)
        if rs is not None:
            s += w_snippet / (k_rrf + rs)
        scores.append((d, s))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scores]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RRF fusion of listwise reranking runs with snippet_doc_fusion runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--listwise-runs-dir", type=Path, required=True,
                    help="Listwise runs directory ({split}.tsv).")
    p.add_argument(
        "--snippet-doc-fusion-runs-dir", "--snippet-rrf-runs-dir",
        type=Path,
        dest="snippet_doc_fusion_runs_dir",
        required=True,
        metavar="PATH",
        help="Snippet doc fusion runs directory (best_rrf_*.tsv). Alias: --snippet-rrf-runs-dir.",
    )
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for fused runs, metrics, and figures.")
    p.add_argument("--pool-top", type=int, default=15,
                    help="Pool size applied to both sides for RRF fusion.")
    p.add_argument("--k-rrf", type=int, default=20,
                    help="RRF k parameter in 1/(k+rank).")
    p.add_argument("--w-snippet-rrf", type=float, default=0.4,
                    help="Weight for snippet_rrf side in RRF.")
    p.add_argument("--w-listwise", type=float, default=0.6,
                    help="Weight for listwise side in RRF.")
    p.add_argument("--train-jsonl", type=Path, default=None, dest="train_jsonl",
                    help="Training queries .jsonl for metrics (optional).")
    p.add_argument("--test-batch-jsonls", type=Path, nargs="*", default=None,
                    dest="test_batch_jsonls",
                    help="Test-batch .jsonl for metrics (optional).")
    p.add_argument("--disable-metrics", action="store_true",
                    help="Skip metrics and plots (only write fused runs).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics helpers (lightweight, same as listwise_rerank.py)
# ---------------------------------------------------------------------------

def _ap_at_k(relevant: set, ranked: list, k: int) -> float:
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k) if relevant else 0.0


def _rr_at_k(relevant: set, ranked: list, k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def _evaluate_run_lightweight(
    gold: Dict[str, List[str]],
    run: Dict[str, List[str]],
    k: int = 10,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return (summary_dict, per_query_df)."""
    rows = []
    for qid, rel in gold.items():
        ranked = run.get(qid, [])
        rel_set = set(rel)
        ap = _ap_at_k(rel_set, ranked, k)
        rr = _rr_at_k(rel_set, ranked, k)
        rows.append({"qid": qid, f"AP@{k}": ap, f"RR@{k}": rr})
    perq = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["qid", f"AP@{k}", f"RR@{k}"])
    summary = {
        f"MAP@{k}": float(perq[f"AP@{k}"].mean()) if len(perq) else 0.0,
        f"MRR@{k}": float(perq[f"RR@{k}"].mean()) if len(perq) else 0.0,
    }
    return summary, perq


def _build_gold_map(questions: List[dict]) -> Dict[str, List[str]]:
    gold: Dict[str, List[str]] = {}
    for q in questions:
        qid = question_qid_str(q)
        docs = q.get("documents", [])
        pmids = [_normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]
        if pmids:
            gold[qid] = pmids
    return gold


def _run_map_from_df(df: pd.DataFrame) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid, g in df.groupby("qid", sort=False):
        run[str(qid)] = g.sort_values("rank")["docno"].tolist()
    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    listwise_dir: Path = args.listwise_runs_dir
    snippet_dir: Path = args.snippet_doc_fusion_runs_dir
    out_dir: Path = args.output_dir
    out_runs = out_dir / "runs"
    out_per_query = out_dir / "per_query"
    out_figures = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_runs.mkdir(parents=True, exist_ok=True)

    # Build a map: split_name -> snippet doc fusion TSV path
    snippet_by_split: Dict[str, Path] = {}
    for p in sorted(snippet_dir.glob("*.tsv")):
        split = _parse_split_from_long_stem(p.stem)
        if split is not None:
            snippet_by_split[split] = p

    if not snippet_by_split:
        print(f"Error: no parseable snippet doc fusion runs in {snippet_dir}")
        sys.exit(1)
    print(f"Found {len(snippet_by_split)} snippet_doc_fusion splits: {list(snippet_by_split.keys())}")

    # Iterate over listwise runs (simple {split}.tsv names)
    fused_dfs: Dict[str, pd.DataFrame] = {}
    listwise_dfs: Dict[str, pd.DataFrame] = {}
    snippet_dfs: Dict[str, pd.DataFrame] = {}

    for lw_path in sorted(listwise_dir.glob("*.tsv")):
        split = lw_path.stem
        if split not in snippet_by_split:
            print(f"skip {lw_path.name}: no matching snippet_doc_fusion run for split '{split}'")
            continue

        sn_path = snippet_by_split[split]
        print(
            f"RRF fusion for split={split}: "
            f"{lw_path.name} + {sn_path.name} "
            f"(pool_top={args.pool_top}, k_rrf={args.k_rrf}, "
            f"w_listwise={args.w_listwise}, w_snippet_rrf={args.w_snippet_rrf})"
        )

        lw_df = load_run_tsv(lw_path)
        sn_df = load_run_tsv(sn_path)
        listwise_dfs[split] = lw_df
        snippet_dfs[split] = sn_df

        fused_rows = []
        for qid, lw_group in lw_df.groupby("qid", sort=False):
            lw_docs = lw_group["docno"].tolist()
            sn_group = sn_df[sn_df["qid"] == qid]
            sn_docs = sn_group["docno"].tolist()

            if not lw_docs and not sn_docs:
                continue

            fused_docs = _rrf_fuse(
                listwise_docs=lw_docs,
                snippet_docs=sn_docs,
                pool_top=args.pool_top,
                k_rrf=args.k_rrf,
                w_listwise=args.w_listwise,
                w_snippet=args.w_snippet_rrf,
            )
            for rank, docno in enumerate(fused_docs, start=1):
                fused_rows.append({"qid": str(qid), "docno": docno, "rank": rank})

        if not fused_rows:
            print(f"  warning: no fused rows for split={split}")
            continue

        fused_df = pd.DataFrame(fused_rows).sort_values(["qid", "rank"]).reset_index(drop=True)
        fused_dfs[split] = fused_df

        out_path = out_runs / f"{split}.tsv"
        fused_df.to_csv(out_path, sep="\t", index=False)
        print(f"  -> {out_path} ({len(fused_df)} rows)")

    if not fused_dfs:
        print("No splits fused. Done.")
        return

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.disable_metrics:
        print("RRF fusion complete. Metrics skipped (--disable-metrics).")
        return

    all_questions: List[dict] = []
    if args.train_jsonl and args.train_jsonl.exists():
        qs = load_questions(args.train_jsonl)
        all_questions.extend(qs)
        print(f"Loaded {len(qs)} questions from {args.train_jsonl}")
    if args.test_batch_jsonls:
        for p in args.test_batch_jsonls:
            if p and p.exists():
                qs = load_questions(p)
                all_questions.extend(qs)
                print(f"Loaded {len(qs)} questions from {p}")

    if not all_questions:
        print("No query .jsonl provided; skipping metrics.")
        return

    gold_map = _build_gold_map(all_questions)
    print(f"Gold relevance for {len(gold_map)} queries")

    out_per_query.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for split in sorted(fused_dfs.keys()):
        fused_df = fused_dfs[split]
        fused_run = _run_map_from_df(fused_df)
        gold_split = {q: gold_map[q] for q in fused_run if q in gold_map}
        if not gold_split:
            print(f"  {split}: no gold queries – skipping metrics")
            continue

        # Fused metrics
        fused_metrics, fused_perq = _evaluate_run_lightweight(gold_split, fused_run)
        fused_perq.to_csv(out_per_query / f"{split}.csv", index=False)

        # Baseline: snippet_rrf
        sn_run = _run_map_from_df(snippet_dfs[split])
        sn_metrics, _ = _evaluate_run_lightweight(gold_split, sn_run)

        # Listwise-only
        lw_run = _run_map_from_df(listwise_dfs[split])
        lw_metrics, _ = _evaluate_run_lightweight(gold_split, lw_run)

        row = {
            "split": split,
            "n_queries": len(gold_split),
            "snippet_rrf_MAP@10": sn_metrics["MAP@10"],
            "listwise_MAP@10": lw_metrics["MAP@10"],
            "fused_MAP@10": fused_metrics["MAP@10"],
            "delta_fused_MAP@10": fused_metrics["MAP@10"] - sn_metrics["MAP@10"],
            "snippet_rrf_MRR@10": sn_metrics["MRR@10"],
            "listwise_MRR@10": lw_metrics["MRR@10"],
            "fused_MRR@10": fused_metrics["MRR@10"],
            "delta_fused_MRR@10": fused_metrics["MRR@10"] - sn_metrics["MRR@10"],
        }
        summary_rows.append(row)
        print(f"  {split}: snippet_rrf MAP@10={sn_metrics['MAP@10']:.4f}  "
              f"listwise MAP@10={lw_metrics['MAP@10']:.4f}  "
              f"fused MAP@10={fused_metrics['MAP@10']:.4f}")

    if not summary_rows:
        print("No metrics computed (no gold overlap). Done.")
        return

    results_df = pd.DataFrame(summary_rows)
    metrics_path = out_dir / "metrics.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"\nWrote metrics -> {metrics_path}")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Comparison bar chart (MAP@10)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(results_df))
    w = 0.25

    ax1 = axes[0]
    ax1.bar(x - w, results_df["snippet_rrf_MAP@10"], w,
            label="Snippet-RRF", color="steelblue")
    ax1.bar(x, results_df["listwise_MAP@10"], w,
            label="Listwise", color="coral")
    ax1.bar(x + w, results_df["fused_MAP@10"], w,
            label="Fused (RRF)", color="seagreen")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("MAP@10")
    ax1.set_title("MAP@10: Snippet-RRF vs Listwise vs Fused")
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["split"], rotation=45, ha="right")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    colors = ["green" if d > 0 else "red" for d in results_df["delta_fused_MAP@10"]]
    ax2.bar(x, results_df["delta_fused_MAP@10"], color=colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Delta MAP@10")
    ax2.set_title("MAP@10 Improvement (Fused - Snippet-RRF)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df["split"], rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_figures / "map10_fusion_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure -> {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
