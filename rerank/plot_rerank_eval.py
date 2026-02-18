#!/usr/bin/env python3
"""
Generate Hybrid vs Reranker eval plots (recall curve + MAP@10) from existing
rerank metrics and hybrid runs. Used when rerank results exist but figure
files are missing. No model load, no reranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

import sys
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "public"))

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    run_df_to_run_map,
)

FIG_NAMES = (
    "hybrid_reranker_recall_map10_train.png",
    "hybrid_reranker_recall_map10_test.png",
)


def _load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(normalize_pmid),
    })
    rank_col = cols.get("rank")
    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def _meanr_columns_to_k_list(df: pd.DataFrame) -> List[int]:
    k_list = []
    for c in df.columns:
        if c.startswith("MeanR@"):
            try:
                k_list.append(int(c.replace("MeanR@", "")))
            except ValueError:
                continue
    return sorted(k_list)


def _is_train_split(split_name: str) -> bool:
    return "train" in split_name.lower()


def build_and_save_hybrid_reranker_plots(
    reranker_metrics_df: pd.DataFrame,
    run_maps: Dict[str, Dict[str, List[str]]],
    gold_map_all: Dict[str, List[str]],
    output_dir: Path,
    candidate_limit: Optional[int] = None,
) -> None:
    """Build Hybrid + Reranker combined table, aggregate train/test, save two figures.
    Recall curve is plotted only up to candidate_limit (reranker cap) when provided.
    """
    if plt is None:
        print("warning: matplotlib not available; skipping eval plots")
        return

    k_list = _meanr_columns_to_k_list(reranker_metrics_df)
    if candidate_limit is not None and candidate_limit > 0:
        k_list = [k for k in k_list if k <= candidate_limit]
    if not k_list:
        print("warning: no MeanR@* columns in reranker metrics; skipping eval plots")
        return

    metric_cols = [f"MeanR@{k}" for k in k_list] + ["MAP@10"]
    rows = []

    for _, row in reranker_metrics_df.iterrows():
        split = row["split"]
        if split not in run_maps:
            continue
        run_map = run_maps[split]
        gold_for_run = {qid: gold_map_all[qid] for qid in run_map if qid in gold_map_all}
        if not gold_for_run:
            continue

        hybrid_metrics, _ = evaluate_run(gold_for_run, run_map, ks_recall=tuple(k_list))
        reranker_vals = {c: row.get(c, np.nan) for c in metric_cols}

        rows.append({"method": "Hybrid", "split": split, **hybrid_metrics})
        rows.append({"method": "Reranker", "split": split, **reranker_vals})

    if not rows:
        print("warning: no overlapping splits with gold; skipping eval plots")
        return

    combined = pd.DataFrame(rows)

    def _aggregate(splits: List[str]) -> pd.DataFrame:
        sub = combined[combined["split"].isin(splits)]
        if sub.empty:
            return pd.DataFrame()
        return sub.groupby("method", as_index=True)[metric_cols].mean()

    train_splits = [s for s in reranker_metrics_df["split"].unique() if _is_train_split(s)]
    test_splits = [s for s in reranker_metrics_df["split"].unique() if not _is_train_split(s)]

    colors = {"Hybrid": "#444444", "Reranker": "#1f77b4"}

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for title_suffix, splits, fname in [
        ("Train", train_splits, FIG_NAMES[0]),
        ("Test (avg)", test_splits, FIG_NAMES[1]),
    ]:
        avg_df = _aggregate(splits)
        if avg_df.empty or "Hybrid" not in avg_df.index or "Reranker" not in avg_df.index:
            continue

        compare = [m for m in ["Hybrid", "Reranker"] if m in avg_df.index]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax_recall = axes[0]
        for method in compare:
            vals = [avg_df.loc[method, f"MeanR@{k}"] for k in k_list]
            ax_recall.plot(k_list, vals, marker="o", label=method, color=colors.get(method))
        ax_recall.set_xlabel("K (Recall Cutoff)")
        ax_recall.set_ylabel("Mean Recall")
        ax_recall.set_title(f"Hybrid vs Reranker ({title_suffix}) Recall")
        ax_recall.set_xscale("log")
        ax_recall.legend(fontsize=9, loc="lower right")

        ax_map = axes[1]
        map_vals = [avg_df.loc[m, "MAP@10"] for m in compare]
        ax_map.bar(compare, map_vals, color=[colors.get(m) for m in compare])
        ax_map.set_ylabel("MAP@10")
        ax_map.set_title(f"Hybrid vs Reranker ({title_suffix}) MAP@10")
        ax_map.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        out_path = figures_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Hybrid vs Reranker eval plots from existing metrics and runs.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Rerank output dir (contains metrics.csv).")
    ap.add_argument("--runs-dir", type=Path, required=True, help="Hybrid runs dir (TSV files).")
    ap.add_argument("--train-subset-json", "--train_subset_json", type=Path, default=None)
    ap.add_argument("--test-batch-jsons", "--test_batch_jsons", type=Path, nargs="*", default=None)
    args = ap.parse_args()

    metrics_path = args.output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Rerank metrics not found: {metrics_path}")

    reranker_metrics_df = pd.read_csv(metrics_path)
    if "split" not in reranker_metrics_df.columns:
        raise ValueError("metrics.csv must have a 'split' column")

    run_files = sorted(args.runs_dir.glob("*.tsv"))
    if not run_files:
        raise FileNotFoundError(f"No TSV run files in {args.runs_dir}")

    run_dfs: Dict[str, pd.DataFrame] = {}
    for path in run_files:
        name = path.stem
        run_dfs[name] = _load_run_tsv(path)
    run_maps = {name: run_df_to_run_map(df) for name, df in run_dfs.items()}

    gold_map_all: Dict[str, List[str]] = {}

    def _add_questions(json_path: Path) -> None:
        if not json_path or not json_path.exists():
            return
        questions = load_questions(json_path)
        _, gold_map = build_topics_and_gold(questions)
        for qid, docs in gold_map.items():
            gold_map_all[qid] = docs

    if args.train_subset_json:
        _add_questions(args.train_subset_json)
    for path in args.test_batch_jsons or []:
        _add_questions(Path(path))

    if not gold_map_all:
        raise ValueError("No gold loaded; provide --train_subset_json and/or --test_batch_jsons")

    candidate_limit = None
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            candidate_limit = config.get("candidate_limit")
        except Exception:
            pass

    build_and_save_hybrid_reranker_plots(
        reranker_metrics_df,
        run_maps,
        gold_map_all,
        args.output_dir,
        candidate_limit=candidate_limit,
    )


if __name__ == "__main__":
    main()
