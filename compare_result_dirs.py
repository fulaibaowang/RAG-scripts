#!/usr/bin/env python3
"""
Compare metrics and optionally plot recall and/or MAP curves across two or more
result directories (e.g. rerank vs rerank_sentence, or hybrid vs dense).
Each dir should contain metrics.csv; for MAP@k curve over k, runs/*.tsv and gold are required.

Example (stage 2 vs stage 3 sentence, with MAP curve 10–200):
  python scripts/public/shared_scripts/compare_result_dirs.py \\
    --dirs output/workflow_local_3pct_hpc_bge/rerank output/workflow_local_3pct_hpc_bge/rerank_sentence \\
    --labels "Stage 2" "Stage 3 sentence" \\
    --plot both \\
    --map-ks 10,20,50,100,200 \\
    --train-json example/training14b_3pct_sample.json \\
    --test-batch-jsons example/13b_golden_50q_sample.json \\
    --output-dir output/workflow_local_3pct_hpc_bge/compare_plots
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

import sys
_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2]
_SHARED_SCRIPTS = _REPO_ROOT / "scripts" / "public" / "shared_scripts"
if _SHARED_SCRIPTS.exists():
    if str(_SHARED_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SHARED_SCRIPTS))
else:
    for _p in [_SCRIPT_DIR.parents[0]] + list(_SCRIPT_DIR.parents):
        if (_p / "retrieval_eval").exists():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            break

from retrieval_eval.common import (
    ap_at_k,
    build_topics_and_gold,
    load_questions,
    normalize_pmid,
    run_df_to_run_map,
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


def compute_map_at_ks(
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
    ks: List[int],
) -> Dict[int, float]:
    """Compute MAP@k for each k. Returns {k: MAP@k}."""
    qids = [q for q in gold_map if q in run_map and gold_map[q]]
    if not qids:
        return {k: 0.0 for k in ks}
    out = {}
    for k in ks:
        aps = [ap_at_k(set(gold_map[q]), run_map[q], k=k) for q in qids]
        out[k] = float(np.mean(aps))
    return out


def load_metrics_from_dirs(
    dirs: List[Path],
    labels: Optional[List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and concatenate metrics.csv from each dir; add column 'result_dir' and optional 'dir_label'."""
    rows = []
    for i, d in enumerate(dirs):
        p = d / "metrics.csv"
        if not p.exists():
            raise FileNotFoundError(f"metrics.csv not found in {d}")
        df = pd.read_csv(p)
        df["result_dir"] = str(d)
        df["dir_label"] = (labels[i] if labels and i < len(labels) else d.name)
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    dir_labels = list(combined["dir_label"].unique())
    return combined, dir_labels


def plot_recall_curves(
    combined: pd.DataFrame,
    dir_labels: List[str],
    output_path: Path,
    k_max: Optional[int] = None,
) -> None:
    """Plot recall (MeanR@k) curves: one line per (dir_label, run), x = k."""
    if plt is None:
        return
    k_list = _meanr_columns_to_k_list(combined)
    if k_max is not None and k_max > 0:
        k_list = [k for k in k_list if k <= k_max]
    if not k_list:
        return
    metric_cols = [f"MeanR@{k}" for k in k_list]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dir_labels), 1)))
    for (dir_lbl, run_id), grp in combined.groupby(["dir_label", "run"], sort=False):
        if grp.shape[0] != 1:
            vals = grp[metric_cols].mean().values
        else:
            vals = grp[metric_cols].iloc[0].values
        label = f"{dir_lbl}: {run_id}" if len(dir_labels) > 1 or len(combined["run"].unique()) > 1 else run_id
        ax.plot(k_list, vals, marker="o", label=label, markersize=4)
    ax.set_xlabel("K (Recall cutoff)")
    ax.set_ylabel("Mean Recall")
    ax.set_title("Recall curve comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


def plot_map_curve(
    map_by_run: Dict[Tuple[str, str], Dict[int, float]],
    ks: List[int],
    output_path: Path,
) -> None:
    """Plot MAP@k curve: one line per (dir_label, run_id), x = k."""
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dir_lbl, run_id), map_vals in map_by_run.items():
        xs = [k for k in ks if k in map_vals]
        ys = [map_vals[k] for k in xs]
        label = f"{dir_lbl}: {run_id}"
        ax.plot(xs, ys, marker="o", label=label, markersize=5)
    ax.set_xlabel("K (MAP cutoff)")
    ax.set_ylabel("MAP@K")
    ax.set_title("MAP@K curve comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare result dirs: plot recall and/or MAP@k curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dirs", type=Path, nargs="+", required=True, help="Result directories (each with metrics.csv; for MAP curve also runs/*.tsv).")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Display labels for each dir (same order as --dirs). Default: dir name.")
    parser.add_argument("--plot", type=str, choices=("recall", "map", "both"), default="both", help="What to plot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save figures. Default: first --dirs parent.")
    parser.add_argument("--map-ks", type=str, default="10,20,50,100,200", help="Comma-separated K values for MAP curve.")
    parser.add_argument("--recall-k-max", type=int, default=None, help="Max K for recall curve (default: use all in metrics).")
    parser.add_argument("--train-json", type=Path, default=None, help="Training questions JSON (for gold, needed for MAP curve).")
    parser.add_argument("--test-batch-jsons", type=Path, nargs="*", default=None, help="Test batch JSONs (for gold, needed for MAP curve).")
    parser.add_argument("--query-field", type=str, default="body", help="Query field in question JSONs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(d).resolve() for d in args.dirs]
    if args.labels and len(args.labels) != len(dirs):
        raise ValueError("--labels must have same length as --dirs")
    output_dir = args.output_dir or (dirs[0].parent if dirs else Path("."))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined, dir_labels = load_metrics_from_dirs(dirs, args.labels)

    if args.plot in ("recall", "both"):
        plot_recall_curves(
            combined,
            dir_labels,
            figures_dir / "compare_recall_curves.png",
            k_max=args.recall_k_max,
        )

    if args.plot in ("map", "both"):
        map_ks = [int(x.strip()) for x in args.map_ks.split(",") if x.strip()]
        if not map_ks:
            map_ks = [10, 20, 50, 100, 200]
        if not args.train_json and not args.test_batch_jsons:
            raise ValueError("For MAP curve provide --train-json and/or --test-batch-jsons to build gold.")
        gold_map: Dict[str, List[str]] = {}
        if args.train_json and args.train_json.exists():
            questions = load_questions(args.train_json)
            _, g = build_topics_and_gold(questions, query_field=args.query_field)
            gold_map.update(g)
        for p in args.test_batch_jsons or []:
            if Path(p).exists():
                questions = load_questions(Path(p))
                _, g = build_topics_and_gold(questions, query_field=args.query_field)
                gold_map.update(g)
        if not gold_map:
            raise ValueError("No gold loaded from --train-json / --test-batch-jsons.")

        map_by_run: Dict[Tuple[str, str], Dict[int, float]] = {}
        for _, row in combined.iterrows():
            dir_lbl = row["dir_label"]
            run_id = row["run"]
            result_dir = Path(row["result_dir"])
            run_path = result_dir / "runs" / f"{run_id}.tsv"
            if not run_path.exists():
                continue
            run_df = _load_run_tsv(run_path)
            run_map = run_df_to_run_map(run_df, qid_col="qid", docno_col="docno")
            map_by_run[(dir_lbl, run_id)] = compute_map_at_ks(gold_map, run_map, map_ks)
        if map_by_run:
            plot_map_curve(map_by_run, map_ks, figures_dir / "compare_map_curves.png")
        else:
            print("warning: no run TSV files found in dirs; skipping MAP curve.")

    print("Done. Figures in", figures_dir)


if __name__ == "__main__":
    main()
