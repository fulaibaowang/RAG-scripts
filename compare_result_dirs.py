#!/usr/bin/env python3
"""
Compare metrics and optionally plot recall and/or MAP curves across two or more
result directories (e.g. rerank vs rerank_sentence, or hybrid vs rerank).
- Dirs with metrics.csv are used as-is.
- Dirs with only runs/*.tsv (e.g. hybrid output) are supported: provide
  --train-json and/or --test-batch-jsons so metrics are computed from runs.
For MAP@k curve, runs/*.tsv and gold are required.

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
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
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


def _metrics_from_runs_dir(
    runs_dir: Path,
    gold_map: Dict[str, List[str]],
    ks_recall: Tuple[int, ...],
    result_dir: str,
    dir_label: str,
) -> pd.DataFrame:
    """Build a metrics-style DataFrame from run TSVs (e.g. hybrid output with runs/ but no metrics.csv)."""
    run_files = sorted(runs_dir.glob("*.tsv"))
    if not run_files:
        return pd.DataFrame()
    rows = []
    for path in run_files:
        run_id = path.stem
        run_df = _load_run_tsv(path)
        run_map = run_df_to_run_map(run_df, qid_col="qid", docno_col="docno")
        gold_for_run = {qid: gold_map[qid] for qid in run_map if qid in gold_map and gold_map[qid]}
        if not gold_for_run:
            rows.append({"run": run_id, "label": run_id, "role": "unknown", "result_dir": result_dir, "dir_label": dir_label})
            continue
        metrics, _ = evaluate_run(gold_for_run, run_map, ks_recall=ks_recall)
        row = {"run": run_id, "label": run_id, "role": "unknown", "result_dir": result_dir, "dir_label": dir_label}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def load_metrics_from_dirs(
    dirs: List[Path],
    labels: Optional[List[str]],
    gold_map: Optional[Dict[str, List[str]]] = None,
    ks_recall: Optional[Tuple[int, ...]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and concatenate metrics from each dir. If a dir has no metrics.csv but has runs/*.tsv,
    compute metrics from runs when gold_map (and ks_recall) are provided."""
    if ks_recall is None:
        ks_recall = RECALL_KS
    rows = []
    for i, d in enumerate(dirs):
        p = d / "metrics.csv"
        runs_dir = d / "runs"
        label = labels[i] if labels and i < len(labels) else d.name
        if p.exists():
            df = pd.read_csv(p)
            df["result_dir"] = str(d)
            df["dir_label"] = label
            rows.append(df)
        elif runs_dir.is_dir() and list(runs_dir.glob("*.tsv")):
            if not gold_map:
                raise ValueError(
                    f"Dir {d} has no metrics.csv but has runs/*.tsv. "
                    "Provide --train-json and/or --test-batch-jsons to compute metrics from runs."
                )
            df = _metrics_from_runs_dir(runs_dir, gold_map, ks_recall, result_dir=str(d), dir_label=label)
            if not df.empty:
                rows.append(df)
        else:
            raise FileNotFoundError(f"metrics.csv not found in {d} and no runs/*.tsv in {d}")
    if not rows:
        raise ValueError("No metrics loaded from any dir.")
    combined = pd.concat(rows, ignore_index=True)
    dir_labels = list(combined["dir_label"].unique())
    return combined, dir_labels


def plot_recall_curves(
    combined: pd.DataFrame,
    dir_labels: List[str],
    output_path: Path,
    k_max: Optional[int] = None,
    log_x: bool = False,
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
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("K (Recall cutoff)")
    ax.set_ylabel("Mean Recall")
    ax.set_title("Recall curve comparison")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


# Key metrics to include in summary stats table (use whatever exists in combined)
SUMMARY_METRIC_COLS = [
    "MAP@10", "MRR@10", "GMAP@10", "Success@10",
    "MeanR@50", "MeanR@100", "MeanR@200", "MeanR@500", "MeanR@1000",
]


def _write_compare_summary(combined: pd.DataFrame, output_dir: Path) -> None:
    """Print and save a short summary table of key metrics per (dir_label, run)."""
    cols = ["dir_label", "run"]
    if "label" in combined.columns:
        cols.append("label")
    if "role" in combined.columns:
        cols.append("role")
    for c in SUMMARY_METRIC_COLS:
        if c in combined.columns:
            cols.append(c)
    out = combined[cols].drop_duplicates()
    out = out.sort_values(["dir_label", "run"]).reset_index(drop=True)
    summary_path = output_dir / "compare_summary.csv"
    out.to_csv(summary_path, index=False)
    print("Summary stats (saved to {}):".format(summary_path))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(out.to_string(index=False))
    print()


def plot_map_curve(
    map_by_run: Dict[Tuple[str, str], Dict[int, float]],
    ks: List[int],
    output_path: Path,
    log_x: bool = False,
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
    if log_x:
        ax.set_xscale("log")
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
    parser.add_argument("--dirs", type=Path, nargs="+", required=True, help="Result dirs: each with metrics.csv, or with runs/*.tsv only (e.g. hybrid; then provide --train-json/--test-batch-jsons).")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Display labels for each dir (same order as --dirs). Default: dir name.")
    parser.add_argument("--plot", type=str, choices=("recall", "map", "both"), default="both", help="What to plot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save figures. Default: first --dirs parent.")
    parser.add_argument("--map-ks", type=str, default="10,20,50,100,200", help="Comma-separated K values for MAP curve.")
    parser.add_argument("--recall-k-max", type=int, default=None, help="Max K for recall curve (default: use all in metrics).")
    parser.add_argument("--ks-recall", type=str, default="", help="Comma-separated K for MeanR@k when building metrics from runs (e.g. hybrid). Default: 50,100,200,...,5000.")
    parser.add_argument("--train-json", type=Path, default=None, help="Training questions JSON (for gold; required for MAP curve and for dirs that only have runs/).")
    parser.add_argument("--test-batch-jsons", type=Path, nargs="*", default=None, help="Test batch JSONs (for gold, needed for MAP curve).")
    parser.add_argument("--query-field", type=str, default="body", help="Query field in question JSONs.")
    parser.add_argument("--log-x", action="store_true", help="Use log scale for x-axis (K) in recall and MAP curves.")
    parser.add_argument("--plots-by-split", action="store_true", help="Output one recall and one MAP plot per split (train/test); uses 'role' from metrics, or 'label' if role missing.")
    return parser.parse_args()


def _parse_ks_recall(raw: str) -> Tuple[int, ...]:
    if not raw or not raw.strip():
        return RECALL_KS
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


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

    # Dirs with runs/ but no metrics.csv need gold to build synthetic metrics
    need_gold = any(
        not (d / "metrics.csv").exists() and (d / "runs").is_dir() and list((d / "runs").glob("*.tsv"))
        for d in dirs
    )
    gold_map: Dict[str, List[str]] = {}
    ks_recall = _parse_ks_recall(args.ks_recall)
    if need_gold or args.plot in ("map", "both"):
        if not args.train_json and not args.test_batch_jsons:
            if need_gold:
                raise ValueError("Some dirs have only runs/ (e.g. hybrid). Provide --train-json and/or --test-batch-jsons to compute metrics.")
            raise ValueError("For MAP curve provide --train-json and/or --test-batch-jsons to build gold.")
        if args.train_json and args.train_json.exists():
            questions = load_questions(args.train_json)
            _, g = build_topics_and_gold(questions, query_field=args.query_field)
            gold_map.update(g)
        for p in args.test_batch_jsons or []:
            if Path(p).exists():
                questions = load_questions(Path(p))
                _, g = build_topics_and_gold(questions, query_field=args.query_field)
                gold_map.update(g)
        if need_gold and not gold_map:
            raise ValueError("No gold loaded; required for dirs that only have runs/*.tsv.")

    combined, dir_labels = load_metrics_from_dirs(
        dirs, args.labels,
        gold_map=gold_map if need_gold else None,
        ks_recall=ks_recall,
    )

    # Summary stats table (always)
    _write_compare_summary(combined, output_dir)

    # Split key for --plots-by-split: "role" (train/test) if present, else "label"
    split_col = "role" if "role" in combined.columns and combined["role"].notna().any() else "label"
    if args.plots_by_split:
        split_values = sorted(combined[split_col].dropna().unique().tolist()) or ["all"]
    else:
        split_values = [None]  # single combined plot

    if args.plot in ("recall", "both"):
        for split_val in split_values:
            if split_val is None:
                subset = combined
                out_path = figures_dir / "compare_recall_curves.png"
            else:
                subset = combined[combined[split_col] == split_val]
                if subset.empty:
                    continue
                safe_name = str(split_val).replace("/", "_").replace(" ", "_")
                out_path = figures_dir / f"compare_recall_curves_{safe_name}.png"
            plot_recall_curves(
                subset,
                dir_labels,
                out_path,
                k_max=args.recall_k_max,
                log_x=args.log_x,
            )

    if args.plot in ("map", "both"):
        map_ks = [int(x.strip()) for x in args.map_ks.split(",") if x.strip()]
        if not map_ks:
            map_ks = [10, 20, 50, 100, 200]
        if not gold_map:
            raise ValueError("For MAP curve provide --train-json and/or --test-batch-jsons to build gold.")

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
            for split_val in split_values:
                if split_val is None:
                    subset_map = map_by_run
                    out_path = figures_dir / "compare_map_curves.png"
                else:
                    keys_in_split = set(
                        (row["dir_label"], row["run"])
                        for _, row in combined[combined[split_col] == split_val].iterrows()
                    )
                    subset_map = {k: v for k, v in map_by_run.items() if k in keys_in_split}
                    if not subset_map:
                        continue
                    safe_name = str(split_val).replace("/", "_").replace(" ", "_")
                    out_path = figures_dir / f"compare_map_curves_{safe_name}.png"
                plot_map_curve(subset_map, map_ks, out_path, log_x=args.log_x)
        else:
            print("warning: no run TSV files found in dirs; skipping MAP curve.")

    print("Done. Figures in", figures_dir)


if __name__ == "__main__":
    main()
