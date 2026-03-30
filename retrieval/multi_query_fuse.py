#!/usr/bin/env python3
"""
N-way weighted RRF fusion of retrieval/rerank run TSVs across parallel directories.

Each directory should contain the same set of run filenames (e.g. dense_train.tsv in
every dir). For each filename, fuses ranked lists using:

    score(doc) += weight_i / (k_rrf + rank_i)

Weights default to 1/N when --weights is omitted. Tie-break: lexicographic docno.

After fusion, optionally evaluates fused runs against gold (--train-json /
--test-batch-jsons) and produces metrics.csv + recall/MAP plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

# Allow importing from sibling retrieval_eval/ package
_SHARED_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))

from retrieval_eval.common import (
    ap_at_k,
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    RECALL_KS,
    run_df_to_run_map,
)


# ---------------------------------------------------------------------------
# TSV loading
# ---------------------------------------------------------------------------
def _load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_c = cols.get("qid") or cols.get("query_id")
    doc_c = cols.get("docno") or cols.get("docid") or cols.get("doc")
    rank_c = cols.get("rank")
    if not qid_c or not doc_c:
        raise ValueError(f"{path}: need qid and docno columns, got {list(df.columns)}")
    out = pd.DataFrame(
        {
            "qid": df[qid_c].astype(str),
            "docno": df[doc_c].astype(str),
        }
    )
    if rank_c and rank_c in df.columns:
        out["rank"] = pd.to_numeric(df[rank_c], errors="coerce").fillna(0).astype(int)
    else:
        out["rank"] = out.groupby("qid", sort=False).cumcount() + 1
    return out


# ---------------------------------------------------------------------------
# N-way RRF fusion
# ---------------------------------------------------------------------------
def fuse_n_way_rrf(
    dfs: Sequence[pd.DataFrame],
    weights: Sequence[float],
    k_rrf: float,
    cap: int | None,
    body_weight: float | None = None,
) -> pd.DataFrame:
    """N-way weighted RRF fusion.

    When *body_weight* is set, per-qid adaptive weighting is used:
    the first sub-run (index 0, "body") gets *body_weight*, and the remaining
    weight is split equally among whichever other sub-runs are active for that
    qid.  If only the body sub-run is active, it receives weight 1.0.
    When *body_weight* is ``None``, the fixed *weights* vector is used.
    """
    if body_weight is None and len(dfs) != len(weights):
        raise ValueError("dfs and weights length mismatch")
    if not dfs:
        raise ValueError("no dataframes")

    qid_order: List[str] = []
    seen_q: set[str] = set()
    for df in dfs:
        for qid in df["qid"].unique():
            s = str(qid)
            if s not in seen_q:
                seen_q.add(s)
                qid_order.append(s)

    # Pre-compute per-df qid sets for fast membership testing
    df_qid_sets: List[set[str]] = [set(df["qid"].astype(str).unique()) for df in dfs]

    rows: List[Dict[str, object]] = []
    kf = float(k_rrf)

    for qid in qid_order:
        # Determine per-qid weights
        if body_weight is not None:
            active = [i for i, qs in enumerate(df_qid_sets) if qid in qs]
            body_idx = 0
            non_body = [i for i in active if i != body_idx]
            n_nb = len(non_body)
            qw: Dict[int, float] = {}
            if body_idx in active:
                qw[body_idx] = body_weight if n_nb else 1.0
                rest = (1.0 - body_weight) if n_nb else 0.0
            else:
                rest = 1.0
            for i in non_body:
                qw[i] = rest / n_nb
        else:
            qw = {i: w for i, w in enumerate(weights)}

        scores: Dict[str, float] = {}
        for i, df in enumerate(dfs):
            wi = qw.get(i, 0.0)
            if wi == 0.0:
                continue
            sub = df[df["qid"].astype(str) == qid]
            for _, r in sub.iterrows():
                doc = str(r["docno"])
                rk = int(r["rank"])
                if rk < 1:
                    rk = 1
                scores[doc] = scores.get(doc, 0.0) + wi / (kf + rk)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if cap is not None:
            ranked = ranked[: int(cap)]
        for i, (doc, sc) in enumerate(ranked, start=1):
            rows.append({"qid": qid, "docno": doc, "rank": i, "score": sc})

    return pd.DataFrame(rows)


def parse_weights(s: str | None, n: int) -> List[float]:
    if not s or not str(s).strip():
        return [1.0 / n] * n
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    w = [float(x) for x in parts]
    if len(w) != n:
        raise ValueError(f"Expected {n} weights (comma-separated), got {len(w)}: {s!r}")
    return w


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _load_gold(
    train_json: Optional[Path],
    test_batch_jsons: Optional[List[Path]],
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Load gold relevance judgments. Returns (gold_map, train_stems, test_stems)."""
    gold_map: Dict[str, List[str]] = {}
    train_stems: List[str] = []
    test_stems: List[str] = []
    if train_json and train_json.exists():
        qs = load_questions(train_json)
        _, gm = build_topics_and_gold(qs)
        gold_map.update(gm)
        train_stems.append(train_json.stem)
    for p in test_batch_jsons or []:
        if p.exists():
            qs = load_questions(p)
            _, gm = build_topics_and_gold(qs)
            gold_map.update(gm)
            test_stems.append(p.stem)
    return gold_map, train_stems, test_stems


def _infer_role(run_id: str, train_stems: List[str], test_stems: List[str]) -> str:
    low = run_id.lower()
    for s in test_stems:
        if s.lower() in low:
            return "test"
    for s in train_stems:
        if s.lower() in low:
            return "train"
    return "unknown"


def _short_label(run_id: str) -> str:
    s = run_id
    for prefix in ("best_rrf_", "dense_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for sep in ("_top", "_rrf_"):
        if sep in s:
            s = s.split(sep)[0]
            break
    return s if s else run_id


def _eval_one_run_map(
    run_map: Dict[str, List[str]],
    gold_map: Dict[str, List[str]],
    ks_recall: Tuple[int, ...],
    map_ks: List[int],
    qid_subset: Optional[set] = None,
) -> Optional[Dict]:
    """Evaluate a single run_map. If *qid_subset* is given, restrict to those qids."""
    if qid_subset is not None:
        run_map = {q: docs for q, docs in run_map.items() if q in qid_subset}
    gold_for_run = {q: gold_map[q] for q in run_map if q in gold_map and gold_map[q]}
    if not gold_for_run:
        return None
    metrics, _ = evaluate_run(gold_for_run, run_map, ks_recall=ks_recall)
    row: Dict = {}
    row.update(metrics)
    for k in map_ks:
        qids = list(gold_for_run.keys())
        aps = [ap_at_k(set(gold_for_run[q]), run_map.get(q, []), k=k) for q in qids]
        row[f"MAP@{k}"] = float(np.mean(aps)) if aps else 0.0
    return row


def _plot_curves(
    rows: List[Dict],
    ks_recall: Tuple[int, ...],
    map_ks: List[int],
    figures_dir: Path,
    prefix: str,
    title_suffix: str,
) -> None:
    """Generate recall and MAP curve PNGs for a list of metric rows."""
    if plt is None or not rows:
        return
    ks_plot = sorted(k for k in ks_recall if any(f"MeanR@{k}" in r for r in rows))
    if ks_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in rows:
            ys = [r.get(f"MeanR@{k}", np.nan) for k in ks_plot]
            ax.plot(ks_plot, ys, marker="o", label=r["label"], markersize=4)
        ax.set_xlabel("K")
        ax.set_ylabel("Mean Recall@K")
        ax.set_title(f"Recall curves {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = figures_dir / f"{prefix}_recall.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[multi_query_fuse] plot -> {p}")
    if map_ks:
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in rows:
            ys = [r.get(f"MAP@{k}", np.nan) for k in map_ks]
            ax.plot(map_ks, ys, marker="o", label=r["label"], markersize=5)
        ax.set_xlabel("K")
        ax.set_ylabel("MAP@K")
        ax.set_title(f"MAP curves {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = figures_dir / f"{prefix}_map.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[multi_query_fuse] plot -> {p}")


def _eval_fused_runs(
    out_dir: Path,
    gold_map: Dict[str, List[str]],
    ks_recall: Tuple[int, ...],
    train_stems: List[str],
    test_stems: List[str],
    map_ks: Optional[List[int]] = None,
    run_dirs: Optional[List[Path]] = None,
    labels: Optional[List[str]] = None,
) -> None:
    """Evaluate fused TSVs and optionally compare with sub-run TSVs.

    When *run_dirs* and *labels* are provided, comparison plots are generated:
      - Plot 1 (``fused_compare_*``): body + fused + eligible per-field curves on
        the "different queries" subset (union of non-body sub-run qid sets).
        A non-body field is included only if its qid set covers all diff_qids.
      - Plot 2 (``fused_all_*``): fused curve on all queries.
    """
    fused_dir = out_dir
    parent_dir = out_dir.parent
    figures_dir = parent_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(fused_dir.glob("*.tsv"))
    if not tsv_files:
        print("[multi_query_fuse] no fused TSVs to evaluate", file=sys.stderr)
        return

    if map_ks is None:
        map_ks = [10, 20, 50, 100, 200]

    all_metrics_rows: List[Dict] = []

    for tsv_path in tsv_files:
        run_id = tsv_path.stem
        role = _infer_role(run_id, train_stems, test_stems)
        fused_df = _load_run_tsv(tsv_path)
        fused_run_map = run_df_to_run_map(fused_df, qid_col="qid", docno_col="docno")

        # -- Fused on ALL queries (Plot 2) --
        row_all = _eval_one_run_map(fused_run_map, gold_map, ks_recall, map_ks)
        if row_all:
            row_all.update({"run": run_id, "label": "fused", "role": role, "scope": "all"})
            all_metrics_rows.append(row_all)

        # -- Comparison with sub-runs (Plot 1) --
        if run_dirs and labels and len(run_dirs) == len(labels):
            sub_dfs: List[Optional[pd.DataFrame]] = []
            for d in run_dirs:
                p = d / tsv_path.name
                sub_dfs.append(_load_run_tsv(p) if p.is_file() else None)

            # Per-field qid sets (None entries get empty set)
            qid_sets = [
                set(df["qid"].astype(str).unique()) if df is not None else set()
                for df in sub_dfs
            ]

            # "different qids" = union of non-body (non-first) sub-run qid sets
            non_body_sets = [qs for i, qs in enumerate(qid_sets) if i > 0 and qs]
            diff_qids = set().union(*non_body_sets) if non_body_sets else set()

            if diff_qids:
                for idx, (lbl, sdf) in enumerate(zip(labels, sub_dfs)):
                    if sdf is None:
                        continue
                    # Include a non-body field only if it covers all diff_qids
                    if idx > 0 and not qid_sets[idx].issuperset(diff_qids):
                        continue
                    rm = run_df_to_run_map(sdf, qid_col="qid", docno_col="docno")
                    row = _eval_one_run_map(rm, gold_map, ks_recall, map_ks, qid_subset=diff_qids)
                    if row:
                        row.update({
                            "run": run_id, "label": lbl, "role": role, "scope": "different",
                        })
                        all_metrics_rows.append(row)

                row_fused_diff = _eval_one_run_map(
                    fused_run_map, gold_map, ks_recall, map_ks, qid_subset=diff_qids,
                )
                if row_fused_diff:
                    row_fused_diff.update({
                        "run": run_id, "label": "fused", "role": role, "scope": "different",
                    })
                    all_metrics_rows.append(row_fused_diff)

    if not all_metrics_rows:
        return

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_path = parent_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[multi_query_fuse] metrics -> {metrics_path}")

    if plt is None:
        print("[multi_query_fuse] matplotlib not available, skipping plots")
        return

    for role_val in ("train", "test"):
        # Plot 2: fused on all queries
        all_rows = [r for r in all_metrics_rows
                     if r.get("role") == role_val and r.get("scope") == "all"]
        if all_rows:
            _plot_curves(all_rows, ks_recall, map_ks, figures_dir,
                         f"fused_all_{role_val}", f"(all queries, {role_val})")

        # Plot 1: comparison on different-queries subset
        diff_rows = [r for r in all_metrics_rows
                      if r.get("role") == role_val and r.get("scope") == "different"]
        if diff_rows:
            _plot_curves(diff_rows, ks_recall, map_ks, figures_dir,
                         f"fused_compare_{role_val}",
                         f"(different queries only, {role_val})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Directories containing run TSVs (each should have matching filenames).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write fused TSVs (created if missing).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.tsv",
        help="Glob pattern relative to each run-dir (default: *.tsv).",
    )
    ap.add_argument(
        "--k-rrf",
        type=float,
        default=60.0,
        help="RRF k in weight/(k+rank) (default: 60).",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated weights, same order as --run-dirs; default 1/N each.",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Max documents per query in fused output (default: no limit).",
    )
    # Eval arguments (all optional; when --train-json or --test-batch-jsons are
    # given, evaluation is performed on fused runs)
    ap.add_argument("--train-json", type=Path, default=None, help="Training questions JSON (for eval).")
    ap.add_argument("--test-batch-jsons", type=Path, nargs="*", default=None, help="Test batch JSONs (for eval).")
    ap.add_argument("--ks", type=str, default="", help="Comma-separated K for recall metrics (default: 50..5000).")
    ap.add_argument("--map-ks", type=str, default="10,20,50,100,200", help="Comma-separated K for MAP curves.")
    ap.add_argument("--no-eval", action="store_true", help="Skip evaluation even when gold JSONs are provided.")
    ap.add_argument(
        "--labels", type=str, default="",
        help="Comma-separated labels matching --run-dirs order (e.g. 'body,body_hyde'). "
        "Enables comparison plots: per-field + fused curves on different-queries subset.",
    )
    ap.add_argument(
        "--body-weight", type=float, default=None,
        help="Adaptive per-qid weighting: the first field (body) gets this weight, "
        "remaining weight is split equally among active non-body fields per query. "
        "Ignored when --weights is explicitly set.",
    )
    args = ap.parse_args()

    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    for d in run_dirs:
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    first = run_dirs[0]
    matched = sorted(first.glob(args.pattern))
    if not matched:
        print(f"Error: no files matching {args.pattern!r} under {first}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(run_dirs)
    has_explicit_weights = bool(args.weights and args.weights.strip())
    weights = parse_weights(args.weights or None, n)
    body_weight: float | None = None
    if has_explicit_weights:
        if args.body_weight is not None:
            print(
                "[multi_query_fuse] WARNING: both --weights and --body-weight set; "
                "--weights takes precedence",
                file=sys.stderr,
            )
    elif args.body_weight is not None:
        body_weight = args.body_weight
        print(f"[multi_query_fuse] adaptive weighting: body={body_weight}, rest shared per-qid")

    for path0 in matched:
        name = path0.name
        paths = [d / name for d in run_dirs]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            print(f"Warning: skip {name} (missing: {missing})", file=sys.stderr)
            continue
        dfs = [_load_run_tsv(p) for p in paths]
        fused = fuse_n_way_rrf(dfs, weights, args.k_rrf, args.cap, body_weight=body_weight)
        out_path = out_dir / name
        fused.to_csv(out_path, sep="\t", index=False)
        print(f"[multi_query_fuse] wrote {out_path} ({len(fused)} rows)")

    # Eval + plots when gold is available
    has_gold = (args.train_json or args.test_batch_jsons) and not args.no_eval
    if has_gold:
        gold_map, train_stems, test_stems = _load_gold(args.train_json, args.test_batch_jsons or [])
        if gold_map:
            ks_raw = args.ks.strip()
            ks_recall = tuple(int(x) for x in ks_raw.split(",") if x.strip()) if ks_raw else RECALL_KS
            map_ks_raw = args.map_ks.strip()
            map_ks = [int(x) for x in map_ks_raw.split(",") if x.strip()] if map_ks_raw else [10, 20, 50, 100, 200]
            lbl_list = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels.strip() else []
            _eval_fused_runs(
                out_dir, gold_map, ks_recall, train_stems, test_stems, map_ks,
                run_dirs=run_dirs if lbl_list else None,
                labels=lbl_list if lbl_list else None,
            )
        else:
            print("[multi_query_fuse] no gold loaded; skipping evaluation")

    print("[multi_query_fuse] done")


if __name__ == "__main__":
    main()
