#!/usr/bin/env python3
"""Apply global rerank score threshold t* to post_rerank_fusion (RRF) runs by joining doc scores from cross-encoder runs.

Fused TSVs may omit scores; this script looks up (qid, docno) -> score from the matching
cross-encoder run (same logical stem without ``_rrf_poolR*_poolH*_k*`` suffix). Optional CAP truncates after
thresholding; FLOOR guarantees at least FLOOR docs by top-up in original fused order.

When ``--rerank-sub-runs-dir`` is given one or more times (multi-query rerank: ``rerank/cross_encoder/_sub_*/runs``),
scores are max over those sub-run TSVs per (qid, docno) — not the RRF-fused scores in
``rerank/cross_encoder/runs``. Otherwise scores come from a single ``--rerank-runs-dir`` TSV.
"""
from __future__ import annotations

import argparse
import json
import re
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

_SHARED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SHARED))

from retrieval_eval.common import (  # type: ignore
    build_topics_and_gold,
    load_questions,
    normalize_pmid,
)


def _fused_stem_to_rerank_stem(fused_stem: str) -> Optional[str]:
    m = re.match(r"^(best_rrf_.+_top\d+)(?:_rrf_.*)?$", fused_stem)
    return m.group(1) if m else None


def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")
    out = pd.DataFrame(
        {
            "qid": df[qid_col].astype(str),
            "docno": df[doc_col].astype(str).map(normalize_pmid),
        }
    )
    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1
    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def build_max_score_map(rerank_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """(qid, docno) -> max score seen."""
    m: Dict[Tuple[str, str], float] = {}
    for _, row in rerank_df.iterrows():
        key = (str(row["qid"]), str(row["docno"]))
        sc = float(row["score"]) if pd.notna(row["score"]) else float("-inf")
        if key not in m or sc > m[key]:
            m[key] = sc
    return m


def build_max_score_map_across_sub_runs(rr_stem: str, sub_run_dirs: Sequence[Path]) -> Dict[Tuple[str, str], float]:
    """(qid, docno) -> max reranker score over per-field sub-run TSVs (multi-query path)."""
    merged: Dict[Tuple[str, str], float] = {}
    for sd in sub_run_dirs:
        path = sd / f"{rr_stem}.tsv"
        if not path.exists():
            print(f"[t*] warning: missing sub rerank run {path} (stem {rr_stem})", file=sys.stderr)
            continue
        sub_map = build_max_score_map(load_run_tsv(path))
        for k, v in sub_map.items():
            prev = merged.get(k, float("-inf"))
            if v > prev:
                merged[k] = v
    return merged


def apply_cutoff_for_query(
    doc_order: List[str],
    score_map: Dict[Tuple[str, str], float],
    qid: str,
    tstar: float,
    floor: int,
    cap: Optional[int],
) -> Tuple[List[str], int, int, int]:
    """Return (output_docs, n_thresh_pre_cap, n_missing_scores, floor_hit).

    *n_thresh_pre_cap*: distinct docs in fused order with score >= t* (before cap / floor).
    """
    missing = 0
    above: List[str] = []
    seen_thr: set[str] = set()
    for d in doc_order:
        sc = score_map.get((qid, d), float("-inf"))
        if sc != float("-inf") and not np.isfinite(sc):
            sc = float("-inf")
        if sc == float("-inf"):
            missing += 1
        if sc >= tstar and d not in seen_thr:
            seen_thr.add(d)
            above.append(d)
    n_thresh_pre_cap = len(above)
    if cap is not None and cap >= 0:
        capped = above[: int(cap)]
    else:
        capped = list(above)
    out: List[str] = []
    seen = set()
    for d in capped:
        if d not in seen:
            seen.add(d)
            out.append(d)
    floor_hit = 0
    if len(out) < floor:
        need = floor - len(out)
        for d in doc_order:
            if need <= 0:
                break
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
            need -= 1
        floor_hit = 1
    return out, n_thresh_pre_cap, missing, floor_hit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-runs-dir", type=Path, required=True, help="Fused post_rerank_fusion runs/*.tsv")
    p.add_argument(
        "--rerank-runs-dir",
        type=Path,
        default=None,
        help="Single rerank runs dir (with scores). Used when no --rerank-sub-runs-dir is passed.",
    )
    p.add_argument(
        "--rerank-sub-runs-dir",
        type=Path,
        action="append",
        default=None,
        metavar="DIR",
        help="Repeat: rerank/_sub_<field>/runs dirs. When set, threshold uses max score across these TSVs per doc.",
    )
    p.add_argument("--output-runs-dir", type=Path, required=True, help="Directory to write filtered TSVs")
    p.add_argument("--tstar", type=float, required=True, help="Global score threshold (reranker scale)")
    p.add_argument("--floor", type=int, default=5, help="Minimum docs per query after cutoff")
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Max docs after threshold (before floor top-up). Omit for no extra cap beyond input length.",
    )
    p.add_argument("--pattern", type=str, default="*.tsv", help="Glob under input-runs-dir")
    p.add_argument("--train-jsonl", "--train-json", type=Path, default=None, dest="train_jsonl")
    p.add_argument(
        "--test-batch-jsonls",
        "--test-batch-jsons",
        type=Path,
        nargs="*",
        default=None,
        dest="test_batch_jsonls",
    )
    p.add_argument("--disable-metrics", action="store_true", help="Skip gold / residual-loss stats")
    return p.parse_args()


def _load_gold_map(
    train_json: Optional[Path],
    test_jsons: Optional[Sequence[Path]],
    query_field: Optional[str],
) -> Dict[str, List[str]]:
    gold_all: Dict[str, List[str]] = {}
    for jp in [train_json, *(test_jsons or [])]:
        if jp is None or not Path(jp).exists():
            continue
        qs = load_questions(Path(jp))
        _, gm = build_topics_and_gold(qs, query_field=query_field, skip_empty=False)
        gold_all.update(gm)
    return gold_all


def _write_kept_histograms(out_base: Path, per_split_kept: Dict[str, List[int]], pooled_kept: List[int]) -> None:
    if plt is None:
        print("warning: matplotlib not installed; skipping kept-count histograms", file=sys.stderr)
        return
    if not per_split_kept and not pooled_kept:
        return

    fig_dir = out_base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(values: List[int], title: str, out_path: Path) -> None:
        if not values:
            return
        vmax = int(max(values))
        bins = np.arange(0, vmax + 2) - 0.5
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Docs kept per query after t*")
        ax.set_ylabel("Query count")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    for split_stem, vals in per_split_kept.items():
        safe = split_stem.replace("/", "_")
        _plot_one(vals, f"t* kept docs histogram ({split_stem})", fig_dir / f"tstar_kept_hist_{safe}.png")

    _plot_one(pooled_kept, "t* kept docs histogram (pooled)", fig_dir / "tstar_kept_hist_pooled.png")


def main() -> int:
    args = parse_args()
    tstar = float(args.tstar)
    floor = int(args.floor)
    cap = args.cap
    if floor < 0:
        print("Error: --floor must be >= 0", file=sys.stderr)
        return 1
    if cap is not None and cap < floor:
        print(f"Error: --cap ({cap}) must be >= --floor ({floor})", file=sys.stderr)
        return 1

    inp = args.input_runs_dir
    rdir = args.rerank_runs_dir
    sub_run_dirs: List[Path] = [Path(p).resolve() for p in (args.rerank_sub_runs_dir or []) if p is not None]
    out_runs = args.output_runs_dir
    out_runs.mkdir(parents=True, exist_ok=True)
    out_base = out_runs.parent

    if not sub_run_dirs and rdir is None:
        print("Error: provide --rerank-runs-dir and/or --rerank-sub-runs-dir", file=sys.stderr)
        return 1

    paths = sorted(inp.glob(args.pattern))
    if not paths:
        print(f"No files matching {args.pattern} in {inp}", file=sys.stderr)
        return 1

    gold_map: Dict[str, List[str]] = {}
    if not args.disable_metrics:
        gold_map = _load_gold_map(args.train_jsonl, args.test_batch_jsonls, query_field=None)

    summary_rows: List[dict] = []
    per_split_kept: Dict[str, List[int]] = {}
    pooled_kept: List[int] = []
    pooled_res: List[float] = []

    for fused_path in paths:
        fused_stem = fused_path.stem
        rr_stem = _fused_stem_to_rerank_stem(fused_stem)
        if not rr_stem:
            print(f"skip {fused_path.name}: could not parse stem")
            continue
        fused_df = load_run_tsv(fused_path)
        if sub_run_dirs:
            score_map = build_max_score_map_across_sub_runs(rr_stem, sub_run_dirs)
            if not score_map:
                print(
                    f"skip {fused_path.name}: no scores from --rerank-sub-runs-dir for stem {rr_stem}",
                    file=sys.stderr,
                )
                continue
        else:
            if rdir is None:
                print("Error: --rerank-runs-dir is required when no --rerank-sub-runs-dir is set", file=sys.stderr)
                return 1
            rerank_path = rdir / f"{rr_stem}.tsv"
            if not rerank_path.exists():
                print(f"skip {fused_path.name}: missing rerank run {rerank_path}", file=sys.stderr)
                continue
            rerank_df = load_run_tsv(rerank_path)
            score_map = build_max_score_map(rerank_df)

        rows_out: List[dict] = []
        kept_counts: List[int] = []
        thresh_fracs: List[float] = []
        floor_hits = 0
        changed_input = 0
        res_losses: List[float] = []

        for qid, grp in fused_df.groupby("qid", sort=False):
            grp = grp.sort_values("rank")
            doc_order = [str(x) for x in grp["docno"].tolist()]
            input_list = list(doc_order)

            out_docs, n_thresh_pre_cap, _miss, used_floor = apply_cutoff_for_query(
                doc_order, score_map, str(qid), tstar, floor, cap,
            )
            if used_floor:
                floor_hits += 1
            n_in = len(doc_order)
            thresh_fracs.append((n_thresh_pre_cap / n_in) if n_in else 0.0)
            kept_counts.append(len(out_docs))
            pooled_kept.append(len(out_docs))

            if input_list != out_docs:
                changed_input += 1

            for rank, docno in enumerate(out_docs, start=1):
                sc = score_map.get((str(qid), docno), float("nan"))
                rows_out.append(
                    {"qid": str(qid), "docno": docno, "rank": rank, "score": sc},
                )

            if gold_map and not args.disable_metrics:
                gold = [normalize_pmid(x) for x in gold_map.get(str(qid), [])]
                gold = [g for g in gold if g]
                pool = set(doc_order)
                g_pool = {g for g in gold if g in pool}
                if g_pool:
                    kset = set(out_docs)
                    lost = len(g_pool - kset)
                    res_losses.append(lost / len(g_pool))
                    pooled_res.append(res_losses[-1])

        out_df = pd.DataFrame(rows_out)
        out_path = out_runs / f"{fused_stem}.tsv"
        out_df.to_csv(out_path, sep="\t", index=False)

        nq = len(kept_counts)
        per_split_kept[fused_stem] = list(kept_counts)
        row = {
            "split_stem": fused_stem,
            "rerank_stem": rr_stem,
            "score_source": "sub_max" if sub_run_dirs else "rerank_runs",
            "n_queries": nq,
            "mean_kept": float(np.mean(kept_counts)) if kept_counts else float("nan"),
            "median_kept": float(np.median(kept_counts)) if kept_counts else float("nan"),
            "p90_kept": float(np.percentile(kept_counts, 90)) if kept_counts else float("nan"),
            "floor_hit_frac": float(floor_hits / nq) if nq else float("nan"),
            "threshold_keep_frac": float(np.mean(thresh_fracs)) if thresh_fracs else float("nan"),
            "changed_vs_input_frac": float(changed_input / nq) if nq else float("nan"),
            "tstar": tstar,
            "floor": floor,
            "cap": cap if cap is not None else "",
        }
        if res_losses:
            arr = np.asarray(res_losses, dtype=np.float64)
            row["n_eval_pool_gold"] = len(res_losses)
            row["mean_res_loss"] = float(arr.mean())
            row["median_res_loss"] = float(np.median(arr))
            row["p90_res_loss"] = float(np.percentile(arr, 90))
        else:
            row["n_eval_pool_gold"] = 0
            row["mean_res_loss"] = ""
            row["median_res_loss"] = ""
            row["p90_res_loss"] = ""
        summary_rows.append(row)
        print(f"wrote {out_path} ({nq} queries)")

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        if pooled_kept:
            prow: dict = {
                "split_stem": "__pooled__",
                "rerank_stem": "",
                "score_source": sdf["score_source"].iloc[0] if "score_source" in sdf.columns else "",
                "n_queries": int(sdf["n_queries"].sum()),
                "mean_kept": float(np.mean(pooled_kept)),
                "median_kept": float(np.median(pooled_kept)),
                "p90_kept": float(np.percentile(pooled_kept, 90)),
                "floor_hit_frac": float(sdf["floor_hit_frac"].astype(float).mean()),
                "threshold_keep_frac": float(sdf["threshold_keep_frac"].astype(float).mean()),
                "changed_vs_input_frac": float(sdf["changed_vs_input_frac"].astype(float).mean()),
                "tstar": tstar,
                "floor": floor,
                "cap": cap if cap is not None else "",
            }
            if pooled_res:
                arr = np.asarray(pooled_res, dtype=np.float64)
                prow["n_eval_pool_gold"] = len(pooled_res)
                prow["mean_res_loss"] = float(arr.mean())
                prow["median_res_loss"] = float(np.median(arr))
                prow["p90_res_loss"] = float(np.percentile(arr, 90))
            else:
                prow["n_eval_pool_gold"] = 0
                prow["mean_res_loss"] = ""
                prow["median_res_loss"] = ""
                prow["p90_res_loss"] = ""
            sdf = pd.concat([sdf, pd.DataFrame([prow])], ignore_index=True)
        sum_path = out_base / "tstar_cutoff_summary.csv"
        sdf.to_csv(sum_path, index=False)
        print(f"wrote {sum_path}")
        _write_kept_histograms(out_base, per_split_kept, pooled_kept)

    meta = {
        "tstar": tstar,
        "floor": floor,
        "cap": cap,
        "input_runs_dir": str(inp),
        "rerank_runs_dir": str(rdir) if rdir is not None else None,
        "rerank_sub_runs_dirs": [str(p) for p in sub_run_dirs],
        "score_source": "sub_max" if sub_run_dirs else "rerank_runs",
        "output_runs_dir": str(out_runs),
    }
    (out_base / "tstar_cutoff_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
