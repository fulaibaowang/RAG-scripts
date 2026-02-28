#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency 'matplotlib' (pip install matplotlib).") from e

# Allow importing retrieval_eval from public scripts root
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval_eval.common import (  # noqa: E402
    build_topics_and_gold,
    collect_qids_from_questions,
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
)


def parse_int_list(raw: str) -> List[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in items]


def parse_weights(raw: str) -> List[Tuple[float, float]]:
    pairs = [x.strip() for x in raw.split(";") if x.strip()]
    out: List[Tuple[float, float]] = []
    for pair in pairs:
        parts = [p.strip() for p in pair.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid weight pair: {pair}")
        out.append((float(parts[0]), float(parts[1])))
    if not out:
        raise ValueError("No weights parsed. Use format like '1,1;2,1;1,2'.")
    return out


def load_bm25_tsv_run(path: Path) -> pd.DataFrame:
    """Load BM25 TSV run; infer columns if needed."""
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
        if score_col:
            tmp = df.copy()
            tmp["_qid"] = tmp[qid_col].astype(str)
            tmp["_score"] = tmp[score_col].astype(float)
            tmp = tmp.sort_values(["_qid", "_score"], ascending=[True, False])
            tmp["_rank"] = tmp.groupby("_qid").cumcount() + 1
            out = pd.DataFrame(
                {
                    "qid": tmp["_qid"],
                    "docno": tmp[doc_col].astype(str).map(normalize_pmid),
                    "rank": tmp["_rank"].astype(int),
                    "score": tmp["_score"].astype(float),
                }
            )
            return out
        out["rank"] = out.groupby("qid").cumcount() + 1

    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan

    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    return out


def load_dense_parquet_run(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"qid", "docno", "rank"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {needed - set(df.columns)}")
    out = df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str).map(normalize_pmid)
    out["rank"] = out["rank"].astype(int)
    if "score" in out.columns:
        out["score"] = out["score"].astype(float)
    else:
        out["score"] = np.nan
    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    return out


def load_dense_tsv_run(path: Path) -> pd.DataFrame:
    """Load Dense run from canonical TSV (qid, docno, rank, score)."""
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")
    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(normalize_pmid),
    })
    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1
    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan
    out = out.sort_values(["qid", "rank"]).reset_index(drop=True)
    return out


def cut_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[df["rank"] <= int(k)].copy()


def df_to_run_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid, grp in df.groupby("qid", sort=False):
        run[str(qid)] = grp.sort_values("rank")["docno"].astype(str).tolist()
    return run


def runmap_to_tsv(run_map: Dict[str, List[str]], out_path: Path) -> None:
    rows = []
    for qid, docs in run_map.items():
        for i, doc in enumerate(docs, start=1):
            rows.append({"qid": qid, "rank": i, "docno": doc, "score": np.nan})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def recall_at_k_eff(gold: set[str], ranked: List[str], k: int) -> Tuple[float, int]:
    k_eff = min(int(k), len(ranked))
    if not gold or k_eff <= 0:
        return 0.0, k_eff
    return len(gold.intersection(ranked[:k_eff])) / len(gold), k_eff


def evaluate_recall_points(
    gold_map: Dict[str, List[str]],
    run_map: Dict[str, List[str]],
    ks: Tuple[int, ...],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    qids = list(gold_map.keys())

    for k in ks:
        recalls: List[float] = []
        shortfalls: List[float] = []
        keffs: List[int] = []
        for qid in qids:
            gold = set(map(str, gold_map.get(qid, [])))
            ranked = list(map(str, run_map.get(qid, [])))
            r, k_eff = recall_at_k_eff(gold, ranked, int(k))
            recalls.append(r)
            keffs.append(k_eff)
            shortfalls.append(1.0 if len(ranked) < int(k) else 0.0)
        out[f"MeanR@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        out[f"ShortfallRate@{k}"] = float(np.mean(shortfalls)) if shortfalls else 0.0
        out[f"MeanKeff@{k}"] = float(np.mean(keffs)) if keffs else 0.0

    return out


def fuse_rrf(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    k_bm25: int,
    k_dense: int,
    k_rrf: int = 60,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
    k_out: int = 5000,
) -> Dict[str, List[str]]:
    b = cut_topk(bm25_df, k_bm25)
    d = cut_topk(dense_df, k_dense)

    run: Dict[str, List[str]] = {}
    b_grp = {qid: grp for qid, grp in b.groupby("qid", sort=False)}
    d_grp = {qid: grp for qid, grp in d.groupby("qid", sort=False)}
    all_qids = list(dict.fromkeys(list(b_grp.keys()) + list(d_grp.keys())))

    for qid in all_qids:
        scores: Dict[str, float] = {}
        if qid in b_grp:
            for _, row in b_grp[qid].iterrows():
                doc = str(row["docno"])
                r = int(row["rank"])
                scores[doc] = scores.get(doc, 0.0) + (float(w_bm25) / (float(k_rrf) + r))
        if qid in d_grp:
            for _, row in d_grp[qid].iterrows():
                doc = str(row["docno"])
                r = int(row["rank"])
                scores[doc] = scores.get(doc, 0.0) + (float(w_dense) / (float(k_rrf) + r))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        run[str(qid)] = [doc for doc, _ in ranked[: int(k_out)]]

    return run


def rank_option1(df: pd.DataFrame, test_splits: List[str], cap_eff: int, k_max_eval_eff: int) -> pd.DataFrame:
    """Rank configs by MeanR@cap (fixed-depth; no adaptive K)."""
    d = df[df["split"].isin(test_splits)].copy()
    recall_cols = [c for c in d.columns if c.startswith("MeanR@")]
    extra = [f"ShortfallRate@{cap_eff}", f"MeanKeff@{cap_eff}"]
    agg_cols = {c: "mean" for c in recall_cols + extra if c in d.columns}
    if not agg_cols:
        agg_cols = {f"MeanR@{cap_eff}": "mean", f"MeanR@{k_max_eval_eff}": "mean"}
    grp = d.groupby(["k_rrf", "w_bm25", "w_dense"], as_index=False).agg(agg_cols)
    sort_cols = [f"MeanR@{cap_eff}", f"MeanR@{k_max_eval_eff}"]
    sort_cols = [c for c in sort_cols if c in grp.columns]
    if not sort_cols:
        sort_cols = [c for c in grp.columns if c.startswith("MeanR@")][:2]
    grp = grp.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return grp


def plot_recall_curves(
    results_df: pd.DataFrame,
    bm25_runs: Dict[str, pd.DataFrame],
    dense_runs: Dict[str, pd.DataFrame],
    gold_maps: Dict[str, Dict[str, List[str]]],
    output_dir: Path,
    curve_splits: List[str],
    test_splits: List[str],
    ks_eval: Tuple[int, ...],
    cap_eff: int,
    k_max_eval_eff: int,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
    p: float,
):
    """Plot recall curves per split (curve_splits: train + test) and avg over test_splits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def baseline_metrics_for_split(split: str, method: str) -> Dict[str, float]:
        if method == "BM25":
            run = df_to_run_map(cut_topk(bm25_runs[split], k_max_eval_eff))
        elif method == "Dense":
            run = df_to_run_map(cut_topk(dense_runs[split], k_max_eval_eff))
        else:
            raise ValueError("method must be BM25 or Dense")
        return evaluate_recall_points(gold_maps[split], run, ks=ks_eval)

    # Use only RECALL_KS-derived K values (filtered by k_max_eval_eff)
    ks = list(ks_eval)

    for split in curve_splits:
        row = results_df[
            (results_df["split"] == split)
            & (results_df["k_rrf"] == k_rrf)
            & (results_df["w_bm25"] == w_bm25)
            & (results_df["w_dense"] == w_dense)
        ].iloc[0]

        bm25_metrics = baseline_metrics_for_split(split, "BM25")
        dense_metrics = baseline_metrics_for_split(split, "Dense")

        hybrid_label = f"Hybrid (k_rrf={k_rrf}, {w_bm25:.1f}:{w_dense:.1f})"
        plt.figure()
        plt.plot(
            ks, [row.get(f"MeanR@{k}", np.nan) for k in ks],
            label=hybrid_label, marker="o", markerfacecolor="none", markeredgewidth=1.5,
        )
        plt.plot(ks, [bm25_metrics.get(f"MeanR@{k}", np.nan) for k in ks], marker="s", label="BM25")
        plt.plot(ks, [dense_metrics.get(f"MeanR@{k}", np.nan) for k in ks], marker="^", label="Dense")

        rmax = row.get(f"MeanR@{k_max_eval_eff}", np.nan)
        if np.isfinite(rmax):
            plt.axhline(p * rmax, linestyle="--", label=f"p*Rmax (p={p})")

        plt.xlabel("K")
        plt.ylabel("Mean Recall@K (k_eff per query)")
        plt.title(f"Recall curves ({split})")
        plt.legend(fontsize="small")
        plt.savefig(output_dir / f"recall_curve_{split}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Only generate test_avg plot when there are 2+ test splits (no need for "average" with one)
    if len(test_splits) >= 2:
        mean_best = {}
        mean_bm25 = {}
        mean_dense = {}
        for k in ks_eval:
            mean_best[f"MeanR@{k}"] = (
                results_df[
                    (results_df["split"].isin(test_splits))
                    & (results_df["k_rrf"] == k_rrf)
                    & (results_df["w_bm25"] == w_bm25)
                    & (results_df["w_dense"] == w_dense)
                ][f"MeanR@{k}"]
                .mean()
            )
            mean_bm25[f"MeanR@{k}"] = float(
                np.mean([baseline_metrics_for_split(s, "BM25").get(f"MeanR@{k}", np.nan) for s in test_splits])
            )
            mean_dense[f"MeanR@{k}"] = float(
                np.mean([baseline_metrics_for_split(s, "Dense").get(f"MeanR@{k}", np.nan) for s in test_splits])
            )

        hybrid_label = f"Hybrid (k_rrf={k_rrf}, {w_bm25:.1f}:{w_dense:.1f})"
        plt.figure()
        plt.plot(
            ks, [mean_best.get(f"MeanR@{k}", np.nan) for k in ks],
            label=hybrid_label, marker="o", markerfacecolor="none", markeredgewidth=1.5,
        )
        plt.plot(ks, [mean_bm25.get(f"MeanR@{k}", np.nan) for k in ks], marker="s", label="BM25")
        plt.plot(ks, [mean_dense.get(f"MeanR@{k}", np.nan) for k in ks], marker="^", label="Dense")

        rmax = mean_best.get(f"MeanR@{k_max_eval_eff}", np.nan)
        target = p * rmax if np.isfinite(rmax) else np.nan
        if np.isfinite(target):
            plt.axhline(target, linestyle="--", label=f"p*Rmax (p={p})")

        plt.xlabel("K")
        plt.ylabel("Mean Recall@K (k_eff per query)")
        plt.title("Recall curves (avg over test splits)")
        plt.legend(fontsize="small")
        plt.savefig(output_dir / "recall_curve_test_avg.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_shortfall(df_cfg: pd.DataFrame, cap: int, topn: int, save_path: Path) -> None:
    d = df_cfg.copy()
    d = d.sort_values(by=f"ShortfallRate@{cap}", ascending=False).head(topn)
    plt.figure()
    if len(d) == 0 or d[f"ShortfallRate@{cap}"].max() == 0:
        # Show zero explicitly instead of empty plot
        plt.bar([0], [0], width=0.5)
        plt.xticks([0], ["all configs"])
        plt.ylabel(f"ShortfallRate@{cap}")
        plt.title("Top shortfall configs (rate = 0)")
        plt.ylim(0, 0.1)
    else:
        plt.bar(range(len(d)), d[f"ShortfallRate@{cap}"].values)
        plt.xticks(range(len(d)), [f"krrf={int(r)}" for r in d["k_rrf"].values], rotation=45, ha="right")
        plt.ylabel(f"ShortfallRate@{cap}")
        plt.title("Top shortfall configs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_keff(df_cfg: pd.DataFrame, cap: int, topn: int, save_path: Path) -> None:
    d = df_cfg.copy()
    d = d.sort_values(by=f"MeanKeff@{cap}", ascending=True).head(topn)
    plt.figure()
    plt.bar(range(len(d)), d[f"MeanKeff@{cap}"].values)
    plt.xticks(range(len(d)), [f"krrf={int(r)}" for r in d["k_rrf"].values], rotation=45, ha="right")
    plt.ylabel(f"MeanKeff@{cap}")
    plt.title("Top low-k_eff configs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _evaluate_single_config(
    split: str,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    gold_map: Dict[str, List[str]],
    kb: int,
    kd: int,
    k_max_eval_eff: int,
    ks_eval: Tuple[int, ...],
    fixed_ks: Tuple[int, ...],
    cap_eff: int,
    p: float,
    cap: int,
    k_max_eval: int,
) -> Dict[str, Any]:
    """Worker function for parallel evaluation."""
    fused = fuse_rrf(
        bm25_df=bm25_df,
        dense_df=dense_df,
        k_bm25=kb,
        k_dense=kd,
        k_rrf=k_rrf,
        w_bm25=w_bm25,
        w_dense=w_dense,
        k_out=k_max_eval_eff,
    )

    metrics = evaluate_recall_points(gold_map, fused, ks=ks_eval)
    eval_summary, _ = evaluate_run(gold_map, fused, ks_recall=fixed_ks)

    row: Dict[str, Any] = {
        "split": split,
        "k_rrf": int(k_rrf),
        "w_bm25": float(w_bm25),
        "w_dense": float(w_dense),
        "KB": int(kb),
        "KD": int(kd),
        "CAP": int(cap),
        "CAP_eff": int(cap_eff),
        "K_MAX_EVAL": int(k_max_eval),
        "K_MAX_EVAL_eff": int(k_max_eval_eff),
        "P": float(p),
        f"ShortfallRate@{cap_eff}": metrics.get(f"ShortfallRate@{cap_eff}", np.nan),
        f"MeanKeff@{cap_eff}": metrics.get(f"MeanKeff@{cap_eff}", np.nan),
        f"MeanR@{cap_eff}": metrics.get(f"MeanR@{cap_eff}", np.nan),
        f"MeanR@{k_max_eval_eff}": metrics.get(f"MeanR@{k_max_eval_eff}", np.nan),
        "MAP@10": eval_summary.get("MAP@10", np.nan),
        "MRR@10": eval_summary.get("MRR@10", np.nan),
        "GMAP@10": eval_summary.get("GMAP@10", np.nan),
        "Success@10": eval_summary.get("Success@10", np.nan),
    }
    for k in ks_eval:
        row[f"MeanR@{k}"] = metrics.get(f"MeanR@{k}", np.nan)
    return row


def plot_param_sensitivity(
    results_df: pd.DataFrame,
    test_splits: List[str],
    cap_eff: int,
    figs_dir: Path,
    baseline_cfg: Tuple[int, float, float],
) -> None:
    df_test = results_df[results_df["split"].isin(test_splits)].copy()
    recall_cols = [c for c in df_test.columns if c.startswith("MeanR@")] 
    recall_cols = sorted(recall_cols, key=lambda c: int(c.split("@")[1]))

    agg = df_test.groupby(["k_rrf", "w_bm25", "w_dense"], as_index=False).agg(
        {c: "mean" for c in recall_cols if c in df_test.columns}
    )
    agg["weight_ratio"] = agg["w_dense"] / agg["w_bm25"]
    agg = agg.sort_values(["weight_ratio", "k_rrf"]).reset_index(drop=True)

    heat_metrics = ["MeanR@200", f"MeanR@{cap_eff}"]
    heat_metrics = [m for m in heat_metrics if m in agg.columns]
    for metric in heat_metrics:
        piv = agg.pivot_table(index="weight_ratio", columns="k_rrf", values=metric, aggfunc="mean")
        plt.figure()
        plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.colorbar(label=metric)
        plt.xticks(range(len(piv.columns)), piv.columns.astype(str))
        plt.yticks(range(len(piv.index)), [str(x) for x in piv.index])
        plt.xlabel("k_rrf")
        plt.ylabel("w_dense / w_bm25")
        plt.title(f"{metric}: weight_ratio x k_rrf")
        plt.savefig(figs_dir / f"heatmap_{metric.replace('@', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close()

    if not recall_cols:
        return

    base_krrf, base_wb, base_wd = baseline_cfg
    baseline = agg[
        (agg["k_rrf"] == int(base_krrf))
        & (agg["w_bm25"] == float(base_wb))
        & (agg["w_dense"] == float(base_wd))
    ]
    if len(baseline) == 0:
        baseline = agg.iloc[[0]]
    baseline = baseline.iloc[0]

    sort_col = f"MeanR@{cap_eff}" if f"MeanR@{cap_eff}" in agg.columns else (recall_cols[0] if recall_cols else None)
    if sort_col is None:
        return
    top_configs = agg.sort_values(sort_col, ascending=False).head(6)
    ks = [int(c.split("@")[1]) for c in recall_cols]
    plt.figure()
    for _, row in top_configs.iterrows():
        deltas = [row[f"MeanR@{k}"] - baseline[f"MeanR@{k}"] for k in ks]
        label = f"krrf={int(row.k_rrf)} wb={row.w_bm25} wd={row.w_dense}"
        plt.plot(ks, deltas, marker="o", label=label)
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xlabel("K")
    plt.ylabel("Delta Recall vs baseline")
    plt.title("Delta-from-baseline recall curves (top configs)")
    plt.legend(fontsize="small")
    plt.savefig(figs_dir / "delta_recall_top_configs.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate hybrid (BM25+Dense) RRF fusion for BioASQ.")
    ap.add_argument("--bm25_runs_dir", required=True, help="Path to BM25 run TSV folder")
    ap.add_argument("--bm25_method", default="BM25_RM3")
    ap.add_argument("--bm25_topk", type=int, default=5000)
    ap.add_argument("--dense_root", required=True, help="Path to dense output folder (dense_*.parquet)")

    ap.add_argument("--train-json", dest="train_json", required=True)
    ap.add_argument("--test_batch_jsons", nargs="+", required=True, help="List of 13B*_golden.json files")

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mode", choices=["sweep", "default"], default="sweep")
    ap.add_argument("--k_rrf_list", default="60,100")
    ap.add_argument("--weights", default="1.0,1.0;2.0,1.0;1.0,2.0")

    ap.add_argument("--k_rrf", type=int, default=150)
    ap.add_argument("--w_bm25", type=float, default=1.0)
    ap.add_argument("--w_dense", type=float, default=1.0)

    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--k_max_eval", type=int, default=5000)
    ap.add_argument("--ks", type=str, default=",".join(map(str, RECALL_KS)), help="Comma-separated K values for recall (default: RECALL_KS)")
    ap.add_argument("--p", type=float, default=0.95)
    ap.add_argument("--kb", type=int, default=None)
    ap.add_argument("--kd", type=int, default=None)

    ap.add_argument("--no_exclude_test_qids", action="store_true")
    ap.add_argument("--no_eval", action="store_true", help="Skip evaluation; use fixed config and write run TSVs only")
    ap.add_argument("--save_plots", action="store_true", help="Force plot generation")
    ap.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    ap.add_argument("--jobs", type=int, default=None, help="Number of parallel processes (default: CPU count)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    figs_dir = out_dir / "figures"
    runs_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    test_files: List[Path]
    test_files = [Path(p).resolve() for p in args.test_batch_jsons]

    for fp in test_files:
        if not fp.exists():
            raise FileNotFoundError(f"Missing test batch: {fp}")

    train_questions_all = load_questions(Path(args.train_json).resolve())
    test_qids = set()
    for fp in test_files:
        test_qids |= collect_qids_from_questions(load_questions(fp))

    if args.no_exclude_test_qids:
        train_questions = train_questions_all
    else:
        train_questions = [
            q
            for i, q in enumerate(train_questions_all)
            if str(q.get("id") or q.get("qid") or i) not in test_qids
        ]

    train_stem = Path(args.train_json).stem

    gold_maps: Dict[str, Dict[str, List[str]]] = {}
    _, gold_maps[train_stem] = build_topics_and_gold(train_questions)

    for fp in test_files:
        questions = load_questions(fp)
        _, gold_map = build_topics_and_gold(questions)
        gold_maps[fp.stem] = gold_map

    splits = [train_stem] + [fp.stem for fp in test_files]

    bm25_runs: Dict[str, pd.DataFrame] = {}
    dense_runs: Dict[str, pd.DataFrame] = {}

    bm25_runs_dir = Path(args.bm25_runs_dir).resolve()
    dense_root = Path(args.dense_root).resolve()

    for split in splits:
        bm25_path = bm25_runs_dir / f"{args.bm25_method}__{split}__top{args.bm25_topk}.tsv"
        if not bm25_path.exists():
            raise FileNotFoundError(f"Missing BM25 run: {bm25_path}")
        bm25_runs[split] = load_bm25_tsv_run(bm25_path)
        dense_tsv = dense_root / "runs" / f"dense_{split}.tsv"
        dense_parquet = dense_root / f"dense_{split}.parquet"
        if dense_tsv.exists():
            dense_runs[split] = load_dense_tsv_run(dense_tsv)
        elif dense_parquet.exists():
            dense_runs[split] = load_dense_parquet_run(dense_parquet)
        else:
            raise FileNotFoundError(f"Missing dense run: {dense_tsv} or {dense_parquet}")

    k_max_eval = int(args.k_max_eval)
    kb = int(args.kb) if args.kb is not None else k_max_eval
    kd = int(args.kd) if args.kd is not None else k_max_eval
    k_max_eval_eff = int(min(k_max_eval, kb + kd))
    cap_eff = int(min(int(args.cap), k_max_eval_eff))

    # Evaluation Ks: use only --ks (RECALL_KS) up to k_max_eval_eff (no extra interpolated Ks)
    ks_from_config = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if not ks_from_config:
        ks_from_config = RECALL_KS
    fixed_ks = tuple(k for k in ks_from_config if k <= k_max_eval_eff)
    ks_eval = fixed_ks

    if not ks_eval:
        raise ValueError("No evaluation K values. Check --cap and --k_max_eval.")

    print(
        f"KB={kb} KD={kd} CAP={args.cap} K_MAX_EVAL={k_max_eval} => cap_eff={cap_eff} k_max_eval_eff={k_max_eval_eff}"
    )
    print("ks_eval:", ks_eval)

    if args.no_eval:
        k_out = min(cap_eff, k_max_eval_eff)
        for split in splits:
            best_run = fuse_rrf(
                bm25_df=bm25_runs[split],
                dense_df=dense_runs[split],
                k_bm25=k_out,
                k_dense=k_out,
                k_rrf=int(args.k_rrf),
                w_bm25=float(args.w_bm25),
                w_dense=float(args.w_dense),
                k_out=k_out,
            )
            runmap_to_tsv(best_run, runs_dir / f"best_rrf_{split}_top{k_out}.tsv")
        config = vars(args)
        config.update({"ks_cap": list(ks_cap), "ks_eval": list(ks_eval)})
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("No-eval mode: runs saved to", runs_dir)
        return

    rows: List[Dict[str, Any]] = []

    if args.mode == "sweep":
        k_rrf_list = parse_int_list(args.k_rrf_list)
        weights = parse_weights(args.weights)
    else:
        k_rrf_list = [int(args.k_rrf)]
        weights = [(float(args.w_bm25), float(args.w_dense))]

    # Prepare tasks for parallel execution
    tasks = []
    for w_bm25, w_dense in weights:
        for k_rrf in k_rrf_list:
            for split in splits:
                tasks.append((
                    split, k_rrf, w_bm25, w_dense,
                    bm25_runs[split], dense_runs[split], gold_maps[split],
                    kb, kd, k_max_eval_eff, ks_eval, fixed_ks, cap_eff,
                    float(args.p), int(args.cap), k_max_eval
                ))

    n_jobs = args.jobs if args.jobs and args.jobs > 0 else None
    print(f"Evaluating {len(tasks)} configs with {n_jobs or 'auto'} workers...")

    if n_jobs == 1:
        # Sequential execution
        for task in tasks:
            row = _evaluate_single_config(*task)
            rows.append(row)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_evaluate_single_config, *task): i for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    rows.append(row)
                    if len(rows) % 10 == 0:
                        print(f"Completed {len(rows)}/{len(tasks)} configs")
                except Exception as e:
                    idx = futures[future]
                    print(f"Error evaluating task {idx}: {e}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "results_all.csv", index=False)

    test_splits = [fp.stem for fp in test_files]
    if not test_splits:
        test_splits = [train_stem]
        print("[warn] No test splits provided; using train split for ranking.")

    ranked = rank_option1(results_df, test_splits, cap_eff=cap_eff, k_max_eval_eff=k_max_eval_eff)
    ranked.to_csv(out_dir / "ranked_test_avg.csv", index=False)

    if ranked.empty:
        raise ValueError("No configs were evaluated; cannot select best config.")

    best_cfg = ranked.iloc[0].to_dict()
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    for split in splits:
        best_row = results_df[
            (results_df["split"] == split)
            & (results_df["k_rrf"] == int(best_cfg["k_rrf"]))
            & (results_df["w_bm25"] == float(best_cfg["w_bm25"]))
            & (results_df["w_dense"] == float(best_cfg["w_dense"]))
        ].iloc[0]

        k_out = min(cap_eff, k_max_eval_eff)
        best_run = fuse_rrf(
            bm25_df=bm25_runs[split],
            dense_df=dense_runs[split],
            k_bm25=k_out,
            k_dense=k_out,
            k_rrf=int(best_row["k_rrf"]),
            w_bm25=float(best_row["w_bm25"]),
            w_dense=float(best_row["w_dense"]),
            k_out=k_out,
        )
        runmap_to_tsv(best_run, runs_dir / f"best_rrf_{split}_top{k_out}.tsv")

    save_plots = bool((not args.no_plots) or args.save_plots)
    if save_plots:
        best_cfg = ranked.iloc[0]
        curve_splits_list = [train_stem] + test_splits
        plot_recall_curves(
            results_df=results_df,
            bm25_runs=bm25_runs,
            dense_runs=dense_runs,
            gold_maps=gold_maps,
            output_dir=figs_dir,
            curve_splits=curve_splits_list,
            test_splits=test_splits,
            ks_eval=ks_eval,
            cap_eff=cap_eff,
            k_max_eval_eff=k_max_eval_eff,
            k_rrf=int(best_cfg["k_rrf"]),
            w_bm25=float(best_cfg["w_bm25"]),
            w_dense=float(best_cfg["w_dense"]),
            p=float(args.p),
        )

        plot_shortfall(ranked, cap=cap_eff, topn=10, save_path=figs_dir / "shortfall_top10.png")
        plot_keff(ranked, cap=cap_eff, topn=10, save_path=figs_dir / "keff_top10.png")

        if args.mode == "sweep":
            plot_param_sensitivity(
                results_df=results_df,
                test_splits=test_splits,
                cap_eff=cap_eff,
                figs_dir=figs_dir,
                baseline_cfg=(
                    int(best_cfg["k_rrf"]),
                    float(best_cfg["w_bm25"]),
                    float(best_cfg["w_dense"]),
                ),
            )

    config = vars(args)
    config.update({"ks_eval": list(ks_eval)})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("[done]", out_dir)


if __name__ == "__main__":
    main()
