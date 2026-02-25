from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

# shared_scripts/ (parent of rerank/) so retrieval_eval is findable when run directly
_SHARED_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SHARED_SCRIPTS))

from retrieval_eval.common import (  # type: ignore
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    run_df_to_run_map,
)


def _parse_split_from_run_stem(run_stem: str) -> Optional[str]:
    m = re.fullmatch(r"best_rrf_(.+)_top\d+", run_stem)
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
            "docno": df[doc_col].astype(str),
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


def _rrf_fuse_docs(
    bge_docs: List[str],
    hybrid_docs: List[str],
    pool_top: int,
    k_rrf: int,
    w_bge: float,
    w_hybrid: float,
) -> List[str]:
    """Union of top-N from BGE + Hybrid, weighted RRF, full fused ranking."""
    bge_top = bge_docs[:pool_top]
    hyb_top = hybrid_docs[:pool_top]
    rank_bge = {d: i + 1 for i, d in enumerate(bge_top)}
    rank_hyb = {d: i + 1 for i, d in enumerate(hyb_top)}

    union: List[str] = list(dict.fromkeys(bge_top + hyb_top))
    scores = []
    for d in union:
        s = 0.0
        rb = rank_bge.get(d)
        rh = rank_hyb.get(d)
        if rb is not None:
            s += w_bge / (k_rrf + rb)
        if rh is not None:
            s += w_hybrid / (k_rrf + rh)
        scores.append((d, s))
    scores.sort(key=lambda x: -x[1])
    return [d for d, _ in scores]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RRF fusion of existing BGE rerank runs and Hybrid runs (top-50 union, weighted RRF, top-10).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hybrid-runs-dir", type=Path, required=True, help="Hybrid runs directory (best_rrf_*.tsv).")
    p.add_argument("--rerank-runs-dir", type=Path, required=True, help="BGE rerank runs directory (best_rrf_*.tsv).")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fused runs and metrics (created if missing).",
    )
    p.add_argument(
        "--pool-top",
        type=int,
        default=50,
        help="Pool size per list: union of top-N from BGE and Hybrid.",
    )
    p.add_argument(
        "--k-rrf",
        type=int,
        default=60,
        help="RRF K parameter in 1/(K+rank).",
    )
    p.add_argument(
        "--w-bge",
        type=float,
        default=0.8,
        help="Weight for BGE rerank scores in RRF.",
    )
    p.add_argument(
        "--w-hybrid",
        type=float,
        default=0.2,
        help="Weight for Hybrid scores in RRF.",
    )
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Training questions JSON (BioASQ format) for metrics (optional).",
    )
    p.add_argument(
        "--test-batch-jsons",
        "--test_batch_jsons",
        type=Path,
        nargs="*",
        default=None,
        help="BioASQ test batch JSONs for metrics (optional).",
    )
    p.add_argument(
        "--ks-recall",
        type=str,
        default="50,100,200,300,400,500,1000,2000,5000",
        help="Recall@K grid (comma-separated) for evaluation.",
    )
    p.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Skip metrics computation (only write fused runs).",
    )
    return p.parse_args()


def _parse_ks_recall(raw: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else ()


def main() -> None:
    args = parse_args()

    hybrid_runs_dir: Path = args.hybrid_runs_dir
    rerank_runs_dir: Path = args.rerank_runs_dir
    out_dir: Path = args.output_dir
    out_runs = out_dir / "runs"
    out_per_query = out_dir / "per_query"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_runs.mkdir(parents=True, exist_ok=True)
    out_per_query.mkdir(parents=True, exist_ok=True)

    ks_recall = _parse_ks_recall(args.ks_recall)

    fused_runs: Dict[str, pd.DataFrame] = {}

    for rerank_path in sorted(rerank_runs_dir.glob("*.tsv")):
        name = rerank_path.stem  # e.g. best_rrf_training14b_10pct_sample_top5000
        split = _parse_split_from_run_stem(name)
        if split is None:
            print(f"skip {rerank_path} (could not parse split)")
            continue

        hybrid_path = hybrid_runs_dir / f"{name}.tsv"
        if not hybrid_path.exists():
            print(f"skip {rerank_path}: missing hybrid run {hybrid_path}")
            continue

        print(
            f"RRF fusion for split={split} using {rerank_path.name} "
            f"+ {hybrid_path.name} (pool_top={args.pool_top}, k_rrf={args.k_rrf}, "
            f"w_bge={args.w_bge}, w_hybrid={args.w_hybrid})"
        )
        bge_df = load_run_tsv(rerank_path)
        hyb_df = load_run_tsv(hybrid_path)

        fused_rows = []
        for qid, bge_group in bge_df.groupby("qid", sort=False):
            bge_docs = bge_group["docno"].tolist()
            hyb_group = hyb_df[hyb_df["qid"] == qid]
            hybrid_docs = hyb_group["docno"].tolist()

            if not bge_docs and not hybrid_docs:
                continue

            fused_docs = _rrf_fuse_docs(
                bge_docs=bge_docs,
                hybrid_docs=hybrid_docs,
                pool_top=int(args.pool_top),
                k_rrf=int(args.k_rrf),
                w_bge=float(args.w_bge),
                w_hybrid=float(args.w_hybrid),
            )

            for rank, docno in enumerate(fused_docs, start=1):
                fused_rows.append({"qid": str(qid), "docno": str(docno), "rank": rank})

        if not fused_rows:
            print(f"warning: no fused rows for split={split}")
            continue

        fused_df = pd.DataFrame(fused_rows).sort_values(["qid", "rank"]).reset_index(drop=True)
        fused_runs[split] = fused_df

        out_path = out_runs / f"{name}_rrf_pool{int(args.pool_top)}_k{int(args.k_rrf)}.tsv"
        fused_df.to_csv(out_path, sep="\t", index=False)

    if args.disable_metrics or not fused_runs:
        print("RRF fusion complete. Metrics skipped or no runs to evaluate.")
        return

    # Build gold and evaluate, if questions are provided
    train_json: Optional[Path] = args.train_json
    test_batch_jsons: Optional[Sequence[Path]] = args.test_batch_jsons

    all_questions: List[dict] = []
    if train_json and train_json.exists():
        all_questions.extend(load_questions(train_json))
    if test_batch_jsons:
        for p in test_batch_jsons:
            if p and Path(p).exists():
                all_questions.extend(load_questions(Path(p)))

    if not all_questions:
        print("No questions JSONs provided; skipping metrics.")
        return

    topics_df, gold_map = build_topics_and_gold(all_questions, query_field=None)
    _ = topics_df  # unused; kept for symmetry

    summary_rows = []

    for split, fused_df in fused_runs.items():
        run_map = run_df_to_run_map(fused_df, qid_col="qid", docno_col="docno")
        # Restrict gold to qids in this run so MAP@10 is per-split (not averaged over all splits)
        gold_for_run = {qid: gold_map[qid] for qid in run_map if qid in gold_map}
        if not gold_for_run:
            continue
        metrics, perq = evaluate_run(gold_for_run, run_map, ks_recall=ks_recall)
        perq.to_csv(
            out_per_query / f"{split}_rrf_pool{int(args.pool_top)}_k{int(args.k_rrf)}.csv",
            index=False,
        )

        row = {"split": split}
        row.update(metrics)
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "metrics.csv", index=False)
        print(f"Wrote RRF fusion metrics to {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

