from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _guard_rail_fuse_topk(
    bge_docs: List[str],
    hybrid_docs: List[str],
    m_bge: int,
    k_top: int,
) -> List[str]:
    """Apply guard-rail fusion for the top-k prefix.

    First take top m_bge docs from BGE, then fill remaining (k_top - m_bge)
    slots with docs from Hybrid that are not already selected. The remainder
    of the ranking is filled with the remaining BGE docs in original order.
    """
    selected: List[str] = []
    seen = set()

    # 1) top-m from BGE
    for doc in bge_docs:
        if doc in seen:
            continue
        selected.append(doc)
        seen.add(doc)
        if len(selected) >= m_bge or len(selected) >= k_top:
            break

    # 2) anchors from Hybrid to reach k_top
    if len(selected) < k_top:
        for doc in hybrid_docs:
            if doc in seen:
                continue
            selected.append(doc)
            seen.add(doc)
            if len(selected) >= k_top:
                break

    # 3) remainder from BGE
    for doc in bge_docs:
        if doc in seen:
            continue
        selected.append(doc)
        seen.add(doc)

    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a top-k guard rail on existing BGE rerank runs using Hybrid anchors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hybrid-runs-dir", type=Path, required=True, help="Hybrid runs directory (best_rrf_*.tsv).")
    p.add_argument("--rerank-runs-dir", type=Path, required=True, help="BGE rerank runs directory (best_rrf_*.tsv).")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for guard-rail runs and metrics (created if missing).",
    )
    p.add_argument(
        "--k-top",
        type=int,
        default=10,
        help="Final top-k cutoff where guard rail is applied (e.g. 10).",
    )
    p.add_argument(
        "--m-bge",
        type=int,
        default=8,
        help="Number of top docs taken from BGE within the final top-k (rest filled from Hybrid).",
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

    # Map split -> guard-rail run_df
    fused_runs: Dict[str, pd.DataFrame] = {}

    for rerank_path in sorted(rerank_runs_dir.glob("*.tsv")):
        name = rerank_path.stem  # e.g. best_rrf_training14b_3pct_sample_top5000
        split = _parse_split_from_run_stem(name)
        if split is None:
            print(f"skip {rerank_path} (could not parse split)")
            continue

        hybrid_path = hybrid_runs_dir / f"{name}.tsv"
        if not hybrid_path.exists():
            print(f"skip {rerank_path}: missing hybrid run {hybrid_path}")
            continue

        print(f"Guard-rail fusion for split={split} using {rerank_path.name} + {hybrid_path.name}")
        bge_df = load_run_tsv(rerank_path)
        hyb_df = load_run_tsv(hybrid_path)

        fused_rows = []
        for qid, bge_group in bge_df.groupby("qid", sort=False):
            bge_docs = bge_group["docno"].tolist()
            hyb_group = hyb_df[hyb_df["qid"] == qid]
            hybrid_docs = hyb_group["docno"].tolist()

            if not bge_docs and not hybrid_docs:
                continue

            fused_docs = _guard_rail_fuse_topk(
                bge_docs=bge_docs,
                hybrid_docs=hybrid_docs,
                m_bge=int(args.m_bge),
                k_top=int(args.k_top),
            )

            for rank, docno in enumerate(fused_docs, start=1):
                fused_rows.append({"qid": str(qid), "docno": str(docno), "rank": rank})

        if not fused_rows:
            print(f"warning: no fused rows for split={split}")
            continue

        fused_df = pd.DataFrame(fused_rows).sort_values(["qid", "rank"]).reset_index(drop=True)
        fused_runs[split] = fused_df

        out_path = out_runs / f"{name}_guardk{int(args.k_top)}_m{int(args.m_bge)}.tsv"
        fused_df.to_csv(out_path, sep="\t", index=False)

    if args.disable_metrics or not fused_runs:
        print("Guard-rail fusion complete. Metrics skipped or no runs to evaluate.")
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

        metrics, perq = evaluate_run(gold_map, run_map, ks_recall=ks_recall)
        perq.to_csv(out_per_query / f"{split}_guardk{int(args.k_top)}_m{int(args.m_bge)}.csv", index=False)

        row = {"split": split}
        row.update(metrics)
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "metrics.csv", index=False)
        print(f"Wrote guard-rail metrics to {out_dir / 'metrics.csv'}")
    else:
        print("No summary metrics to write.")


if __name__ == "__main__":
    main()

