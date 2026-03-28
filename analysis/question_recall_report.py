#!/usr/bin/env python3
"""Full per-question recall report across retrieval stages.

Builds a table for **all** questions with:
  - question metadata (qid, body, type, n_rel)
  - hybrid Recall@cap  (cap read from hybrid/config.json)
  - rerank Recall at multiple cut-offs (R@10, R@100, R@200, R@500, R@1000)
  - golden-truth PMIDs and their PubMed titles

Reuses shared helpers from low_recall_report.py in the same package.

Usage examples
--------------
# Default: hybrid + rerank, titles from NCBI
python question_recall_report.py --output-dir bioasq14_output/batch_1

# With local corpus for titles
python question_recall_report.py --output-dir bioasq14_output/batch_1 \\
    --docs-jsonl "/pubmed/jsonl_2026/*.jsonl"

# Custom rerank cut-offs and explicit ground truth
python question_recall_report.py --output-dir bioasq14_output/batch_1 \\
    --rerank-ks 10,50,100,500,1000,2000 \\
    --ground-truth bioasq_data/14b/BioASQ-task14bPhaseB-testset1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse helpers from the sibling low_recall_report module
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_DIR = _SCRIPT_DIR.parent  # shared_scripts/
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from low_recall_report import (  # noqa: E402
    _resolve_ground_truth,
    _load_run_tsv,
    _recall_from_run,
    _build_question_meta,
    fetch_titles_jsonl,
    fetch_titles_ncbi,
)
from retrieval_eval.common import (  # noqa: E402
    load_questions,
    build_topics_and_gold,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Hybrid recall loading (reads cap from config.json)
# ---------------------------------------------------------------------------

def _read_hybrid_cap(output_dir: Path) -> int:
    """Read the 'cap' value from hybrid/config.json."""
    cfg_path = output_dir / "hybrid" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"hybrid/config.json not found in {output_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cap = cfg.get("cap") or cfg.get("k_max_eval") or cfg.get("bm25_topk")
    if cap is None:
        raise KeyError(f"No 'cap' key found in {cfg_path}")
    return int(cap)


def _load_hybrid_recall(
    output_dir: Path,
    gold_map: Dict[str, List[str]],
    cap: int,
) -> Dict[str, float]:
    """Compute per-query Recall@cap from hybrid run files."""
    runs_dir = output_dir / "hybrid" / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"hybrid/runs/ not found in {output_dir}")
    tsvs = list(runs_dir.glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No TSV files in {runs_dir}")
    print(f"  Loading hybrid run: {tsvs[0].name}")
    run_df = _load_run_tsv(tsvs[0])
    return _recall_from_run(run_df, gold_map, cap)

# ---------------------------------------------------------------------------
# Rerank recall loading (multiple K from per_query CSV)
# ---------------------------------------------------------------------------

def _load_rerank_recalls(
    output_dir: Path,
    rerank_stage: str,
    ks: List[int],
) -> pd.DataFrame:
    """Load per-query CSV from the rerank stage and extract R@K columns.

    Returns a DataFrame with 'qid' + one column per requested K.
    """
    perq_dir = output_dir / rerank_stage / "per_query"
    if not perq_dir.is_dir():
        raise FileNotFoundError(
            f"{rerank_stage}/per_query/ not found in {output_dir}"
        )
    csvs = list(perq_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {perq_dir}")
    csv_path = csvs[0]
    print(f"  Loading rerank per-query CSV: {csv_path.name}")

    df = pd.read_csv(csv_path)
    df["qid"] = df["qid"].astype(str)

    available_r = [c for c in df.columns if c.startswith("R@")]
    selected = []
    for k in ks:
        col = f"R@{k}"
        if col in df.columns:
            selected.append(col)
        else:
            print(f"    WARNING: {col} not in CSV (available: {available_r})")

    return df[["qid"] + selected].copy()

# ---------------------------------------------------------------------------
# Build the full report
# ---------------------------------------------------------------------------

def _build_full_report(
    qids: List[str],
    question_meta: Dict[str, dict],
    gold_map: Dict[str, List[str]],
    hybrid_recall: Dict[str, float],
    rerank_df: pd.DataFrame,
    titles: Dict[str, str],
    cap: int,
) -> pd.DataFrame:
    rerank_lookup: Dict[str, dict] = {}
    rerank_cols = [c for c in rerank_df.columns if c != "qid"]
    for _, row in rerank_df.iterrows():
        rerank_lookup[str(row["qid"])] = {c: row[c] for c in rerank_cols}

    rows = []
    for qid in qids:
        meta = question_meta.get(qid, {})
        gold_pmids = gold_map.get(qid, [])
        pmid_str = ", ".join(gold_pmids)
        title_str = ", ".join(titles.get(p, "") for p in gold_pmids)

        row: dict = {
            "qid": qid,
            "question": meta.get("body", ""),
            "type": meta.get("type", "unknown"),
            "n_rel": len(gold_pmids),
            f"hybrid_R@{cap}": round(hybrid_recall.get(qid, float("nan")), 6),
        }

        rr = rerank_lookup.get(qid, {})
        for col in rerank_cols:
            row[f"rerank_{col}"] = round(rr.get(col, float("nan")), 6)

        row["golden_pmids"] = pmid_str
        row["golden_titles"] = title_str
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"hybrid_R@{cap}").reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Full per-question recall report with hybrid + rerank metrics, "
                    "golden PMIDs, and titles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--output-dir", required=True, type=Path,
        help="Batch output folder (e.g. bioasq14_output/batch_1)",
    )
    ap.add_argument(
        "--rerank-stage", default="rerank",
        help="Rerank stage subfolder (default: rerank)",
    )
    ap.add_argument(
        "--rerank-ks", default="10,100,200,500,1000",
        help="Comma-separated K values for rerank recall columns "
             "(default: 10,100,200,500,1000)",
    )
    ap.add_argument(
        "--ground-truth", default=None,
        help="Explicit ground-truth JSON path; if omitted, parsed from config.env",
    )
    ap.add_argument(
        "--docs-jsonl", default=None,
        help="Glob to JSONL corpus for title lookup "
             '(e.g. "/pubmed/jsonl_2026/*.jsonl")',
    )
    ap.add_argument(
        "--output", default=None, type=Path,
        help="Output CSV path (default: {output_dir}/analysis/question_recall_report.csv)",
    )
    ap.add_argument(
        "--repo-root", default=None,
        help="Override $REPO_ROOT for config.env resolution",
    )
    ap.add_argument(
        "--no-titles", action="store_true",
        help="Skip title fetching entirely",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    rerank_stage: str = args.rerank_stage
    rerank_ks = [int(x) for x in args.rerank_ks.split(",")]

    # 1. Read hybrid cap from config.json
    cap = _read_hybrid_cap(output_dir)
    print(f"=== Question Recall Report ===")
    print(f"  Output dir    : {output_dir}")
    print(f"  Hybrid cap    : {cap}")
    print(f"  Rerank stage  : {rerank_stage}")
    print(f"  Rerank K's    : {rerank_ks}")
    print()

    # 2. Ground truth
    gt_path = _resolve_ground_truth(output_dir, args.ground_truth, args.repo_root)
    print(f"Ground truth : {gt_path}")
    questions = load_questions(gt_path)
    _, gold_map = build_topics_and_gold(questions)
    question_meta = _build_question_meta(questions)
    all_qids = [str(q.get("id") or q.get("qid") or i) for i, q in enumerate(questions)]
    print(f"  {len(questions)} questions loaded")
    print()

    # 3. Hybrid recall
    print(f"Loading hybrid Recall@{cap}...")
    hybrid_recall = _load_hybrid_recall(output_dir, gold_map, cap)
    print(f"  {len(hybrid_recall)} queries")
    print()

    # 4. Rerank recall
    print(f"Loading rerank recalls ({rerank_stage})...")
    rerank_df = _load_rerank_recalls(output_dir, rerank_stage, rerank_ks)
    print(f"  {len(rerank_df)} queries, columns: {[c for c in rerank_df.columns if c != 'qid']}")
    print()

    # 5. Collect all gold PMIDs for title fetching
    all_gold_pmids: Set[str] = set()
    for qid in all_qids:
        all_gold_pmids.update(gold_map.get(qid, []))

    # 6. Title lookup
    titles: Dict[str, str] = {}
    if not args.no_titles and all_gold_pmids:
        print(f"Resolving titles for {len(all_gold_pmids)} unique PMIDs...")
        if args.docs_jsonl:
            titles = fetch_titles_jsonl(all_gold_pmids, args.docs_jsonl)
            missing = all_gold_pmids - set(titles.keys())
            if missing:
                print(f"  {len(missing)} PMIDs not found in JSONL, "
                      "falling back to NCBI API...")
                titles.update(fetch_titles_ncbi(missing))
        else:
            titles = fetch_titles_ncbi(all_gold_pmids)
        print()

    # 7. Build report
    report_df = _build_full_report(
        all_qids, question_meta, gold_map,
        hybrid_recall, rerank_df, titles, cap,
    )

    # 8. Save
    out_path: Path = args.output or (
        output_dir / "analysis" / "question_recall_report.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    print(f"Saved {len(report_df)} rows → {out_path}")
    print()

    # 9. Print summary to stdout
    recall_cols = [f"hybrid_R@{cap}"] + [
        c for c in report_df.columns if c.startswith("rerank_R@")
    ]
    display_cols = ["qid", "type", "n_rel"] + recall_cols + ["question", "golden_pmids"]
    with pd.option_context(
        "display.max_rows", None,
        "display.max_colwidth", 80,
        "display.width", 300,
    ):
        print(report_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
