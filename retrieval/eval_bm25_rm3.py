#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pyterrier as pt

# Add public scripts root to path so we can import retrieval_eval
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval_eval.common import (
    BatchResult,
    build_topics_and_gold,
    collect_qids_from_questions,
    evaluate_run,
    load_questions,
    RECALL_KS,
    run_df_to_run_map,
    zero_recall_qids,
)

# ---------------------------
# Query augmentation (same logic as your notebook)
# ---------------------------
CODE_RE = re.compile(r"\b([A-Za-z]{2,20})?[-–_:/\s]*([0-9]{5,})\b")

def chunk_digits(d: str, k: int = 4) -> list[str]:
    return [d[i : i + k] for i in range(0, len(d), k)]


def augment_text_for_codes(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    extras = []
    for pfx, digits in CODE_RE.findall(text):
        p = (pfx or "").lower()
        if p:
            extras.append(p)

        if len(digits) >= 5:
            chunks = chunk_digits(digits, 4)
            extras.extend(chunks)
            if p:
                extras.append(p + " " + " ".join(chunks))
        else:
            extras.append(digits)
            if p:
                extras.append(f"{p}{digits}")
                extras.append(f"{p}-{digits}")
                extras.append(f"{p} {digits}")

    if extras:
        return text + " " + " ".join(extras)
    return text


def apply_augment_text_for_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query"] = out["query"].map(augment_text_for_codes)
    return out


# ---------------------------
# RM3 seed cleaning (same idea as your notebook)
# ---------------------------
_PREFIX_PATTERNS = [
    r"^\s*what\s+is\s+",
    r"^\s*what\s+are\s+",
    r"^\s*what\s+was\s+",
    r"^\s*what\s+were\s+",
    r"^\s*what\s+does\s+",
    r"^\s*which\s+(is|are)\s+",
    r"^\s*when\s+(is|was)\s+",
    r"^\s*how\s+many\s+",
    r"^\s*list\s+",
    r"^\s*describe\s+",
    r"^\s*define\s+",
]


def clean_seed_query(q: str) -> str:
    if q is None:
        return ""
    s = str(q).strip()
    s = re.sub(r"\s+", " ", s).rstrip(" ?")

    low = s.lower()
    for pat in _PREFIX_PATTERNS:
        m = re.match(pat, low)
        if m:
            s = s[m.end() :].strip()
            break

    if not s:
        s = re.sub(r"[?]+$", "", str(q)).strip()
    return s


def use_seed_query(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query_raw"] = out["query"]
    out["query"] = out["seed_query"]
    return out


# ---------------------------
# Retrieval & evaluation helpers
# ---------------------------
def cut_to_k(res: pd.DataFrame, k: int) -> pd.DataFrame:
    # Ensure rank exists and cut
    if "rank" not in res.columns:
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res["rank"] = res.groupby("qid").cumcount() + 1
    return res[res["rank"] <= k].copy()


def run_retrieval_only(
    topics_df: pd.DataFrame,
    pipe: pt.Transformer,
    k_eval: int,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Run retrieval only (no evaluation). Returns run_map and res_df."""
    res_df = pipe(topics_df)
    res_df = cut_to_k(res_df, k_eval)
    run_map = run_df_to_run_map(res_df, qid_col="qid", docno_col="docno")
    return run_map, res_df


def eval_one(
    method: str,
    batch_name: str,
    topics_df: pd.DataFrame,
    gold_map: Dict[str, List[str]],
    pipe: pt.Transformer,
    k_eval: int,
    ks_recall: Optional[Sequence[int]] = None,
    eps: float = 1e-5,
) -> Tuple[BatchResult, pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    res_df = pipe(topics_df)
    res_df = cut_to_k(res_df, k_eval)
    run_map = run_df_to_run_map(res_df, qid_col="qid", docno_col="docno")
    summary, perq = evaluate_run(gold_map, run_map, ks_recall=ks_recall or RECALL_KS, eps=eps)
    br = BatchResult(method=method, batch=batch_name, n_queries=len(topics_df), metrics=summary)
    return br, perq, run_map, res_df


def ensure_pt(java_mem: str | None = None):
    if not pt.java.started():
        jvm_opts = []
        if java_mem:
            jvm_opts.append(f"-Xmx{java_mem}")
        pt.java.init() if not jvm_opts else pt.init(jvm_args=jvm_opts)


def main():
    ap = argparse.ArgumentParser(description="Evaluate BM25+RM3 on BioASQ train subset + test batches.")
    ap.add_argument("--index_path", required=True, help="Path to Terrier index directory")
    ap.add_argument("--train_json", required=True, help="Path to training subset json (e.g. training14b_10pct_sample.json)")
    ap.add_argument(
        "--test_batch_jsons",
        nargs="*",
        default=[],
        help="List of test batch JSON files (e.g. bioasq_data/Task13BGoldenEnriched/13B1_golden.json ...).",
    )
    ap.add_argument(
        "--test_dir",
        default=None,
        help="Backward-compatible: directory containing 13B1_golden.json .. 13B4_golden.json (used only if --test_batch_jsons not provided)",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--threads", type=int, default=4, help="Terrier retrieval threads")
    ap.add_argument("--java_mem", default=None, help='Optional JVM heap, e.g. "8g"')

    ap.add_argument("--k_eval", type=int, default=5000, help="Retrieve/evaluate top K")
    ap.add_argument("--ks", type=str, default=",".join(map(str, RECALL_KS)), help="Comma-separated K values for recall (default: RECALL_KS)")
    ap.add_argument("--k_feedback", type=int, default=50, help="BM25 feedback pool for RM3")
    ap.add_argument("--rm3_fb_docs", type=int, default=20)
    ap.add_argument("--rm3_fb_terms", type=int, default=30)
    ap.add_argument("--rm3_lambda", type=float, default=0.6)

    ap.add_argument(
        "--include_bm25",
        action="store_true",
        help="Also evaluate BM25 baseline (default: only BM25_RM3).",
    )

    ap.add_argument("--no_exclude_test_qids", action="store_true", help="Do not remove test qids from train set")
    ap.add_argument("--no_eval", action="store_true", help="Skip evaluation; only run retrieval and write run TSVs")
    ap.add_argument("--save_runs", action="store_true", help="Save run TSVs (qid docno rank score)")
    ap.add_argument("--save_per_query", action="store_true", help="Save per-query metrics CSV")
    ap.add_argument("--save_zero_recall", action="store_true", help="Save zero-recall reports (default off)")
    args = ap.parse_args()

    ks_recall = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if not ks_recall:
        ks_recall = RECALL_KS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "per_query").mkdir(exist_ok=True)
    (out_dir / "zero_recall").mkdir(exist_ok=True)

    ensure_pt(java_mem=args.java_mem)

    # Load index - handle both directory path and properties file path
    index_path = Path(args.index_path).resolve()  # Convert to absolute path
    if index_path.is_dir():
        index_path = index_path / "data.properties"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index properties file not found: {index_path}")
    
    index = pt.IndexFactory.of(str(index_path))

    # Build pipelines
    bm25_final = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.k_eval)
    bm25_feedback = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.k_feedback)

    pipe_bm25 = pt.apply.generic(apply_augment_text_for_codes) >> bm25_final

    rm3 = pt.rewrite.RM3(
        index,
        fb_docs=args.rm3_fb_docs,
        fb_terms=args.rm3_fb_terms,
        fb_lambda=args.rm3_lambda,
    )

    pipe_bm25_rm3 = (
        pt.apply.generic(lambda df: df.assign(seed_query=df["query"].map(clean_seed_query)))
        >> pt.apply.generic(use_seed_query)
        >> pt.apply.generic(apply_augment_text_for_codes)  # important: same code augmentation before feedback
        >> bm25_feedback
        >> rm3
        >> bm25_final
    )

    # Load datasets
    test_files: List[Path] = []
    if args.test_batch_jsons:
        test_files = [Path(fp).resolve() for fp in args.test_batch_jsons]
    elif args.test_dir:
        # Backward-compatible default behavior
        test_dir = Path(args.test_dir).resolve()
        test_files = [
            test_dir / "13B1_golden.json",
            test_dir / "13B2_golden.json",
            test_dir / "13B3_golden.json",
            test_dir / "13B4_golden.json",
        ]
    else:
        raise ValueError("Provide --test_batch_jsons (preferred) or --test_dir (legacy).")

    for fp in test_files:
        if not fp.exists():
            raise FileNotFoundError(f"Missing test batch: {fp}")

    train_questions = load_questions(Path(args.train_json).resolve())

    # Collect test qids for exclusion
    test_qids = set()
    test_batches: List[Tuple[str, List[dict]]] = []
    for fp in test_files:
        qs = load_questions(fp)
        test_batches.append((fp.stem, qs))
        test_qids |= collect_qids_from_questions(qs)

    # Build train topics/gold and optionally exclude test qids
    train_topics, train_gold = build_topics_and_gold(train_questions)
    if not args.no_exclude_test_qids:
        mask = ~train_topics["qid"].astype(str).isin(test_qids)
        train_topics = train_topics.loc[mask].reset_index(drop=True)
        # filter gold accordingly
        keep = set(train_topics["qid"].astype(str).tolist())
        train_gold = {qid: pmids for qid, pmids in train_gold.items() if qid in keep}

    train_batch = Path(args.train_json).stem

    # Evaluate
    all_rows = []
    config = vars(args)
    config["index_path"] = str(args.index_path)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_runs = args.save_runs or args.no_eval

    def save_run_tsv(method: str, batch: str, res_df: pd.DataFrame) -> None:
        run_path = out_dir / "runs" / f"{method}__{batch}__top{args.k_eval}.tsv"
        if "rank" not in res_df.columns:
            tmp = res_df.sort_values(["qid", "score"], ascending=[True, False]).copy()
            tmp["rank"] = tmp.groupby("qid").cumcount() + 1
        else:
            tmp = res_df
        tmp = tmp.loc[tmp["rank"] <= args.k_eval, ["qid", "docno", "rank", "score"]]
        tmp.to_csv(run_path, sep="\t", index=False)

    def maybe_save(method: str, batch: str, res_run: Dict[str, List[str]], res_df: pd.DataFrame, perq: pd.DataFrame, gold: Dict[str, List[str]]):
        if save_runs:
            save_run_tsv(method, batch, res_df)
        if args.save_per_query and not args.no_eval:
            perq_path = out_dir / "per_query" / f"{method}__{batch}__perq.csv"
            perq.to_csv(perq_path, index=False)
        if args.save_zero_recall and not args.no_eval:
            zr = zero_recall_qids(gold, res_run, k=args.k_eval)
            zr_path = out_dir / "zero_recall" / f"{method}__{batch}__zero_recall_at{args.k_eval}.txt"
            with open(zr_path, "w", encoding="utf-8") as f:
                f.write(f"ZERO-RECALL QUESTIONS (n={len(zr)}) at {args.k_eval}\n\n")
                for qid in zr:
                    f.write(qid + "\n")

    methods_to_run = [("BM25_RM3", pipe_bm25_rm3)]
    if args.include_bm25:
        methods_to_run = [("BM25", pipe_bm25)] + methods_to_run

    if args.no_eval:
        for method, pipe in methods_to_run:
            run_map, res_df = run_retrieval_only(train_topics, pipe, args.k_eval)
            save_run_tsv(method, train_batch, res_df)
        for batch_name, questions in test_batches:
            topics, _ = build_topics_and_gold(questions)
            for method, pipe in methods_to_run:
                run_map, res_df = run_retrieval_only(topics, pipe, args.k_eval)
                save_run_tsv(method, batch_name, res_df)
        print("No-eval mode: runs saved to", out_dir / "runs")
        return

    # Train subset
    for method, pipe in methods_to_run:
        br, perq, run_map, res_df = eval_one(method, train_batch, train_topics, train_gold, pipe, args.k_eval, ks_recall=ks_recall)
        all_rows.append(br.to_row())
        maybe_save(method, train_batch, run_map, res_df, perq, train_gold)

    # Test batches
    for batch_name, questions in test_batches:
        topics, gold = build_topics_and_gold(questions)
        for method, pipe in methods_to_run:
            br, perq, run_map, res_df = eval_one(method, batch_name, topics, gold, pipe, args.k_eval, ks_recall=ks_recall)
            all_rows.append(br.to_row())
            maybe_save(method, batch_name, run_map, res_df, perq, gold)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df)
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
