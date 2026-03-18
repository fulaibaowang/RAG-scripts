#!/usr/bin/env python3
"""
Listwise reranking with RankZephyr 7B via RankLLM.

Reads snippet_rrf runs + snippet_rerank windows produced by the main pipeline
and produces reranked runs in the same TSV format (qid, docno, rank) that
downstream evidence/generation scripts expect.

Two modes are executed:
  1. Single-window  – top-k docs, one reranking pass.
  2. Sliding-window – larger pool with sliding window (pool, window, stride).

Requires the vLLM-based listwise container (Dockerfile.listwise).
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Listwise reranking (RankZephyr) on snippet-rrf pipeline output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input paths
    p.add_argument("--runs-dir", type=Path, required=True,
                   help="Directory with snippet_rrf run TSVs (qid, docno, rank).")
    p.add_argument("--windows-dir", type=Path, required=True,
                   help="Directory with snippet_rerank window JSONLs ({split}.jsonl).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root output directory (e.g. <workflow>/listwise_rerank).")

    # Query JSONs (for evaluation)
    p.add_argument("--train-json", type=Path, default=None,
                   help="Training query JSON (questions with id/body).")
    p.add_argument("--test-batch-jsons", type=Path, nargs="*", default=[],
                   help="Test-batch query JSONs.")

    # Single-window config
    p.add_argument("--single-k", type=int, default=15,
                   help="Number of docs for single-window reranking.")

    # Sliding-window config
    p.add_argument("--no-sliding", action="store_true",
                   help="Skip sliding-window reranking (only run single-window).")
    p.add_argument("--pool", type=int, default=50,
                   help="Pool size for sliding-window reranking.")
    p.add_argument("--window-size", type=int, default=15,
                   help="Window size for sliding-window reranking.")
    p.add_argument("--stride", type=int, default=5,
                   help="Stride for sliding-window reranking.")

    # Model config
    p.add_argument("--model", type=str,
                   default="castorini/rank_zephyr_7b_v1_full",
                   help="HuggingFace model name for RankZephyr.")
    p.add_argument("--context-size", type=int, default=4096,
                   help="Model context window size in tokens.")
    p.add_argument("--max-snippet-tokens", type=int, default=250,
                   help="Max tokens per snippet (truncated if longer).")

    # Evaluation toggle
    p.add_argument("--disable-metrics", action="store_true",
                   help="Skip evaluation (no ground-truth available).")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_split_from_run_stem(run_stem: str) -> Optional[str]:
    """Extract split name from a run filename stem.

    Handles patterns produced by the main pipeline, e.g.:
      best_rrf_13B1_golden_top5000_rrf_poolR200_poolH200_k60_rrf_poolR100_poolH100_k60
    Returns: ``13B1_golden`` (or ``None`` on no match).
    """
    m = re.fullmatch(
        r"best_rrf_(.+?)_top\d+(?:_rrf_pool[^\s]+)?",
        run_stem,
    )
    return m.group(1) if m else None


def normalize_pmid(x) -> str:
    """Best-effort normalisation of a document id / PubMed URL to a bare PMID."""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"/pubmed/(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+", s):
        return s
    return s


def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(normalize_pmid),
    })
    if rank_col:
        out["rank"] = df[rank_col].astype(int)
    else:
        out["rank"] = out.groupby("qid").cumcount() + 1
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


def load_windows_jsonl(path: Path) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """Return ``(qid, docno) -> (best_window_text, ce_score)`` keeping top-1 window."""
    best: Dict[Tuple[str, str], Tuple[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec["qid"])
            docno = normalize_pmid(rec["docno"])
            text = rec["window_text"]
            score = float(rec["ce_score"])
            key = (qid, docno)
            if key not in best or score > best[key][1]:
                best[key] = (text, score)
    return best


def load_questions(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "questions" not in data:
        raise KeyError(f"{json_path} missing top-level 'questions'")
    return data["questions"]


def estimate_query_token_len(tokenizer, query_text: str) -> int:
    return len(tokenizer.encode(query_text, add_special_tokens=False))


def truncate_snippet(snippet: str, max_tokens: int, tokenizer) -> str:
    tokens = tokenizer.encode(snippet, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return snippet
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation (lightweight – no external IR-metrics dependency)
# ---------------------------------------------------------------------------

def _ap_at_k(relevant: set, ranked: list, k: int) -> float:
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k) if relevant else 0.0


def _rr_at_k(relevant: set, ranked: list, k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate_run(
    gold: Dict[str, List[str]],
    run: Dict[str, List[str]],
    k: int = 10,
) -> Dict[str, float]:
    aps, rrs = [], []
    for qid, rel in gold.items():
        ranked = run.get(qid, [])
        rel_set = set(rel)
        aps.append(_ap_at_k(rel_set, ranked, k))
        rrs.append(_rr_at_k(rel_set, ranked, k))
    return {
        f"MAP@{k}": float(np.mean(aps)) if aps else 0.0,
        f"MRR@{k}": float(np.mean(rrs)) if rrs else 0.0,
    }


def build_gold_map(questions: List[dict]) -> Dict[str, List[str]]:
    gold: Dict[str, List[str]] = {}
    for q in questions:
        qid = str(q.get("id") or q.get("qid"))
        docs = q.get("documents", [])
        pmids = [normalize_pmid(d) for d in docs]
        pmids = [p for p in pmids if p]
        if pmids:
            gold[qid] = pmids
    return gold


def run_df_to_run_map(run_df: pd.DataFrame) -> Dict[str, List[str]]:
    run_map: Dict[str, List[str]] = {}
    for qid, g in run_df.groupby("qid", sort=False):
        run_map[str(qid)] = g.sort_values("rank")["docno"].tolist()
    return run_map


# ---------------------------------------------------------------------------
# RankLLM wrappers
# ---------------------------------------------------------------------------

def _import_rankllm():
    """Import RankLLM components; returns ``(Request, Candidate, Query, ZephyrReranker)``."""
    from rank_llm.data import Request, Candidate, Query
    from rank_llm.rerank.listwise import ZephyrReranker
    return Request, Candidate, Query, ZephyrReranker


def patch_vllm_tqdm_zerodiv() -> None:
    """Patch vLLM's LLM.generate to force ``use_tqdm=False``.

    vLLM ≥0.11 has a progress-bar code path in ``_run_engine`` that divides
    by ``pbar.format_dict["elapsed"]``, which can be zero when the batch
    finishes faster than the timer resolution.  This causes a
    ``ZeroDivisionError``.  Disabling tqdm avoids the buggy path entirely.
    """
    try:
        from vllm.entrypoints.llm import LLM as _VllmLLM
    except ImportError:
        logger.warning("Could not import vllm.entrypoints.llm.LLM; skipping tqdm patch")
        return

    if getattr(_VllmLLM, "_listwise_tqdm_patched", False):
        return

    _original_generate = _VllmLLM.generate

    def _generate_no_tqdm(self, *args, **kwargs):
        kwargs["use_tqdm"] = False
        return _original_generate(self, *args, **kwargs)

    _VllmLLM.generate = _generate_no_tqdm
    _VllmLLM._listwise_tqdm_patched = True
    logger.info("Patched vllm.LLM.generate: force use_tqdm=False")


def build_zephyr_reranker(ZephyrReranker, model_name: str, context_size: int = 4096):
    sig = inspect.signature(ZephyrReranker)
    params = sig.parameters

    candidate_kwargs: dict = {}
    if "model" in params:
        candidate_kwargs["model"] = model_name
    elif "model_path" in params:
        candidate_kwargs["model_path"] = model_name
    if "context_size" in params:
        candidate_kwargs["context_size"] = context_size

    attempts = []
    if candidate_kwargs:
        attempts.append(candidate_kwargs)
    attempts.append({})

    last_err = None
    for kwargs in attempts:
        try:
            logger.info("Trying ZephyrReranker init with kwargs=%s", kwargs)
            return ZephyrReranker(**kwargs)
        except TypeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("Could not init ZephyrReranker")


def build_requests_for_split(
    data: dict,
    qid_to_query: Dict[str, str],
    k: int,
    tokenizer,
    max_snippet_tokens: int,
    Candidate,
    Query,
    Request,
) -> Tuple[list, List[Tuple[str, List[str]]], int]:
    requests = []
    qid_docno_order: List[Tuple[str, List[str]]] = []
    n_truncated = 0

    for qid, top_docnos in data["top_docs_per_query"].items():
        query_text = qid_to_query.get(qid, "")
        if not query_text:
            continue

        top_k_docnos = top_docnos[:k]
        doc_snippets = data["snippets_per_query"].get(qid, {})

        candidates = []
        valid_docnos = []
        for docno in top_k_docnos:
            if docno in doc_snippets:
                snippet = doc_snippets[docno]
                orig_len = len(tokenizer.encode(snippet, add_special_tokens=False))
                if orig_len > max_snippet_tokens:
                    snippet = truncate_snippet(snippet, max_snippet_tokens, tokenizer)
                    n_truncated += 1
                candidates.append(Candidate(docid=docno, doc={"text": snippet}, score=0.0))
                valid_docnos.append(docno)

        if not candidates:
            continue

        query_obj = Query(text=query_text, qid=qid)
        req = Request(query=query_obj, candidates=candidates)
        requests.append(req)
        qid_docno_order.append((qid, valid_docnos))

    return requests, qid_docno_order, n_truncated


def rerank_single_window(reranker, requests, qid_docno_order):
    if not requests:
        return {}
    results = reranker.rerank_batch(
        requests=requests, rank_start=0,
        rank_end=len(requests[0].candidates),
    )
    run_map: Dict[str, List[str]] = {}
    for (qid, _), result in zip(qid_docno_order, results):
        run_map[qid] = [str(c.docid) for c in result.candidates]
    return run_map


def rerank_sliding_window(reranker, requests, qid_docno_order, window_size, stride):
    if not requests:
        return {}
    pool_size = len(requests[0].candidates)
    results = reranker.rerank_batch(
        requests=requests, rank_start=0, rank_end=pool_size,
        window_size=window_size, step=stride,
    )
    run_map: Dict[str, List[str]] = {}
    for (qid, _), result in zip(qid_docno_order, results):
        run_map[qid] = [str(c.docid) for c in result.candidates]
    return run_map


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_runs_to_tsv(
    runs: Dict[str, Dict[str, List[str]]],
    out_dir: Path,
    label: str,
):
    for split, run_map in runs.items():
        rows = []
        for qid, docnos in run_map.items():
            for rank, docno in enumerate(docnos, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank})
        if not rows:
            continue
        df_split = pd.DataFrame(rows)
        out_path = out_dir / f"{split}.tsv"
        df_split.to_csv(out_path, sep="\t", index=False)
        logger.info("Saved %s run for %s -> %s (%d rows)", label, split, out_path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Validate inputs ---------------------------------------------------
    if not args.runs_dir.is_dir():
        logger.error("--runs-dir does not exist: %s", args.runs_dir)
        sys.exit(1)
    if not args.windows_dir.is_dir():
        logger.error("--windows-dir does not exist: %s", args.windows_dir)
        sys.exit(1)

    # ---- Output dirs -------------------------------------------------------
    out = args.output_dir
    fig_dir = out / "figures"
    runs_single = out / "single_window" / "runs"
    runs_sliding = out / "sliding_window" / "runs"
    for d in (out, fig_dir, runs_single, runs_sliding):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load queries (for eval + reranking) --------------------------------
    all_questions: List[dict] = []
    qid_to_query: Dict[str, str] = {}
    query_json_paths: List[Path] = []
    if args.train_json and args.train_json.exists():
        query_json_paths.append(args.train_json)
    query_json_paths.extend(p for p in args.test_batch_jsons if p.exists())

    for p in query_json_paths:
        qs = load_questions(p)
        all_questions.extend(qs)
        logger.info("Loaded %d questions from %s", len(qs), p)

    for q in all_questions:
        qid = str(q.get("id") or q.get("qid"))
        body = str(q.get("body") or q.get("query") or q.get("question") or "")
        qid_to_query[qid] = body
    logger.info("Total: %d queries loaded", len(qid_to_query))

    # ---- Load tokenizer ----------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Loaded tokenizer for %s", args.model)

    # ---- Load splits -------------------------------------------------------
    top_k_load = max(args.pool, args.single_k)
    run_files = sorted(args.runs_dir.glob("*.tsv"))
    logger.info("Found %d run TSVs in %s", len(run_files), args.runs_dir)

    splits_data: Dict[str, dict] = {}
    for run_path in run_files:
        split = parse_split_from_run_stem(run_path.stem)
        if split is None:
            logger.warning("Skipping %s (could not parse split)", run_path.name)
            continue

        windows_path = args.windows_dir / f"{split}.jsonl"
        if not windows_path.exists():
            logger.warning("Skipping %s: no windows file at %s", split, windows_path)
            continue

        run_df = load_run_tsv(run_path)
        windows = load_windows_jsonl(windows_path)

        top_docs_per_query: Dict[str, List[str]] = {}
        for qid, g in run_df.groupby("qid", sort=False):
            top_docs_per_query[str(qid)] = g.head(top_k_load)["docno"].tolist()

        snippets_per_query: Dict[str, Dict[str, str]] = {}
        for qid, docnos in top_docs_per_query.items():
            snippets_per_query[qid] = {}
            for docno in docnos:
                if (qid, docno) in windows:
                    snippets_per_query[qid][docno] = windows[(qid, docno)][0]

        splits_data[split] = {
            "run_df": run_df,
            "top_docs_per_query": top_docs_per_query,
            "snippets_per_query": snippets_per_query,
        }
        n_qs = len(top_docs_per_query)
        n_sn = sum(len(v) for v in snippets_per_query.values())
        logger.info("  %s: %d queries, %d snippets (top-%d)", split, n_qs, n_sn, top_k_load)

    if not splits_data:
        logger.error("No splits loaded – nothing to do.")
        sys.exit(1)

    # ---- Snippet token statistics + histogram ------------------------------
    all_token_lengths = []
    for data in splits_data.values():
        for doc_snippets in data["snippets_per_query"].values():
            for txt in doc_snippets.values():
                all_token_lengths.append(len(tokenizer.encode(txt, add_special_tokens=False)))

    if all_token_lengths:
        stats = {
            "n_snippets": len(all_token_lengths),
            "mean": float(np.mean(all_token_lengths)),
            "median": float(np.median(all_token_lengths)),
            "p95": float(np.percentile(all_token_lengths, 95)),
            "max": float(np.max(all_token_lengths)),
            "single_k": args.single_k,
            "sliding_pool": args.pool,
            "sliding_window": args.window_size,
            "sliding_stride": args.stride,
            "context_size": args.context_size,
            "max_snippet_tokens": args.max_snippet_tokens,
        }
        stats_path = out / "snippet_token_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Snippet token stats: %s", stats)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_token_lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(stats["mean"], color="red", linestyle="--",
                   label=f"Mean: {stats['mean']:.1f}")
        ax.axvline(stats["p95"], color="orange", linestyle="--",
                   label=f"P95: {stats['p95']:.1f}")
        ax.set_xlabel("Snippet Token Length")
        ax.set_ylabel("Count")
        ax.set_title(f"Snippet Token Lengths (n={len(all_token_lengths)})")
        ax.legend()
        plt.tight_layout()
        hist_path = fig_dir / "snippet_token_hist.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        logger.info("Saved histogram -> %s", hist_path)

    # ---- Import & initialise RankLLM --------------------------------------
    logger.info("Importing RankLLM...")
    Request, Candidate, Query, ZephyrReranker = _import_rankllm()
    patch_vllm_tqdm_zerodiv()
    logger.info("Building ZephyrReranker (model=%s, context=%d)...", args.model, args.context_size)
    reranker = build_zephyr_reranker(ZephyrReranker, args.model, args.context_size)
    logger.info("Reranker ready.")

    # ---- Rerank ------------------------------------------------------------
    single_runs: Dict[str, Dict[str, List[str]]] = {}
    sliding_runs: Dict[str, Dict[str, List[str]]] = {}
    total_trunc_single = 0
    total_trunc_sliding = 0

    for split, data in tqdm(splits_data.items(), desc="Reranking"):
        logger.info("=== %s ===", split)

        # -- single window --
        logger.info("[single] k=%d", args.single_k)
        reqs_s, order_s, nt_s = build_requests_for_split(
            data, qid_to_query, args.single_k, tokenizer,
            args.max_snippet_tokens, Candidate, Query, Request,
        )
        total_trunc_single += nt_s
        logger.info("[single] %d requests (%d truncated)", len(reqs_s), nt_s)
        if reqs_s:
            single_runs[split] = rerank_single_window(reranker, reqs_s, order_s)
            logger.info("[single] reranked %d queries", len(single_runs[split]))
        else:
            single_runs[split] = {}

        # -- sliding window --
        if args.no_sliding:
            logger.info("[sliding] skipped (--no-sliding)")
        else:
            logger.info("[sliding] pool=%d window=%d stride=%d", args.pool, args.window_size, args.stride)
            reqs_sl, order_sl, nt_sl = build_requests_for_split(
                data, qid_to_query, args.pool, tokenizer,
                args.max_snippet_tokens, Candidate, Query, Request,
            )
            total_trunc_sliding += nt_sl
            logger.info("[sliding] %d requests (%d truncated)", len(reqs_sl), nt_sl)
            if reqs_sl:
                sliding_runs[split] = rerank_sliding_window(
                    reranker, reqs_sl, order_sl, args.window_size, args.stride,
                )
                logger.info("[sliding] reranked %d queries", len(sliding_runs[split]))
            else:
                sliding_runs[split] = {}

    logger.info("Total truncated – single: %d, sliding: %d", total_trunc_single, total_trunc_sliding)

    # ---- Save runs ---------------------------------------------------------
    save_runs_to_tsv(single_runs, runs_single, "single-window")
    if not args.no_sliding:
        save_runs_to_tsv(sliding_runs, runs_sliding, "sliding-window")

    # ---- Evaluation --------------------------------------------------------
    if args.disable_metrics or not all_questions:
        logger.info("Skipping evaluation (--disable-metrics or no query JSONs)")
    else:
        gold_map = build_gold_map(all_questions)
        logger.info("Gold relevance for %d queries", len(gold_map))

        rows = []
        for split, data in splits_data.items():
            baseline_run_map = run_df_to_run_map(data["run_df"])
            gold_split = {q: gold_map[q] for q in baseline_run_map if q in gold_map}
            if not gold_split:
                continue

            bl = evaluate_run(gold_split, baseline_run_map)
            sg = evaluate_run(gold_split, single_runs.get(split, {}))

            row = {
                "split": split,
                "n_queries": len(gold_split),
                "baseline_MAP@10": bl["MAP@10"],
                "single_MAP@10": sg["MAP@10"],
                "delta_single_MAP@10": sg["MAP@10"] - bl["MAP@10"],
                "baseline_MRR@10": bl["MRR@10"],
                "single_MRR@10": sg["MRR@10"],
                "delta_single_MRR@10": sg["MRR@10"] - bl["MRR@10"],
            }
            if not args.no_sliding:
                sl = evaluate_run(gold_split, sliding_runs.get(split, {}))
                row["sliding_MAP@10"] = sl["MAP@10"]
                row["delta_sliding_MAP@10"] = sl["MAP@10"] - bl["MAP@10"]
                row["sliding_MRR@10"] = sl["MRR@10"]
                row["delta_sliding_MRR@10"] = sl["MRR@10"] - bl["MRR@10"]
            rows.append(row)

        if rows:
            results_df = pd.DataFrame(rows)
            logger.info("\n%s", results_df.to_string(index=False))
            metrics_path = out / "metrics.csv"
            results_df.to_csv(metrics_path, index=False)
            logger.info("Saved metrics -> %s", metrics_path)

            # Comparison plot
            if args.no_sliding:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                x = np.arange(len(results_df))
                w = 0.35
                ax1 = axes[0]
                ax1.bar(x - w/2, results_df["baseline_MAP@10"], w, label="Baseline", color="steelblue")
                ax1.bar(x + w/2, results_df["single_MAP@10"], w,
                        label=f"Single (k={args.single_k})", color="coral")
                ax1.set_xlabel("Split"); ax1.set_ylabel("MAP@10")
                ax1.set_title("MAP@10: Baseline vs Single")
                ax1.set_xticks(x)
                ax1.set_xticklabels(results_df["split"], rotation=45, ha="right")
                ax1.legend(loc="upper left", fontsize=8); ax1.grid(axis="y", alpha=0.3)

                ax2 = axes[1]
                colors = ["green" if d > 0 else "red" for d in results_df["delta_single_MAP@10"]]
                ax2.bar(x, results_df["delta_single_MAP@10"], color=colors)
                ax2.axhline(0, color="black", linewidth=0.5)
                ax2.set_xlabel("Split"); ax2.set_ylabel("Delta MAP@10")
                ax2.set_title("MAP@10 Improvement (Single - Baseline)")
                ax2.set_xticks(x)
                ax2.set_xticklabels(results_df["split"], rotation=45, ha="right")
                ax2.grid(axis="y", alpha=0.3)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                x = np.arange(len(results_df))
                w = 0.25
                ax1 = axes[0]
                ax1.bar(x - w, results_df["baseline_MAP@10"], w, label="Baseline", color="steelblue")
                ax1.bar(x, results_df["single_MAP@10"], w,
                        label=f"Single (k={args.single_k})", color="coral")
                ax1.bar(x + w, results_df["sliding_MAP@10"], w,
                        label=f"Sliding (p={args.pool} w={args.window_size} s={args.stride})",
                        color="seagreen")
                ax1.set_xlabel("Split"); ax1.set_ylabel("MAP@10")
                ax1.set_title("MAP@10: Baseline vs Single vs Sliding")
                ax1.set_xticks(x)
                ax1.set_xticklabels(results_df["split"], rotation=45, ha="right")
                ax1.legend(loc="upper left", fontsize=8); ax1.grid(axis="y", alpha=0.3)

                ax2 = axes[1]
                ax2.bar(x - w / 2, results_df["delta_single_MAP@10"], w,
                        label="Single - Baseline", color="coral")
                ax2.bar(x + w / 2, results_df["delta_sliding_MAP@10"], w,
                        label="Sliding - Baseline", color="seagreen")
                ax2.axhline(0, color="black", linewidth=0.5)
                ax2.set_xlabel("Split"); ax2.set_ylabel("Delta MAP@10")
                ax2.set_title("MAP@10 Improvement over Baseline")
                ax2.set_xticks(x)
                ax2.set_xticklabels(results_df["split"], rotation=45, ha="right")
                ax2.legend(loc="upper left", fontsize=8); ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fig_path = fig_dir / "map10_comparison.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logger.info("Saved comparison figure -> %s", fig_path)

    logger.info("Done.  Output directory: %s", out)


if __name__ == "__main__":
    main()
