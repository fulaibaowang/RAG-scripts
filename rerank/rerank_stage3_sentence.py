#!/usr/bin/env python3
"""
Stage-3 reranking: CE on (query, title + selected sentence) with max or top-2-mean
doc pooling. Uses same inputs as stage-2; sentence picks from sentence_pick.py
(or computed in-process).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
<<<<<<< HEAD
_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
# Find directory containing retrieval_eval so imports work when run directly (no pipeline PYTHONPATH)
_SHARED_SCRIPTS = _SCRIPT_DIR.parents[1]
for _p in [_SHARED_SCRIPTS] + list(_SCRIPT_DIR.parents):
    if (_p / "retrieval_eval").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        _SHARED_SCRIPTS = _p
        break
else:
    sys.path.insert(0, str(_SHARED_SCRIPTS))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
=======
_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_SCRIPTS = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SHARED_SCRIPTS))
sys.path.insert(0, str(_SCRIPT_DIR))
>>>>>>> 5e739c8f00994910061ab1ac95e8492136fd86a9

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
    run_df_to_run_map,
)
from rerank_stage2 import (
    _build_output_config,
    _build_split_to_role_and_label,
    _parse_split_from_run_stem,
    load_run_tsv,
    OutputConfig,
)


def _parse_ks_recall(raw: str) -> Tuple[int, ...]:
    if not raw:
        return ()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _resolve_device(model_device: str) -> str:
    if model_device != "auto":
        return model_device
    if torch and torch.cuda.is_available():
        return "cuda"
    if torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_doc_titles(docnos: iter, jsonl_path: Path) -> Dict[str, str]:
    """Load docno -> title using sentence_pick helper."""
    from sentence_pick import load_doc_title_sentences
    doc_title_sentences = load_doc_title_sentences(docnos, jsonl_path)
    return {docno: title for docno, (title, _) in doc_title_sentences.items()}


def rerank_run_sentence(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    picks: Dict[Tuple[str, str], List[str]],
    doc_titles: Dict[str, str],
    model: CrossEncoder,
    batch_size: int,
    max_length: int,
    pool_mode: str,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Rerank using CE on (query, title + sentence) and pool to doc score.
    pool_mode: "max" or "top2mean"
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    items = list(run_map.items())
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            out[qid] = [(doc, float("nan")) for doc in docs]
            continue

        pairs: List[Tuple[str, str]] = []
        doc_sent_ranges: List[Tuple[int, int]] = []  # (start, end) into pairs for each doc
        for doc in docs:
            title = doc_titles.get(doc, "")
            sents = picks.get((qid, doc), [])
            start_idx = len(pairs)
            if sents:
                for sent in sents:
                    passage = f"{title} {sent}".strip() if title else sent
                    pairs.append((query, passage))
            else:
                pairs.append((query, title.strip() or " "))
            doc_sent_ranges.append((start_idx, len(pairs)))

        if not pairs:
            out[qid] = [(doc, 0.0) for doc in docs]
            if idx % 10 == 0 or idx == total:
                elapsed = max(1e-9, time() - start)
                rate = idx / elapsed
                print(f"[rerank_sentence] {idx}/{total} queries | {rate:.2f} q/s")
            continue

        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float64)

        doc_scores_list: List[float] = []
        for (start_idx, end_idx) in doc_sent_ranges:
            sent_scores = list(scores[start_idx:end_idx])
            if not sent_scores:
                sent_scores = [0.0]
            if pool_mode == "max":
                doc_score = float(max(sent_scores))
            else:
                top2 = sorted(sent_scores, reverse=True)[:2]
                doc_score = float(sum(top2) / len(top2)) if top2 else 0.0
            doc_scores_list.append(doc_score)

        scored = list(zip(docs, doc_scores_list))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[qid] = scored

        if idx % 10 == 0 or idx == total:
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[rerank_sentence] {idx}/{total} queries | {rate:.2f} q/s")
    return out


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-3 sentence reranking: CE on (query, title+sentence), max/top2mean pooling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--runs-dir", type=Path, default=None, help="Directory with stage-1 run TSV files.")
    inputs.add_argument("--run-files", type=Path, nargs="*", default=None, help="Explicit run TSV files.")
    inputs.add_argument("--run-glob", type=str, default="*.tsv", help="Glob for run files under --runs-dir.")
    inputs.add_argument("--docs-jsonl", type=Path, default=None, help="JSONL corpus with PubMed texts.")
    inputs.add_argument("--train-subset-json", "--train_subset_json", type=Path, default=None)
    inputs.add_argument("--test-batch-jsons", "--test_batch_jsons", type=Path, nargs="*", default=None)
    inputs.add_argument("--query-field", type=str, default="body")
    inputs.add_argument("--candidate-limit", type=int, default=1000, help="Stage-1 candidate cutoff per query.")
    inputs.add_argument("--max-queries", type=int, default=None)
    inputs.add_argument("--dense-model", type=str, default=None, help="Dense encoder model name for sentence pick (e.g. abhinand/MedEmbed-small-v0.1). Use this OR --dense-index-dir when not using --sentence-picks-dir.")
    inputs.add_argument("--dense-index-dir", type=Path, default=None, help="Dense index dir to load model from meta.json. Use this OR --dense-model when not using --sentence-picks-dir.")
    inputs.add_argument("--sentence-picks-dir", type=Path, default=None, help="Precomputed sentence picks (JSON per run stem). If set, skip sentence pick.")
    inputs.add_argument("--sentence-top-k", type=int, default=3, help="Sentences to pick per doc (used when computing picks).")
    inputs.add_argument("--doc-score", type=str, default="both", choices=("max", "top2mean", "both"), help="Doc score pooling: max, top2mean, or both.")
    model = parser.add_argument_group("model")
    model.add_argument("--model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    model.add_argument("--model-device", type=str, default="auto")
    model.add_argument("--model-batch", type=int, default=16)
    model.add_argument("--model-max-length", type=int, default=512)
    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--disable-metrics", action="store_true")
    evaluation.add_argument("--ks-recall", type=str, default="50,100,200,300,400,500,1000,2000,5000")
    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve_repo_root()

    runs_dir = args.runs_dir or root / "output" / "eval_hybird_production_test" / "runs"
    docs_jsonl = args.docs_jsonl or root / "output" / "subset_pubmed.jsonl"
    train_subset_json = args.train_subset_json
    test_batch_jsons = args.test_batch_jsons or []
    output_dir = args.output_dir or root / "output" / "rerank_sentence"

    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    else:
        run_files = sorted(runs_dir.glob(args.run_glob))
    if not run_files:
        raise FileNotFoundError("No run files found. Provide --run-files or --runs-dir/--run-glob.")

    if not args.sentence_picks_dir and not args.dense_index_dir and not args.dense_model:
        raise ValueError("Provide --dense-model or --dense-index-dir (to compute sentence picks) or --sentence-picks-dir (precomputed).")

    ks_recall = _parse_ks_recall(args.ks_recall) or RECALL_KS
    cap = int(args.candidate_limit) if args.candidate_limit else None
    if cap and cap > 0:
        ks_recall = tuple(k for k in ks_recall if k <= cap) or (cap,)

    output_cfg = _build_output_config(output_dir)

    run_dfs: Dict[str, pd.DataFrame] = {}
    for path in run_files:
        name = path.stem
        df = load_run_tsv(path)
        if args.candidate_limit:
            df = df[df["rank"] <= int(args.candidate_limit)]
        if args.max_queries:
            qids = sorted(df["qid"].unique())[: int(args.max_queries)]
            df = df[df["qid"].isin(qids)]
        run_dfs[name] = df

    run_maps = {name: run_df_to_run_map(df, qid_col="qid", docno_col="docno") for name, df in run_dfs.items()}
    run_names = list(run_maps.keys())

    topics_map: Dict[str, str] = {}
    gold_map_all: Dict[str, List[str]] = {}

    def _add_questions(json_path: Optional[Path], query_field: Optional[str] = None) -> None:
        if not json_path or not json_path.exists():
            return
        questions = load_questions(json_path)
        topics_df, gold_map = build_topics_and_gold(questions, query_field=query_field)
        topics_map.update(dict(zip(topics_df["qid"].astype(str), topics_df["query"].astype(str))))
        for qid, docs in gold_map.items():
            gold_map_all[qid] = docs

    _add_questions(train_subset_json, args.query_field)
    for p in test_batch_jsons:
        _add_questions(Path(p), args.query_field)

    # Load or compute sentence picks per run
    all_picks: Dict[str, Dict[Tuple[str, str], List[str]]] = {}
    if args.sentence_picks_dir:
        from sentence_pick import load_picks_json
        for name in run_names:
            path = args.sentence_picks_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Precomputed picks not found: {path}")
            all_picks[name] = load_picks_json(path)
        print("loaded sentence picks from", args.sentence_picks_dir)
    else:
        # Run sentence pick in-process; load dense model by name or from index dir
        from sentence_pick import (
            load_doc_title_sentences,
            run_sentence_pick,
            save_picks_json,
        )
        if args.dense_model:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e
            model_dense = SentenceTransformer(args.dense_model, device="cuda")
            normalize_emb = True
        elif args.dense_index_dir and args.dense_index_dir.exists():
            _RETRIEVAL_DIR = _SHARED_SCRIPTS / "retrieval"
            if str(_RETRIEVAL_DIR) not in sys.path:
                sys.path.insert(0, str(_RETRIEVAL_DIR))
            from eval_dense import load_dense_runtime
            model_dense, _, _, meta = load_dense_runtime(args.dense_index_dir, "cuda")
            normalize_emb = meta.get("loaded_normalize_embeddings", True)
        else:
            raise ValueError("Provide --dense-model or --dense-index-dir to compute sentence picks.")
        for name in run_names:
            run_map = run_maps[name]
            candidate_docnos = set()
            for doc_list in run_map.values():
                candidate_docnos.update(doc_list)
            doc_title_sentences = load_doc_title_sentences(candidate_docnos, docs_jsonl)
            picks = run_sentence_pick(
                run_map=run_map,
                topics=topics_map,
                doc_title_sentences=doc_title_sentences,
                model=model_dense,
                top_k=args.sentence_top_k,
                k_rrf=60,
                w_bm25=1.0,
                w_dense=1.0,
                dense_batch_size=256,
                normalize_embeddings=normalize_emb,
            )
            all_picks[name] = picks
            out_path = output_cfg.output_dir / "sentence_picks" / f"{name}.json"
            save_picks_json(picks, out_path)
        print("computed and saved sentence picks")

    candidate_docnos = set()
    for docs in run_maps.values():
        for doc_list in docs.values():
            candidate_docnos.update(doc_list)
    doc_titles = load_doc_titles(candidate_docnos, docs_jsonl)
    print("loaded doc titles:", len(doc_titles))

    model_device = _resolve_device(args.model_device)
    reranker = CrossEncoder(
        args.model,
        device=model_device,
        max_length=None if args.model_max_length <= 0 else args.model_max_length,
    )

    pool_modes = ["max", "top2mean"] if args.doc_score == "both" else [args.doc_score]
    reranked_runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}  # run_name -> reranked
    for name in run_names:
        picks = all_picks[name]
        for pool_mode in pool_modes:
            run_name = f"{name}_sent_{pool_mode}"
            reranked = rerank_run_sentence(
                run_maps[name],
                topics=topics_map,
                picks=picks,
                doc_titles=doc_titles,
                model=reranker,
                batch_size=args.model_batch,
                max_length=args.model_max_length,
                pool_mode=pool_mode,
            )
            reranked_runs[run_name] = reranked
            print("reranked", run_name, "queries:", len(reranked))

    for run_name, reranked in reranked_runs.items():
        rows = []
        for qid, docs in reranked.items():
            for rank, (docno, score) in enumerate(docs, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        run_df = pd.DataFrame(rows)
        out_path = output_cfg.runs_dir / f"{run_name}.tsv"
        run_df.to_csv(out_path, sep="\t", index=False)

    summary_rows = []
    if not args.disable_metrics and gold_map_all:
        split_to_role, split_to_label = _build_split_to_role_and_label(
            train_subset_json, [Path(p) for p in test_batch_jsons]
        )
        for run_name, reranked in reranked_runs.items():
            reranked_map = {qid: [doc for doc, _ in docs] for qid, docs in reranked.items()}
            gold_for_run = {qid: gold_map_all[qid] for qid in reranked_map if qid in gold_map_all}
            if not gold_for_run:
                continue
            metrics, perq = evaluate_run(gold_for_run, reranked_map, ks_recall=ks_recall)
            perq.to_csv(output_cfg.per_query_dir / f"{run_name}.csv", index=False)
            base_name = run_name.replace("_sent_max", "").replace("_sent_top2mean", "")
            split = _parse_split_from_run_stem(base_name)
            if split is not None and split in split_to_role:
                role = split_to_role[split]
                label = split_to_label[split]
            else:
                role = "unknown"
                label = run_name
            row = {"run": run_name, "label": label, "role": role}
            row.update(metrics)
            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_cfg.metrics_path, index=False)
            try:
                from plot_rerank_eval import build_and_save_hybrid_reranker_plots
                run_maps_for_plot = {}
                for name in run_names:
                    for suffix in ("_sent_max", "_sent_top2mean"):
                        if f"{name}{suffix}" in reranked_runs:
                            run_maps_for_plot[f"{name}{suffix}"] = run_maps[name]
                build_and_save_hybrid_reranker_plots(
                    summary_df, run_maps_for_plot, gold_map_all, output_cfg.output_dir,
                    candidate_limit=cap, config={"split_to_role": split_to_role, "split_to_label": split_to_label},
                )
            except Exception as e:
                print("warning: could not generate eval plots:", e)

    split_to_role_cfg: Dict[str, str] = {}
    split_to_label_cfg: Dict[str, str] = {}
    if train_subset_json or test_batch_jsons:
        try:
            split_to_role_cfg, split_to_label_cfg = _build_split_to_role_and_label(
                train_subset_json, [Path(p) for p in test_batch_jsons]
            )
        except ValueError:
            pass

    config = {
        "model": args.model,
        "model_device": model_device,
        "model_batch": args.model_batch,
        "model_max_length": args.model_max_length,
        "candidate_limit": args.candidate_limit,
        "runs_dir": str(runs_dir),
        "run_files": [str(p) for p in run_files],
        "docs_jsonl": str(docs_jsonl),
        "train_subset_json": str(train_subset_json) if train_subset_json else "",
        "test_batch_jsons": [str(p) for p in test_batch_jsons],
        "ks_recall": list(ks_recall),
        "split_to_role": split_to_role_cfg,
        "split_to_label": split_to_label_cfg,
        "dense_model": args.dense_model or "",
        "dense_index_dir": str(args.dense_index_dir) if args.dense_index_dir else "",
        "sentence_picks_dir": str(args.sentence_picks_dir) if args.sentence_picks_dir else "",
        "sentence_top_k": args.sentence_top_k,
        "doc_score": args.doc_score,
    }
    output_cfg.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("saved outputs to", output_cfg.output_dir)


if __name__ == "__main__":
    main()
