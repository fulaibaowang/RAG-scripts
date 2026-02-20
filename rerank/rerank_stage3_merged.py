#!/usr/bin/env python3
"""
Stage-3 reranking (merged variant): merge title + top-k sentences into one passage,
then one CE call per doc (query, merged_passage). No pooling.

Reuse: If sentence picks exist but merged reranking is missing, re-run with
--sentence-picks-dir pointing at the sentence_picks folder; no dense model needed.
Otherwise set --dense-model or --dense-index-dir to compute picks in-process.
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
_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
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
    RECALL_KS,
    run_df_to_run_map,
)
from rerank_stage2 import (
    _build_output_config,
    _build_split_to_role_and_label,
    _parse_split_from_run_stem,
    load_run_tsv,
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
    from sentence_pick import load_doc_title_sentences
    doc_title_sentences = load_doc_title_sentences(docnos, jsonl_path)
    return {docno: title for docno, (title, _) in doc_title_sentences.items()}


def rerank_run_merged(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    picks: Dict[Tuple[str, str], List[str]],
    doc_titles: Dict[str, str],
    model: CrossEncoder,
    batch_size: int,
) -> Dict[str, List[Tuple[str, float]]]:
    """One CE call per doc: passage = title + ' '.join(top-k sentences)."""
    out: Dict[str, List[Tuple[str, float]]] = {}
    items = list(run_map.items())
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        passages = []
        for doc in docs:
            title = doc_titles.get(doc, "")
            sents = picks.get((qid, doc), [])
            if sents:
                merged = f"{title} {' '.join(sents)}".strip() if title else " ".join(sents)
            else:
                merged = title.strip() if title else " "
            passages.append((query, merged))
        scores = model.predict(passages, batch_size=batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float64)
        scored = list(zip(docs, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[qid] = scored
        if idx % 10 == 0 or idx == total:
            elapsed = max(1e-9, time() - start)
            print(f"[rerank_merged] {idx}/{total} queries | {idx / elapsed:.2f} q/s")
    return out


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-3 merged: CE on (query, title + top-k sentences as one passage). Reuse picks with --sentence-picks-dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--runs-dir", type=Path, default=None)
    inputs.add_argument("--run-files", type=Path, nargs="*", default=None)
    inputs.add_argument("--run-glob", type=str, default="*.tsv")
    inputs.add_argument("--docs-jsonl", type=Path, default=None)
    inputs.add_argument("--train-subset-json", "--train_subset_json", type=Path, default=None)
    inputs.add_argument("--test-batch-jsons", "--test_batch_jsons", type=Path, nargs="*", default=None)
    inputs.add_argument("--query-field", type=str, default="body")
    inputs.add_argument("--candidate-limit", type=int, default=1000)
    inputs.add_argument("--max-queries", type=int, default=None)
    inputs.add_argument("--dense-model", type=str, default=None, help="For sentence pick when not using --sentence-picks-dir.")
    inputs.add_argument("--dense-index-dir", type=Path, default=None)
    inputs.add_argument("--sentence-picks-dir", type=Path, default=None, help="Reuse precomputed picks; skip sentence pick.")
    inputs.add_argument("--sentence-top-k", type=int, default=3)
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
    output_dir = args.output_dir or root / "output" / "rerank_merged"

    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    else:
        run_files = sorted(runs_dir.glob(args.run_glob))
    if not run_files:
        raise FileNotFoundError("No run files found.")

    if not args.sentence_picks_dir and not args.dense_index_dir and not args.dense_model:
        raise ValueError("Provide --dense-model or --dense-index-dir (to compute picks) or --sentence-picks-dir (reuse).")

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

    # Load or compute sentence picks (same as rerank_stage3_sentence)
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
        from sentence_pick import load_doc_title_sentences, run_sentence_pick, save_picks_json
        if args.dense_model:
            from sentence_transformers import SentenceTransformer
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
            raise ValueError("Provide --dense-model or --dense-index-dir to compute picks.")
        for name in run_names:
            run_map = run_maps[name]
            candidate_docnos = set()
            for doc_list in run_map.values():
                candidate_docnos.update(doc_list)
            doc_title_sentences = load_doc_title_sentences(candidate_docnos, docs_jsonl)
            picks = run_sentence_pick(
                run_map=run_map, topics=topics_map, doc_title_sentences=doc_title_sentences,
                model=model_dense, top_k=args.sentence_top_k, k_rrf=60, w_bm25=1.0, w_dense=1.0,
                dense_batch_size=256, normalize_embeddings=normalize_emb,
            )
            all_picks[name] = picks
            picks_dir = output_cfg.output_dir / "sentence_picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            save_picks_json(picks, picks_dir / f"{name}.json")
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

    reranked_runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for name in run_names:
        run_name = f"{name}_merged"
        reranked = rerank_run_merged(
            run_maps[name], topics=topics_map, picks=all_picks[name], doc_titles=doc_titles,
            model=reranker, batch_size=args.model_batch,
        )
        reranked_runs[run_name] = reranked
        print("reranked", run_name, "queries:", len(reranked))

    for run_name, reranked in reranked_runs.items():
        rows = []
        for qid, docs in reranked.items():
            for rank, (docno, score) in enumerate(docs, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        run_df = pd.DataFrame(rows)
        run_df.to_csv(output_cfg.runs_dir / f"{run_name}.tsv", sep="\t", index=False)

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
            base_name = run_name.replace("_merged", "")
            split = _parse_split_from_run_stem(base_name)
            if split is not None and split in split_to_role:
                role, label = split_to_role[split], split_to_label[split]
            else:
                role, label = "unknown", run_name
            summary_rows.append({"run": run_name, "label": label, "role": role, **metrics})

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(output_cfg.metrics_path, index=False)
            try:
                from plot_rerank_eval import build_and_save_hybrid_reranker_plots
                run_maps_for_plot = {rn: run_maps[rn.replace("_merged", "")] for rn in reranked_runs if rn.replace("_merged", "") in run_maps}
                build_and_save_hybrid_reranker_plots(
                    pd.DataFrame(summary_rows), run_maps_for_plot, gold_map_all, output_cfg.output_dir,
                    candidate_limit=cap, config={"split_to_role": split_to_role, "split_to_label": split_to_label},
                )
            except Exception as e:
                print("warning: could not generate eval plots:", e)

    split_to_role_cfg, split_to_label_cfg = {}, {}
    if train_subset_json or test_batch_jsons:
        try:
            split_to_role_cfg, split_to_label_cfg = _build_split_to_role_and_label(
                train_subset_json, [Path(p) for p in test_batch_jsons]
            )
        except ValueError:
            pass

    config = {
        "model": args.model, "model_device": model_device, "model_batch": args.model_batch,
        "model_max_length": args.model_max_length, "candidate_limit": args.candidate_limit,
        "runs_dir": str(runs_dir), "run_files": [str(p) for p in run_files], "docs_jsonl": str(docs_jsonl),
        "train_subset_json": str(train_subset_json) if train_subset_json else "",
        "test_batch_jsons": [str(p) for p in test_batch_jsons], "ks_recall": list(ks_recall),
        "split_to_role": split_to_role_cfg, "split_to_label": split_to_label_cfg,
        "dense_model": args.dense_model or "", "dense_index_dir": str(args.dense_index_dir) if args.dense_index_dir else "",
        "sentence_picks_dir": str(args.sentence_picks_dir) if args.sentence_picks_dir else "",
        "sentence_top_k": args.sentence_top_k, "variant": "merged",
    }
    output_cfg.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("saved outputs to", output_cfg.output_dir)


if __name__ == "__main__":
    main()
