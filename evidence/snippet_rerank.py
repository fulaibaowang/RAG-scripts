#!/usr/bin/env python3
"""
Snippet (window) extraction and CE reranking.

For each query, takes top-N docs from a rerank_hybrid run, generates overlapping
sentence windows from each abstract, applies two-stage selection:
  Stage A  – BM25 + dense hybrid (RRF) to keep top-W windows per doc
  Stage B  – Cross-encoder rerank on the kept windows

Outputs:
  - Reranked windows  (JSONL per split)
  - Reranked docs     (TSV per split, doc score = max window CE score)
  - MAP@10 evaluation (per-query CSV + summary metrics CSV)
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# PYTHONPATH is set by run_retrieval_rerank_pipeline.sh; retrieval_eval.common is expected there.

try:
    import torch
except ImportError:
    torch = None

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
    run_df_to_run_map,
)

# ---------------------------------------------------------------------------
# NLTK sentence tokenizer
# ---------------------------------------------------------------------------
_nltk_punkt_ensured: bool = False


def _ensure_nltk_punkt() -> None:
    global _nltk_punkt_ensured
    if _nltk_punkt_ensured:
        return
    import nltk
    try:
        nltk.sent_tokenize("Hello world.")
    except LookupError:
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    _nltk_punkt_ensured = True


# ---------------------------------------------------------------------------
# Document I/O  (mirrors rerank_stage2 helpers, supports glob)
# ---------------------------------------------------------------------------
def _resolve_jsonl_paths(path_or_glob: Path) -> List[Path]:
    s = str(path_or_glob)
    if "*" in s or "?" in s:
        paths = sorted(Path(p) for p in glob_mod.glob(s) if Path(p).is_file())
        if not paths:
            raise FileNotFoundError(f"No files matched JSONL glob: {s}")
        return paths
    if not path_or_glob.exists():
        raise FileNotFoundError(f"JSONL file not found: {path_or_glob}")
    return [path_or_glob]


def _extract_docno(rec: dict) -> str:
    for key in ("docno", "pmid", "id"):
        if key in rec:
            return normalize_pmid(rec[key])
    return ""


def load_doc_title_sentences(
    docnos: set[str],
    jsonl_path: Path,
) -> Dict[str, Tuple[str, List[str]]]:
    """Load docno -> (title, [sentence, ...]) for requested docs."""
    _ensure_nltk_punkt()
    import nltk

    wanted = set(map(str, docnos))
    out: Dict[str, Tuple[str, List[str]]] = {}
    paths = _resolve_jsonl_paths(jsonl_path)
    n_files = len(paths)
    if n_files > 1:
        print(f"[docs] scanning {n_files} JSONL files from glob: {jsonl_path}")
    for fi, fp in enumerate(paths):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                docno = _extract_docno(rec)
                if docno not in wanted or docno in out:
                    continue
                title = str(rec.get("title", "")).strip()
                abstract = str(rec.get("abstract", "") or rec.get("abstractText", "")).strip()
                sentences = [s.strip() for s in nltk.sent_tokenize(abstract) if s.strip()] if abstract else []
                out[docno] = (title, sentences)
                if len(out) == len(wanted):
                    break
        if len(out) == len(wanted):
            break
        if n_files > 1:
            print(f"[docs] scanned {fi+1}/{n_files} files, found {len(out)}/{len(wanted)} docs")
    missing = wanted - set(out.keys())
    if missing:
        print(f"WARNING: {len(missing)}/{len(wanted)} docs not found in corpus. Using empty text.")
    return out


# ---------------------------------------------------------------------------
# Run TSV loading  (mirrors rerank_stage2.load_run_tsv)
# ---------------------------------------------------------------------------
def load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid") or cols.get("query_id") or df.columns[0]
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc") or df.columns[1]
    rank_col = cols.get("rank")
    score_col = cols.get("score")
    out = pd.DataFrame({
        "qid": df[qid_col].astype(str),
        "docno": df[doc_col].astype(str).map(normalize_pmid),
    })
    out["rank"] = df[rank_col].astype(int) if rank_col else out.groupby("qid").cumcount() + 1
    if score_col:
        out["score"] = df[score_col].astype(float)
    else:
        out["score"] = np.nan
    return out.sort_values(["qid", "rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------
def build_windows(
    title: str,
    sentences: List[str],
    window_size: int,
    stride: int,
) -> List[str]:
    """
    Generate overlapping sentence windows.
    Each window = title + consecutive sentences.
    Docs with fewer than window_size sentences produce a single window.
    """
    if not sentences:
        return [title] if title else []
    if len(sentences) <= window_size:
        return [f"{title} {' '.join(sentences)}".strip()]
    windows = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        chunk = " ".join(sentences[i : i + window_size])
        windows.append(f"{title} {chunk}".strip())
    return windows


# ---------------------------------------------------------------------------
# Stage A: BM25 + Dense hybrid scoring (RRF)
# ---------------------------------------------------------------------------
def _tokenize_for_bm25(text: str) -> List[str]:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()


def _window_scores_bm25(query: str, windows: List[str], bm25_class) -> np.ndarray:
    if not windows:
        return np.array([], dtype=np.float64)
    corpus = [_tokenize_for_bm25(w) for w in windows]
    non_empty = [i for i, t in enumerate(corpus) if t]
    if not non_empty:
        return np.zeros(len(windows), dtype=np.float64)
    sub_corpus = [corpus[i] for i in non_empty]
    bm25 = bm25_class(sub_corpus)
    q_tok = _tokenize_for_bm25(query)
    if not q_tok:
        return np.zeros(len(windows), dtype=np.float64)
    sub_scores = bm25.get_scores(q_tok)
    out = np.zeros(len(windows), dtype=np.float64)
    for j, i in enumerate(non_empty):
        out[i] = float(sub_scores[j])
    return out


def _window_scores_dense(
    query: str,
    windows: List[str],
    model,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    if not windows:
        return np.array([], dtype=np.float64)
    q_emb = model.encode(
        [query], batch_size=1, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=False,
    ).astype(np.float32)[0]
    w_embs = model.encode(
        windows, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=False,
    ).astype(np.float32)
    if normalize:
        return np.dot(w_embs, q_emb)
    qn = np.linalg.norm(q_emb)
    if qn < 1e-9:
        return np.zeros(len(windows), dtype=np.float64)
    return np.dot(w_embs, q_emb) / (np.linalg.norm(w_embs, axis=1) * qn + 1e-9)


def _rrf_fuse(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k_rrf: int,
    w_a: float,
    w_b: float,
) -> np.ndarray:
    n = len(scores_a)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    ranks_a = np.argsort(np.argsort(-scores_a)) + 1
    ranks_b = np.argsort(np.argsort(-scores_b)) + 1
    return w_a / (k_rrf + ranks_a) + w_b / (k_rrf + ranks_b)


def select_top_windows(
    query: str,
    windows: List[str],
    dense_model,
    bm25_class,
    top_w: int,
    k_rrf: int = 60,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
    dense_batch: int = 256,
    normalize_emb: bool = True,
) -> List[int]:
    """Return indices of the top-W windows after BM25+dense RRF fusion."""
    if not windows:
        return []
    if len(windows) <= top_w:
        return list(range(len(windows)))
    bm25_sc = _window_scores_bm25(query, windows, bm25_class)
    dense_sc = _window_scores_dense(query, windows, dense_model, dense_batch, normalize_emb)
    fused = _rrf_fuse(bm25_sc, dense_sc, k_rrf, w_bm25, w_dense)
    return [int(i) for i in np.argsort(-fused)[:top_w]]


# ---------------------------------------------------------------------------
# Stage B multi-GPU helpers
# ---------------------------------------------------------------------------
def _chunk_ce_items(
    items: List[Tuple[str, List[Tuple[str, int, str]]]],
    n: int,
) -> List[List[Tuple[str, List[Tuple[str, int, str]]]]]:
    """Split (qid, win_list) items into n chunks for multi-GPU."""
    if n <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / n))
    return [
        items[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
        if items[i * chunk_size : (i + 1) * chunk_size]
    ]


def _ce_worker(
    gpu_id: int,
    items: List[Tuple[str, List[Tuple[str, int, str]]]],
    topics: Dict[str, str],
    model_name: str,
    batch_size: int,
    max_length: int,
    return_dict,
) -> None:
    """Run CE rerank on a chunk of (qid, win_list) on one GPU."""
    from sentence_transformers import CrossEncoder

    device = f"cuda:{gpu_id}" if torch and torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_name,
        device=device,
        max_length=None if max_length <= 0 else max_length,
    )
    local_out: Dict[str, List[Tuple[str, int, str, float]]] = {}
    total = len(items)
    t0 = time()
    for idx, (qid, win_list) in enumerate(items, 1):
        query = topics.get(qid, "").strip()
        if not win_list or not query:
            local_out[qid] = [(d, wi, wt, 0.0) for d, wi, wt in win_list]
            continue
        pairs = [(query, wt) for _, _, wt in win_list]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float64)
        scored = [
            (docno, wi, wt, float(sc))
            for (docno, wi, wt), sc in zip(win_list, scores)
        ]
        scored.sort(key=lambda x: x[3], reverse=True)
        local_out[qid] = scored
        if idx == 1 or idx % 10 == 0 or idx == total:
            elapsed = max(1e-9, time() - t0)
            print(f"[Stage B gpu {gpu_id}] {idx}/{total} queries | {idx/elapsed:.2f} q/s")
    return_dict[gpu_id] = local_out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    output_dir: Path
    runs_dir: Path
    windows_dir: Path
    per_query_dir: Path
    metrics_path: Path
    config_path: Path


def _build_output_config(base_dir: Path) -> OutputConfig:
    d = base_dir
    runs = d / "runs"
    wins = d / "windows"
    pq = d / "per_query"
    for p in (d, runs, wins, pq):
        p.mkdir(parents=True, exist_ok=True)
    return OutputConfig(
        output_dir=d, runs_dir=runs, windows_dir=wins,
        per_query_dir=pq, metrics_path=d / "metrics.csv", config_path=d / "config.json",
    )


def _parse_split_from_run_stem(run_stem: str) -> Optional[str]:
    m = re.fullmatch(r"best_rrf_(.+)_top\d+(?:_rrf_pool\d+_k\d+)?", run_stem)
    return m.group(1) if m else None


def _build_split_to_role_and_label(
    train_json: Optional[Path],
    test_batch_jsons: List[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    split_to_role: Dict[str, str] = {}
    split_to_label: Dict[str, str] = {}
    if train_json and train_json.exists():
        stem = train_json.stem
        split_to_role[stem] = "train"
        split_to_label[stem] = stem
    for p in test_batch_jsons:
        stem = Path(p).stem
        split_to_role[stem] = "test"
        split_to_label[stem] = stem
    return split_to_role, split_to_label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Snippet window extraction + two-stage (hybrid→CE) reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = p.add_argument_group("inputs")
    inp.add_argument("--runs-dir", type=Path, default=None, help="Dir with rerank_hybrid run TSVs.")
    inp.add_argument("--run-files", type=Path, nargs="*", default=None, help="Explicit run TSV files.")
    inp.add_argument("--run-glob", type=str, default="*.tsv", help="Glob under --runs-dir.")
    inp.add_argument("--docs-jsonl", type=str, required=True, help="JSONL corpus path or glob.")
    inp.add_argument("--train-json", type=Path, default=None)
    inp.add_argument("--test-batch-jsons", "--test_batch_jsons", type=Path, nargs="*", default=None)
    inp.add_argument("--query-field", type=str, default="body")
    inp.add_argument("--n-docs", type=int, default=100, help="Top docs per query from input run.")

    win = p.add_argument_group("windows")
    win.add_argument("--window-size", type=int, default=3, help="Sentences per window.")
    win.add_argument("--window-stride", type=int, default=1, help="Stride between windows.")
    win.add_argument("--top-w", type=int, default=8, help="Windows to keep per doc after Stage A.")

    stageA = p.add_argument_group("stage-A (hybrid filter)")
    stageA.add_argument("--dense-model", type=str, required=True, help="SentenceTransformer for dense scoring.")
    stageA.add_argument("--dense-device", type=str, default="cpu", help="Device for dense model.")
    stageA.add_argument("--dense-batch", type=int, default=256)
    stageA.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter.")
    stageA.add_argument("--w-bm25", type=float, default=1.0, help="BM25 weight in RRF.")
    stageA.add_argument("--w-dense", type=float, default=1.0, help="Dense weight in RRF.")

    stageB = p.add_argument_group("stage-B (CE rerank)")
    stageB.add_argument("--ce-model", type=str, default="BAAI/bge-reranker-v2-m3")
    stageB.add_argument("--ce-device", type=str, default="cuda")
    stageB.add_argument("--ce-batch", type=int, default=64)
    stageB.add_argument("--ce-max-length", type=int, default=512)
    stageB.add_argument("--ce-use-multi-gpu", action="store_true", help="Enable multi-GPU for Stage B CE rerank.")
    stageB.add_argument("--ce-num-gpus", type=int, default=0, help="Max GPUs for CE (0 = all).")

    ev = p.add_argument_group("evaluation")
    ev.add_argument("--disable-metrics", action="store_true")
    ev.add_argument("--ks-recall", type=str, default="50,100,200,300,400,500,1000,2000,5000")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", type=Path, required=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- resolve run files ---
    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    elif args.runs_dir:
        run_files = sorted(args.runs_dir.glob(args.run_glob))
    else:
        raise FileNotFoundError("Provide --runs-dir or --run-files.")
    if not run_files:
        raise FileNotFoundError("No run files found.")
    print(f"[init] {len(run_files)} run file(s): {[p.name for p in run_files]}")

    # --- topics + gold ---
    topics: Dict[str, str] = {}
    gold_all: Dict[str, List[str]] = {}

    def _add_questions(json_path: Optional[Path]) -> None:
        if not json_path or not json_path.exists():
            return
        qs = load_questions(json_path)
        tdf, gm = build_topics_and_gold(qs, query_field=args.query_field)
        topics.update(dict(zip(tdf["qid"].astype(str), tdf["query"].astype(str))))
        for qid, docs in gm.items():
            gold_all[qid] = docs

    _add_questions(args.train_json)
    for p in args.test_batch_jsons or []:
        _add_questions(Path(p))
    print(f"[init] {len(topics)} queries loaded")

    # --- load dense model (Stage A) ---
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(args.dense_model, device=args.dense_device)
    normalize_emb = True
    print(f"[init] dense model loaded: {args.dense_model} on {args.dense_device}")

    # --- load CE model (Stage B); skip when multi-GPU (each worker loads its own) ---
    from sentence_transformers import CrossEncoder

    ce_model = None
    if not args.ce_use_multi_gpu:
        ce_device = args.ce_device
        if ce_device == "auto":
            if torch and torch.cuda.is_available():
                ce_device = "cuda"
            else:
                ce_device = "cpu"
        ce_model = CrossEncoder(
            args.ce_model, device=ce_device,
            max_length=None if args.ce_max_length <= 0 else args.ce_max_length,
        )
        print(f"[init] CE model loaded: {args.ce_model} on {ce_device}")
    else:
        if not torch or not torch.cuda.is_available():
            raise RuntimeError("CE multi-GPU requested but CUDA is not available.")
        n_dev = torch.cuda.device_count()
        if n_dev < 2:
            raise RuntimeError("CE multi-GPU requested but fewer than 2 CUDA devices found.")
        use_n = n_dev if not args.ce_num_gpus or args.ce_num_gpus < 1 else min(args.ce_num_gpus, n_dev)
        print(f"[init] CE multi-GPU: {use_n} GPUs (model loaded per worker)")

    from rank_bm25 import BM25Okapi

    docs_jsonl = Path(args.docs_jsonl)
    output_cfg = _build_output_config(args.output_dir)

    ks_recall = tuple(int(k) for k in args.ks_recall.split(",") if k.strip()) or RECALL_KS

    # --- save config ---
    def _jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return v
    cfg_dump = {k: _jsonable(v) for k, v in vars(args).items()}
    output_cfg.config_path.write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")

    summary_rows: List[dict] = []
    split_to_role, split_to_label = _build_split_to_role_and_label(
        args.train_json, [Path(p) for p in (args.test_batch_jsons or [])],
    )

    # --- process each run split ---
    for run_path in run_files:
        split_name = run_path.stem
        print(f"\n{'='*60}\n[{split_name}] processing\n{'='*60}")

        # load run, limit to top n_docs per query
        df = load_run_tsv(run_path)
        df = df[df["rank"] <= args.n_docs]
        run_map = run_df_to_run_map(df, qid_col="qid", docno_col="docno")
        n_queries = len(run_map)
        print(f"[{split_name}] {n_queries} queries, top-{args.n_docs} docs each")

        # collect all candidate docnos
        all_docnos: set[str] = set()
        for doc_list in run_map.values():
            all_docnos.update(doc_list)
        print(f"[{split_name}] loading {len(all_docnos)} unique docs from corpus...")
        doc_data = load_doc_title_sentences(all_docnos, docs_jsonl)
        print(f"[{split_name}] loaded title+sentences for {len(doc_data)} docs")

        # ----------------------------------------------------------------
        # Stage A: window generation + hybrid filter
        # ----------------------------------------------------------------
        # kept_windows[qid] = [(docno, window_idx, window_text), ...]
        kept_windows: Dict[str, List[Tuple[str, int, str]]] = {}
        t0 = time()
        for qi, (qid, docnos) in enumerate(run_map.items(), 1):
            query = topics.get(qid, "").strip()
            qid_windows: List[Tuple[str, int, str]] = []
            for docno in docnos:
                title, sents = doc_data.get(docno, ("", []))
                windows = build_windows(title, sents, args.window_size, args.window_stride)
                if not windows:
                    continue
                top_idx = select_top_windows(
                    query, windows, dense_model, BM25Okapi,
                    top_w=args.top_w, k_rrf=args.rrf_k,
                    w_bm25=args.w_bm25, w_dense=args.w_dense,
                    dense_batch=args.dense_batch, normalize_emb=normalize_emb,
                )
                for wi in top_idx:
                    qid_windows.append((docno, wi, windows[wi]))
            kept_windows[qid] = qid_windows

            if qi % 10 == 0 or qi == n_queries:
                elapsed = max(1e-9, time() - t0)
                print(
                    f"[{split_name}][Stage A] {qi}/{n_queries} queries "
                    f"| {qi/elapsed:.2f} q/s "
                    f"| {sum(len(v) for v in kept_windows.values())} windows kept so far"
                )

        total_kept = sum(len(v) for v in kept_windows.values())
        print(f"[{split_name}][Stage A] done – {total_kept} windows across {n_queries} queries")

        # ----------------------------------------------------------------
        # Stage B: CE rerank windows (single- or multi-GPU)
        # ----------------------------------------------------------------
        # ce_results[qid] = [(docno, window_idx, window_text, ce_score), ...]
        ce_results: Dict[str, List[Tuple[str, int, str, float]]] = {}
        items = list(kept_windows.items())
        if args.ce_use_multi_gpu:
            use_n = torch.cuda.device_count() if not args.ce_num_gpus or args.ce_num_gpus < 1 else min(args.ce_num_gpus, torch.cuda.device_count())
            chunks = _chunk_ce_items(items, use_n)
            ctx = torch.multiprocessing.get_context("spawn")
            manager = ctx.Manager()
            return_dict = manager.dict()
            procs = []
            for gpu_id, chunk in enumerate(chunks):
                p = ctx.Process(
                    target=_ce_worker,
                    args=(gpu_id, chunk, topics, args.ce_model, args.ce_batch, args.ce_max_length, return_dict),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            for part in return_dict.values():
                ce_results.update(part)
        else:
            t0 = time()
            for qi, (qid, win_list) in enumerate(items, 1):
                query = topics.get(qid, "").strip()
                if not win_list or not query:
                    ce_results[qid] = [(d, wi, wt, 0.0) for d, wi, wt in win_list]
                    continue
                pairs = [(query, wt) for _, _, wt in win_list]
                scores = ce_model.predict(pairs, batch_size=args.ce_batch, show_progress_bar=False)
                scores = np.asarray(scores, dtype=np.float64)
                scored = [
                    (docno, wi, wt, float(sc))
                    for (docno, wi, wt), sc in zip(win_list, scores)
                ]
                scored.sort(key=lambda x: x[3], reverse=True)
                ce_results[qid] = scored

                if qi % 10 == 0 or qi == n_queries:
                    elapsed = max(1e-9, time() - t0)
                    print(
                        f"[{split_name}][Stage B] {qi}/{n_queries} queries "
                        f"| {qi/elapsed:.2f} q/s"
                    )

        # ----------------------------------------------------------------
        # Save reranked windows (JSONL)
        # ----------------------------------------------------------------
        win_path = output_cfg.windows_dir / f"{split_name}.jsonl"
        with win_path.open("w", encoding="utf-8") as wf:
            for qid, scored_list in ce_results.items():
                for docno, wi, wt, sc in scored_list:
                    wf.write(json.dumps({
                        "qid": qid, "docno": docno,
                        "window_idx": wi, "window_text": wt, "ce_score": sc,
                    }, ensure_ascii=False) + "\n")
        print(f"[{split_name}] saved window results -> {win_path}")

        # ----------------------------------------------------------------
        # Doc-level scores  (max window CE per doc)
        # ----------------------------------------------------------------
        doc_scores: Dict[str, List[Tuple[str, float]]] = {}
        for qid, scored_list in ce_results.items():
            best: Dict[str, float] = {}
            for docno, _wi, _wt, sc in scored_list:
                if docno not in best or sc > best[docno]:
                    best[docno] = sc
            ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
            doc_scores[qid] = ranked

        # save doc run TSV
        rows = []
        for qid, ranked in doc_scores.items():
            for rank, (docno, score) in enumerate(ranked, 1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        run_df = pd.DataFrame(rows)
        run_tsv_path = output_cfg.runs_dir / f"{split_name}.tsv"
        run_df.to_csv(run_tsv_path, sep="\t", index=False)
        print(f"[{split_name}] saved doc run -> {run_tsv_path}")

        # ----------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------
        if not args.disable_metrics and gold_all:
            reranked_map = {qid: [d for d, _ in ranked] for qid, ranked in doc_scores.items()}
            gold_for_run = {qid: gold_all[qid] for qid in reranked_map if qid in gold_all}
            if gold_for_run:
                metrics, perq = evaluate_run(gold_for_run, reranked_map, ks_recall=ks_recall)
                perq.to_csv(output_cfg.per_query_dir / f"{split_name}.csv", index=False)

                split_key = _parse_split_from_run_stem(split_name)
                role = split_to_role.get(split_key, "unknown") if split_key else "unknown"
                label = split_to_label.get(split_key, split_name) if split_key else split_name
                row = {"split": split_name, "label": label, "role": role}
                row.update(metrics)
                summary_rows.append(row)

                print(f"[{split_name}] MAP@10={metrics.get('MAP@10', 0):.4f}  "
                      f"MRR@10={metrics.get('MRR@10', 0):.4f}  "
                      f"Success@10={metrics.get('Success@10', 0):.4f}")

    # --- save summary ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_cfg.metrics_path, index=False)
        print(f"\n[done] metrics -> {output_cfg.metrics_path}")
    print(f"[done] all outputs in {output_cfg.output_dir}")


if __name__ == "__main__":
    main()
