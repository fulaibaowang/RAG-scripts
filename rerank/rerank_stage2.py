from __future__ import annotations

import argparse
import glob as glob_mod
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Suppress Hugging Face "Loading weights" progress in logs (e.g. sbatch .err)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pandas as pd

# Allow importing retrieval_eval from shared_scripts (parent of rerank/).
import sys
_SHARED_SCRIPTS = Path(__file__).resolve().parents[1]  # rerank/ -> shared_scripts/
sys.path.insert(0, str(_SHARED_SCRIPTS))

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e

# FlagLLMReranker is imported lazily when --reranker-type llm is used (see main()).
FlagLLMReranker: Any = None

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
    run_df_to_run_map,
)


@dataclass
class OutputConfig:
    output_dir: Path
    runs_dir: Path
    per_query_dir: Path
    metrics_path: Path
    config_path: Path


def _resolve_repo_root() -> Path:
    """Walk up from this file to find the repo root (.git marker)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent


# Hybrid stage writes best_rrf_{split}_top{k}.tsv; extract logical split for role/label mapping.
def _parse_split_from_run_stem(run_stem: str) -> Optional[str]:
    m = re.fullmatch(r"best_rrf_(.+)_top\d+", run_stem)
    return m.group(1) if m else None


def _build_split_to_role_and_label(
    train_json: Optional[Path],
    test_batch_jsons: List[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build split -> role (train/test) and split -> label (file stem for display)."""
    split_to_role: Dict[str, str] = {}
    split_to_label: Dict[str, str] = {}
    if train_json and train_json.exists():
        train_stem = train_json.stem
        split_to_role[train_stem] = "train"
        split_to_label[train_stem] = train_stem
    for p in test_batch_jsons:
        path = Path(p)
        stem = path.stem
        split_to_role[stem] = "test"
        split_to_label[stem] = stem
    labels = list(split_to_label.values())
    if len(labels) != len(set(labels)):
        raise ValueError(
            "Duplicate dataset labels; train and test file names (stems) must be distinct. "
            f"Labels: {labels}"
        )
    return split_to_role, split_to_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 reranking with a cross-encoder reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--runs-dir", type=Path, required=True, help="Directory with stage-1 run TSV files.")
    inputs.add_argument("--run-files", type=Path, nargs="*", default=None, help="Explicit run TSV files.")
    inputs.add_argument("--run-glob", type=str, default="*.tsv", help="Glob for run files under --runs-dir.")
    inputs.add_argument(
        "--docs-jsonl",
        type=str,
        required=True,
        help="JSONL corpus path or glob pattern (e.g. /pubmed/*.jsonl).",
    )
    inputs.add_argument(
        "--train-jsonl",
        "--train-json",
        type=Path,
        default=None,
        dest="train_jsonl",
        help="Training queries .jsonl.",
    )
    inputs.add_argument(
        "--test-batch-jsonls",
        "--test-batch-jsons",
        "--test_batch_jsonls",
        "--test_batch_jsons",
        type=Path,
        nargs="*",
        default=None,
        dest="test_batch_jsonls",
        help="Test batch .jsonl files (queries + gold).",
    )
    inputs.add_argument("--candidate-limit", type=int, default=2000, help="Stage-1 candidate cutoff per query.")
    inputs.add_argument("--max-queries", type=int, default=None, help="Max queries per split.")
    inputs.add_argument(
        "--query-field",
        type=str,
        default="query_text",
        help="Question key to use as query text for reranker (strict query JSONL: query_text; HyDE etc.: query_text_hyde). Default: query_text.",
    )
    inputs.add_argument(
        "--skip-empty-query-field",
        action="store_true",
        help="Skip questions where --query-field is empty/null instead of raising an error, "
        "and omit those qids from the reranked output. Used by multi-query fusion sub-runs.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    model.add_argument(
        "--reranker-type",
        type=str,
        choices=("cross_encoder", "llm"),
        default="cross_encoder",
        help="Backend: cross_encoder (CrossEncoder) or llm (FlagLLMReranker for e.g. bge-reranker-v2-gemma).",
    )
    model.add_argument(
        "--model-device",
        type=str,
        default="auto",
        help="Model device: auto, cuda, mps, or cpu.",
    )
    model.add_argument("--model-batch", type=int, default=16, help="Batch size for cross-encoder scoring.")
    model.add_argument(
        "--model-max-length",
        type=int,
        default=512,
        help="Max token length for the cross-encoder (set to 0 to use model default).",
    )
    model.add_argument(
        "--llm-use-fp16",
        action="store_true",
        default=True,
        help="Use FP16 for LLM reranker (default True). Ignored when --reranker-type cross_encoder.",
    )
    model.add_argument(
        "--no-llm-use-fp16",
        action="store_false",
        dest="llm_use_fp16",
        help="Disable FP16 for LLM reranker.",
    )
    model.add_argument(
        "--llm-use-bf16",
        action="store_true",
        default=False,
        help="Use BF16 for LLM reranker. Ignored when --reranker-type cross_encoder.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--use-multi-gpu", action="store_true", help="Enable multi-GPU reranking.")
    runtime.add_argument("--num-gpus", type=int, default=0, help="Max GPUs to use (0 = all).")
    runtime.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log rerank progress every N queries (0 = no per-query progress, only start/end). Default: 10.",
    )

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--disable-metrics", action="store_true", help="Skip metrics (no ground truth).")
    evaluation.add_argument(
        "--ks-recall",
        type=str,
        default="50,100,200,300,400,500,1000,2000,5000",
        help="Recall K values as a comma-separated list (default: RECALL_KS).",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, required=True, help="Base output directory.")

    return parser.parse_args()


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


def extract_docno(rec: dict) -> str:
    for key in ("docno", "pmid", "id"):
        if key in rec:
            return normalize_pmid(rec[key])
    return ""


def extract_text(rec: dict) -> str:
    if "text" in rec and rec["text"]:
        return str(rec["text"]).strip()
    parts = []
    if rec.get("title"):
        parts.append(str(rec["title"]).strip())
    if rec.get("abstract"):
        parts.append(str(rec["abstract"]).strip())
    if rec.get("abstractText"):
        parts.append(str(rec["abstractText"]).strip())
    return " ".join([p for p in parts if p])


def _resolve_jsonl_paths(path_or_glob: Path) -> List[Path]:
    """Resolve a single file path or a glob pattern to a sorted list of JSONL files."""
    s = str(path_or_glob)
    if "*" in s or "?" in s:
        paths = sorted(Path(p) for p in glob_mod.glob(s) if Path(p).is_file())
        if not paths:
            raise FileNotFoundError(f"No files matched JSONL glob: {s}")
        return paths
    if not path_or_glob.exists():
        raise FileNotFoundError(f"JSONL file not found: {path_or_glob}")
    return [path_or_glob]


def load_doc_texts(docnos: Iterable[str], jsonl_path: Path) -> Dict[str, str]:
    wanted = set(map(str, docnos))
    out: Dict[str, str] = {}
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
                docno = extract_docno(rec)
                if docno in wanted and docno not in out:
                    out[docno] = extract_text(rec)
                    if len(out) == len(wanted):
                        break
        if len(out) == len(wanted):
            break
        if n_files > 1:
            step = 50
            if (fi + 1) % step == 0 or (fi + 1) == n_files:
                print(f"[docs] scanned {fi+1}/{n_files} files, found {len(out)}/{len(wanted)} docs", flush=True)
    missing = wanted - set(out.keys())
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(wanted)} candidate PMIDs not found in corpus "
            f"({jsonl_path}). Reranker will use empty text for these documents.",
            flush=True,
        )
        if len(missing) <= 20:
            print(f"  missing PMIDs: {sorted(missing)}", flush=True)
        else:
            sample = sorted(missing)[:20]
            print(f"  missing PMIDs (first 20): {sample}", flush=True)
    return out


def _visible_gpu_physical_ids(n: int) -> List[int]:
    """Physical GPU indices for the first n logical devices (Slurm-safe).
    When CUDA_VISIBLE_DEVICES is set (e.g. 3,5), return [3, 5] so workers use the right GPUs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        physical = [int(x.strip()) for x in visible.split(",") if x.strip()]
        return [physical[i] for i in range(min(n, len(physical)))]
    if torch and torch.cuda.is_available():
        return list(range(min(n, torch.cuda.device_count())))
    return list(range(n))


def _chunk_items(items: List[Tuple[str, List[str]]], n: int) -> List[List[Tuple[str, List[str]]]]:
    if n <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / n))
    return [
        items[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
        if items[i * chunk_size : (i + 1) * chunk_size]
    ]


def _rerank_worker(
    gpu_id: int,
    items: List[Tuple[str, List[str]]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model_name: str,
    batch_size: int,
    max_length: int,
    progress_every: int,
    return_dict,
) -> None:
    device = f"cuda:{gpu_id}" if torch and torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_name,
        device=device,
        max_length=None if max_length <= 0 else max_length,
        trust_remote_code=True,
    )
    print(f"[gpu {gpu_id}] model loaded on {device}, reranking {len(items)} queries", flush=True)
    local_out: Dict[str, List[Tuple[str, float]]] = {}
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            local_out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        pairs = [(query, doc_texts.get(doc, "")) for doc in docs]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = [(doc, float(score)) for doc, score in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        local_out[qid] = scored
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == total):
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[gpu {gpu_id}] {idx}/{total} queries | {rate:.2f} q/s", flush=True)
    return_dict[gpu_id] = local_out


def rerank_run(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model: Union[CrossEncoder, Any, None],
    model_name: str,
    batch_size: int,
    max_length: int,
    use_multi_gpu: bool,
    num_gpus: int,
    reranker_type: str = "cross_encoder",
    llm_use_fp16: bool = True,
    llm_use_bf16: bool = False,
    progress_every: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    items = list(run_map.items())
    if use_multi_gpu and reranker_type == "cross_encoder":
        if not torch or not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU requested but CUDA is not available.")
        available = torch.cuda.device_count()
        if available < 2:
            raise RuntimeError("Multi-GPU requested but fewer than 2 CUDA devices found.")
        use_n = available if not num_gpus or num_gpus < 1 else min(num_gpus, available)
        chunks = _chunk_items(items, use_n)
        ctx = torch.multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        procs = []
        for gpu_id, chunk in enumerate(chunks):
            p = ctx.Process(
                target=_rerank_worker,
                args=(gpu_id, chunk, topics, doc_texts, model_name, batch_size, max_length, progress_every, return_dict),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        # Fail fast if any worker crashed (e.g. model load error); avoid silently using partial results.
        for gpu_id, p in enumerate(procs):
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Rerank worker on GPU {gpu_id} exited with code {p.exitcode}. "
                    "Check logs for the actual error (e.g. trust_remote_code, OOM, CUDA)."
                )
        if len(return_dict) != len(procs):
            raise RuntimeError(
                f"Rerank workers returned incomplete results: got {len(return_dict)}/{len(procs)} parts. "
                "One or more workers may have crashed before writing to return_dict."
            )
        merged: Dict[str, List[Tuple[str, float]]] = {}
        for part in return_dict.values():
            merged.update(part)
        return {qid: merged.get(qid, [(doc, float("nan")) for doc in docs]) for qid, docs in items}

    if use_multi_gpu and reranker_type == "llm":
        # CE-style: split (qid, docs) items across GPUs, one model replica per GPU, merge
        if not torch or not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU requested but CUDA is not available.")
        available = torch.cuda.device_count()
        if available < 2:
            raise RuntimeError("Multi-GPU requested but fewer than 2 CUDA devices found.")
        use_n = available if not num_gpus or num_gpus < 1 else min(num_gpus, available)
        chunks = _chunk_items(items, use_n)
        # Slurm/HPC: use physical GPU indices so CUDA_VISIBLE_DEVICES in child maps correctly
        physical_ids = _visible_gpu_physical_ids(use_n)
        ctx = torch.multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        procs = []
        for gpu_id, chunk in enumerate(chunks):
            physical_id = physical_ids[gpu_id] if gpu_id < len(physical_ids) else gpu_id
            p = ctx.Process(
                target=_llm_rerank_worker_pairs,
                args=(
                    gpu_id,
                    physical_id,
                    chunk,
                    topics,
                    doc_texts,
                    model_name,
                    batch_size,
                    llm_use_fp16,
                    llm_use_bf16,
                    progress_every,
                    return_dict,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for gpu_id, p in enumerate(procs):
            if p.exitcode != 0:
                raise RuntimeError(
                    f"LLM rerank worker on GPU {gpu_id} exited with code {p.exitcode}. "
                    "Check logs for the actual error."
                )
        if len(return_dict) != len(procs):
            raise RuntimeError(
                f"LLM rerank workers returned incomplete results: got {len(return_dict)}/{len(procs)} parts."
            )
        merged = {}
        for part in return_dict.values():
            merged.update(part)
        return {qid: merged.get(qid, [(doc, float("nan")) for doc in docs]) for qid, docs in items}

    if model is None:
        raise RuntimeError("Single-GPU path requires a loaded model instance.")

    out: Dict[str, List[Tuple[str, float]]] = {}
    total = len(items)
    start = time()
    use_llm = reranker_type == "llm"
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        pairs = [(query, doc_texts.get(doc, "")) for doc in docs]
        if use_llm:
            # FlagLLMReranker.compute_score([['q','p1'],['q','p2'],...]) returns list of scores
            pair_list = [[q, p] for q, p in pairs]
            scores_list: List[float] = []
            for i in range(0, len(pair_list), batch_size):
                batch = pair_list[i : i + batch_size]
                batch_scores = model.compute_score(batch)
                if isinstance(batch_scores, (int, float)):
                    scores_list.append(float(batch_scores))
                else:
                    scores_list.extend(float(s) for s in batch_scores)
            scores = scores_list
        else:
            scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = [(doc, float(score)) for doc, score in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        out[qid] = scored
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == total):
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[rerank] {idx}/{total} queries | {rate:.2f} q/s")
    return out


def _ensure_flag_llm_reranker() -> Any:
    """Import FlagLLMReranker when using --reranker-type llm. Raises if not installed."""
    global FlagLLMReranker
    if FlagLLMReranker is not None:
        return FlagLLMReranker
    try:
        from FlagEmbedding import FlagLLMReranker as _Cls
        FlagLLMReranker = _Cls  # type: ignore[assignment]
        return FlagLLMReranker
    except ImportError as e:
        raise ImportError(
            "LLM reranker requires FlagEmbedding. Run: pip install FlagEmbedding"
        ) from e


def _llm_rerank_worker_pairs(
    gpu_id: int,
    physical_id: int,
    items: List[Tuple[str, List[str]]],
    topics: Dict[str, str],
    doc_texts: Dict[str, str],
    model_name: str,
    batch_size: int,
    llm_use_fp16: bool,
    llm_use_bf16: bool,
    progress_every: int,
    return_dict: Any,
) -> None:
    """CE-style: one process per GPU, score a chunk of (qid, docs) items with FlagLLMReranker.
    physical_id is the system GPU index (Slurm-safe: use when CUDA_VISIBLE_DEVICES is set)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_id)
    _ensure_flag_llm_reranker()
    model = FlagLLMReranker(
        model_name,
        use_fp16=llm_use_fp16,
        use_bf16=llm_use_bf16,
    )
    device = f"cuda:0"  # after CUDA_VISIBLE_DEVICES, cuda:0 is this process's GPU
    print(f"[gpu {gpu_id}] model loaded on {device}, reranking {len(items)} queries", flush=True)
    local_out: Dict[str, List[Tuple[str, float]]] = {}
    total = len(items)
    start = time()
    for idx, (qid, docs) in enumerate(items, start=1):
        query = topics.get(qid, "").strip()
        if not query:
            local_out[qid] = [(doc, float("nan")) for doc in docs]
            continue
        pairs = [(query, doc_texts.get(doc, "")) for doc in docs]
        pair_list = [[q, p] for q, p in pairs]
        scores_list: List[float] = []
        for i in range(0, len(pair_list), batch_size):
            batch = pair_list[i : i + batch_size]
            batch_scores = model.compute_score(batch)
            if isinstance(batch_scores, (int, float)):
                scores_list.append(float(batch_scores))
            else:
                scores_list.extend(float(s) for s in batch_scores)
        scored = [(doc, float(score)) for doc, score in zip(docs, scores_list)]
        scored.sort(key=lambda x: x[1], reverse=True)
        local_out[qid] = scored
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == total):
            elapsed = max(1e-9, time() - start)
            rate = idx / elapsed
            print(f"[gpu {gpu_id}] {idx}/{total} queries | {rate:.2f} q/s", flush=True)
    return_dict[gpu_id] = local_out


def _llm_rerank_worker(
    gpu_id: int,
    run_names_subset: List[str],
    runs_dir: Path,
    docs_jsonl: Path,
    output_runs_dir: Path,
    worker_args: Dict[str, Any],
) -> None:
    """One process per GPU: load FlagLLMReranker on this GPU, rerank assigned splits, write TSVs.
    If worker_args has 'physical_id', use it for CUDA_VISIBLE_DEVICES (Slurm-safe)."""
    physical_id = worker_args.get("physical_id")
    if physical_id is None:
        physical_id = gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_id)
    _ensure_flag_llm_reranker()
    reranker = FlagLLMReranker(
        worker_args["model"],
        use_fp16=worker_args["llm_use_fp16"],
        use_bf16=worker_args["llm_use_bf16"],
    )
    run_maps: Dict[str, Dict[str, List[str]]] = {}
    for name in run_names_subset:
        path = runs_dir / f"{name}.tsv"
        if not path.exists():
            continue
        df = load_run_tsv(path)
        if worker_args.get("candidate_limit"):
            df = df[df["rank"] <= int(worker_args["candidate_limit"])]
        if worker_args.get("max_queries"):
            qids = sorted(df["qid"].unique())[: int(worker_args["max_queries"])]
            df = df[df["qid"].isin(qids)]
        run_maps[name] = run_df_to_run_map(df)
    if not run_maps:
        return
    topics_map: Dict[str, str] = {}
    _tj = worker_args.get("train_jsonl") or worker_args.get("train_json")
    _tb = worker_args.get("test_batch_jsonls") or worker_args.get("test_batch_jsons") or []
    for json_path in [_tj] + list(_tb):
        if not json_path or not Path(json_path).exists():
            continue
        questions = load_questions(Path(json_path))
        topics_df, _ = build_topics_and_gold(
            questions,
            query_field=worker_args.get("query_field") or "query_text",
            skip_empty=bool(worker_args.get("skip_empty_query_field")),
        )
        topics_map.update(dict(zip(topics_df["qid"], topics_df["query"])))
    candidate_docnos = set()
    for rm in run_maps.values():
        for doc_list in rm.values():
            candidate_docnos.update(doc_list)
    doc_texts = load_doc_texts(candidate_docnos, docs_jsonl)
    batch_size = int(worker_args.get("model_batch") or 16)
    for name in run_names_subset:
        if name not in run_maps:
            continue
        reranked = rerank_run(
            run_maps[name],
            topics=topics_map,
            doc_texts=doc_texts,
            model=reranker,
            model_name=worker_args["model"],
            batch_size=batch_size,
            max_length=0,
            use_multi_gpu=False,
            num_gpus=0,
            reranker_type="llm",
        )
        rows = []
        for qid, docs in reranked.items():
            for rank, (docno, score) in enumerate(docs, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        out_path = output_runs_dir / f"{name}.tsv"
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
        print(f"[gpu {gpu_id}] saved {out_path}", flush=True)


def _build_output_config(base_dir: Path) -> OutputConfig:
    output_dir = base_dir
    runs_dir = output_dir / "runs"
    per_query_dir = output_dir / "per_query"

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir.mkdir(parents=True, exist_ok=True)

    return OutputConfig(
        output_dir=output_dir,
        runs_dir=runs_dir,
        per_query_dir=per_query_dir,
        metrics_path=output_dir / "metrics.csv",
        config_path=output_dir / "config.json",
    )


def main() -> None:
    args = parse_args()
    root = _resolve_repo_root()

    print("[debug] use_multi_gpu:", args.use_multi_gpu, "num_gpus:", args.num_gpus)
    if torch is None:
        print("[debug] torch: not available")
    else:
        print(
            "[debug] torch.cuda.is_available():",
            torch.cuda.is_available(),
            "device_count:",
            torch.cuda.device_count(),
        )

    runs_dir = args.runs_dir
    docs_jsonl = Path(args.docs_jsonl)
    train_json = args.train_jsonl
    test_batch_jsons = args.test_batch_jsonls or []
    output_dir = args.output_dir

    output_cfg = _build_output_config(output_dir)

    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    else:
        run_files = sorted(runs_dir.glob(args.run_glob))

    if not run_files:
        raise FileNotFoundError("No run files found. Provide --run-files or --runs-dir/--run-glob.")

    ks_recall = _parse_ks_recall(args.ks_recall) or RECALL_KS
    # Only report MeanR@K for K <= candidate_limit (above cap it equals MeanR@cap)
    cap = int(args.candidate_limit) if args.candidate_limit else None
    if cap is not None and cap > 0:
        ks_recall = tuple(k for k in ks_recall if k <= cap)
        if not ks_recall:
            ks_recall = (cap,)

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

    run_maps = {name: run_df_to_run_map(df) for name, df in run_dfs.items()}
    run_names = list(run_maps.keys())

    # Resume: skip splits that already have output; load them from disk for metrics later
    runs_dir_out = output_cfg.runs_dir
    run_names_done = [n for n in run_names if (runs_dir_out / f"{n}.tsv").is_file()]
    run_names_todo = [n for n in run_names if n not in run_names_done]
    reranked_runs: Dict[str, List[Tuple[str, float]]] = {}
    if run_names_done:
        print("resume: skipping", len(run_names_done), "existing run(s):", run_names_done)
        for name in run_names_done:
            df = load_run_tsv(runs_dir_out / f"{name}.tsv")
            reranked = {}
            for qid, grp in df.groupby("qid", sort=False):
                grp = grp.sort_values("rank")
                reranked[str(qid)] = list(zip(grp["docno"].astype(str), grp["score"].astype(float)))
            reranked_runs[name] = reranked

    candidate_docnos = set()
    for name in run_names_todo:
        for doc_list in run_maps[name].values():
            candidate_docnos.update(doc_list)

    print("candidate docnos:", len(candidate_docnos))
    doc_texts = load_doc_texts(candidate_docnos, docs_jsonl)
    print("loaded texts:", len(doc_texts))

    topics_map: Dict[str, str] = {}
    gold_map_all: Dict[str, List[str]] = {}

    def _add_questions(json_path: Path, query_field: Optional[str] = None) -> None:
        if not json_path.exists():
            return
        questions = load_questions(json_path)
        topics_df, gold_map = build_topics_and_gold(
            questions, query_field=query_field, skip_empty=args.skip_empty_query_field,
        )
        topics_map.update(dict(zip(topics_df["qid"], topics_df["query"])))
        for qid, docs in gold_map.items():
            gold_map_all[qid] = docs

    if train_json:
        _add_questions(train_json, query_field=args.query_field)

    for path in test_batch_jsons:
        _add_questions(Path(path), query_field=args.query_field)

    if not topics_map:
        print("warning: no query text loaded; reranking will preserve original order.")

    if args.reranker_type == "llm":
        _ensure_flag_llm_reranker()

    model_device = _resolve_device(args.model_device)

    reranker = None
    if not args.use_multi_gpu:
        if args.reranker_type == "llm":
            reranker = FlagLLMReranker(
                args.model,
                use_fp16=args.llm_use_fp16,
                use_bf16=args.llm_use_bf16,
            )
        else:
            reranker = CrossEncoder(
                args.model,
                device=model_device,
                max_length=None if args.model_max_length <= 0 else args.model_max_length,
                trust_remote_code=True,
            )

    for name in run_names_todo:
        reranked = rerank_run(
            run_maps[name],
            topics=topics_map,
            doc_texts=doc_texts,
            model=reranker,
            model_name=args.model,
            batch_size=args.model_batch,
            max_length=args.model_max_length,
            use_multi_gpu=args.use_multi_gpu,
            num_gpus=args.num_gpus,
            reranker_type=args.reranker_type,
            llm_use_fp16=args.llm_use_fp16,
            llm_use_bf16=args.llm_use_bf16,
            progress_every=args.progress_every,
        )
        if args.skip_empty_query_field:
            before = len(reranked)
            reranked = {qid: docs for qid, docs in reranked.items() if qid in topics_map}
            dropped = before - len(reranked)
            if dropped:
                print(f"  [skip-empty] dropped {dropped}/{before} qids not in topics_map")
        reranked_runs[name] = reranked
        print("reranked", name, "queries:", len(reranked))
        rows = []
        for qid, docs in reranked.items():
            for rank, (docno, score) in enumerate(docs, start=1):
                rows.append({"qid": qid, "docno": docno, "rank": rank, "score": score})
        run_df = pd.DataFrame(rows)
        out_path = output_cfg.runs_dir / f"{name}.tsv"
        run_df.to_csv(out_path, sep="\t", index=False)
        print("saved", out_path)

    summary_rows = []

    if not args.disable_metrics and gold_map_all:
        split_to_role, split_to_label = _build_split_to_role_and_label(
            train_json, [Path(p) for p in test_batch_jsons]
        )
        for name, reranked in reranked_runs.items():
            reranked_map = {qid: [doc for doc, _ in docs] for qid, docs in reranked.items()}
            gold_for_run = {qid: gold_map_all[qid] for qid in reranked_map if qid in gold_map_all}
            if not gold_for_run:
                print("skip metrics, no gold overlap for", name)
                continue

            metrics, perq = evaluate_run(gold_for_run, reranked_map, ks_recall=ks_recall)
            perq.to_csv(output_cfg.per_query_dir / f"{name}.csv", index=False)

            split = _parse_split_from_run_stem(name)
            if split is not None and split in split_to_role:
                role = split_to_role[split]
                label = split_to_label[split]
            else:
                role = "unknown"
                label = name

            row = {"run": name, "label": label, "role": role}
            row.update(metrics)
            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_cfg.metrics_path, index=False)
            # Eval plots (Hybrid vs Reranker) when running with ground truth
            try:
                from rerank.plot_rerank_eval import build_and_save_hybrid_reranker_plots
                cap = int(args.candidate_limit) if args.candidate_limit else None
                plot_config = {"split_to_role": split_to_role, "split_to_label": split_to_label}
                build_and_save_hybrid_reranker_plots(
                    summary_df, run_maps, gold_map_all, output_cfg.output_dir,
                    candidate_limit=cap,
                    config=plot_config,
                )
            except Exception as e:
                print("warning: could not generate eval plots:", e)
    elif args.disable_metrics:
        print("metrics disabled")
    else:
        print("no gold provided; skipping metrics")

    split_to_role_cfg: Dict[str, str] = {}
    split_to_label_cfg: Dict[str, str] = {}
    if train_json or test_batch_jsons:
        try:
            split_to_role_cfg, split_to_label_cfg = _build_split_to_role_and_label(
                train_json, [Path(p) for p in test_batch_jsons]
            )
        except ValueError:
            pass

    config = {
        "model": args.model,
        "reranker_type": args.reranker_type,
        "model_device": model_device,
        "model_batch": args.model_batch,
        "model_max_length": args.model_max_length,
        "llm_use_fp16": getattr(args, "llm_use_fp16", False),
        "llm_use_bf16": getattr(args, "llm_use_bf16", False),
        "use_multi_gpu": args.use_multi_gpu,
        "num_gpus": args.num_gpus,
        "candidate_limit": args.candidate_limit,
        "max_queries": args.max_queries,
        "runs_dir": str(runs_dir),
        "run_files": [str(p) for p in run_files],
        "docs_jsonl": str(docs_jsonl),
        "train_jsonl": str(train_json) if train_json else "",
        "test_batch_jsonls": [str(p) for p in test_batch_jsons],
        "ks_recall": list(ks_recall),
        "split_to_role": split_to_role_cfg,
        "split_to_label": split_to_label_cfg,
    }
    output_cfg.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("saved outputs to", output_cfg.output_dir)


if __name__ == "__main__":
    main()
