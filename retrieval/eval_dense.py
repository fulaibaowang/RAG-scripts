#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import hnswlib  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency 'hnswlib' (pip install hnswlib).") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency 'sentence-transformers' (pip install sentence-transformers).") from e


# Allow importing retrieval_eval from public scripts root
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval_eval.common import (  # noqa: E402
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    RECALL_KS,
    run_df_to_run_map,
)


def _load_rowid_to_pmid_tsv(path: Path) -> list[str]:
    rowid_to_pmid: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rowid_s, pmid = line.split("\t", 1)
            rowid = int(rowid_s)
            if rowid != len(rowid_to_pmid):
                raise ValueError(
                    f"Non-contiguous rowid mapping at rowid={rowid}, expected={len(rowid_to_pmid)}"
                )
            rowid_to_pmid.append(pmid.strip())
    if not rowid_to_pmid:
        raise ValueError(f"Empty rowid_to_pmid mapping: {path}")
    return rowid_to_pmid


def load_dense_runtime(
    index_dir: Path,
    device: str,
    model_name_override: str | None = None,
    ef_search_override: int | None = None,
) -> tuple[SentenceTransformer, hnswlib.Index, list[str], dict[str, Any]]:
    """Load SentenceTransformer + HNSW index + rowid->PMID mapping from a build_dense_hnsw_index output dir."""
    meta_path = index_dir / "meta.json"
    idx_path = index_dir / "hnsw_index.bin"
    map_path = index_dir / "rowid_to_pmid.tsv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in index_dir: {meta_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing hnsw_index.bin in index_dir: {idx_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing rowid_to_pmid.tsv in index_dir: {map_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_name = model_name_override or meta.get("model_name") or "abhinand/MedEmbed-small-v0.1"
    max_seq_length = int(meta.get("max_seq_length") or 512)
    normalize_embeddings = bool(meta.get("normalize_embeddings", True))
    dim = int(meta.get("dim") or 0)
    space = str(meta.get("hnsw_space") or ("cosine" if normalize_embeddings else "l2"))

    rowid_to_pmid = _load_rowid_to_pmid_tsv(map_path)

    model = SentenceTransformer(str(model_name), device=device)
    model.max_seq_length = max_seq_length
    if dim <= 0:
        dim = int(model.get_sentence_embedding_dimension())

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(idx_path), max_elements=len(rowid_to_pmid))

    ef_search = ef_search_override if ef_search_override is not None else int(meta.get("hnsw_ef_search") or 100)
    index.set_ef(int(ef_search))

    runtime_meta = dict(meta)
    runtime_meta.update(
        {
            "loaded_model_name": str(model_name),
            "loaded_device": device,
            "loaded_max_seq_length": max_seq_length,
            "loaded_dim": dim,
            "loaded_hnsw_space": space,
            "loaded_ef_search": int(ef_search),
            "loaded_normalize_embeddings": bool(normalize_embeddings),
            "index_dir": str(index_dir),
        }
    )

    print(
        "[dense-runtime]",
        {
            "model_name": str(model_name),
            "device": device,
            "dim": dim,
            "space": space,
            "normalize": bool(normalize_embeddings),
            "mapping": len(rowid_to_pmid),
            "ef_search": int(ef_search),
        },
    )

    return model, index, rowid_to_pmid, runtime_meta


def dense_retrieve_topics(
    model: SentenceTransformer,
    index: hnswlib.Index,
    rowid_to_pmid: list[str],
    topics_df: pd.DataFrame,
    topk: int,
    batch_size: int,
    normalize_embeddings: bool,
    space: str,
    ef: int | None = None,
) -> pd.DataFrame:
    if "qid" not in topics_df.columns or "query" not in topics_df.columns:
        raise ValueError("topics_df must have columns: qid, query")

    topk = int(topk)
    if topk <= 0:
        raise ValueError("topk must be positive")

    if ef is not None:
        # efSearch controls how many candidates HNSW explores.
        # Larger efSearch tends to improve recall but costs time.
        index.set_ef(int(ef))

    qids = topics_df["qid"].astype(str).tolist()
    queries = topics_df["query"].astype(str).tolist()

    rows: list[dict[str, Any]] = []

    for i in range(0, len(queries), int(batch_size)):
        batch_qids = qids[i : i + int(batch_size)]
        batch_queries = queries[i : i + int(batch_size)]

        q_emb = model.encode(
            batch_queries,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize_embeddings),
            show_progress_bar=False,
        ).astype(np.float32)

        labels, distances = index.knn_query(q_emb, k=topk)

        for local_j, qid in enumerate(batch_qids):
            for rank in range(topk):
                rid = int(labels[local_j, rank])
                dist = float(distances[local_j, rank])
                pmid = rowid_to_pmid[rid] if 0 <= rid < len(rowid_to_pmid) else ""
                pmid = normalize_pmid(pmid)
                if not pmid:
                    continue

                # store larger-is-better score
                if space == "cosine":
                    score = 1.0 - dist
                else:
                    score = -dist

                rows.append({"qid": str(qid), "docno": pmid, "rank": int(rank + 1), "score": float(score)})

    return pd.DataFrame(rows)


# =========================
# Saving helpers
# =========================
def ensure_dense_schema(res_df: pd.DataFrame) -> pd.DataFrame:
    req = {"qid", "docno", "rank", "score"}
    missing = req - set(res_df.columns)
    if missing:
        raise ValueError(f"Dense results missing columns: {missing}")

    out = res_df.copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out["rank"] = out["rank"].astype(int)
    out["score"] = out["score"].astype(float)
    out = out.sort_values(["qid", "rank"], ascending=[True, True])
    out = out.drop_duplicates(["qid", "docno"], keep="first")
    out["rank"] = out.groupby("qid").cumcount() + 1
    out = out.reset_index(drop=True)
    return out


def save_dense_outputs(
    out_dir: Path,
    split: str,
    res_df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
    save_run_map: bool = True,
):
    """Write run as TSV only (canonical format). Optionally write meta.json."""
    save_dense_run_tsv(out_dir, split, res_df)
    if meta is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"dense_{split}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def save_dense_run_tsv(out_dir: Path, split: str, res_df: pd.DataFrame) -> None:
    """Write run as TSV only (canonical format: qid, docno, rank, score)."""
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    res_df = ensure_dense_schema(res_df)
    tsv_path = runs_dir / f"dense_{split}.tsv"
    res_df[["qid", "docno", "rank", "score"]].to_csv(tsv_path, sep="\t", index=False)
    print(f"[saved] {tsv_path}")


# =========================
# Eval driver
# =========================
def evaluate_and_save_dense_on_questions(
    questions: list[dict],
    split: str,
    out_dir: Path,
    model: SentenceTransformer,
    index: hnswlib.Index,
    rowid_to_pmid: list[str],
    normalize_embeddings: bool,
    space: str,
    topk: int,
    ks_recall: tuple[int, ...],
    ef_search: int | None,
    batch_size: int,
    meta_base: dict[str, Any] | None = None,
    save: bool = True,
    save_per_query: bool = False,
) -> dict[str, Any]:
    topics_df, gold_map = build_topics_and_gold(questions)

    res_df = dense_retrieve_topics(
        model=model,
        index=index,
        rowid_to_pmid=rowid_to_pmid,
        topics_df=topics_df,
        topk=topk,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        space=space,
        ef=ef_search,
    )

    # enforce cutoff
    res_df = res_df[res_df["rank"] <= topk].copy()

    # save artifacts
    if save:
        meta = dict(meta_base or {})
        meta.update({
            "split": split,
            "n_queries": int(topics_df.shape[0]),
            "topk": int(topk),
            "ef_search": int(ef_search) if ef_search is not None else None,
            "ks_recall": list(ks_recall),
        })
        save_dense_outputs(out_dir, split=split, res_df=res_df, meta=meta, save_run_map=True)

    run_map = run_df_to_run_map(res_df, qid_col="qid", docno_col="docno")
    summary, perq = evaluate_run(gold_map, run_map, ks_recall=ks_recall, eps=1e-5)

    if save and save_per_query:
        pq_dir = out_dir / "per_query"
        pq_dir.mkdir(parents=True, exist_ok=True)
        perq.to_csv(pq_dir / f"dense_{split}.csv", index=False)

    return {"method": "Dense", "batch": split, "n_queries": int(topics_df.shape[0]), **summary}


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("Eval dense retrieval (no build). Produces eval_dense-style outputs.")
    ap.add_argument(
        "--index_dir",
        required=True,
        help="Dense index dir containing hnsw_index.bin, rowid_to_pmid.tsv, meta.json",
    )
    ap.add_argument(
        "--out_dir",
        default="../output/eval_dense_MedEmbed",
        help="Output directory (dense_*.parquet, *_meta.json, *_run_map.json)",
    )
    ap.add_argument("--train_subset_json", default="../example/training14b_10pct_sample.json")
    ap.add_argument("--test_batch_jsons", nargs="*", default=[], help="List of 13B*_golden.json files")

    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--ks", type=str, default=",".join(map(str, RECALL_KS)), help="Comma-separated K values for recall (default: RECALL_KS)")
    ap.add_argument(
        "--ef_search",
        type=int,
        default=None,
        help="Override HNSW efSearch (if omitted, use meta.json/default); effective efSearch defaults to >= topk unless limited by --ef_cap",
    )
    ap.add_argument(
        "--ef_cap",
        type=int,
        default=None,
        help="Optional cap on effective efSearch to bound runtime. If ef_cap < topk, deep recall@topk may degrade.",
    )
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--device", type=str, default="cpu", help='"cpu", "cuda", or "mps"')
    ap.add_argument("--model_name", type=str, default="", help="Override SentenceTransformer model name")

    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--save_per_query", action="store_true", help="Save per-query metrics CSVs")
    ap.add_argument("--no_eval", action="store_true", help="Skip evaluation; only run retrieval and write run TSV")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks_recall = tuple(int(x) for x in args.ks.split(",") if x.strip())

    model, index, rowid_to_pmid, runtime_meta = load_dense_runtime(
        index_dir=Path(args.index_dir),
        device=args.device,
        model_name_override=(args.model_name or None),
        ef_search_override=args.ef_search,
    )
    normalize_embeddings = bool(runtime_meta.get("loaded_normalize_embeddings", True))
    space = str(runtime_meta.get("loaded_hnsw_space") or "cosine")

    meta_base = {
        "notes": args.notes,
        "runtime": runtime_meta,
    }

    # Choose efSearch for this run.
    # Strategy: aim for efSearch >= topk for strong deep recall, but allow --ef_cap to bound runtime.
    ef_requested = int(args.ef_search) if args.ef_search is not None else None
    ef_base = ef_requested
    if ef_base is None:
        ef_base = int(runtime_meta.get("loaded_ef_search") or 0) or None
    if ef_base is None:
        ef_base = int(args.topk)

    ef_desired = int(max(int(ef_base), int(args.topk)))
    ef_effective = int(ef_desired)
    if args.ef_cap is not None:
        ef_effective = int(min(int(ef_effective), int(args.ef_cap)))

    print(
        "[dense-runtime] ef_search:",
        {
            "requested": ef_requested,
            "base": int(ef_base),
            "desired_atleast_topk": int(ef_desired),
            "cap": int(args.ef_cap) if args.ef_cap is not None else None,
            "effective": int(ef_effective),
            "topk": int(args.topk),
        },
    )
    if args.ef_cap is not None and int(args.ef_cap) < int(args.topk):
        print(
            "[dense-runtime][warn] ef_cap < topk; expect lower recall@topk and/or less stable deep ranks.",
            file=sys.stderr,
        )

    meta_base["ef_search"] = {
        "requested": ef_requested,
        "base": int(ef_base),
        "desired_atleast_topk": int(ef_desired),
        "cap": int(args.ef_cap) if args.ef_cap is not None else None,
        "effective": int(ef_effective),
    }

    train_stem = Path(args.train_subset_json).stem

    if args.no_eval:
        train_data = json.loads(Path(args.train_subset_json).read_text(encoding="utf-8"))
        topics_df, _ = build_topics_and_gold(train_data["questions"])
        res_df = dense_retrieve_topics(
            model=model,
            index=index,
            rowid_to_pmid=rowid_to_pmid,
            topics_df=topics_df,
            topk=args.topk,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            space=space,
            ef=ef_effective,
        )
        save_dense_run_tsv(out_dir, train_stem, res_df)
        for fp in args.test_batch_jsons:
            p = Path(fp)
            data = json.loads(p.read_text(encoding="utf-8"))
            topics_df, _ = build_topics_and_gold(data["questions"])
            res_df = dense_retrieve_topics(
                model=model,
                index=index,
                rowid_to_pmid=rowid_to_pmid,
                topics_df=topics_df,
                topk=args.topk,
                batch_size=args.batch_size,
                normalize_embeddings=normalize_embeddings,
                space=space,
                ef=ef_effective,
            )
            save_dense_run_tsv(out_dir, p.stem, res_df)
        config = vars(args)
        config["index_dir"] = str(args.index_dir)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("No-eval mode: runs saved to", out_dir / "runs")
        return

    all_rows = []

    # train subset
    train_data = json.loads(Path(args.train_subset_json).read_text(encoding="utf-8"))
    all_rows.append(
        evaluate_and_save_dense_on_questions(
            train_data["questions"],
            split=train_stem,
            out_dir=out_dir,
            model=model,
            index=index,
            rowid_to_pmid=rowid_to_pmid,
            normalize_embeddings=normalize_embeddings,
            space=space,
            topk=args.topk,
            ks_recall=ks_recall,
            ef_search=ef_effective,
            batch_size=args.batch_size,
            meta_base=meta_base,
            save=True,
            save_per_query=bool(args.save_per_query),
        )
    )

    # tests
    for fp in args.test_batch_jsons:
        p = Path(fp)
        data = json.loads(p.read_text(encoding="utf-8"))
        all_rows.append(
            evaluate_and_save_dense_on_questions(
                data["questions"],
                split=p.stem,
                out_dir=out_dir,
                model=model,
                index=index,
                rowid_to_pmid=rowid_to_pmid,
                normalize_embeddings=normalize_embeddings,
                space=space,
                topk=args.topk,
                ks_recall=ks_recall,
                ef_search=ef_effective,
                batch_size=args.batch_size,
                meta_base=meta_base,
                save=True,
                save_per_query=bool(args.save_per_query),
            )
        )

    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print("\n=== Dense metrics ===")
    print(metrics_df)
    print(f"[saved] {metrics_csv}")


if __name__ == "__main__":
    main()
