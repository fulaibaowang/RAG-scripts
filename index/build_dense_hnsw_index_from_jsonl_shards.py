#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
import time
import numpy as np
from tqdm import tqdm

try:
    import orjson
    _json_loads = orjson.loads
    _FAST_JSON = True
except Exception:
    import json as _json
    _json_loads = _json.loads
    _FAST_JSON = False

import hnswlib
from sentence_transformers import SentenceTransformer


# -----------------------------
# Text construction
# -----------------------------
def parse_mesh_terms(mesh_terms: str) -> List[str]:
    if not mesh_terms:
        return []
    parts = [p.strip() for p in mesh_terms.split(";") if p.strip()]
    names = []
    for p in parts:
        if ":" in p:
            names.append(p.split(":", 1)[1].strip())
        else:
            names.append(p)
    return names


def build_doc_text(d: Dict, include_mesh: bool = False) -> str:
    title = (d.get("title") or "").strip()
    abstract = (d.get("abstract") or "").strip()

    if title and abstract:
        text = f"{title}\n\n{abstract}"
    else:
        text = title or abstract

    if include_mesh:
        mesh_names = parse_mesh_terms(d.get("mesh_terms") or "")
        if mesh_names:
            text = f"{text}\n\nMeSH: " + "; ".join(mesh_names)

    return (text or "").strip()


def get_pmid(d: Dict) -> Optional[str]:
    pmid = d.get("pmid")
    if pmid is None or str(pmid).strip() == "":
        pmid = d.get("docno")
    if pmid is None:
        return None
    pmid = str(pmid).strip()
    return pmid or None


# -----------------------------
# IO helpers
# -----------------------------
def iter_jsonl_records(jsonl_glob: str) -> Iterable[Tuple[str, Dict]]:
    """Yield (filepath, record_dict) from multiple JSONL files."""
    files = sorted(glob.glob(jsonl_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {jsonl_glob}")
    for fp in files:
        with open(fp, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json_loads(line)
                except Exception:
                    # fall back to decoding as utf-8 if needed
                    rec = _json_loads(line.decode("utf-8"))
                yield fp, rec


def count_unique_pmids(
    jsonl_glob: str,
    dedup_pmids: bool,
    max_docs: Optional[int] = None,
) -> int:
    """First pass: count docs (unique pmids if dedup)."""
    seen = set() if dedup_pmids else None
    n = 0
    for _, rec in tqdm(iter_jsonl_records(jsonl_glob), desc="Counting docs", unit="rec"):
        pmid = get_pmid(rec)
        if pmid is None:
            continue
        if seen is not None:
            # pmids are numeric strings in PubMed; int saves memory
            try:
                key = int(pmid)
            except Exception:
                key = pmid
            if key in seen:
                continue
            seen.add(key)

        n += 1
        if max_docs is not None and n >= max_docs:
            break
    return n


@dataclass
class IndexMeta:
    created_at: str
    jsonl_glob: str
    model_name: str
    device: str
    max_seq_length: int
    include_mesh: bool
    normalize_embeddings: bool
    dim: int
    dtype: str
    dedup_pmids: bool
    max_docs: Optional[int]
    n_docs_indexed: int
    hnsw_space: str
    hnsw_M: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    batch_size: int


# -----------------------------
# Main build
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build dense HNSW index from JSONL shards (PubMed-style).")
    ap.add_argument("--jsonl_glob", required=True, help='Glob for JSONL shards, e.g. "/path/*.jsonl"')
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--model_name", default="abhinand/MedEmbed-small-v0.1", help="SentenceTransformer model name")
    ap.add_argument("--device", default="cuda", help='Device for embedding, e.g. "cuda", "cpu", "mps"')
    ap.add_argument("--max_seq_length", type=int, default=512, help="Max tokens for encoder (truncation)")
    ap.add_argument("--batch_size", type=int, default=128, help="Embedding batch size")
    ap.add_argument("--include_mesh", action="store_true", help="Append MeSH names into doc text before embedding")
    ap.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization (default: normalize)")
    ap.add_argument("--dedup_pmids", action="store_true", help="De-duplicate documents by PMID across shards")
    ap.add_argument("--max_docs", type=int, default=None, help="Index only first N docs (after dedup), for testing")
    ap.add_argument("--M", type=int, default=32, help="HNSW M (graph degree)")
    ap.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction")
    ap.add_argument("--ef_search", type=int, default=100, help="HNSW efSearch (query-time)")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save checkpoint every N docs")
    ap.add_argument("--max_elements", type=int, default=None, help="If set, skip counting pass and use this max_elements for HNSW.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Count docs for max_elements (HNSW requires it)
    if args.max_elements is not None:
        n_docs_target = args.max_elements
        print(f"[count] Skipping counting pass. Using max_elements={n_docs_target:,}")
    else:
        n_docs_target = count_unique_pmids(
            args.jsonl_glob,
            dedup_pmids=args.dedup_pmids,
            max_docs=args.max_docs,
        )
    if n_docs_target <= 0:
        raise RuntimeError("No docs found to index (after filtering/dedup).")
    print(f"[count] Will index n_docs={n_docs_target:,} (dedup_pmids={args.dedup_pmids})")

    # 2) Load model
    normalize = not args.no_normalize
    model = SentenceTransformer(args.model_name, device=args.device)
    model.max_seq_length = args.max_seq_length

    # Probe dim
    probe = model.encode(["hello world"], convert_to_numpy=True, normalize_embeddings=normalize)
    dim = int(probe.shape[1])
    print(f"[model] {args.model_name} dim={dim} max_seq_length={args.max_seq_length} device={args.device}")

    # 3) Init HNSW
    # Use cosine space if you normalize embeddings; otherwise use "l2" typically.
    space = "cosine" if normalize else "l2"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=n_docs_target,
        ef_construction=args.ef_construction,
        M=args.M,
    )
    index.set_ef(args.ef_search)

    # 4) Build index (2nd pass): stream shards, batch embed, add
    seen = set() if args.dedup_pmids else None

    # rowid -> pmid mapping (stored as numpy object array for flexibility)
    rowid_to_pmid: List[str] = []
    next_id = 0

    # batching
    batch_texts: List[str] = []
    batch_ids: List[int] = []
    batch_pmids: List[str] = []

    def flush_batch() -> int:
        n = len(batch_texts)
        if n == 0:
            return 0

        t0 = time.time()
        emb = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        t1 = time.time()

        index.add_items(emb, np.array(batch_ids, dtype=np.int64))
        t2 = time.time()

        # "indexed_in_index" = how many have been flushed into rowid_to_pmid so far (before extending)
        indexed_in_index_before = len(rowid_to_pmid)

        print(
            f"[flush] indexed_in_index_before={indexed_in_index_before} next_id_assigned={next_id} "
            f"n={n} encode={t1-t0:.1f}s add={t2-t1:.1f}s total={t2-t0:.1f}s",
            flush=True,
        )

        rowid_to_pmid.extend(batch_pmids)
        batch_texts.clear()
        batch_ids.clear()
        batch_pmids.clear()
        return n
    
    pbar = tqdm(total=n_docs_target, desc="Indexing", unit="doc")
    for _, rec in iter_jsonl_records(args.jsonl_glob):
        pmid = get_pmid(rec)
        if pmid is None:
            continue

        if seen is not None:
            try:
                key = int(pmid)
            except Exception:
                key = pmid
            if key in seen:
                continue
            seen.add(key)

        text = build_doc_text(rec, include_mesh=args.include_mesh)
        # You said no filter about deleted/abstract, so we keep even empty abstract/title-only.
        # But we should skip truly empty text (rare).
        if not text:
            # still keep mapping? better to skip empty, otherwise embedding becomes meaningless
            continue

        doc_id = next_id
        next_id += 1

        batch_texts.append(text)
        batch_ids.append(doc_id)
        batch_pmids.append(pmid)

        if len(batch_texts) >= args.batch_size:
            n_flushed = flush_batch()
            pbar.update(n_flushed)

            if args.save_every and next_id % args.save_every == 0:
                # checkpoint
                idx_path = os.path.join(args.out_dir, "hnsw_index.partial.bin")
                index.save_index(idx_path)
                map_path = os.path.join(args.out_dir, "rowid_to_pmid.partial.tsv")
                with open(map_path, "w", encoding="utf-8") as f:
                    for i, p in enumerate(rowid_to_pmid):
                        f.write(f"{i}\t{p}\n")

        if args.max_docs is not None and next_id >= args.max_docs:
            break

        if next_id >= n_docs_target:
            print(f"[warn] Reached max_elements={n_docs_target:,}; stopping early.", flush=True)
            break

    # flush remaining
    n_flushed = flush_batch()
    pbar.update(n_flushed)
    pbar.close()

    n_indexed = next_id
    print(f"[done] Indexed docs (assigned ids): {n_indexed:,}")
    print(f"[done] rowid_to_pmid: {len(rowid_to_pmid):,}  hnsw_count: {index.get_current_count():,}")

    # 5) Save index + mapping + metadata
    index_path = os.path.join(args.out_dir, "hnsw_index.bin")
    index.save_index(index_path)

    map_path = os.path.join(args.out_dir, "rowid_to_pmid.tsv")
    with open(map_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(rowid_to_pmid):
            f.write(f"{i}\t{p}\n")

    meta = IndexMeta(
        created_at=datetime.utcnow().isoformat() + "Z",
        jsonl_glob=args.jsonl_glob,
        model_name=args.model_name,
        device=args.device,
        max_seq_length=args.max_seq_length,
        include_mesh=bool(args.include_mesh),
        normalize_embeddings=bool(normalize),
        dim=dim,
        dtype="float32",
        dedup_pmids=bool(args.dedup_pmids),
        max_docs=args.max_docs,
        n_docs_indexed=n_indexed,
        hnsw_space=space,
        hnsw_M=args.M,
        hnsw_ef_construction=args.ef_construction,
        hnsw_ef_search=args.ef_search,
        batch_size=args.batch_size,
    )
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[saved] {index_path}")
    print(f"[saved] {map_path}")
    print(f"[saved] {meta_path}")
    print(f"[json] parser={'orjson' if _FAST_JSON else 'json'}")

if __name__ == "__main__":
    main()
