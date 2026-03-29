#!/usr/bin/env python3
"""
N-way weighted RRF fusion of retrieval/rerank run TSVs across parallel directories.

Each directory should contain the same set of run filenames (e.g. dense_train.tsv in
every dir). For each filename, fuses ranked lists using:

    score(doc) += weight_i / (k_rrf + rank_i)

Weights default to 1/N when --weights is omitted. Tie-break: lexicographic docno.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def _load_run_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_c = cols.get("qid") or cols.get("query_id")
    doc_c = cols.get("docno") or cols.get("docid") or cols.get("doc")
    rank_c = cols.get("rank")
    if not qid_c or not doc_c:
        raise ValueError(f"{path}: need qid and docno columns, got {list(df.columns)}")
    out = pd.DataFrame(
        {
            "qid": df[qid_c].astype(str),
            "docno": df[doc_c].astype(str),
        }
    )
    if rank_c and rank_c in df.columns:
        out["rank"] = pd.to_numeric(df[rank_c], errors="coerce").fillna(0).astype(int)
    else:
        tmp = df.copy()
        tmp["_ord"] = range(len(tmp))
        out["rank"] = tmp.groupby(qid_c, sort=False).cumcount() + 1
    return out


def fuse_n_way_rrf(
    dfs: Sequence[pd.DataFrame],
    weights: Sequence[float],
    k_rrf: float,
    cap: int | None,
) -> pd.DataFrame:
    if len(dfs) != len(weights):
        raise ValueError("dfs and weights length mismatch")
    if not dfs:
        raise ValueError("no dataframes")

    qid_order: List[str] = []
    seen_q: set[str] = set()
    for df in dfs:
        for qid in df["qid"].unique():
            s = str(qid)
            if s not in seen_q:
                seen_q.add(s)
                qid_order.append(s)

    rows: List[Dict[str, object]] = []
    kf = float(k_rrf)

    for qid in qid_order:
        scores: Dict[str, float] = {}
        for df, w in zip(dfs, weights):
            sub = df[df["qid"].astype(str) == qid]
            wi = float(w)
            for _, r in sub.iterrows():
                doc = str(r["docno"])
                rk = int(r["rank"])
                if rk < 1:
                    rk = 1
                scores[doc] = scores.get(doc, 0.0) + wi / (kf + rk)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if cap is not None:
            ranked = ranked[: int(cap)]
        for i, (doc, sc) in enumerate(ranked, start=1):
            rows.append({"qid": qid, "docno": doc, "rank": i, "score": sc})

    return pd.DataFrame(rows)


def parse_weights(s: str | None, n: int) -> List[float]:
    if not s or not str(s).strip():
        return [1.0 / n] * n
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    w = [float(x) for x in parts]
    if len(w) != n:
        raise ValueError(f"Expected {n} weights (comma-separated), got {len(w)}: {s!r}")
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Directories containing run TSVs (each should have matching filenames).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write fused TSVs (created if missing).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.tsv",
        help="Glob pattern relative to each run-dir (default: *.tsv).",
    )
    ap.add_argument(
        "--k-rrf",
        type=float,
        default=60.0,
        help="RRF k in weight/(k+rank) (default: 60).",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated weights, same order as --run-dirs; default 1/N each.",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Max documents per query in fused output (default: no limit).",
    )
    args = ap.parse_args()

    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    for d in run_dirs:
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    first = run_dirs[0]
    matched = sorted(first.glob(args.pattern))
    if not matched:
        print(f"Error: no files matching {args.pattern!r} under {first}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(run_dirs)
    weights = parse_weights(args.weights or None, n)

    for path0 in matched:
        name = path0.name
        paths = [d / name for d in run_dirs]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            print(f"Warning: skip {name} (missing: {missing})", file=sys.stderr)
            continue
        dfs = [_load_run_tsv(p) for p in paths]
        fused = fuse_n_way_rrf(dfs, weights, args.k_rrf, args.cap)
        out_path = out_dir / name
        fused.to_csv(out_path, sep="\t", index=False)
        print(f"[multi_query_fuse] wrote {out_path} ({len(fused)} rows)")

    print("[multi_query_fuse] done")


if __name__ == "__main__":
    main()
