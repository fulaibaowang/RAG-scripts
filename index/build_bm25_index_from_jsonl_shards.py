#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable

import pyterrier as pt

import re

# letters + optional separators + digits (capture long digit runs)
CODE_RE = re.compile(r"\b([A-Za-z]{2,12})\s*[-–-]?\s*(\d{2,})\b")

def chunk_digits(d: str, k: int = 4) -> list[str]:
    # chunk into <=4 digits; keeps leading zeros
    return [d[i:i+k] for i in range(0, len(d), k)]

def augment_text_for_codes(text: str) -> str:
    extras = []
    for pfx, digits in CODE_RE.findall(text):
        p = pfx.lower()
        # always keep prefix as a token (often survives even when digits are dropped)
        extras.append(p)

        if len(digits) >= 5:
            # critical: create <=4-digit chunks so Terrier won't discard them
            chunks = chunk_digits(digits, 4)
            extras.extend(chunks)                 # e.g. 0066101 -> 0066 101
            extras.append(p + " " + " ".join(chunks))
        else:
            # for short digits (2-4), variants usually survive
            extras.append(digits)
            extras.append(f"{p}{digits}")
            extras.append(f"{p}-{digits}")
            extras.append(f"{p} {digits}")

    if extras:
        # append extras as additional terms (index-only hints)
        return text + "\n\n" + " ".join(sorted(set(extras)))
    return text

def iter_docs(jsonl_glob: str) -> Iterable[Dict]:
    """
    Stream documents from many JSONL shards.

    Filters:
      - Only valid PMID required

    Yields:
      {"docno": pmid, "text": title + "\\n\\n" + abstract}
    """
    for fp in sorted(glob.glob(jsonl_glob)):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                pmid = (d.get("pmid") or d.get("docno") or "").strip()
                if not pmid:
                    continue

                title = (d.get("title") or "").strip()
                abstract = (d.get("abstract") or "").strip()
                text = (title + "\n\n" + abstract).strip()
                
                # Skip if both title and abstract are empty
                if not text:
                    continue
                text = augment_text_for_codes(text)
    
                yield {"docno": pmid, "text": text}


def build_index(index_path: str, jsonl_glob: str, overwrite: bool, threads: int):
    os.makedirs(index_path, exist_ok=True)

    # Meta sizes are fixed-width in Terrier. Keep them small.
    # We only need docno; text is stored in the direct index, not meta.
    indexer = pt.IterDictIndexer(
        index_path,
        text_attrs=["text"],
        meta={"docno": 32},  # PMID fits easily
        overwrite=overwrite,
        threads=threads,
    )

    indexref = indexer.index(iter_docs(jsonl_glob))
    return indexref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_glob", required=True, help='e.g. "/data/pubmed_jsonl/baseline/*.jsonl"')
    ap.add_argument("--index_path", required=True, help='e.g. "/data/terrier_indexes/pubmed_baseline_bm25"')
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not pt.started():
        pt.init()

    indexref = build_index(
        index_path=args.index_path,
        jsonl_glob=args.jsonl_glob,
        overwrite=args.overwrite,
        threads=args.threads,
    )

    print("DONE")
    print("IndexRef:", indexref)


if __name__ == "__main__":
    main()
