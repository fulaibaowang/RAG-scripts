#!/usr/bin/env python3
"""
Post-hybrid-reranking: attach top-k documents from rerank run to query JSONL.

Reads a hybrid rerank TSV (qid, docno, rank), picks at most top-k docno (PMID) per query,
and writes JSONL (one question object per line) with a "documents" field (list of
PubMed URLs). Oracle fields (snippets, documents, ideal_answer, exact_answer)
are removed from each source question so the output is suitable for
post-reranking use without gold labels.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

ORACLE_KEYS = frozenset({"snippets", "documents", "ideal_answer", "exact_answer"})
PUBMED_URL_PREFIX = "http://www.ncbi.nlm.nih.gov/pubmed/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-rerank JSONL: top-k documents from rerank TSV attached to query JSONL."
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Path to rerank-hybrid TSV (columns: qid, docno, rank).",
    )
    parser.add_argument(
        "--query-jsonl",
        "--query-json",
        type=Path,
        required=True,
        dest="query_jsonl",
        help="Path to query .jsonl (one question record per line). --query-json is deprecated.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output .jsonl (e.g. post_rerank_13B1_golden.jsonl).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Max docs per query from the run TSV (rank <= top-k). Fewer if the run has fewer rows; no error.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def load_rerank_topk_urls(run_path: Path, top_k: int) -> Dict[str, List[str]]:
    """
    Read rerank TSV and return qid -> list of top-k PubMed URLs (ordered by rank).
    """
    qid_to_docs: Dict[str, List[Tuple[int, str]]] = {}

    with open(run_path, "r") as f:
        lines = iter(f)
        header = next(lines, None)
        if not header or "qid" not in header:
            raise ValueError(f"Expected TSV header with qid, docno, rank; got: {header!r}")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, docno, rank_str = parts[0], parts[1], parts[2]
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            if rank > top_k:
                continue
            pmid = str(docno).strip()
            if qid not in qid_to_docs:
                qid_to_docs[qid] = []
            if len(qid_to_docs[qid]) < top_k:
                qid_to_docs[qid].append((rank, pmid))

    # Sort by rank and convert to URLs
    result: Dict[str, List[str]] = {}
    for qid, pairs in qid_to_docs.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])[:top_k]
        result[qid] = [f"{PUBMED_URL_PREFIX}{pmid}" for _, pmid in pairs_sorted]
    return result


def main() -> int:
    _shared = Path(__file__).resolve().parents[1]
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    try:
        from logging_config import configure_logging_from_env
        configure_logging_from_env()
    except ImportError:
        pass
    from retrieval_eval.common import iter_questions_jsonl, write_questions_jsonl

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.run_path.exists():
        logger.error("Rerank run file not found: %s", args.run_path)
        return 1
    if not args.query_jsonl.exists():
        logger.error("Query JSONL not found: %s", args.query_jsonl)
        return 1
    if args.output_path.suffix.lower() != ".jsonl":
        logger.error("--output-path must end with .jsonl, got %s", args.output_path)
        return 1

    logger.info("Reading rerank run: %s (top-k=%d)", args.run_path, args.top_k)
    qid_to_urls = load_rerank_topk_urls(args.run_path, args.top_k)
    logger.info("Queries in rerank: %d", len(qid_to_urls))

    out_questions = []
    for q in iter_questions_jsonl(args.query_jsonl):
        qid = q.get("id")
        if qid is None:
            continue
        new_q = {k: v for k, v in q.items() if k not in ORACLE_KEYS}
        new_q["documents"] = qid_to_urls.get(str(qid), [])
        out_questions.append(new_q)

    if not out_questions:
        logger.error("No questions in query JSONL")
        return 1

    write_questions_jsonl(args.output_path, out_questions)
    logger.info("Wrote %d questions to %s", len(out_questions), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
