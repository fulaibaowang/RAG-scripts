#!/usr/bin/env python3
"""
Post-hybrid-reranking: attach top-k document ids from rerank run to query JSONL.

Reads a hybrid rerank TSV (qid, docno, rank), picks at most top-k docno strings per query,
and writes JSONL (one question object per line) with ``doc_ids`` (ordered list, same
values as TSV docno — for PubMed today, numeric PMIDs). Optional ``--windows-jsonl``
merges max-pooled CE snippet windows into ``doc_snippet_windows`` per question in compact form (``pmid`` → ``{"selected_windows": [...]}``;
snippet route standalone artifact). Use ``--window-size`` / ``--top-windows`` with ``--windows-jsonl``.

Oracle fields (snippets, documents, ideal_answer, exact_answer) are removed from each
source question so the output is suitable for post-reranking use without gold labels.
``doc_ids`` / ``docnos`` / embedded window fields from the query source are stripped
before attaching fresh ``doc_ids`` (and optional ``doc_snippet_windows``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

ORACLE_KEYS = frozenset(
    {
        "snippets",
        "documents",
        "ideal_answer",
        "exact_answer",
        "doc_ids",
        "docnos",
        "doc_snippet_windows",
        "contexts",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-rerank JSONL: top-k documents from rerank TSV attached to query JSONL; "
        "optional merge of snippet CE windows into compact doc_snippet_windows.",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help="Path to rerank-hybrid TSV (columns: qid, docno, rank).",
    )
    parser.add_argument(
        "--query-jsonl",
        type=Path,
        required=True,
        dest="query_jsonl",
        help="Path to query .jsonl (one question record per line).",
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
        default=30,
        help="Max docs per query from the run TSV (rank <= top-k). Evidence cap is applied later in "
        "build_contexts_from_*.py (--evidence-top-k). Fewer if the run has fewer rows; no error.",
    )
    parser.add_argument(
        "--windows-jsonl",
        type=Path,
        default=None,
        help="Optional snippet_rerank windows JSONL for this split; merges pre-selected windows into "
        "doc_snippet_windows (compact shape; only for doc_ids in the run).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sentence span length per CE window when merging --windows-jsonl (default: 3).",
    )
    parser.add_argument(
        "--top-windows",
        type=int,
        choices=(1, 2),
        default=2,
        help="Max disjoint windows per doc after merge (default: 2).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def load_rerank_topk_doc_ids(run_path: Path, top_k: int) -> Dict[str, List[str]]:
    """Read rerank TSV and return qid -> list of top-k docno strings (ordered by rank)."""
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
            doc_id = str(docno).strip()
            if not doc_id:
                continue
            if qid not in qid_to_docs:
                qid_to_docs[qid] = []
            if len(qid_to_docs[qid]) < top_k:
                qid_to_docs[qid].append((rank, doc_id))

    result: Dict[str, List[str]] = {}
    for qid, pairs in qid_to_docs.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])[:top_k]
        result[qid] = [doc_id for _, doc_id in pairs_sorted]
    return result


def main() -> int:
    _shared = Path(__file__).resolve().parents[1]
    _evidence = Path(__file__).resolve().parent
    for p in (_shared, _evidence):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from utils.logging_config import configure_logging_from_env
        configure_logging_from_env()
    except ImportError:
        pass
    from retrieval_eval.common import iter_questions_jsonl, question_qid, write_questions_jsonl

    from score_snippet_windows import (
        embed_compact_doc_snippet_windows_for_question,
        load_by_pair_from_windows_jsonl,
    )

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

    by_pair_global = None
    if args.windows_jsonl is not None:
        if not args.windows_jsonl.exists():
            logger.error("--windows-jsonl not found: %s", args.windows_jsonl)
            return 1
        logger.info("Loading windows for merge: %s", args.windows_jsonl)
        by_pair_global = load_by_pair_from_windows_jsonl(args.windows_jsonl)
        logger.info("(qid, docno) pairs with windows after max-pool: %d", len(by_pair_global))

    logger.info("Reading rerank run: %s (top-k=%d)", args.run_path, args.top_k)
    qid_to_doc_ids = load_rerank_topk_doc_ids(args.run_path, args.top_k)
    logger.info("Queries in rerank: %d", len(qid_to_doc_ids))

    out_questions = []
    n_with_windows = 0
    for q in iter_questions_jsonl(args.query_jsonl):
        qid = question_qid(q)
        if qid is None:
            continue
        new_q = {k: v for k, v in q.items() if k not in ORACLE_KEYS}
        doc_ids = qid_to_doc_ids.get(str(qid), [])
        new_q["doc_ids"] = doc_ids
        if by_pair_global is not None and doc_ids:
            emb = embed_compact_doc_snippet_windows_for_question(
                str(qid),
                doc_ids,
                by_pair_global,
                args.window_size,
                args.top_windows,
            )
            if emb:
                new_q["doc_snippet_windows"] = emb
                n_with_windows += 1
        out_questions.append(new_q)

    if not out_questions:
        logger.error("No questions in query JSONL")
        return 1

    if by_pair_global is not None:
        logger.info("Questions with non-empty doc_snippet_windows: %d / %d", n_with_windows, len(out_questions))

    write_questions_jsonl(args.output_path, out_questions)
    logger.info("Wrote %d questions to %s", len(out_questions), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
