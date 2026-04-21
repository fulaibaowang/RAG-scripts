#!/usr/bin/env python3
"""
Build contexts (title + abstract) from post-rerank JSONL and literature corpus.

Reads post-rerank JSONL (``post_rerank_jsonl.py`` output: ``doc_ids`` per question), looks up
title and abstract for each doc id in a PubMed JSONL corpus (``pmid`` field must match doc_id
for PubMed), appends a ``contexts`` list with ``doc_id`` per row (no PubMed URLs). Only the
first ``--evidence-top-k`` ranked ``doc_ids`` are used (post-rerank may list a larger pool).
Legacy post-rerank
with URL ``documents`` or ``docnos`` is still accepted. Each output question includes ``context_mode``: ``document``.
"""

import argparse
import glob as glob_mod
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SHARED_SCRIPTS = Path(__file__).resolve().parents[1]
_EVIDENCE_DIR = Path(__file__).resolve().parent
for _p in (_SHARED_SCRIPTS, _EVIDENCE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retrieval_eval.common import iter_questions_jsonl, question_qid, write_questions_jsonl
from snippet_window_ce import CONTEXT_MODE_DOCUMENT
from retrieval_eval.doc_id_util import ranked_doc_ids_for_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contexts from post-rerank JSONL and literature corpus."
    )
    parser.add_argument(
        "--post-rerank-jsonl",
        "--post-rerank-json",
        type=Path,
        required=True,
        dest="post_rerank_jsonl",
        help="Path to post-rerank .jsonl (output of post_rerank_jsonl.py).",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=10,
        help="Max doc_ids per question used for contexts and corpus indexing (default: 10).",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path or glob pattern to corpus JSONL (e.g. /pubmed/*.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output .jsonl (e.g. output/<workflow>/evidence/..._contexts.jsonl).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def load_post_rerank_questions(
    post_rerank_path: Path,
    evidence_top_k: Optional[int],
) -> Tuple[List[dict], Set[str]]:
    """Load post-rerank JSONL; ``needed`` is PMIDs for capped doc lists."""
    questions = list(iter_questions_jsonl(post_rerank_path))
    needed: Set[str] = set()
    for q in questions:
        for doc_id in ranked_doc_ids_for_evidence(q, evidence_top_k):
            needed.add(doc_id)
    return questions, needed


def _resolve_corpus_paths(path_or_glob: str) -> List[Path]:
    """Resolve a single file path or a glob pattern to a sorted list of JSONL files."""
    if "*" in path_or_glob or "?" in path_or_glob:
        paths = sorted(Path(p) for p in glob_mod.glob(path_or_glob) if Path(p).is_file())
        if not paths:
            raise FileNotFoundError(f"No files matched corpus glob: {path_or_glob}")
        return paths
    p = Path(path_or_glob)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    return [p]


def build_pmid_to_text(corpus_path: str, needed_pmids: Set[str]) -> Dict[str, Tuple[str, str]]:
    """
    Stream JSONL file(s) and build pmid -> (title, abstract) only for needed PMIDs.
    Accepts a single path or a glob pattern.
    """
    paths = _resolve_corpus_paths(corpus_path)
    n_files = len(paths)
    if n_files > 1:
        logger.info("Scanning %d JSONL files from glob: %s", n_files, corpus_path)

    pmid_to_text: Dict[str, Tuple[str, str]] = {}
    found = 0

    for fi, fp in enumerate(paths):
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pmid_raw = obj.get("pmid")
                if pmid_raw is None:
                    continue
                pmid = str(pmid_raw).strip()
                if pmid not in needed_pmids:
                    continue
                title = obj.get("title") or ""
                abstract = obj.get("abstract") or ""
                if isinstance(title, list):
                    title = " ".join(str(t) for t in title)
                if isinstance(abstract, list):
                    abstract = " ".join(str(a) for a in abstract)
                pmid_to_text[pmid] = (str(title), str(abstract))
                found += 1
                if found == len(needed_pmids):
                    break
        if found == len(needed_pmids):
            break
        if n_files > 1:
            step = 50
            if (fi + 1) % step == 0 or (fi + 1) == n_files:
                logger.info("Scanned %d/%d files, found %d/%d PMIDs so far", fi + 1, n_files, found, len(needed_pmids))

    missing = needed_pmids - set(pmid_to_text.keys())
    if missing:
        logger.warning(
            "%d/%d PMIDs not found in corpus (%s). These documents will have no context.",
            len(missing), len(needed_pmids), corpus_path,
        )
        if len(missing) <= 20:
            logger.warning("  missing PMIDs: %s", sorted(missing))
        else:
            logger.warning("  missing PMIDs (first 20): %s", sorted(missing)[:20])

    return pmid_to_text


def _normalize_unicode_whitespace(text: str) -> str:
    """Replace exotic Unicode whitespace / separator codepoints with ASCII space.

    PubMed abstracts sometimes contain characters like U+2029 (Paragraph
    Separator), U+2009 (Thin Space), U+00A0 (No-Break Space), etc.  These
    cause editor warnings (unusual line terminators) and may confuse
    downstream LLM tokenizers.  We collapse them all to plain ASCII space and
    then clean up any resulting multi-space runs.
    """
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Zs = space separator, Zl = line separator, Zp = paragraph separator
        if cat in ("Zl", "Zp") or (cat == "Zs" and ch != " "):
            out.append(" ")
        else:
            out.append(ch)
    # collapse multi-space runs introduced by replacements
    return re.sub(r"  +", " ", "".join(out))


def build_context_text(title: str, abstract: str) -> str:
    """Combine title and abstract for context text."""
    parts = [s.strip() for s in (title, abstract) if s and s.strip()]
    text = ". ".join(parts) if parts else ""
    return _normalize_unicode_whitespace(text)


def main() -> int:
    import sys
    _shared = Path(__file__).resolve().parents[1]
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    try:
        from logging_config import configure_logging_from_env
        configure_logging_from_env()
    except ImportError:
        pass
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.output_path.suffix.lower() != ".jsonl":
        logger.error("--output-path must end with .jsonl, got %s", args.output_path)
        return 1

    if not args.post_rerank_jsonl.exists():
        logger.error("Post-rerank JSONL not found: %s", args.post_rerank_jsonl)
        return 1

    etk = args.evidence_top_k if args.evidence_top_k > 0 else None
    logger.info("Loading post-rerank JSONL: %s", args.post_rerank_jsonl)
    questions, needed_pmids = load_post_rerank_questions(args.post_rerank_jsonl, etk)
    logger.info("Questions: %d, unique PMIDs (evidence cap): %d", len(questions), len(needed_pmids))

    logger.info("Indexing corpus: %s", args.corpus_path)
    pmid_to_text = build_pmid_to_text(args.corpus_path, needed_pmids)
    logger.info("Found %d / %d PMIDs in corpus", len(pmid_to_text), len(needed_pmids))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_total = 0
    out_questions: List[dict] = []
    for q in questions:
        qid = question_qid(q)
        if qid is None:
            continue
        contexts = []
        for doc_id in ranked_doc_ids_for_evidence(q, etk):
            pair = pmid_to_text.get(doc_id)
            if pair is None:
                missing_total += 1
                continue
            title, abstract = pair
            text = build_context_text(title, abstract)
            contexts.append(
                {
                    "id": f"{doc_id}-1",
                    "doc_id": doc_id,
                    "text": text,
                }
            )
        # Full question object (body, type, id, documents, etc.) with contexts appended
        out_q = dict(q)
        out_q["context_mode"] = CONTEXT_MODE_DOCUMENT
        out_q["contexts"] = contexts
        out_questions.append(out_q)

    write_questions_jsonl(args.output_path, out_questions)

    if missing_total:
        logger.warning("PMIDs missing from corpus: %d", missing_total)
    logger.info("Wrote %d query records to %s", len(out_questions), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
