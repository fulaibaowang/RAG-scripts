#!/usr/bin/env python3
"""
Build contexts (title + abstract) from post-rerank JSON and literature corpus.

Reads the JSON produced by post_rerank_json.py (questions with body, type, id,
documents), looks up title and abstract for each document PMID from a PubMed
JSONL corpus, appends a "contexts" field to each question, and writes JSONL
(one line per question) with the full question object (body, type, id,
documents, contexts) preserved.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

PUBMED_URL_PATTERN = re.compile(r"pubmed/(\d+)/?$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contexts from post-rerank JSON and literature corpus."
    )
    parser.add_argument(
        "--post-rerank-json",
        type=Path,
        required=True,
        help="Path to post-rerank JSON (output of post_rerank_json.py).",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("output/subset_pubmed.jsonl"),
        help="Path to subset_pubmed.jsonl (default: output/subset_pubmed.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output JSONL (e.g. output/<workflow>/rerank_hybrid/evidence/..._contexts.jsonl).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def pmid_from_url(url: str):
    """Extract PMID from a PubMed URL, or return None."""
    if not url:
        return None
    m = PUBMED_URL_PATTERN.search(url.strip())
    return m.group(1) if m else None


def load_post_rerank_questions(post_rerank_path: Path) -> Tuple[List[dict], Set[str]]:
    """
    Load post-rerank JSON and return (questions list, set of all PMIDs in documents).
    """
    with open(post_rerank_path, "r") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    needed_pmids: Set[str] = set()
    for q in questions:
        for url in q.get("documents") or []:
            pmid = pmid_from_url(url)
            if pmid:
                needed_pmids.add(pmid)
    return questions, needed_pmids


def build_pmid_to_text(corpus_path: Path, needed_pmids: Set[str]) -> Dict[str, Tuple[str, str]]:
    """
    Stream subset_pubmed.jsonl and build pmid -> (title, abstract) only for needed PMIDs.
    """
    pmid_to_text: Dict[str, Tuple[str, str]] = {}
    found = 0

    with open(corpus_path, "r") as f:
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

    return pmid_to_text


def build_context_text(title: str, abstract: str) -> str:
    """Combine title and abstract for context text."""
    parts = [s.strip() for s in (title, abstract) if s and s.strip()]
    return ". ".join(parts) if parts else ""


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.post_rerank_json.exists():
        logger.error("Post-rerank JSON not found: %s", args.post_rerank_json)
        return 1
    if not args.corpus_path.exists():
        logger.error("Corpus file not found: %s", args.corpus_path)
        return 1

    logger.info("Loading post-rerank JSON: %s", args.post_rerank_json)
    questions, needed_pmids = load_post_rerank_questions(args.post_rerank_json)
    logger.info("Questions: %d, unique PMIDs: %d", len(questions), len(needed_pmids))

    logger.info("Indexing corpus: %s", args.corpus_path)
    pmid_to_text = build_pmid_to_text(args.corpus_path, needed_pmids)
    logger.info("Found %d / %d PMIDs in corpus", len(pmid_to_text), len(needed_pmids))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_total = 0
    num_written = 0
    with open(args.output_path, "w") as f:
        for q in questions:
            qid = q.get("id")
            if qid is None:
                continue
            contexts = []
            for rank, url in enumerate(q.get("documents") or [], start=1):
                pmid = pmid_from_url(url)
                if not pmid:
                    continue
                pair = pmid_to_text.get(pmid)
                if pair is None:
                    missing_total += 1
                    continue
                title, abstract = pair
                text = build_context_text(title, abstract)
                # One context per document (one title+abstract per PMID); id is pmid-1
                contexts.append(
                    {
                        "id": f"{pmid}-1",
                        "doc": f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                        "text": text,
                    }
                )
            # Output full question object (body, type, id, documents, etc.) with contexts appended
            out_q = dict(q)
            out_q["contexts"] = contexts
            f.write(json.dumps(out_q, ensure_ascii=False) + "\n")
            num_written += 1

    if missing_total:
        logger.warning("PMIDs missing from corpus: %d", missing_total)
    logger.info("Wrote %d query records to %s", num_written, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
