#!/usr/bin/env python3
"""
Build contexts from post-rerank JSON using top CE windows from snippet reranking.

Reads the JSON produced by post_rerank_json.py and the window JSONL from
snippet_rerank (qid, docno, window_idx, ce_score). For each (qid, doc) in the
post-rerank documents, selects the top-K windows by CE score, merges their
sentence indices (non-overlapping), and builds context as title + selected
sentences. Fallback: if no windows exist for a doc, uses full title + abstract.
Output format matches build_contexts_from_documents.py.
"""

import argparse
import glob as glob_mod
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

PUBMED_URL_PATTERN = re.compile(r"pubmed/(\d+)/?$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contexts from post-rerank JSON and snippet window JSONL (top CE windows, non-overlapping sentences).",
    )
    parser.add_argument(
        "--post-rerank-json",
        type=Path,
        required=True,
        help="Path to post-rerank JSON (output of post_rerank_json.py).",
    )
    parser.add_argument(
        "--snippet-windows-dir",
        type=Path,
        required=True,
        help="Path to snippet_rerank windows directory (contains per-split JSONL).",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        help="Split name matching the JSONL filename (e.g. best_rrf_13B1_golden_top5000_rrf_pool200_k60).",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path or glob pattern to corpus JSONL.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output JSON (e.g. output/<workflow>/evidence/..._contexts.json).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sentence span per window (used to compute indices from window_idx).",
    )
    parser.add_argument(
        "--top-windows",
        type=int,
        default=2,
        help="Number of top CE windows per document to use for context.",
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
    """Load post-rerank JSON and return (questions list, set of all PMIDs in documents)."""
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


def load_snippet_windows(
    windows_path: Path,
    window_size: int,
    top_windows: int,
) -> Dict[Tuple[str, str], List[int]]:
    """
    Load snippet JSONL and return (qid, docno) -> sorted list of sentence indices to include.

    Each line: qid, docno, window_idx, window_text, ce_score.
    For each (qid, docno) we take top_windows by ce_score; each window covers
    sentence indices [window_idx, window_idx+1, ..., window_idx+window_size-1].
    Union of those indices is returned, sorted ascending.
    """
    # (qid, docno) -> list of (ce_score, window_idx) for sorting
    by_qid_doc: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}
    with open(windows_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("qid")
            docno = obj.get("docno")
            window_idx = obj.get("window_idx", 0)
            ce_score = float(obj.get("ce_score", 0.0))
            if qid is None or docno is None:
                continue
            key = (str(qid), str(docno))
            if key not in by_qid_doc:
                by_qid_doc[key] = []
            by_qid_doc[key].append((ce_score, window_idx))

    out: Dict[Tuple[str, str], List[int]] = {}
    for (qid, docno), scored in by_qid_doc.items():
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:top_windows]
        indices: Set[int] = set()
        for _, wi in top:
            for j in range(window_size):
                indices.add(wi + j)
        out[(qid, docno)] = sorted(indices)
    return out


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


def build_pmid_to_title_sentences(
    corpus_path: str,
    needed_pmids: Set[str],
) -> Dict[str, Tuple[str, List[str]]]:
    """
    Stream JSONL and build pmid -> (title, list of sentences) for needed PMIDs.
    Uses NLTK sent_tokenize on abstract.
    """
    try:
        import nltk
        nltk.sent_tokenize("Hello.")
    except LookupError:
        for res in ("punkt_tab", "punkt"):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

    paths = _resolve_corpus_paths(corpus_path)
    pmid_to_data: Dict[str, Tuple[str, List[str]]] = {}
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
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
                if pmid not in needed_pmids or pmid in pmid_to_data:
                    continue
                title = obj.get("title") or ""
                abstract = obj.get("abstract") or obj.get("abstractText") or ""
                if isinstance(title, list):
                    title = " ".join(str(t) for t in title)
                if isinstance(abstract, list):
                    abstract = " ".join(str(a) for a in abstract)
                title = str(title).strip()
                abstract = str(abstract).strip()
                sentences = [s.strip() for s in nltk.sent_tokenize(abstract) if s.strip()] if abstract else []
                pmid_to_data[pmid] = (title, sentences)
                if len(pmid_to_data) == len(needed_pmids):
                    break
        if len(pmid_to_data) == len(needed_pmids):
            break
    return pmid_to_data


def _normalize_unicode_whitespace(text: str) -> str:
    """Collapse exotic Unicode whitespace to ASCII space and multi-space runs."""
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Zl", "Zp") or (cat == "Zs" and ch != " "):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"  +", " ", "".join(out))


def build_context_from_sentences(
    title: str,
    sentences: List[str],
    indices: List[int],
) -> str:
    """Build context as title + selected sentences (by index), normalized."""
    parts = [title.strip()] if title and title.strip() else []
    for i in indices:
        if 0 <= i < len(sentences):
            parts.append(sentences[i])
    text = ". ".join(parts) if parts else ""
    return _normalize_unicode_whitespace(text)


def build_context_title_abstract(title: str, abstract: str) -> str:
    """Fallback: full title + abstract, same as build_contexts_from_documents."""
    parts = [s.strip() for s in (title, abstract) if s and s.strip()]
    text = ". ".join(parts) if parts else ""
    return _normalize_unicode_whitespace(text)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.post_rerank_json.exists():
        logger.error("Post-rerank JSON not found: %s", args.post_rerank_json)
        return 1

    windows_path = args.snippet_windows_dir / f"{args.split_name}.jsonl"
    if not windows_path.exists():
        logger.error("Snippet windows JSONL not found: %s", windows_path)
        return 1

    logger.info("Loading post-rerank JSON: %s", args.post_rerank_json)
    questions, needed_pmids = load_post_rerank_questions(args.post_rerank_json)
    logger.info("Questions: %d, unique PMIDs: %d", len(questions), len(needed_pmids))

    logger.info("Loading snippet windows: %s", windows_path)
    snippet_indices = load_snippet_windows(windows_path, args.window_size, args.top_windows)
    logger.info("Snippet indices for %d (qid, docno) pairs", len(snippet_indices))

    logger.info("Indexing corpus: %s", args.corpus_path)
    pmid_to_title_sents = build_pmid_to_title_sentences(args.corpus_path, needed_pmids)
    logger.info("Found %d / %d PMIDs in corpus", len(pmid_to_title_sents), len(needed_pmids))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_total = 0
    out_questions: List[dict] = []
    for q in questions:
        qid = q.get("id")
        if qid is None:
            continue
        contexts = []
        for url in q.get("documents") or []:
            pmid = pmid_from_url(url)
            if not pmid:
                continue
            pair = pmid_to_title_sents.get(pmid)
            if pair is None:
                missing_total += 1
                continue
            title, sentences = pair
            key = (str(qid), str(pmid))
            indices = snippet_indices.get(key)
            if indices is not None and sentences:
                text = build_context_from_sentences(title, sentences, indices)
            else:
                abstract = " ".join(sentences) if sentences else ""
                text = build_context_title_abstract(title, abstract)
            contexts.append(
                {
                    "id": f"{pmid}-1",
                    "doc": f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                    "text": text,
                }
            )
        out_q = dict(q)
        out_q["contexts"] = contexts
        out_questions.append(out_q)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"questions": out_questions}, f, ensure_ascii=False, indent=2)

    if missing_total:
        logger.warning("PMIDs missing from corpus: %d", missing_total)
    logger.info("Wrote %d query records to %s", len(out_questions), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
