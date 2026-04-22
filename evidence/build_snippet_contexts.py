#!/usr/bin/env python3
"""
Build contexts from post-rerank JSONL using top CE windows from snippet reranking.

Reads post-rerank JSONL (from ``build_retrieval_jsonl.py``). Window CE may be embedded as
compact ``doc_snippet_windows`` (``pmid`` → ``{"selected_windows": [...]}`` from post_rerank),
or as the legacy flat list per PMID (unsupported for new post-rerank outputs; regenerate),
or supplied via ``--snippet-windows-dir`` / ``{split}.jsonl`` when post-rerank has no windows.
Each output question includes ``context_mode``: ``snippet``.

For each (qid, doc_id) in the first ``--evidence-top-k`` post-rerank ``doc_ids``, ranks
distinct windows, then:

- top-windows=1: keep the best window only.
- top-windows=2: keep the best window; add the second-ranked window only if its
  sentence span is disjoint from the first (overlapping second is dropped).

Merges kept windows' sentence indices for context text. Fallback: if no windows
exist for a doc, uses full title + abstract.

Emits per-context selected_windows, rejected_windows (rank-2 dropped for sentence
overlap with rank-1), and optional per-split stats JSON.
"""

from __future__ import annotations

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
from retrieval_eval.common import iter_jsonl_dicts, question_qid, write_questions_jsonl
from retrieval_eval.doc_id_util import ranked_doc_ids_for_evidence

from score_snippet_windows import (
    CONTEXT_MODE_SNIPPET,
    DOC_SNIPPET_WINDOWS_KEY,
    is_compact_doc_snippet_windows,
    merge_window_selection_from_embedded_questions,
    questions_use_embedded_windows,
    select_windows_max_pool_from_path,
)


def _embedded_window_source_label(questions: List[dict]) -> str:
    """How doc_snippet_windows is represented when embedded (stats only)."""
    seen_compact = False
    seen_legacy = False
    for q in questions:
        raw = q.get(DOC_SNIPPET_WINDOWS_KEY)
        if not isinstance(raw, dict) or not raw:
            continue
        if is_compact_doc_snippet_windows(raw):
            seen_compact = True
        else:
            seen_legacy = True
    if seen_compact and seen_legacy:
        return "embedded_mixed"
    if seen_compact:
        return "embedded_compact"
    if seen_legacy:
        return "embedded_legacy"
    return "embedded"


def _top_windows_int(value: str) -> int:
    v = int(value)
    if v not in (1, 2):
        raise argparse.ArgumentTypeError("--top-windows must be 1 or 2")
    return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contexts from post-rerank JSONL and snippet CE windows "
        "(embedded doc_snippet_windows or legacy windows JSONL; max-pool, top 1–2 distinct windows).",
    )
    parser.add_argument(
        "--post-rerank-jsonl",
        "--post-rerank-json",
        type=Path,
        required=True,
        dest="post_rerank_jsonl",
        help="Path to post-rerank .jsonl (output of build_retrieval_jsonl.py). --post-rerank-json is deprecated.",
    )
    parser.add_argument(
        "--snippet-windows-dir",
        type=Path,
        default=None,
        help="Path to snippet/snippet_rerank/windows directory (legacy). Not required when "
        "post-rerank JSONL contains doc_snippet_windows.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        help="Split name matching the JSONL filename (e.g. 13B1_golden).",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Path or glob pattern to corpus JSONL (not used with --stats-only).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to output contexts .jsonl (not used with --stats-only).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sentence span per window (used to compute indices from window_idx).",
    )
    parser.add_argument(
        "--top-windows",
        type=_top_windows_int,
        default=2,
        help="1 or 2: top CE windows per doc; with 2, second window is kept only if disjoint from the first.",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=10,
        help="Max doc_ids per question used for contexts and corpus indexing (default: 10). "
        "Post-rerank JSONL may list a larger pool.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only read post-rerank + windows and write stats JSON (no corpus, no contexts).",
    )
    parser.add_argument(
        "--stats-output-path",
        type=Path,
        default=None,
        help="Path for per-split stats JSON. Required with --stats-only; optional otherwise (derived from --output-path).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def default_stats_output_path(output_path: Path) -> Path:
    stem = output_path.stem
    if stem.endswith("_contexts"):
        base = stem[: -len("_contexts")]
    else:
        base = stem
    return output_path.parent / f"{base}_snippet_window_stats.json"


def load_post_rerank_questions(
    post_rerank_path: Path,
    evidence_top_k: Optional[int],
) -> Tuple[List[dict], Set[str]]:
    """Load post-rerank JSONL; ``needed`` is PMIDs referenced by capped doc lists."""
    questions = list(iter_jsonl_dicts(post_rerank_path, label="post-rerank JSONL"))
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
        raise FileNotFoundError(f"Corpus file not found: {path_or_glob}")
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
    """Fallback: full title + abstract, same as build_doc_contexts."""
    parts = [s.strip() for s in (title, abstract) if s and s.strip()]
    text = ". ".join(parts) if parts else ""
    return _normalize_unicode_whitespace(text)


def compute_snippet_window_stats(
    questions: List[dict],
    selected_by_pair: Dict[Tuple[str, str], List[dict]],
    pair_aux: Dict[Tuple[str, str], dict],
    split_name: str,
    top_windows: int,
    window_size: int,
    track_corpus_fallback: bool,
    pmid_to_title_sents: Optional[Dict[str, Tuple[str, List[str]]]],
    evidence_top_k: Optional[int],
) -> dict:
    """
    Aggregate stats over (qid, pmid) doc slots in post-rerank JSON (capped by evidence_top_k).

    With top-windows=2, final kept windows are never sentence-overlapping; overlap is
    reported on the *candidate* rank-1 vs rank-2 pair before the drop rule.

    If track_corpus_fallback and pmid_to_title_sents is set, counts snippet vs fallback
    using the same rules as context building (empty sentences -> fallback).
    """
    doc_pairs_considered = 0
    pairs_final_two_windows = 0
    pairs_final_one_window = 0
    pairs_fallback_no_windows = 0
    pairs_ranked_two_candidates = 0
    pairs_second_dropped_overlap = 0
    candidate_overlap_sent_count_sum = 0
    pairs_candidate_top2_sentence_overlap = 0

    for q in questions:
        qid = question_qid(q)
        if qid is None:
            continue
        qid_s = str(qid)
        for doc_id in ranked_doc_ids_for_evidence(q, evidence_top_k):
            doc_pairs_considered += 1
            key = (qid_s, doc_id)
            sw = selected_by_pair.get(key, [])
            aux = pair_aux.get(
                key,
                {
                    "had_two_ranked": False,
                    "dropped_second_overlap": False,
                    "candidate_overlap_sent_count": 0,
                },
            )

            use_fallback = len(sw) == 0
            if not use_fallback and track_corpus_fallback and pmid_to_title_sents is not None:
                pair = pmid_to_title_sents.get(doc_id)
                if pair is None:
                    use_fallback = True
                else:
                    _, sentences = pair
                    if not sentences:
                        use_fallback = True

            if use_fallback:
                pairs_fallback_no_windows += 1
                continue

            if aux.get("had_two_ranked"):
                pairs_ranked_two_candidates += 1
            if aux.get("dropped_second_overlap"):
                pairs_second_dropped_overlap += 1
            n_ov = int(aux.get("candidate_overlap_sent_count") or 0)
            if n_ov > 0:
                pairs_candidate_top2_sentence_overlap += 1
                candidate_overlap_sent_count_sum += n_ov

            if len(sw) >= 2:
                pairs_final_two_windows += 1
            elif len(sw) == 1:
                pairs_final_one_window += 1

    mean_among_overlapping_candidates: Optional[float]
    if pairs_candidate_top2_sentence_overlap > 0:
        mean_among_overlapping_candidates = (
            candidate_overlap_sent_count_sum / pairs_candidate_top2_sentence_overlap
        )
    else:
        mean_among_overlapping_candidates = None

    dpc = doc_pairs_considered
    pr2 = pairs_ranked_two_candidates

    return {
        "split_name": split_name,
        "top_windows": top_windows,
        "window_size": window_size,
        "evidence_top_k": evidence_top_k,
        "doc_pairs_considered": doc_pairs_considered,
        "pairs_final_two_windows": pairs_final_two_windows,
        "pairs_final_one_window": pairs_final_one_window,
        "pairs_lt2_windows": pairs_final_one_window,
        "pairs_fallback_no_windows": pairs_fallback_no_windows,
        "pairs_ranked_two_candidates": pairs_ranked_two_candidates,
        "pairs_second_dropped_overlap": pairs_second_dropped_overlap,
        "pairs_candidate_top2_sentence_overlap": pairs_candidate_top2_sentence_overlap,
        "candidate_top2_overlap_sent_count_sum": candidate_overlap_sent_count_sum,
        "candidate_top2_overlap_sent_count_mean_among_overlapping": mean_among_overlapping_candidates,
        "rate_pairs_final_two_windows": (pairs_final_two_windows / dpc) if dpc else 0.0,
        "rate_pairs_final_one_window": (pairs_final_one_window / dpc) if dpc else 0.0,
        "rate_pairs_fallback_no_windows": (pairs_fallback_no_windows / dpc) if dpc else 0.0,
        "rate_second_dropped_of_ranked_two": (pairs_second_dropped_overlap / pr2) if pr2 else 0.0,
        "rate_candidate_overlap_of_ranked_two": (pairs_candidate_top2_sentence_overlap / pr2) if pr2 else 0.0,
    }


def write_stats_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
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

    if args.stats_only:
        if not args.stats_output_path:
            logger.error("--stats-only requires --stats-output-path")
            return 1
        if args.corpus_path or args.output_path:
            logger.warning("--stats-only: ignoring corpus-path / output-path if set")
    else:
        if not args.corpus_path or not args.output_path:
            logger.error("Without --stats-only, --corpus-path and --output-path are required")
            return 1

    if not args.post_rerank_jsonl.exists():
        logger.error("Post-rerank JSONL not found: %s", args.post_rerank_jsonl)
        return 1

    if not args.stats_only and args.output_path is not None:
        if args.output_path.suffix.lower() != ".jsonl":
            logger.error("--output-path must end with .jsonl, got %s", args.output_path)
            return 1

    etk = args.evidence_top_k if args.evidence_top_k > 0 else None

    logger.info("Loading post-rerank JSONL: %s", args.post_rerank_jsonl)
    questions, needed_pmids = load_post_rerank_questions(args.post_rerank_jsonl, etk)
    logger.info("Questions: %d, unique PMIDs (evidence cap): %d", len(questions), len(needed_pmids))

    use_embedded = questions_use_embedded_windows(questions)
    if use_embedded:
        if args.snippet_windows_dir is not None:
            logger.info("Using embedded %s; ignoring --snippet-windows-dir", DOC_SNIPPET_WINDOWS_KEY)
        logger.info("Window source: embedded post-rerank (%s)", DOC_SNIPPET_WINDOWS_KEY)
        merged_by_pair, selected_by_pair, rejected_by_pair, pair_aux = merge_window_selection_from_embedded_questions(
            questions, args.window_size, args.top_windows,
        )
    else:
        if args.snippet_windows_dir is None:
            logger.error(
                "No %s in post-rerank JSONL; provide --snippet-windows-dir (legacy) or regenerate post-rerank "
                "with --windows-jsonl.",
                DOC_SNIPPET_WINDOWS_KEY,
            )
            return 1
        windows_path = args.snippet_windows_dir / f"{args.split_name}.jsonl"
        if not windows_path.exists():
            logger.error("Snippet windows JSONL not found: %s", windows_path)
            return 1
        logger.info("Window source: file %s", windows_path)
        merged_by_pair, selected_by_pair, rejected_by_pair, pair_aux = select_windows_max_pool_from_path(
            windows_path, args.window_size, args.top_windows,
        )
    logger.info("Snippet selection for %d (qid, docno) pairs", len(merged_by_pair))

    stats_path = args.stats_output_path
    if stats_path is None and args.output_path is not None:
        stats_path = default_stats_output_path(args.output_path)

    pmid_to_title_sents: Dict[str, Tuple[str, List[str]]] = {}
    if not args.stats_only:
        logger.info("Indexing corpus: %s", args.corpus_path)
        pmid_to_title_sents = build_pmid_to_title_sentences(args.corpus_path, needed_pmids)
        logger.info("Found %d / %d PMIDs in corpus", len(pmid_to_title_sents), len(needed_pmids))

    stats_payload = compute_snippet_window_stats(
        questions,
        selected_by_pair,
        pair_aux,
        args.split_name,
        args.top_windows,
        args.window_size,
        track_corpus_fallback=not args.stats_only,
        pmid_to_title_sents=pmid_to_title_sents if not args.stats_only else None,
        evidence_top_k=etk,
    )
    stats_payload["window_source"] = (
        _embedded_window_source_label(questions) if use_embedded else "file"
    )
    if stats_path is not None:
        write_stats_json(stats_path, stats_payload)
        logger.info("Wrote stats: %s", stats_path)

    logger.info(
        "Stats: doc_pairs_considered=%d final_two=%d final_one=%d fallback=%d "
        "ranked_two=%d dropped_second=%d candidate_overlap_pairs=%d overlap_sent_sum=%d",
        stats_payload["doc_pairs_considered"],
        stats_payload["pairs_final_two_windows"],
        stats_payload["pairs_final_one_window"],
        stats_payload["pairs_fallback_no_windows"],
        stats_payload["pairs_ranked_two_candidates"],
        stats_payload["pairs_second_dropped_overlap"],
        stats_payload["pairs_candidate_top2_sentence_overlap"],
        stats_payload["candidate_top2_overlap_sent_count_sum"],
    )
    if stats_payload["candidate_top2_overlap_sent_count_mean_among_overlapping"] is not None:
        logger.info(
            "Stats: mean candidate overlap sent count (among overlapping rank-1 vs rank-2)=%.4f",
            stats_payload["candidate_top2_overlap_sent_count_mean_among_overlapping"],
        )

    if args.stats_only:
        return 0

    assert args.output_path is not None
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_total = 0
    out_questions: List[dict] = []
    for q in questions:
        qid = question_qid(q)
        if qid is None:
            continue
        contexts: List[dict] = []
        for doc_id in ranked_doc_ids_for_evidence(q, etk):
            pair = pmid_to_title_sents.get(doc_id)
            if pair is None:
                missing_total += 1
                continue
            title, sentences = pair
            key = (str(qid), str(doc_id))
            indices = merged_by_pair.get(key)
            sw = selected_by_pair.get(key, [])
            rw = rejected_by_pair.get(key, [])
            if indices is not None and sentences and len(sw) > 0:
                text = build_context_from_sentences(title, sentences, indices)
                ctx: dict = {
                    "id": f"{doc_id}-1",
                    "doc_id": doc_id,
                    "text": text,
                    "selected_windows": sw,
                    "rejected_windows": rw,
                }
            else:
                abstract = " ".join(sentences) if sentences else ""
                text = build_context_title_abstract(title, abstract)
                ctx = {
                    "id": f"{doc_id}-1",
                    "doc_id": doc_id,
                    "text": text,
                    "selected_windows": [],
                    "rejected_windows": [],
                }
            contexts.append(ctx)
        out_q = dict(q)
        out_q["context_mode"] = CONTEXT_MODE_SNIPPET
        out_q["contexts"] = contexts
        out_questions.append(out_q)

    write_questions_jsonl(args.output_path, out_questions)

    if missing_total:
        logger.warning("PMIDs missing from corpus: %d", missing_total)
    logger.info("Wrote %d query records to %s", len(out_questions), args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
