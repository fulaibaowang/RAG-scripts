#!/usr/bin/env python3
"""
Sentence picking for stage-3 rerank: hybrid (BM25 + dense) over document sentences,
output top-k sentences per (qid, docno). Outputs JSON for use by the stage-3 rerank scripts in scripts/deprecated/ (if needed).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow importing retrieval_eval and retrieval from shared_scripts; rerank for rerank_stage2
import sys
_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
_SHARED_SCRIPTS = _SCRIPT_DIR.parents[1]
for _p in [_SHARED_SCRIPTS] + list(_SCRIPT_DIR.parents):
    if (_p / "retrieval_eval").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        _SHARED_SCRIPTS = _p
        break
else:
    sys.path.insert(0, str(_SHARED_SCRIPTS))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from retrieval_eval.common import (
    build_topics_and_gold,
    load_questions,
    normalize_pmid,
    run_df_to_run_map,
)
from rerank_stage2 import extract_docno, load_run_tsv


_nltk_punkt_ensured: bool = False


def _ensure_nltk_punkt() -> None:
    """Ensure NLTK sentence tokenizer data is available. Newer NLTK uses punkt_tab; older uses punkt."""
    global _nltk_punkt_ensured
    if _nltk_punkt_ensured:
        return
    import nltk
    try:
        nltk.sent_tokenize("Hello world.")
    except LookupError:
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    _nltk_punkt_ensured = True


def extract_title_and_sentences(rec: dict) -> Tuple[str, List[str]]:
    """Extract title and list of abstract sentences from a JSONL record (same keys as extract_text)."""
    title = ""
    if rec.get("title"):
        title = str(rec["title"]).strip()

    abstract = ""
    if rec.get("abstract"):
        abstract = str(rec["abstract"]).strip()
    if rec.get("abstractText"):
        abstract = abstract or str(rec["abstractText"]).strip()

    if not abstract:
        return title, []

    _ensure_nltk_punkt()
    import nltk
    sentences = [s.strip() for s in nltk.sent_tokenize(abstract) if s.strip()]
    return title, sentences


def load_doc_title_sentences(
    docnos: iter,
    jsonl_path: Path,
) -> Dict[str, Tuple[str, List[str]]]:
    """Load docno -> (title, list of sentences) for each doc in docnos."""
    wanted = set(map(str, docnos))
    out: Dict[str, Tuple[str, List[str]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docno = extract_docno(rec)
            if docno in wanted and docno not in out:
                out[docno] = extract_title_and_sentences(rec)
            if len(out) == len(wanted):
                break
    return out


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenization for rank_bm25 (lowercase, split on non-alphanumeric)."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()


def _sentence_scores_bm25(
    query: str,
    sentences: List[str],
    bm25_class,
) -> np.ndarray:
    """BM25 scores for query vs list of sentences. Returns array of length len(sentences)."""
    if not sentences:
        return np.array([], dtype=np.float64)
    tokenized_corpus = [_tokenize_for_bm25(s) for s in sentences]
    # Filter empty token lists so BM25Okapi doesn't choke
    empty = [i for i, t in enumerate(tokenized_corpus) if not t]
    if len(empty) == len(sentences):
        return np.zeros(len(sentences), dtype=np.float64)
    # Build corpus without fully empty docs; rank_bm25 needs at least one non-empty
    non_empty_idx = [i for i in range(len(sentences)) if tokenized_corpus[i]]
    if not non_empty_idx:
        return np.zeros(len(sentences), dtype=np.float64)
    corpus_sub = [tokenized_corpus[i] for i in non_empty_idx]
    bm25 = bm25_class(corpus_sub)
    q_tok = _tokenize_for_bm25(query)
    if not q_tok:
        return np.zeros(len(sentences), dtype=np.float64)
    scores_sub = bm25.get_scores(q_tok)
    out = np.zeros(len(sentences), dtype=np.float64)
    for j, i in enumerate(non_empty_idx):
        out[i] = float(scores_sub[j])
    return out


def _sentence_scores_dense(
    query: str,
    sentences: List[str],
    model,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """Dense similarity (cosine) between query and each sentence. Returns array of length len(sentences)."""
    if not sentences:
        return np.array([], dtype=np.float64)
    q_emb = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).astype(np.float32)
    q_emb = q_emb[0]
    sent_embs = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).astype(np.float32)
    if normalize:
        # cosine sim = dot product when normalized
        return np.dot(sent_embs, q_emb)
    # unnormalized: return cosine similarity
    qn = np.linalg.norm(q_emb)
    if qn < 1e-9:
        return np.zeros(len(sentences), dtype=np.float64)
    return np.dot(sent_embs, q_emb) / (np.linalg.norm(sent_embs, axis=1) * qn + 1e-9)


def _rrf_fuse_sentence_scores(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
) -> np.ndarray:
    """Fuse BM25 and dense scores with RRF: score += w / (k_rrf + rank)."""
    n = len(bm25_scores)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # Rank by BM25 (descending)
    bm25_ranks = np.argsort(np.argsort(-bm25_scores))
    # Rank by dense (descending)
    dense_ranks = np.argsort(np.argsort(-dense_scores))
    fused = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_bm25 = int(bm25_ranks[i]) + 1
        r_dense = int(dense_ranks[i]) + 1
        fused[i] = (w_bm25 / (k_rrf + r_bm25)) + (w_dense / (k_rrf + r_dense))
    return fused


def pick_topk_sentences(
    query: str,
    sentences: List[str],
    model,
    bm25_class,
    top_k: int,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
    dense_batch_size: int,
    normalize_embeddings: bool,
) -> List[str]:
    """Return top-k sentence strings (in order) for this (query, sentences)."""
    if not sentences or top_k <= 0:
        return []
    if len(sentences) <= top_k:
        return list(sentences)

    bm25_scores = _sentence_scores_bm25(query, sentences, bm25_class)
    dense_scores = _sentence_scores_dense(
        query, sentences, model, dense_batch_size, normalize_embeddings
    )
    fused = _rrf_fuse_sentence_scores(bm25_scores, dense_scores, k_rrf, w_bm25, w_dense)
    top_indices = np.argsort(-fused)[:top_k]
    return [sentences[i] for i in top_indices]


def run_sentence_pick(
    run_map: Dict[str, List[str]],
    topics: Dict[str, str],
    doc_title_sentences: Dict[str, Tuple[str, List[str]]],
    model,
    top_k: int,
    k_rrf: int,
    w_bm25: float,
    w_dense: float,
    dense_batch_size: int,
    normalize_embeddings: bool,
):
    """
    For each (qid, docno) in run_map, pick top-k sentences.
    Returns Dict[(qid, docno), List[str]] of selected sentence strings.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError("Missing rank_bm25. Run: pip install rank_bm25") from e

    result: Dict[Tuple[str, str], List[str]] = {}
    total_pairs = sum(len(docs) for docs in run_map.values())
    done = 0
    for qid, docnos in run_map.items():
        query = topics.get(qid, "").strip()
        for docno in docnos:
            title, sentences = doc_title_sentences.get(docno, ("", []))
            selected = pick_topk_sentences(
                query=query,
                sentences=sentences,
                model=model,
                bm25_class=BM25Okapi,
                top_k=top_k,
                k_rrf=k_rrf,
                w_bm25=w_bm25,
                w_dense=w_dense,
                dense_batch_size=dense_batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            result[(qid, docno)] = selected
            done += 1
            if done % 5000 == 0 or done == total_pairs:
                print(f"[sentence_pick] {done}/{total_pairs} (qid, doc) pairs")
    return result


def save_picks_json(picks: Dict[Tuple[str, str], List[str]], path: Path) -> None:
    """Save picks as JSON: list of {qid, docno, sentences} for human readability."""
    rows = [
        {"qid": qid, "docno": docno, "sentences": sents}
        for (qid, docno), sents in picks.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=0, ensure_ascii=False)


def load_picks_json(path: Path) -> Dict[Tuple[str, str], List[str]]:
    """Load picks from JSON produced by save_picks_json."""
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {(r["qid"], r["docno"]): r["sentences"] for r in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentence picking for stage-3 rerank: hybrid BM25 + dense over doc sentences, output top-k per (qid, docno).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="Directory with run TSV files.")
    parser.add_argument("--run-files", type=Path, nargs="*", default=None, help="Explicit run TSV files.")
    parser.add_argument("--run-glob", type=str, default="*.tsv", help="Glob for run files under --runs-dir.")
    parser.add_argument("--docs-jsonl", type=Path, required=True, help="JSONL corpus with title/abstract.")
    parser.add_argument("--train-json", type=Path, default=None, help="Training questions JSON (BioASQ).")
    parser.add_argument("--test-batch-jsons", type=Path, nargs="*", default=None, help="Test batch JSONs (BioASQ).")
    parser.add_argument("--query-field", type=str, default="body", help="Question key for query text.")
    parser.add_argument("--candidate-limit", type=int, default=1000, help="Max candidates per query from run.")
    parser.add_argument("--dense-model", type=str, default=None, help="Dense encoder model name (e.g. abhinand/MedEmbed-small-v0.1). Use this OR --dense-index-dir to load the model.")
    parser.add_argument("--dense-index-dir", type=Path, default=None, help="Dense index dir (meta.json) to load model name from. Use this OR --dense-model.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of sentences to pick per document.")
    parser.add_argument("--sentence-k-rrf", type=int, default=60, help="RRF k for sentence-level fusion.")
    parser.add_argument("--sentence-w-bm25", type=float, default=1.0, help="BM25 weight in RRF.")
    parser.add_argument("--sentence-w-dense", type=float, default=1.0, help="Dense weight in RRF.")
    parser.add_argument("--dense-batch-size", type=int, default=256, help="Batch size for encoding sentences.")
    parser.add_argument("--dense-device", type=str, default="cuda", help="Device for dense model.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; sentence_picks/ will be under it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_files:
        run_files = [Path(p) for p in args.run_files]
    else:
        if not args.runs_dir:
            raise FileNotFoundError("Provide --runs-dir or --run-files.")
        run_files = sorted(args.runs_dir.glob(args.run_glob))
    if not run_files:
        raise FileNotFoundError("No run files found.")

    # Topics from train + test JSONs
    topics: Dict[str, str] = {}
    if args.train_json and args.train_json.exists():
        questions = load_questions(args.train_json)
        topics_df, _ = build_topics_and_gold(questions, query_field=args.query_field)
        topics.update(dict(zip(topics_df["qid"].astype(str), topics_df["query"].astype(str))))
    for path in args.test_batch_jsons or []:
        if Path(path).exists():
            questions = load_questions(Path(path))
            topics_df, _ = build_topics_and_gold(questions, query_field=args.query_field)
            topics.update(dict(zip(topics_df["qid"].astype(str), topics_df["query"].astype(str))))

    if not topics:
        print("warning: no queries loaded; sentence pick will use empty query.")

    # Load dense model: by name (--dense-model) or from index dir (--dense-index-dir)
    if args.dense_model:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Missing sentence-transformers. Run: pip install sentence-transformers") from e
        model = SentenceTransformer(args.dense_model, device=args.dense_device)
        normalize_embeddings = True
        print("[dense] loaded model by name:", args.dense_model)
    elif args.dense_index_dir and args.dense_index_dir.exists():
        _RETRIEVAL_DIR = _SHARED_SCRIPTS / "retrieval"
        if str(_RETRIEVAL_DIR) not in sys.path:
            sys.path.insert(0, str(_RETRIEVAL_DIR))
        from eval_dense import load_dense_runtime
        model, _index, _rowid_to_pmid, meta = load_dense_runtime(
            args.dense_index_dir,
            args.dense_device,
        )
        normalize_embeddings = meta.get("loaded_normalize_embeddings", True)
    else:
        raise ValueError("Provide --dense-model (e.g. abhinand/MedEmbed-small-v0.1) or --dense-index-dir to load the encoder for sentence pick.")

    output_dir = args.output_dir or Path(".")
    picks_dir = output_dir / "sentence_picks"
    picks_dir.mkdir(parents=True, exist_ok=True)

    for path in run_files:
        stem = path.stem
        df = load_run_tsv(path)
        if args.candidate_limit:
            df = df[df["rank"] <= int(args.candidate_limit)]
        run_map = run_df_to_run_map(df, qid_col="qid", docno_col="docno")

        candidate_docnos = set()
        for doc_list in run_map.values():
            candidate_docnos.update(doc_list)
        print(f"[{stem}] loading docs: {len(candidate_docnos)}")
        doc_title_sentences = load_doc_title_sentences(candidate_docnos, args.docs_jsonl)
        print(f"[{stem}] loaded title+sentences: {len(doc_title_sentences)}")

        picks = run_sentence_pick(
            run_map=run_map,
            topics=topics,
            doc_title_sentences=doc_title_sentences,
            model=model,
            top_k=args.top_k,
            k_rrf=args.sentence_k_rrf,
            w_bm25=args.sentence_w_bm25,
            w_dense=args.sentence_w_dense,
            dense_batch_size=args.dense_batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        out_path = picks_dir / f"{stem}.json"
        save_picks_json(picks, out_path)
        print(f"[{stem}] saved {len(picks)} picks to {out_path}")


if __name__ == "__main__":
    main()
