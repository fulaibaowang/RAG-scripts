#!/usr/bin/env python3
"""Identify questions with very low Recall@K from a pipeline output folder.

For each low-recall question the report includes the question text, golden
truth PMIDs, and (optionally) their PubMed titles.  Titles are resolved from
a local JSONL corpus (``--docs-jsonl``) or, as a fallback, via the NCBI
E-utilities API.

Usage examples
--------------
# Hybrid stage, default K=5000, both zero and bottom-10% modes:
python low_recall_report.py --output-dir bioasq14_output/batch_1

# BM25 stage with a local corpus for titles:
python low_recall_report.py --output-dir bioasq14_output/batch_1 \
    --stage bm25 --docs-jsonl "/pubmed/jsonl_2026/*.jsonl"

# Only zero-recall questions, explicit ground truth:
python low_recall_report.py --output-dir bioasq14_output/batch_1 \
    --stage dense --mode zero \
    --ground-truth bioasq_data/14b/BioASQ-task14bPhaseB-testset1
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve imports from the shared retrieval_eval package
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED_DIR = _SCRIPT_DIR.parent  # shared_scripts/
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from retrieval_eval.common import (  # noqa: E402
    load_questions,
    build_topics_and_gold,
    normalize_pmid,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# config.env parsing
# ---------------------------------------------------------------------------

def _parse_config_env(path: Path) -> Dict[str, str]:
    """Minimal shell-variable parser for config.env files."""
    env: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            val = val.strip().strip('"').strip("'")
            env[key] = val
    return env


def _substitute_env(value: str, env: Dict[str, str]) -> str:
    """Replace $VAR and ${VAR} references with their values from *env*."""
    def _repl(m: re.Match) -> str:
        return env.get(m.group(1) or m.group(2), m.group(0))
    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", _repl, value)


def _detect_repo_root(start: Path) -> Path:
    """Walk upward from *start* to find the repository root (.git or pyproject.toml)."""
    cur = start.resolve()
    for _ in range(20):
        if (cur / ".git").exists() or (cur / "pyproject.toml").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return start.resolve()

# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _resolve_ground_truth(
    output_dir: Path,
    ground_truth_arg: Optional[str],
    repo_root_arg: Optional[str],
) -> Path:
    """Return the resolved ground-truth JSON path."""
    if ground_truth_arg:
        p = Path(ground_truth_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Ground truth not found: {p}")

    config_path = output_dir / "config.env"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.env in {output_dir} and --ground-truth not provided"
        )

    env = _parse_config_env(config_path)
    train_json_raw = env.get("TRAIN_JSON")
    if not train_json_raw:
        raise KeyError("TRAIN_JSON not found in config.env; pass --ground-truth explicitly")

    repo_root = Path(repo_root_arg) if repo_root_arg else _detect_repo_root(output_dir)
    env["REPO_ROOT"] = str(repo_root)
    resolved = Path(_substitute_env(train_json_raw, env))
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Resolved ground-truth path does not exist: {resolved}\n"
        f"  (from TRAIN_JSON={train_json_raw!r}, REPO_ROOT={repo_root})\n"
        "  Pass --ground-truth or --repo-root to override."
    )

# ---------------------------------------------------------------------------
# Per-query recall loading
# ---------------------------------------------------------------------------

def _load_run_tsv(path: Path) -> pd.DataFrame:
    """Load a run TSV (qid, docno, rank, score) in any column order."""
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid")
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc")
    rank_col = cols.get("rank")
    if qid_col is None or doc_col is None:
        raise ValueError(f"Missing qid/doc columns in {path}: {list(df.columns)}")
    df[qid_col] = df[qid_col].astype(str)
    df[doc_col] = df[doc_col].astype(str)
    if rank_col:
        df = df.sort_values([qid_col, rank_col])
    return df[[qid_col, doc_col]]


def _recall_from_run(
    run_df: pd.DataFrame,
    gold_map: Dict[str, List[str]],
    k: int,
) -> Dict[str, float]:
    """Compute per-query Recall@K from a run DataFrame and gold map."""
    qid_col, doc_col = run_df.columns.tolist()
    result: Dict[str, float] = {}
    for qid, group in run_df.groupby(qid_col, sort=False):
        qid_str = str(qid)
        gold = set(map(str, gold_map.get(qid_str, [])))
        if not gold:
            continue
        ranked = group[doc_col].tolist()
        result[qid_str] = recall_at_k(gold, ranked, k)
    return result


def _load_perquery_csv(path: Path, k: int) -> Dict[str, float]:
    """Read a per-query CSV and extract qid -> R@K."""
    df = pd.read_csv(path)
    col = f"R@{k}"
    if col not in df.columns:
        available = [c for c in df.columns if c.startswith("R@")]
        raise KeyError(f"Column {col} not in {path}; available recall cols: {available}")
    df["qid"] = df["qid"].astype(str)
    return dict(zip(df["qid"], df[col]))


def load_recall_per_query(
    output_dir: Path,
    stage: str,
    k: int,
    gold_map: Dict[str, List[str]],
) -> Dict[str, float]:
    """Return {qid: recall@K} for the chosen stage.

    Uses a pre-computed per_query CSV if available; otherwise computes from
    the run TSV.
    """
    stage_dir = output_dir / stage

    perq_dir = stage_dir / "per_query"
    if perq_dir.is_dir():
        csvs = list(perq_dir.glob("*.csv"))
        if csvs:
            csv_path = csvs[0]
            print(f"  Loading per-query CSV: {csv_path.name}")
            return _load_perquery_csv(csv_path, k)

    runs_dir = stage_dir / "runs"
    if runs_dir.is_dir():
        tsvs = list(runs_dir.glob("*.tsv"))
        if tsvs:
            tsv_path = tsvs[0]
            print(f"  Computing R@{k} from run file: {tsv_path.name}")
            run_df = _load_run_tsv(tsv_path)
            return _recall_from_run(run_df, gold_map, k)

    raise FileNotFoundError(
        f"No per_query CSV or run TSV found for stage '{stage}' in {stage_dir}"
    )

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_low_recall(
    recall_map: Dict[str, float],
    mode: str,
) -> Set[str]:
    """Return the set of qids matching the low-recall filter."""
    if not recall_map:
        return set()

    values = np.array(list(recall_map.values()))

    zero_qids: Set[str] = set()
    bottom_qids: Set[str] = set()

    if mode in ("zero", "both"):
        zero_qids = {qid for qid, r in recall_map.items() if r == 0.0}

    if mode in ("bottom10", "both"):
        threshold = float(np.percentile(values, 10))
        bottom_qids = {qid for qid, r in recall_map.items() if r <= threshold}

    return zero_qids | bottom_qids

# ---------------------------------------------------------------------------
# Title fetching — JSONL corpus
# ---------------------------------------------------------------------------

def fetch_titles_jsonl(
    pmids: Set[str],
    jsonl_glob: str,
) -> Dict[str, str]:
    """Scan JSONL files to extract titles for the given PMIDs."""
    titles: Dict[str, str] = {}
    remaining = set(pmids)
    total_scanned = 0

    for fp in sorted(glob_mod.glob(jsonl_glob)):
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                total_scanned += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_pmid = str(d.get("pmid") or d.get("docno") or "").strip()
                if doc_pmid in remaining:
                    title = (d.get("title") or "").strip()
                    if title:
                        titles[doc_pmid] = title
                    remaining.discard(doc_pmid)
                    if not remaining:
                        break
        if not remaining:
            break

    print(f"  JSONL scan: {len(titles)}/{len(pmids)} titles found "
          f"({total_scanned} lines scanned)")
    return titles

# ---------------------------------------------------------------------------
# Title fetching — NCBI E-utilities
# ---------------------------------------------------------------------------

_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_BATCH_SIZE = 200
_RATE_LIMIT_DELAY = 0.34  # ~3 req/s without API key


def fetch_titles_ncbi(pmids: Set[str]) -> Dict[str, str]:
    """Fetch PubMed titles via NCBI esummary in batches."""
    if not pmids:
        return {}

    titles: Dict[str, str] = {}
    pmid_list = sorted(pmids)
    n_batches = (len(pmid_list) + _BATCH_SIZE - 1) // _BATCH_SIZE

    print(f"  Fetching titles from NCBI for {len(pmid_list)} PMIDs "
          f"({n_batches} batch(es))...")

    for i in range(0, len(pmid_list), _BATCH_SIZE):
        batch = pmid_list[i : i + _BATCH_SIZE]
        ids_param = ",".join(batch)
        url = f"{_ESUMMARY_URL}?db=pubmed&id={ids_param}&retmode=json"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "BioASQ-LowRecall/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"    WARNING: NCBI request failed: {exc}")
            continue

        result = data.get("result", {})
        for pmid in batch:
            info = result.get(pmid)
            if info and isinstance(info, dict):
                title = info.get("title", "").strip()
                if title:
                    titles[pmid] = title

        if i + _BATCH_SIZE < len(pmid_list):
            time.sleep(_RATE_LIMIT_DELAY)

    print(f"  NCBI: {len(titles)}/{len(pmids)} titles resolved")
    return titles

# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------

def _build_question_meta(questions: List[dict]) -> Dict[str, dict]:
    """Map qid -> {body, type, documents (raw URLs)}."""
    meta: Dict[str, dict] = {}
    for q in questions:
        qid = str(q.get("id") or q.get("qid") or "")
        if not qid:
            continue
        meta[qid] = {
            "body": (q.get("body") or q.get("query") or q.get("question") or "").strip(),
            "type": (q.get("type") or "unknown").lower(),
            "documents": q.get("documents") or [],
        }
    return meta


def build_report(
    low_qids: Set[str],
    recall_map: Dict[str, float],
    gold_map: Dict[str, List[str]],
    question_meta: Dict[str, dict],
    titles: Dict[str, str],
    k: int,
) -> pd.DataFrame:
    rows = []
    for qid in sorted(low_qids):
        meta = question_meta.get(qid, {})
        gold_pmids = gold_map.get(qid, [])
        pmid_str = ", ".join(gold_pmids)
        title_str = ", ".join(titles.get(p, "") for p in gold_pmids)

        rows.append({
            "qid": qid,
            "question": meta.get("body", ""),
            "type": meta.get("type", "unknown"),
            "n_rel": len(gold_pmids),
            f"R@{k}": round(recall_map.get(qid, 0.0), 6),
            "golden_pmids": pmid_str,
            "golden_titles": title_str,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"R@{k}").reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Report questions with low Recall@K from a pipeline output folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--output-dir", required=True, type=Path,
        help="Batch output folder (e.g. bioasq14_output/batch_1)",
    )
    ap.add_argument(
        "--stage", default="hybrid",
        help="Retrieval stage to analyze (default: hybrid). "
             "Common: bm25, dense, hybrid, rerank, rerank_hybrid, "
             "rerank_hybrid_200, snippet_rerank, snippet_rrf",
    )
    ap.add_argument(
        "--recall-k", type=int, default=5000,
        help="K for Recall@K (default: 5000)",
    )
    ap.add_argument(
        "--mode", choices=("zero", "bottom10", "both"), default="both",
        help="Filter mode: zero (R@K==0), bottom10 (bottom 10%%), or both (default: both)",
    )
    ap.add_argument(
        "--ground-truth", default=None,
        help="Explicit ground-truth JSON path; if omitted, parsed from config.env",
    )
    ap.add_argument(
        "--docs-jsonl", default=None,
        help="Glob to JSONL corpus for title lookup "
             '(e.g. "/pubmed/jsonl_2026/*.jsonl")',
    )
    ap.add_argument(
        "--output", default=None, type=Path,
        help="Output CSV path (default: {output_dir}/analysis/low_recall_{stage}.csv)",
    )
    ap.add_argument(
        "--repo-root", default=None,
        help="Override $REPO_ROOT for config.env resolution",
    )
    ap.add_argument(
        "--no-titles", action="store_true",
        help="Skip title fetching entirely",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    stage: str = args.stage
    k: int = args.recall_k
    mode: str = args.mode

    print(f"=== Low Recall@{k} Report ===")
    print(f"  Output dir : {output_dir}")
    print(f"  Stage      : {stage}")
    print(f"  Mode       : {mode}")
    print()

    # 1. Ground truth
    gt_path = _resolve_ground_truth(output_dir, args.ground_truth, args.repo_root)
    print(f"Ground truth : {gt_path}")
    questions = load_questions(gt_path)
    _, gold_map = build_topics_and_gold(questions)
    question_meta = _build_question_meta(questions)
    print(f"  {len(questions)} questions loaded, {len(gold_map)} with gold docs")
    print()

    # 2. Per-query recall
    print(f"Loading Recall@{k} for stage '{stage}'...")
    recall_map = load_recall_per_query(output_dir, stage, k, gold_map)
    print(f"  {len(recall_map)} queries with recall values")
    print()

    # 3. Filter
    low_qids = filter_low_recall(recall_map, mode)
    if not low_qids:
        print("No low-recall questions found. Nothing to report.")
        return

    n_zero = sum(1 for q in low_qids if recall_map.get(q, 0) == 0.0)
    print(f"Low-recall questions: {len(low_qids)} "
          f"(zero={n_zero}, non-zero={len(low_qids) - n_zero})")
    print()

    # 4. Collect gold PMIDs for title fetching
    all_gold_pmids: Set[str] = set()
    for qid in low_qids:
        all_gold_pmids.update(gold_map.get(qid, []))

    # 5. Title lookup
    titles: Dict[str, str] = {}
    if not args.no_titles and all_gold_pmids:
        print(f"Resolving titles for {len(all_gold_pmids)} unique PMIDs...")
        if args.docs_jsonl:
            titles = fetch_titles_jsonl(all_gold_pmids, args.docs_jsonl)
            missing = all_gold_pmids - set(titles.keys())
            if missing:
                print(f"  {len(missing)} PMIDs not found in JSONL, "
                      "falling back to NCBI API...")
                titles.update(fetch_titles_ncbi(missing))
        else:
            titles = fetch_titles_ncbi(all_gold_pmids)
        print()

    # 6. Build report
    report_df = build_report(low_qids, recall_map, gold_map, question_meta, titles, k)

    # 7. Save
    out_path: Path = args.output or (output_dir / "analysis" / f"low_recall_{stage}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)
    print(f"Saved {len(report_df)} rows → {out_path}")
    print()

    # 8. Print summary to stdout
    display_cols = ["qid", "type", "n_rel", f"R@{k}", "question", "golden_pmids"]
    if titles:
        display_cols.append("golden_titles")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_colwidth", 100,
        "display.width", 240,
    ):
        print(report_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
