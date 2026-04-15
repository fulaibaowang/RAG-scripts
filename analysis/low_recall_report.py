#!/usr/bin/env python3
"""Identify questions with very low Recall@K from a pipeline output folder.

If a stage has several ``runs/*.tsv`` (or several ``per_query/*.csv``), each
file is processed in sorted name order and gets its own report CSV (unless
there is only one artifact, in which case the filename stays
``low_recall_{stage}.csv``).

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

# Write CSVs only (no table on stdout; use -q to silence progress too):
python low_recall_report.py --output-dir bioasq14_output/batch_1 -q

# Per-run ground truth: optional ``output_dir/low_recall_ground_truth_map.json``::
#
#   {"by_run_file": {"dense_13b.tsv": "example/a.json"}, "default": "example/b.json"}
#
# Or set TEST_BATCH_JSONS in config*.env; any run filename containing a batch
# file's stem (e.g. ``13b_golden_50q_sample``) uses that JSON before the default.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob as glob_mod
import json
import re
import shlex
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

def _find_pipeline_config_env(output_dir: Path) -> Optional[Path]:
    """Locate a pipeline env file under *output_dir*.

    Prefer ``config.env`` (batch default). Otherwise use any ``config*.env``
    (e.g. ``config_14b.env``) so ad-hoc or copied output trees still resolve
    ``TRAIN_JSON`` without ``--ground-truth``.
    """
    preferred = output_dir / "config.env"
    if preferred.is_file():
        return preferred

    candidates = sorted(output_dir.glob("config*.env"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for p in candidates:
        try:
            env = _parse_config_env(p)
            if env.get("TRAIN_JSON"):
                return p
        except OSError:
            continue
    return candidates[0]


def _resolve_ground_truth(
    output_dir: Path,
    ground_truth_arg: Optional[str],
    repo_root_arg: Optional[str],
) -> Tuple[Path, Optional[Path], Dict[str, str]]:
    """Return (default_ground_truth_json, config_env_path_or_none, env_with_REPO_ROOT).

    *env_with_REPO_ROOT* is suitable for ``_substitute_env`` when resolving map paths.
    """
    repo_root = Path(repo_root_arg) if repo_root_arg else _detect_repo_root(output_dir)

    if ground_truth_arg:
        p = Path(ground_truth_arg)
        if not p.exists():
            raise FileNotFoundError(f"Ground truth not found: {p}")
        rp = p.resolve()
        cfg_path = _find_pipeline_config_env(output_dir)
        file_env = _parse_config_env(cfg_path) if cfg_path else {}
        env = {**file_env, "REPO_ROOT": str(repo_root)}
        return rp, cfg_path, env

    config_path = _find_pipeline_config_env(output_dir)
    if config_path is None:
        raise FileNotFoundError(
            f"No config.env or config*.env in {output_dir} and --ground-truth not provided"
        )

    env = _parse_config_env(config_path)
    train_json_raw = env.get("TRAIN_JSON")
    if not train_json_raw:
        raise KeyError(
            f"TRAIN_JSON not found in {config_path.name}; pass --ground-truth explicitly"
        )

    env["REPO_ROOT"] = str(repo_root)
    resolved = Path(_substitute_env(train_json_raw, env))
    if resolved.exists():
        return resolved.resolve(), config_path, env

    raise FileNotFoundError(
        f"Resolved ground-truth path does not exist: {resolved}\n"
        f"  (from TRAIN_JSON={train_json_raw!r}, REPO_ROOT={repo_root})\n"
        "  Pass --ground-truth or --repo-root to override."
    )


def _resolve_path_against_repo(
    raw: str,
    env: Dict[str, str],
    repo_root: Path,
) -> Path:
    """Resolve a config or map path (may be relative to repo root)."""
    expanded = _substitute_env(raw.strip(), env)
    p = Path(expanded)
    if p.is_absolute():
        out = p
    else:
        out = repo_root / p
    return out.resolve()


def _parse_test_batch_jsons(
    raw: str,
    env: Dict[str, str],
    repo_root: Path,
) -> List[Path]:
    """Paths from TEST_BATCH_JSONS that exist (shell-tokenized)."""
    if not (raw or "").strip():
        return []
    out: List[Path] = []
    for part in shlex.split(raw.strip()):
        p = _resolve_path_against_repo(part, env, repo_root)
        if p.is_file():
            out.append(p)
    return out


def _load_low_recall_gt_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Ground-truth map must be a JSON object: {path}")
    return data


def _pick_ground_truth_for_run(
    run_filename: str,
    *,
    default_gt: Path,
    config_env: Dict[str, str],
    repo_root: Path,
    map_data: Optional[dict],
) -> Tuple[Path, str]:
    """Pick JSON ground truth for this run / per-query CSV filename."""
    if map_data:
        by_run = map_data.get("by_run_file") or map_data.get("runs")
        if isinstance(by_run, dict) and run_filename in by_run:
            p = _resolve_path_against_repo(str(by_run[run_filename]), config_env, repo_root)
            if p.is_file():
                return p, f"ground-truth map key {run_filename!r}"

        patterns = map_data.get("patterns")
        if isinstance(patterns, list):
            for entry in patterns:
                if not isinstance(entry, dict):
                    continue
                g = entry.get("glob") or entry.get("match")
                gt = entry.get("ground_truth")
                if g and gt and fnmatch.fnmatch(run_filename, g):
                    p = _resolve_path_against_repo(str(gt), config_env, repo_root)
                    if p.is_file():
                        return p, f"ground-truth map glob {g!r}"

        d = map_data.get("default")
        if d is not None and str(d).strip():
            p = _resolve_path_against_repo(str(d), config_env, repo_root)
            if p.is_file():
                return p, "ground-truth map default"

    batches = _parse_test_batch_jsons(
        config_env.get("TEST_BATCH_JSONS", ""), config_env, repo_root
    )
    for bp in batches:
        if bp.stem in run_filename:
            return bp, f"TEST_BATCH_JSONS → {bp.name}"

    return default_gt, "default (TRAIN_JSON or --ground-truth)"

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


def discover_recall_sources(stage_dir: Path) -> List[Tuple[str, Path]]:
    """List recall inputs for *stage_dir*, deterministic order.

    If ``per_query/*.csv`` exists, returns those only (sorted by name).
    Otherwise returns ``runs/*.tsv`` (sorted by name).
    """
    perq_dir = stage_dir / "per_query"
    if perq_dir.is_dir():
        csvs = sorted(perq_dir.glob("*.csv"))
        if csvs:
            return [("per_query", p) for p in csvs]

    runs_dir = stage_dir / "runs"
    if runs_dir.is_dir():
        tsvs = sorted(runs_dir.glob("*.tsv"))
        if tsvs:
            return [("run_tsv", p) for p in tsvs]

    raise FileNotFoundError(
        f"No per_query CSV or run TSV found for stage '{stage_dir.name}' in {stage_dir}"
    )


def load_recall_for_source(
    kind: str,
    path: Path,
    k: int,
    gold_map: Dict[str, List[str]],
) -> Dict[str, float]:
    """Load {qid: recall@K} from one per-query CSV or one run TSV."""
    if kind == "per_query":
        return _load_perquery_csv(path, k)
    run_df = _load_run_tsv(path)
    return _recall_from_run(run_df, gold_map, k)

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
    *,
    quiet: bool = False,
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

    if not quiet:
        print(f"  JSONL scan: {len(titles)}/{len(pmids)} titles found "
              f"({total_scanned} lines scanned)")
    return titles

# ---------------------------------------------------------------------------
# Title fetching — NCBI E-utilities
# ---------------------------------------------------------------------------

_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_BATCH_SIZE = 200
_RATE_LIMIT_DELAY = 0.34  # ~3 req/s without API key


def fetch_titles_ncbi(pmids: Set[str], *, quiet: bool = False) -> Dict[str, str]:
    """Fetch PubMed titles via NCBI esummary in batches."""
    if not pmids:
        return {}

    titles: Dict[str, str] = {}
    pmid_list = sorted(pmids)
    n_batches = (len(pmid_list) + _BATCH_SIZE - 1) // _BATCH_SIZE

    if not quiet:
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
            if not quiet:
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

    if not quiet:
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
             "Common: retrieval/bm25, retrieval/dense, retrieval/fusion, "
             "rerank/cross_encoder, rerank/post_rerank_fusion, rerank/post_rerank_fusion_snippet, "
             "snippet/snippet_rerank, snippet/snippet_doc_fusion",
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
        help="Default ground-truth JSON; per-run overrides via map file or TEST_BATCH_JSONS",
    )
    ap.add_argument(
        "--ground-truth-map", default=None, type=Path,
        help="JSON map (see module docstring); default tries low_recall_ground_truth_map.json "
             "in --output-dir unless --no-ground-truth-map",
    )
    ap.add_argument(
        "--no-ground-truth-map", action="store_true",
        help="Do not load low_recall_ground_truth_map.json from the output dir",
    )
    ap.add_argument(
        "--docs-jsonl", default=None,
        help="Glob to JSONL corpus for title lookup "
             '(e.g. "/pubmed/jsonl_2026/*.jsonl")',
    )
    ap.add_argument(
        "--output", default=None, type=Path,
        help="Output CSV path (single run/CSV only; not allowed with multiple artifacts)",
    )
    ap.add_argument(
        "--repo-root", default=None,
        help="Override $REPO_ROOT when resolving TRAIN_JSON from pipeline config.env",
    )
    ap.add_argument(
        "--no-titles", action="store_true",
        help="Skip title fetching entirely",
    )
    ap.add_argument(
        "--quiet", "-q", action="store_true",
        help="Do not print progress or tables; only write CSV(s)",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    stage: str = args.stage
    k: int = args.recall_k
    mode: str = args.mode

    quiet = args.quiet
    if not quiet:
        print(f"=== Low Recall@{k} Report ===")
        print(f"  Output dir : {output_dir}")
        print(f"  Stage      : {stage}")
        print(f"  Mode       : {mode}")
        print()

    # 1. Default ground truth + env (for TEST_BATCH_JSONS and map substitution)
    default_gt, cfg_env, config_env = _resolve_ground_truth(
        output_dir, args.ground_truth, args.repo_root
    )
    repo_root = Path(config_env["REPO_ROOT"])

    map_data: Optional[dict] = None
    map_path_used: Optional[Path] = None
    if not args.no_ground_truth_map:
        map_candidate = (
            args.ground_truth_map.resolve()
            if args.ground_truth_map
            else (output_dir / "low_recall_ground_truth_map.json")
        )
        if args.ground_truth_map and not map_candidate.is_file():
            raise SystemExit(f"--ground-truth-map not found: {map_candidate}")
        if map_candidate.is_file():
            map_data = _load_low_recall_gt_map(map_candidate)
            map_path_used = map_candidate

    if not quiet:
        print(f"Default ground truth : {default_gt}")
        if cfg_env is not None and cfg_env.name != "config.env":
            print(f"  (TRAIN_JSON from {cfg_env.name})")
        if map_path_used is not None:
            print(f"  Ground-truth map    : {map_path_used.name}")
        elif not args.no_ground_truth_map and config_env.get("TEST_BATCH_JSONS"):
            print("  Per-run override   : TEST_BATCH_JSONS stem ⊆ run filename")
        print()

    gt_bundle_cache: Dict[Path, Tuple[Dict[str, List[str]], Dict[str, dict]]] = {}

    def _get_gold_bundle(gt_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
        if gt_path not in gt_bundle_cache:
            qs = load_questions(gt_path)
            _, gm = build_topics_and_gold(qs)
            meta = _build_question_meta(qs)
            gt_bundle_cache[gt_path] = (gm, meta)
        return gt_bundle_cache[gt_path]

    stage_dir = output_dir / stage
    sources = discover_recall_sources(stage_dir)
    if args.output is not None and len(sources) > 1:
        raise SystemExit(
            "Cannot use --output when multiple per_query CSVs or run TSVs exist; "
            "omit --output to write one analysis file per artifact."
        )

    titles_cache: Dict[str, str] = {}
    any_report = False

    for si, (kind, src_path) in enumerate(sources):
        run_key = src_path.name
        gt_path, gt_reason = _pick_ground_truth_for_run(
            run_key,
            default_gt=default_gt,
            config_env=config_env,
            repo_root=repo_root,
            map_data=map_data,
        )
        gold_map, question_meta = _get_gold_bundle(gt_path)

        if not quiet:
            print(f"--- Recall source {si + 1}/{len(sources)}: {run_key} ---")
            print(f"  Ground truth file : {gt_path} ({gt_reason})")
            print(f"  {len(gold_map)} qids with gold in this JSON")
            if kind == "per_query":
                print("  Loading per-query CSV…")
            else:
                print(f"  Computing R@{k} from run TSV…")
        recall_map = load_recall_for_source(kind, src_path, k, gold_map)
        if not quiet:
            print(f"  {len(recall_map)} queries with recall values")
        if not recall_map:
            if not quiet:
                print(
                    "  Warning: no overlapping qids with gold (wrong TRAIN_JSON / run pair?)"
                )
                print()
            continue

        low_qids = filter_low_recall(recall_map, mode)
        if not low_qids:
            if not quiet:
                print("  No low-recall questions for this run.")
                print()
            continue

        n_zero = sum(1 for q in low_qids if recall_map.get(q, 0) == 0.0)
        if not quiet:
            print(f"  Low-recall questions: {len(low_qids)} "
                  f"(zero={n_zero}, non-zero={len(low_qids) - n_zero})")

        all_gold_pmids: Set[str] = set()
        for qid in low_qids:
            all_gold_pmids.update(gold_map.get(qid, []))

        if not args.no_titles and all_gold_pmids:
            missing = all_gold_pmids - set(titles_cache.keys())
            if missing:
                if not quiet:
                    print(f"  Resolving titles for {len(missing)} new unique PMIDs…")
                if args.docs_jsonl:
                    new_titles = fetch_titles_jsonl(
                        missing, args.docs_jsonl, quiet=quiet
                    )
                    still = missing - set(new_titles.keys())
                    if still:
                        if not quiet:
                            print(f"    {len(still)} PMIDs not in JSONL, NCBI fallback…")
                        new_titles.update(fetch_titles_ncbi(still, quiet=quiet))
                    titles_cache.update(new_titles)
                else:
                    titles_cache.update(fetch_titles_ncbi(missing, quiet=quiet))

        report_df = build_report(
            low_qids, recall_map, gold_map, question_meta, titles_cache, k
        )

        if args.output is not None:
            out_path = args.output
        elif len(sources) == 1:
            out_path = output_dir / "analysis" / f"low_recall_{stage}.csv"
        else:
            safe_stem = src_path.stem.replace("/", "_")
            out_path = output_dir / "analysis" / f"low_recall_{stage}__{safe_stem}.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        if not quiet:
            print(f"  Saved {len(report_df)} rows → {out_path}")
        any_report = True
        if not quiet:
            print()

    if not any_report and not quiet:
        print("No low-recall report written (no matching recall or all above threshold).")


if __name__ == "__main__":
    main()
