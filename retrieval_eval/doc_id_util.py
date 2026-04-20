"""Shared helpers for corpus-agnostic doc_id / doc_ids in evidence pipelines."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_PUBMED_URL_PATTERN = re.compile(r"pubmed/(\d+)/?$", re.I)


def doc_id_from_legacy_document_entry(entry: str) -> Optional[str]:
    """Parse legacy ``documents[]`` entry: PubMed URL or bare doc id string."""
    s = str(entry).strip()
    if not s:
        return None
    if s.lower().startswith("http"):
        m = _PUBMED_URL_PATTERN.search(s)
        return m.group(1) if m else None
    return s


def ranked_doc_ids_from_question(q: Dict[str, Any]) -> List[str]:
    """Ordered doc ids: ``doc_ids``, deprecated ``docnos``, or legacy URL/plain ``documents``."""
    raw = q.get("doc_ids")
    if isinstance(raw, list) and raw:
        return [str(x).strip() for x in raw if str(x).strip()]
    legacy_nos = q.get("docnos")
    if isinstance(legacy_nos, list) and legacy_nos:
        return [str(x).strip() for x in legacy_nos if str(x).strip()]
    out: List[str] = []
    for entry in q.get("documents") or []:
        did = doc_id_from_legacy_document_entry(str(entry))
        if did:
            out.append(did)
    return out


def ranked_doc_ids_for_evidence(q: Dict[str, Any], evidence_top_k: Optional[int]) -> List[str]:
    """Like ``ranked_doc_ids_from_question`` but capped to the first ``evidence_top_k`` when that is set and positive."""
    ids = ranked_doc_ids_from_question(q)
    if evidence_top_k is not None and evidence_top_k > 0:
        return ids[:evidence_top_k]
    return ids
