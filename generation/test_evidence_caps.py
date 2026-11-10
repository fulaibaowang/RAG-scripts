#!/usr/bin/env python3
"""Pins for the evidence caps in generate_answers.py (--max-contexts, --max-chars-per-context).

1. NO-OP. A positive cap renders the evidence block exactly as before, so existing configs keep
   producing byte-identical prompts.
2. NO SILENT CUT. Whenever a cap removes evidence from the prompt, evidence_caps_warning returns
   a message with the counts; when the prompt carries every context whole, it returns None.
3. 0 MEANS NO CAP, for either flag.

Run with plain python (no pytest needed):
    python generation/test_evidence_caps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_answers as G  # noqa: E402


def _legacy_block(contexts, max_contexts, max_chars_per_context):
    """The evidence block as rendered before the 0 = no cap option existed."""
    lines = []
    for ctx in contexts[:max_contexts]:
        cid = str(ctx.get("id", "")) or "(no id)"
        text = str(ctx.get("text", "")).strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + "..."
        lines.append(f"[{cid}],\n{text}" if text else f"[{cid}],")
    return "\n\n".join(lines)


def main() -> None:
    ctxs = [{"id": f"d{i}-0", "text": "x" * n} for i, n in enumerate([5, 50, 500, 0, 51])]
    ctxs.append({"text": "  padded  "})

    # 1. positive caps: identical to the old rendering
    for mc in (1, 3, 6, 100):
        for mx in (1, 50, 51, 10_000):
            assert G.format_evidence_block(ctxs, mc, mx) == _legacy_block(ctxs, mc, mx), (mc, mx)

    # 2. the effect is reported exactly, and only when there is one
    assert G.evidence_cap_effect(ctxs, 100, 10_000) == (0, 0)
    assert G.evidence_cap_effect(ctxs, 6, 500) == (0, 0)      # caps equal to the data: no cut
    assert G.evidence_cap_effect(ctxs, 2, 10_000) == (4, 0)
    assert G.evidence_cap_effect(ctxs, 100, 50) == (0, 2)     # 500 and 51
    assert G.evidence_cap_effect(ctxs, 2, 10) == (4, 1)       # cut counted among kept contexts only
    assert G.evidence_cap_effect(ctxs, 100, 7) == (0, 3)      # "  padded  " is measured stripped (6)
    assert G.evidence_caps_warning([ctxs, ctxs[:1]], 100, 10_000) is None
    msg = G.evidence_caps_warning([ctxs, ctxs[:1], []], 2, 10)
    assert msg and "1/3 question(s)" in msg and "4 context(s) dropped" in msg and "1 context(s) cut" in msg, msg

    # 3. 0 = no cap
    assert G.evidence_cap_effect(ctxs, 0, 0) == (0, 0)
    assert G.format_evidence_block(ctxs, 0, 0) == _legacy_block(ctxs, 10**9, 10**9)
    assert G.format_evidence_block(ctxs, 0, 50) == _legacy_block(ctxs, 10**9, 50)
    assert G.format_evidence_block(ctxs, 2, 0) == _legacy_block(ctxs, 2, 10**9)
    assert G.evidence_caps_warning([ctxs], 0, 0) is None

    print("test_evidence_caps: OK")


if __name__ == "__main__":
    main()
