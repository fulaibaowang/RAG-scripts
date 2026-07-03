#!/usr/bin/env python3
"""Regression pins for the distillation cache-key contracts.

The claim and summary caches are expensive to rebuild (one LLM call per entry); these pins
guarantee the key functions never drift, so existing caches stay readable. Run with plain
python (no pytest needed): python generation/test_distill_common.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_common import member_hash, norm, sha_key, strip_preamble  # noqa: E402


def main() -> None:
    # claim-cache key: sha1(qid + "\0" + text)
    assert sha_key("q1", "hello world") == "6ecf7c5aedc0fb1fb615e547bc4ab67cf96cf3c5"

    # summary-cache key component: first 16 hex of sha1 of sorted normalized member texts
    assert member_hash(["Beta claim, two.", "Alpha claim!"]) == "eb230ea79e38dee7"
    # order-independent, normalization-sensitive
    assert member_hash(["Alpha claim!", "Beta claim, two."]) == "eb230ea79e38dee7"

    assert norm("  A-B,  c!! ") == "a b c"

    assert strip_preamble("Here is the combined statement: Facts stand.") == "Facts stand."
    assert strip_preamble("Combined statement: Facts stand.") == "Facts stand."
    assert strip_preamble("Facts stand.") == "Facts stand."

    print("test_distill_common: OK")


if __name__ == "__main__":
    main()
