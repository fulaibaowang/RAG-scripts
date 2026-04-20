#!/usr/bin/env python3
"""Deprecated entrypoint: use post_rerank_jsonl.py (same CLI). Kept for backward compatibility."""

import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    _target = _here / "post_rerank_jsonl.py"
    spec = importlib.util.spec_from_file_location("post_rerank_jsonl", _target)
    if spec is None or spec.loader is None:
        print("Failed to load post_rerank_jsonl.py", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main())
