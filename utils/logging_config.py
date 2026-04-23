#!/usr/bin/env python3
"""
Configure logging from environment (LOG_LEVEL, LOG_FILE).

When the pipeline sets LOG_FILE (e.g. to WORKFLOW_OUTPUT_DIR/pipeline.log),
scripts that call configure_logging_from_env() will also write logs to that file.
Call this at the start of main() before logging.basicConfig() or other config.
"""

import logging
import os
from pathlib import Path


def configure_logging_from_env() -> None:
    """Configure root logger from LOG_LEVEL and optional LOG_FILE env vars."""
    root = logging.getLogger()
    level_name = (os.environ.get("LOG_LEVEL") or "").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    log_file = (os.environ.get("LOG_FILE") or "").strip()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
