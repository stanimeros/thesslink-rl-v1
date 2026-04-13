#!/usr/bin/env python3
"""Remove every empty directory under ``results/`` at the repo root.

Walks deepest-first so nested empties collapse. Does not delete ``results/`` itself.

Usage (from repo root)::

    python scripts/clean_empty_results_dirs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    results = repo_root / "results"

    if not results.is_dir():
        print(f"No results/ directory at {results} — nothing to do.")
        return 0

    results_resolved = results.resolve()
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(
        results_resolved, topdown=False, followlinks=False,
    ):
        p = Path(dirpath)
        if p == results_resolved:
            continue
        try:
            if any(p.iterdir()):
                continue
        except OSError:
            continue
        try:
            p.rmdir()
            removed += 1
        except OSError:
            pass

    print(f"Removed {removed} empty director(y/ies) under results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
