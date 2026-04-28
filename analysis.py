#!/usr/bin/env python3
"""Shim: use the `analysis` package CLI.

  python analysis.py runs --version w6
  python analysis.py runs --version g64 --metrics-source history
  python analysis.py runs --version g64 --metrics-source summary
  python -m analysis compare-versions -V w6 w7 --state finished
"""

from analysis.cli import main

if __name__ == "__main__":
    main()
