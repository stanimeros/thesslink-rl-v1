#!/usr/bin/env python3
"""ThessLink RL — W&B analysis.

  python analysis.py w6            # last run per algo for w6
  python analysis.py w6 w7         # compare w6 (neg+nav specialists) vs w7 (full)
  python analysis.py w6 --best     # best run per algo instead of last
"""

from analysis.cli import main

if __name__ == "__main__":
    main()
