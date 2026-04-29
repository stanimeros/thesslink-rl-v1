"""CLI: show last/best runs for a version, or compare two versions."""

from __future__ import annotations

import argparse
import sys

from .compare import print_compare, print_runs
from .config import ALGOS, WANDB_ENTITY, WANDB_PROJECT
from .metrics_display import clear_history_cache


def _wandb_api():
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)
    return wandb.Api()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analysis",
        description="ThessLink RL — W&B run inspection and comparison.",
    )
    p.add_argument("versions", nargs="+", metavar="VERSION",
                   help="One version (show runs) or two versions (compare specialist vs full).")
    p.add_argument("--best", action="store_true",
                   help="Pick best-metric run per algo (default: most recent).")
    p.add_argument("--state", default="all",
                   choices=["finished", "running", "crashed", "failed", "all", "active"])
    p.add_argument("--algo", "-a", default=None, choices=list(ALGOS) + [a.upper() for a in ALGOS])
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--project", default=WANDB_PROJECT)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.versions) > 2:
        parser.error("Provide 1 or 2 version strings.")

    api = _wandb_api()
    clear_history_cache()
    pick = "best" if args.best else "last"

    if len(args.versions) == 1:
        print_runs(api, args.entity, args.project, args.versions[0], args.state, args.algo, pick=pick)
    else:
        print_compare(api, args.entity, args.project, args.versions[0], args.versions[1],
                      args.state, args.algo, pick=pick)


if __name__ == "__main__":
    main()
