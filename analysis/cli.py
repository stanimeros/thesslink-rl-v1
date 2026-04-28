"""CLI: current runs, compare algos (same version), compare versions."""

from __future__ import annotations

import argparse
import sys

from .compare import print_algo_comparison, print_version_comparison
from .config import ALGOS, FULL_METRICS, NAV_METRICS, NEG_METRICS, OTHER_METRICS, WANDB_ENTITY, WANDB_PROJECT
from .metrics_display import clear_history_cache, print_section
from .partition import partition_runs
from .wandb_runs import fetch_runs


def _wandb_api():
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)
    return wandb.Api()


def cmd_runs(args: argparse.Namespace) -> None:
    """List runs with detailed sections (same output as legacy analysis.py)."""
    api = _wandb_api()
    clear_history_cache()
    print(f"W&B  entity={args.entity}  project={args.project}")
    print(
        "Fetching runs: version="
        + repr(args.version)
        + (f"  algo={args.algo}" if args.algo else "")
        + f"  state={args.state} …"
    )

    runs = fetch_runs(api, args.entity, args.project, args.version, args.algo, args.state)

    if not runs:
        print(f"\nNo runs found for version={args.version!r}.")
        print("Hint: check that the version string appears in run name, tags, group, or env_config.")
        return

    print(f"Found {len(runs)} run(s).\n")

    parts = partition_runs(runs)

    print_section(
        "FULL EPISODE  (neg → nav)  —  neg / nav test metrics / battle_won",
        parts.full,
        FULL_METRICS,
        args.top,
    )
    print_section(
        "NAVIGATION  —  quality / quality-on-win / length / battle_won",
        parts.nav,
        NAV_METRICS,
        args.top,
    )
    print_section(
        "NEGOTIATION  —  quality / agreed / length / battle_won",
        parts.neg,
        NEG_METRICS,
        args.top,
    )
    if parts.other:
        print_section("OTHER", parts.other, OTHER_METRICS, args.top)


def cmd_compare_algos(args: argparse.Namespace) -> None:
    api = _wandb_api()
    clear_history_cache()
    print(f"W&B  entity={args.entity}  project={args.project}")
    runs = fetch_runs(api, args.entity, args.project, args.version, args.algo, args.state)
    if not runs:
        print(f"No runs found for version={args.version!r}.")
        return
    parts = partition_runs(runs)
    print_algo_comparison(args.version, parts)


def cmd_compare_versions(args: argparse.Namespace) -> None:
    api = _wandb_api()
    clear_history_cache()
    print(f"W&B  entity={args.entity}  project={args.project}")
    print_version_comparison(
        api,
        args.entity,
        args.project,
        list(args.versions),
        args.state,
        args.algo,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analysis",
        description="ThessLink RL — W&B run inspection and comparisons.",
    )
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--project", default=WANDB_PROJECT)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", default="all", choices=["finished", "running", "crashed", "failed", "all"])

    runs_p = sub.add_parser("runs", parents=[common], help="List runs for one version tag (detailed tables).")
    runs_p.add_argument("--version", "-v", required=True, help="Filter substring, e.g. w6, w7, w6_nav")
    runs_p.add_argument("--algo", "-a", default=None, choices=list(ALGOS) + [a.upper() for a in ALGOS])
    runs_p.add_argument("--top", type=int, default=0, help="Top-N runs per algo (0 = all)")
    runs_p.set_defaults(func=cmd_runs)

    ca = sub.add_parser(
        "compare-algos",
        parents=[common],
        help="Same version: best run per algorithm (wide tables per phase).",
    )
    ca.add_argument("--version", "-v", required=True)
    ca.add_argument("--algo", "-a", default=None, choices=list(ALGOS) + [a.upper() for a in ALGOS])
    ca.set_defaults(func=cmd_compare_algos)

    cv = sub.add_parser(
        "compare-versions",
        parents=[common],
        help="Metrics × algorithms × several version filters (e.g. w6 w7).",
    )
    cv.add_argument(
        "--versions",
        "-V",
        nargs="+",
        required=True,
        metavar="VER",
        help="Version substrings as in run names (e.g. w6 w7)",
    )
    cv.add_argument("--algo", "-a", default=None, choices=list(ALGOS) + [a.upper() for a in ALGOS])
    cv.set_defaults(func=cmd_compare_versions)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
