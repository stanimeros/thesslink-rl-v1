#!/usr/bin/env python3
"""Fetch and display W&B metrics for a selected ThessLink RL version.

Usage:
    python analysis.py --version v7
    python analysis.py --version v7 --algo iql
    python analysis.py --version v7 --metric test_return_mean
    python analysis.py --version v7 --entity my-entity --project my-project
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "aid26006-university-of-macedonia")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "thesslink-rl")

ALGOS = ("iql", "qmix", "mappo", "ippo", "maddpg")

KEY_METRICS = [
    "test_return_mean",
    "test_total_return_mean",
    "test_return_std",
    "test_battle_won_mean",
    "test_negotiation_agreed_mean",
    "test_negotiation_quality_mean",
    "test_negotiation_length_mean",
    "test_navigation_quality_mean",
    "test_navigation_length_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse W&B runs for a ThessLink version.")
    p.add_argument("--version", "-v", required=True,
                   help="Env version tag to filter (e.g. v7, v4_neg, v6_nav)")
    p.add_argument("--algo", "-a", default=None,
                   choices=list(ALGOS) + [a.upper() for a in ALGOS],
                   help="Restrict to a single algorithm (default: all)")
    p.add_argument("--metric", "-m", default=None,
                   help="Extra metric key to include in the summary")
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--project", default=WANDB_PROJECT)
    p.add_argument("--state", default="all",
                   choices=["finished", "running", "crashed", "failed", "all"],
                   help="Run state filter (default: all)")
    p.add_argument("--top", type=int, default=0,
                   help="Show the top-N runs per algo by test_return_mean (0 = all)")
    return p.parse_args()


def fetch_runs(api, entity: str, project: str, version: str, algo: str | None, state: str):
    """Return W&B runs matching the version tag and optional algo filter."""
    filters: dict = {}
    if state != "all":
        filters["state"] = state

    all_runs = api.runs(f"{entity}/{project}", filters=filters)

    matched = []
    version_lower = version.lower()
    for run in all_runs:
        tags_lower = [t.lower() for t in (run.tags or [])]
        group_lower = (run.group or "").lower()
        name_lower = run.name.lower()
        config_env = (run.config.get("env_config") or run.config.get("env", "") or "").lower()

        version_match = (
            version_lower in tags_lower
            or version_lower in group_lower
            or version_lower in name_lower
            or version_lower in config_env
        )
        if not version_match:
            continue

        if algo:
            algo_l = algo.lower()
            algo_match = (
                algo_l in (run.config.get("name") or "").lower()
                or algo_l in name_lower
                or any(algo_l == t for t in tags_lower)
                or algo_l in group_lower
            )
            if not algo_match:
                continue

        matched.append(run)

    return matched


def summarise_runs(runs, extra_metric: str | None, top_n: int) -> None:
    metrics_to_show = list(KEY_METRICS)
    if extra_metric and extra_metric not in metrics_to_show:
        metrics_to_show.append(extra_metric)

    by_algo: dict[str, list] = defaultdict(list)
    ungrouped = []

    for run in runs:
        detected = None
        for algo in ALGOS:
            if algo in (run.name or "").lower() or algo in (run.group or "").lower():
                detected = algo
                break
        (by_algo[detected] if detected else ungrouped).append(run)

    groups = [(a, by_algo[a]) for a in ALGOS if by_algo[a]]
    if ungrouped:
        groups.append((None, ungrouped))

    for algo_name, algo_runs in groups:
        label = algo_name.upper() if algo_name else "UNKNOWN"

        def sort_key(r):
            v = r.summary.get("test_return_mean") or r.summary.get("test_total_return_mean")
            return v if v is not None else float("-inf")

        algo_runs.sort(key=sort_key, reverse=True)
        if top_n:
            algo_runs = algo_runs[:top_n]

        print(f"\n{'─' * 70}")
        print(f"  {label}  ({len(algo_runs)} run{'s' if len(algo_runs) != 1 else ''})")
        print(f"{'─' * 70}")

        col_w = 32
        state_w = 10
        val_w = 12
        header = f"  {'Run name':<{col_w}}{'state':<{state_w}}" + "".join(f"{m.split('_',1)[-1]:>{val_w}}" for m in metrics_to_show)
        print(header)
        print("  " + "-" * (col_w + state_w + val_w * len(metrics_to_show)))

        for run in algo_runs:
            row = f"  {run.name:<{col_w}}{run.state:<{state_w}}"
            for m in metrics_to_show:
                val = run.summary.get(m)
                if val is None:
                    row += f"{'—':>{val_w}}"
                else:
                    row += f"{val:>{val_w}.4f}"
            print(row)

        if len(algo_runs) > 1:
            print("  " + "-" * (col_w + state_w + val_w * len(metrics_to_show)))
            best_row = f"  {'best':>{col_w}}{'':>{state_w}}"
            for m in metrics_to_show:
                vals = [r.summary.get(m) for r in algo_runs if r.summary.get(m) is not None]
                best_row += f"{max(vals):>{val_w}.4f}" if vals else f"{'—':>{val_w}}"
            print(best_row)


def main() -> None:
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    entity, project = args.entity, args.project
    print(f"W&B  entity={entity}  project={project}")
    print(f"Fetching runs for version={args.version}"
          + (f"  algo={args.algo}" if args.algo else "")
          + f"  state={args.state} …")

    runs = fetch_runs(api, entity, project, args.version, args.algo, args.state)

    if not runs:
        print(f"\nNo runs found for version={args.version!r}.")
        print("Hint: check that the version string appears in run tags, group, or name.")
        sys.exit(0)

    print(f"Found {len(runs)} run(s).")
    summarise_runs(runs, extra_metric=args.metric, top_n=args.top)
    print()


if __name__ == "__main__":
    main()
