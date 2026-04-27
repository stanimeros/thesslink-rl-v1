#!/usr/bin/env python3
"""Fetch and display W&B metrics for a selected ThessLink RL version.

Usage:
    python analysis.py --version w5
    python analysis.py --version w6 --state running
    python analysis.py --version w6 --algo ippo
    python analysis.py --version w6 --entity my-entity --project my-project
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "aid26006-university-of-macedonia")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "thesslink-rl")

ALGOS = ("iql", "qmix", "mappo", "ippo", "maddpg")

NAV_METRICS = [
    ("nav_quality",  "test_navigation_quality_mean"),
    ("q_on_win",     None),   # computed: nav_quality / battle_won
    ("nav_length",   "test_navigation_length_mean"),
    ("battle_won",   "test_battle_won_mean"),
]

NEG_METRICS = [
    ("neg_quality",  "test_negotiation_quality_mean"),
    ("neg_agreed",   "test_negotiation_agreed_mean"),
    ("neg_length",   "test_negotiation_length_mean"),
    ("battle_won",   "test_battle_won_mean"),
]

STATE_ICON = {"running": "⟳", "finished": "✓", "crashed": "✗", "failed": "✗"}

_history_cache: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse W&B runs for a ThessLink version.")
    p.add_argument("--version", "-v", required=True,
                   help="Env version tag to filter (e.g. w5, w6, v6_nav)")
    p.add_argument("--algo", "-a", default=None,
                   choices=list(ALGOS) + [a.upper() for a in ALGOS],
                   help="Restrict to a single algorithm (default: all)")
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--project", default=WANDB_PROJECT)
    p.add_argument("--state", default="all",
                   choices=["finished", "running", "crashed", "failed", "all"],
                   help="Run state filter (default: all)")
    p.add_argument("--top", type=int, default=0,
                   help="Show top-N runs per algo by primary metric (0 = all)")
    return p.parse_args()


def fetch_runs(api, entity, project, version, algo, state):
    filters: dict = {}
    if state != "all":
        filters["state"] = state
    all_runs = api.runs(f"{entity}/{project}", filters=filters)

    matched = []
    vl = version.lower()
    for run in all_runs:
        tags_l      = [t.lower() for t in (run.tags or [])]
        group_l     = (run.group or "").lower()
        name_l      = run.name.lower()
        config_env  = (run.config.get("env_config") or run.config.get("env", "") or "").lower()
        combined    = " ".join([name_l, group_l, config_env] + tags_l)

        if vl not in combined:
            continue

        if algo:
            al = algo.lower()
            algo_match = (
                al in (run.config.get("name") or "").lower()
                or al in name_l
                or any(al == t for t in tags_l)
                or al in group_l
            )
            if not algo_match:
                continue

        matched.append(run)
    return matched


def _detect_algo(run) -> str | None:
    for a in ALGOS:
        if a in (run.name or "").lower() or a in (run.group or "").lower():
            return a
    return None


def _progress_bar(steps: int | None, t_max: int | None, width: int = 12) -> str:
    if not steps or not t_max:
        return f"{'?':^{width + 7}}"
    pct = min(1.0, steps / t_max)
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct * 100:3.0f}%"


def _val(run, key: str) -> float | None:
    if key is None:
        return None
    return run.summary.get(key)


def _fmt(v: float | None, w: int) -> str:
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.3f}"


def _get_history(run, key: str, samples: int = 40) -> list:
    """Fetch sampled metric history for a run (cached per run+key)."""
    cache_key = (run.id, key)
    if cache_key not in _history_cache:
        try:
            rows = run.history(samples=samples, keys=[key], pandas=False)
            vals = [row[key] for row in rows if row.get(key) is not None]
        except Exception:
            vals = []
        _history_cache[cache_key] = vals
    return _history_cache[cache_key]


def _trend_indicator(run, key: str, window_frac: float = 0.25, min_delta: float = 0.01) -> str:
    """Single-char trend over the most recent quarter of training history.

    ↑  improving  (mean of last window > mean of prior window by > min_delta)
    ~  plateau    (change within ±min_delta)
    ↓  declining
    ?  not enough data
    """
    if key is None:
        return "?"
    vals = _get_history(run, key)
    if len(vals) < 6:
        return "?"
    n = max(2, int(len(vals) * window_frac))
    recent  = vals[-n:]
    earlier = vals[-2 * n: -n]
    if len(earlier) < 2:
        return "?"
    delta = sum(recent) / len(recent) - sum(earlier) / len(earlier)
    if   delta >  min_delta: return "↑"
    elif delta < -min_delta: return "↓"
    else:                    return "~"


def print_section(title: str, runs: list, metrics: list, top_n: int) -> None:
    if not runs:
        return

    # Group by algo, sort within group by first real metric descending
    by_algo: dict[str, list] = defaultdict(list)
    ungrouped = []
    for run in runs:
        a = _detect_algo(run)
        (by_algo[a] if a else ungrouped).append(run)

    groups = [(a, by_algo[a]) for a in ALGOS if by_algo[a]]
    if ungrouped:
        groups.append((None, ungrouped))

    # Column widths
    name_w  = max(len(r.name) for r in runs) + 1
    name_w  = max(name_w, 18)
    prog_w  = 19   # "[████████████]  99%"
    val_w   = 10
    algo_w  = 7
    icon_w  = 2
    trend_w = 2

    # Primary metric key for trend detection (first with a real W&B key)
    primary_key = next((key for _, key in metrics if key is not None), None)

    labels = [m[0] for m in metrics]
    header = (f"  {'algo':<{algo_w}} {'run name':<{name_w}} {'progress':<{prog_w}}"
              + "".join(f"{l:>{val_w}}" for l in labels)
              + f"  {'tr':<{trend_w}}  {'st':<{icon_w}}")

    divider_w = algo_w + name_w + prog_w + val_w * len(metrics) + icon_w + trend_w + 7
    print(f"\n{'═' * divider_w}")
    print(f"  {title}")
    print(f"{'═' * divider_w}")
    print(header)

    for algo_name, algo_runs in groups:
        alabel = algo_name.upper() if algo_name else "???"

        # Sort by first numeric metric (descending)
        def sort_key(r):
            for lbl, key in metrics:
                if key is not None:
                    v = _val(r, key)
                    if v is not None:
                        return v
            return float("-inf")

        algo_runs = sorted(algo_runs, key=sort_key, reverse=True)
        if top_n:
            algo_runs = algo_runs[:top_n]

        print("  " + "─" * (divider_w - 2))

        for run in algo_runs:
            steps = run.summary.get("t_env") or run.summary.get("_step")
            t_max = run.config.get("t_max")
            prog  = _progress_bar(steps, t_max)
            icon  = STATE_ICON.get(run.state, "?")

            trend = _trend_indicator(run, primary_key)
            row = f"  {alabel:<{algo_w}} {run.name:<{name_w}} {prog:<{prog_w}}"
            for lbl, key in metrics:
                if lbl == "q_on_win":
                    nq = _val(run, "test_navigation_quality_mean")
                    bw = _val(run, "test_battle_won_mean")
                    v  = (nq / bw) if (nq is not None and bw and bw > 0) else None
                else:
                    v = _val(run, key)
                row += _fmt(v, val_w)
            row += f"  {trend:<{trend_w}}  {icon}"
            print(row)

        # Best row if multiple runs
        if len(algo_runs) > 1:
            row = f"  {'':>{algo_w}} {'best':<{name_w}} {'':>{prog_w}}"
            for lbl, key in metrics:
                if lbl == "q_on_win":
                    vals = []
                    for r in algo_runs:
                        nq = _val(r, "test_navigation_quality_mean")
                        bw = _val(r, "test_battle_won_mean")
                        if nq is not None and bw and bw > 0:
                            vals.append(nq / bw)
                else:
                    vals = [_val(r, key) for r in algo_runs if _val(r, key) is not None]
                row += (_fmt(max(vals), val_w) if vals else f"{'—':>{val_w}}")
            print(row)

    print()


def main() -> None:
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    print(f"W&B  entity={args.entity}  project={args.project}")
    print(f"Fetching runs: version={args.version}"
          + (f"  algo={args.algo}" if args.algo else "")
          + f"  state={args.state} …")

    runs = fetch_runs(api, args.entity, args.project, args.version, args.algo, args.state)

    if not runs:
        print(f"\nNo runs found for version={args.version!r}.")
        print("Hint: check that the version string appears in run name, tags, group, or env_config.")
        sys.exit(0)

    print(f"Found {len(runs)} run(s).\n")

    nav_runs = [r for r in runs if "nav" in r.name.lower()]
    neg_runs = [r for r in runs if "neg" in r.name.lower()]
    other    = [r for r in runs if r not in nav_runs and r not in neg_runs]

    print_section("NAVIGATION  —  quality / quality-on-win / length / battle_won",
                  nav_runs, NAV_METRICS, args.top)
    print_section("NEGOTIATION  —  quality / agreed / length / battle_won",
                  neg_runs, NEG_METRICS, args.top)
    if other:
        print_section("OTHER", other,
                      [("return", "test_return_mean"), ("total_ret", "test_total_return_mean"),
                       ("battle_won", "test_battle_won_mean")], args.top)


if __name__ == "__main__":
    main()
