"""Summary values, W&B history sampling, and section tables."""

from __future__ import annotations

from collections import defaultdict

from .config import ALGOS, STATE_ICON
from .wandb_runs import detect_algo

_history_cache: dict = {}


def clear_history_cache() -> None:
    _history_cache.clear()


def val(run, key: str | None) -> float | None:
    if key is None:
        return None
    return run.summary.get(key)


def fmt(v: float | None, w: int) -> str:
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.3f}"


def progress_bar(steps: int | None, t_max: int | None, width: int = 12) -> str:
    if not steps or not t_max:
        return f"{'?':^{width + 7}}"
    pct = min(1.0, steps / t_max)
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct * 100:3.0f}%"


def get_history(run, key: str, samples: int = 40) -> list:
    cache_key = (run.id, key)
    if cache_key not in _history_cache:
        try:
            rows = run.history(samples=samples, keys=[key], pandas=False)
            vals = [row[key] for row in rows if row.get(key) is not None]
        except Exception:
            vals = []
        _history_cache[cache_key] = vals
    return _history_cache[cache_key]


def trend_indicator(run, key: str | None, window_frac: float = 0.25, min_delta: float = 0.01) -> str:
    if key is None:
        return "?"
    vals = get_history(run, key)
    if len(vals) < 6:
        return "?"
    n = max(2, int(len(vals) * window_frac))
    recent = vals[-n:]
    earlier = vals[-2 * n : -n]
    if len(earlier) < 2:
        return "?"
    delta = sum(recent) / len(recent) - sum(earlier) / len(earlier)
    if delta > min_delta:
        return "↑"
    if delta < -min_delta:
        return "↓"
    return "~"


def metric_value(run, lbl: str, key: str | None) -> float | None:
    if lbl == "q_on_win":
        nq = val(run, "test_navigation_quality_mean")
        bw = val(run, "test_battle_won_mean")
        return (nq / bw) if (nq is not None and bw and bw > 0) else None
    return val(run, key)


def print_section(title: str, runs: list, metrics: list[tuple[str, str | None]], top_n: int) -> None:
    if not runs:
        return

    by_algo: dict[str | None, list] = defaultdict(list)
    ungrouped = []
    for run in runs:
        a = detect_algo(run)
        (by_algo[a] if a else ungrouped).append(run)

    groups = [(a, by_algo[a]) for a in ALGOS if by_algo[a]]
    if ungrouped:
        groups.append((None, ungrouped))

    name_w = max(len(r.name) for r in runs) + 1
    name_w = max(name_w, 18)
    prog_w = 19
    val_w = 10
    algo_w = 7
    icon_w = 2
    trend_w = 2

    primary_key = next((key for _, key in metrics if key is not None), None)

    labels = [m[0] for m in metrics]
    header = (
        f"  {'algo':<{algo_w}} {'run name':<{name_w}} {'progress':<{prog_w}}"
        + "".join(f"{l:>{val_w}}" for l in labels)
        + f"  {'tr':<{trend_w}}  {'st':<{icon_w}}"
    )

    divider_w = algo_w + name_w + prog_w + val_w * len(metrics) + icon_w + trend_w + 7
    print(f"\n{'═' * divider_w}")
    print(f"  {title}")
    print(f"{'═' * divider_w}")
    print(header)

    for algo_name, algo_runs in groups:
        alabel = algo_name.upper() if algo_name else "???"

        def sort_key(r):
            for _, key in metrics:
                if key is not None:
                    v = val(r, key)
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
            prog = progress_bar(steps, t_max)
            icon = STATE_ICON.get(run.state, "?")

            trend = trend_indicator(run, primary_key)
            row = f"  {alabel:<{algo_w}} {run.name:<{name_w}} {prog:<{prog_w}}"
            for lbl, key in metrics:
                v = metric_value(run, lbl, key)
                row += fmt(v, val_w)
            row += f"  {trend:<{trend_w}}  {icon}"
            print(row)

        if len(algo_runs) > 1:
            row = f"  {'':>{algo_w}} {'best':<{name_w}} {'':>{prog_w}}"
            for lbl, key in metrics:
                if lbl == "q_on_win":
                    vals = []
                    for r in algo_runs:
                        nq = val(r, "test_navigation_quality_mean")
                        bw = val(r, "test_battle_won_mean")
                        if nq is not None and bw and bw > 0:
                            vals.append(nq / bw)
                else:
                    vals = [val(r, key) for r in algo_runs if val(r, key) is not None]
                row += fmt(max(vals), val_w) if vals else f"{'—':>{val_w}}"
            print(row)

    print()


def best_run_per_algo(
    runs: list, metrics: list[tuple[str, str | None]]
) -> dict[str, object]:
    """One run per algorithm (best by first available summary metric)."""
    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        a = detect_algo(run)
        if a:
            by_algo[a].append(run)
    out: dict[str, object] = {}

    def sort_key(r):
        for _, key in metrics:
            if key is not None:
                v = val(r, key)
                if v is not None:
                    return v
        return float("-inf")

    for a in ALGOS:
        if a not in by_algo:
            continue
        ranked = sorted(by_algo[a], key=sort_key, reverse=True)
        out[a] = ranked[0]
    return out
