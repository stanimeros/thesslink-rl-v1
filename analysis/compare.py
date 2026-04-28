"""Wide tables: compare algorithms within one version, or versions side by side."""

from __future__ import annotations

from .config import ALGOS, FULL_METRICS, NAV_METRICS, NEG_METRICS
from .metrics_display import best_run_per_algo, fmt, metric_value
from .partition import RunPartition, partition_runs
from .wandb_runs import fetch_runs


def _divider(title: str, w: int = 88) -> None:
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def _pick_run(runs: list, metrics: list[tuple[str, str | None]], algo: str):
    best = best_run_per_algo(runs, metrics)
    return best.get(algo)


def print_algo_comparison(version: str, parts: RunPartition) -> None:
    """Best run per algorithm for each bucket (wide grid)."""
    _divider(f"COMPARE ALGORITHMS  |  version={version}  |  best run per algo (by primary metric)")

    sections = [
        ("FULL EPISODE (neg → nav)", parts.full, FULL_METRICS),
        ("NAVIGATION (nav-only env)", parts.nav, NAV_METRICS),
        ("NEGOTIATION (neg-only env)", parts.neg, NEG_METRICS),
    ]

    val_w = 10
    algo_w = 7

    for title, runs, metrics in sections:
        if not runs:
            continue
        print(f"\n--- {title} ---")
        best = best_run_per_algo(runs, metrics)
        hdr_cells = [f"{lbl:>{val_w}}" for lbl, _ in metrics]
        header = f"  {'algo':<{algo_w}}" + "  ".join(hdr_cells)
        print(header)
        print("  " + "─" * (algo_w + (val_w + 2) * len(metrics)))
        for a in ALGOS:
            if a not in best:
                continue
            run = best[a]
            cells = [fmt(metric_value(run, lbl, key), val_w) for lbl, key in metrics]
            print(f"  {a.upper():<{algo_w}}" + "  ".join(cells))


def _print_metric_pivot(
    title: str,
    versions: list[str],
    per_v: dict[str, RunPartition],
    bucket_attr: str,
    metrics: list[tuple[str, str | None]],
    missing_note: str = "",
) -> None:
    buckets = {v: getattr(per_v[v], bucket_attr) for v in versions}
    if not any(buckets.values()):
        return

    cw = max(12, max(len(v) for v in versions) + 2)
    print(f"\n--- {title} ---")
    if missing_note:
        print(f"  ({missing_note})")

    for lbl, key in metrics:
        if lbl == "q_on_win":
            continue

        has_any = False
        for v in versions:
            for a in ALGOS:
                run = _pick_run(buckets[v], metrics, a)
                if run is None:
                    continue
                if metric_value(run, lbl, key) is not None:
                    has_any = True
                    break
            if has_any:
                break
        if not has_any:
            continue

        print(f"\n  {lbl}")
        hdr = f"  {'algo':<7}" + "".join(f"{v:>{cw}}" for v in versions)
        print(hdr)
        print("  " + "─" * (7 + cw * len(versions)))
        for a in ALGOS:
            cells = []
            anyv = False
            for v in versions:
                run = _pick_run(buckets[v], metrics, a)
                if run is None:
                    cells.append(f"{'—':>{cw}}")
                    continue
                mv = metric_value(run, lbl, key)
                if mv is None:
                    cells.append(f"{'—':>{cw}}")
                else:
                    anyv = True
                    cells.append(f"{mv:>{cw}.3f}")
            if anyv:
                print(f"  {a.upper():<7}" + "".join(cells))


def print_version_comparison(
    api,
    entity: str,
    project: str,
    versions: list[str],
    state: str,
    algo_filter: str | None,
) -> None:
    """Pivot tables: metrics × algos × version filters."""
    _divider(
        f"COMPARE VERSIONS  |  {' vs '.join(versions)}  |  state={state}",
        w=92,
    )

    per_v: dict[str, RunPartition] = {}
    for v in versions:
        runs = fetch_runs(api, entity, project, v, algo_filter, state)
        per_v[v] = partition_runs(runs)

    print(
        "\n  Legend: nav-only / neg-only rows use dedicated env runs when present.\n"
        "  Full-episode metrics come from neg→nav runs; interpret cross-version deltas carefully."
    )

    _print_metric_pivot(
        "FULL EPISODE (full neg→nav runs)",
        versions,
        per_v,
        "full",
        FULL_METRICS,
        "Versions with no full runs show as —.",
    )

    _print_metric_pivot(
        "NAVIGATION — nav-only env (e.g. w6_nav)",
        versions,
        per_v,
        "nav",
        NAV_METRICS,
    )

    if any(per_v[v].full for v in versions):
        _print_metric_pivot(
            "NAVIGATION — test_navigation_quality_mean from FULL runs (nav segment, not nav-only env)",
            versions,
            per_v,
            "full",
            [("nav_q", "test_navigation_quality_mean")],
        )

    _print_metric_pivot(
        "NEGOTIATION — neg-only env (e.g. w6_neg)",
        versions,
        per_v,
        "neg",
        NEG_METRICS,
    )

    neg_from_full = [m for m in FULL_METRICS if m[0] in ("neg_q", "neg_agree", "neg_len")]
    _print_metric_pivot(
        "NEGOTIATION — metrics from FULL runs (phase 1 segment)",
        versions,
        per_v,
        "full",
        neg_from_full,
    )
