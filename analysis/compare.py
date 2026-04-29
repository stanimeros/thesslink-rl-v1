"""print_runs (single version) and print_compare (specialist vs full)."""

from __future__ import annotations

from .config import ALGOS, FULL_METRICS, NAV_METRICS, NEG_METRICS
from .metrics_display import (
    best_run_per_algo,
    last_run_per_algo,
    metric_cell,
    metric_value,
    peak_last_for_key,
    peak_last_q_on_win,
)
from .partition import partition_runs
from .wandb_runs import fetch_runs

# Key quality metrics shown in both modes.
QUALITY_METRICS: list[tuple[str, str | None]] = [
    ("neg_quality", "test_negotiation_quality_mean"),
    ("nav_quality", "test_navigation_quality_mean"),
    ("neg_len", "test_negotiation_length_mean"),
    ("nav_len", "test_navigation_length_mean"),
    ("battle_won", "test_battle_won_mean"),
]

_MS = "history"  # fixed metrics source


def _divider(title: str, w: int = 92) -> None:
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def _pick(runs: list, algo: str, *, pick: str) -> object | None:
    if pick == "last":
        return last_run_per_algo(runs).get(algo)
    return best_run_per_algo(runs, QUALITY_METRICS, metrics_source=_MS).get(algo)


def _table(
    title: str,
    col_headers: list[str],
    rows: list[tuple[str, list[object | None]]],  # (algo_label, [run | None, ...])
    metrics: list[tuple[str, str | None]],
) -> None:
    val_w = 14
    algo_w = 7
    col_w = max(val_w, max(len(h) for h in col_headers) + 2)

    print(f"\n--- {title} ---")
    hdr = f"  {'algo':<{algo_w}}" + "".join(f"{h:>{col_w * len(metrics)}}" for h in col_headers)
    # Better: one column-group per col_header, sub-columns per metric
    # Simple flat: metric labels row, then data
    metric_hdr = f"  {'':>{algo_w}}" + "".join(
        "".join(f"{lbl:>{val_w}}" for lbl, _ in metrics)
        for _ in col_headers
    )
    col_sep_hdr = f"  {'algo':<{algo_w}}"
    for h in col_headers:
        col_sep_hdr += f"  [{h}]" + " " * (val_w * len(metrics) - len(h) - 4)
    print(col_sep_hdr)
    print(metric_hdr)
    print("  " + "─" * (algo_w + val_w * len(metrics) * len(col_headers) + 2 * len(col_headers)))

    for algo_lbl, run_list in rows:
        any_data = any(r is not None for r in run_list)
        if not any_data:
            continue
        row = f"  {algo_lbl:<{algo_w}}"
        for run in run_list:
            if run is None:
                row += "".join(f"{'—':>{val_w}}" for _ in metrics)
            else:
                row += "".join(
                    metric_cell(run, lbl, key, metrics_source=_MS, val_w=val_w)
                    for lbl, key in metrics
                )
        print(row)


def print_runs(
    api,
    entity: str,
    project: str,
    version: str,
    state: str,
    algo_filter: str | None,
    *,
    pick: str = "last",
) -> None:
    """Show last (or best) run per algo for a version with quality metrics."""
    runs = fetch_runs(api, entity, project, version, algo_filter, state)
    if not runs:
        print(f"No runs found for version={version!r}.")
        return

    parts = partition_runs(runs)
    _divider(f"RUNS  |  version={version}  |  pick={pick}  |  state={state}")

    sections = [
        ("FULL (neg→nav)", parts.full, FULL_METRICS),
        ("NAVIGATION (nav-only)", parts.nav, NAV_METRICS),
        ("NEGOTIATION (neg-only)", parts.neg, NEG_METRICS),
    ]

    val_w = 13
    algo_w = 7
    name_w = max((len(r.name) for r in runs), default=20) + 1

    for title, bucket, metrics in sections:
        if not bucket:
            continue
        print(f"\n--- {title} ---")
        labels = [lbl for lbl, _ in metrics]
        print(f"  {'algo':<{algo_w}} {'run':<{name_w}}" + "".join(f"{l:>{val_w}}" for l in labels))
        print("  " + "─" * (algo_w + name_w + val_w * len(metrics)))

        picked = {a: _pick(bucket, a, pick=pick) for a in ALGOS}
        for a in ALGOS:
            run = picked.get(a)
            if run is None:
                continue
            cells = "".join(metric_cell(run, lbl, key, metrics_source=_MS, val_w=val_w) for lbl, key in metrics)
            print(f"  {a.upper():<{algo_w}} {run.name:<{name_w}}{cells}")

    print()


def print_compare(
    api,
    entity: str,
    project: str,
    specialist_ver: str,
    full_ver: str,
    state: str,
    algo_filter: str | None,
    *,
    pick: str = "last",
) -> None:
    """Compare specialist neg+nav (ver1) vs full episode (ver2) on quality metrics."""
    runs_sp = fetch_runs(api, entity, project, specialist_ver, algo_filter, state)
    runs_fu = fetch_runs(api, entity, project, full_ver, algo_filter, state)
    sp = partition_runs(runs_sp)
    fu = partition_runs(runs_fu)

    _divider(
        f"COMPARE  |  {specialist_ver} (neg+nav specialists)  vs  {full_ver} (full)  "
        f"|  pick={pick}  state={state}"
    )

    val_w = 13
    algo_w = 7

    # --- NEGOTIATION quality: neg-only env vs full run phase-1 ---
    neg_metrics: list[tuple[str, str | None]] = [
        ("neg_quality", "test_negotiation_quality_mean"),
        ("neg_len", "test_negotiation_length_mean"),
        ("battle_won", "test_battle_won_mean"),
    ]
    neg_cols = [f"{specialist_ver}_neg", f"{full_ver}_full"]

    # --- NAVIGATION quality: nav-only env vs full run nav segment ---
    nav_metrics: list[tuple[str, str | None]] = [
        ("nav_quality", "test_navigation_quality_mean"),
        ("nav_len", "test_navigation_length_mean"),
        ("battle_won", "test_battle_won_mean"),
    ]
    nav_cols = [f"{specialist_ver}_nav", f"{full_ver}_full"]

    for title, bucket_sp, bucket_fu, metrics, cols in [
        (f"NEGOTIATION  —  {specialist_ver}_neg vs {full_ver}_full (phase-1)",
         sp.neg, fu.full, neg_metrics, neg_cols),
        (f"NAVIGATION  —  {specialist_ver}_nav vs {full_ver}_full (nav segment)",
         sp.nav, fu.full, nav_metrics, nav_cols),
    ]:
        if not bucket_sp and not bucket_fu:
            continue
        labels = [lbl for lbl, _ in metrics]
        col_w = val_w
        print(f"\n--- {title} ---")
        # header: algo | [col1] metrics... | [col2] metrics...
        col_hdr = f"  {'algo':<{algo_w}}"
        for col in cols:
            col_hdr += f"  [{col}]" + " " * max(0, col_w * len(metrics) - len(col) - 4)
        print(col_hdr)
        metric_hdr = f"  {'':>{algo_w}}" + "".join(
            "".join(f"{l:>{col_w}}" for l in labels) for _ in cols
        )
        print(metric_hdr)
        print("  " + "─" * (algo_w + col_w * len(metrics) * 2 + 4))

        picked_sp = {a: _pick(bucket_sp, a, pick=pick) for a in ALGOS}
        picked_fu = {a: _pick(bucket_fu, a, pick=pick) for a in ALGOS}

        for a in ALGOS:
            r_sp = picked_sp.get(a)
            r_fu = picked_fu.get(a)
            if r_sp is None and r_fu is None:
                continue
            row = f"  {a.upper():<{algo_w}}"
            for run in (r_sp, r_fu):
                if run is None:
                    row += "".join(f"{'—':>{col_w}}" for _ in metrics)
                else:
                    row += "".join(
                        metric_cell(run, lbl, key, metrics_source=_MS, val_w=col_w)
                        for lbl, key in metrics
                    )
            print(row)

    # --- End-to-end (full run only) ---
    if fu.full:
        e2e_metrics: list[tuple[str, str | None]] = [
            ("neg_quality", "test_negotiation_quality_mean"),
            ("nav_quality", "test_navigation_quality_mean"),
            ("neg_len", "test_negotiation_length_mean"),
            ("nav_len", "test_navigation_length_mean"),
            ("battle_won", "test_battle_won_mean"),
        ]
        print(f"\n--- END-TO-END  —  {full_ver}_full ---")
        labels = [lbl for lbl, _ in e2e_metrics]
        print(f"  {'algo':<{algo_w}}" + "".join(f"{l:>{val_w}}" for l in labels))
        print("  " + "─" * (algo_w + val_w * len(e2e_metrics)))
        picked_fu_e2e = {a: _pick(fu.full, a, pick=pick) for a in ALGOS}
        for a in ALGOS:
            run = picked_fu_e2e.get(a)
            if run is None:
                continue
            cells = "".join(metric_cell(run, lbl, key, metrics_source=_MS, val_w=val_w) for lbl, key in e2e_metrics)
            print(f"  {a.upper():<{algo_w}}{cells}")

    print()
