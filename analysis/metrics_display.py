"""Summary values, W&B history scan (peaks / best logged points), and section tables."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .config import ALGOS, HISTORY_SAMPLES, STATE_ICON, all_logged_test_metric_keys, metric_objective
from .wandb_runs import detect_algo

_history_cache: dict = {}
# One sampled ``run.history`` per run id (uniform samples; fast on long / live runs).
_run_series: dict[str, dict[str, list[float]]] = {}
_run_q_on_win_ratios: dict[str, list[float]] = {}


def clear_history_cache() -> None:
    _history_cache.clear()
    _run_series.clear()
    _run_q_on_win_ratios.clear()


def val(run, key: str | None) -> float | None:
    if key is None:
        return None
    return run.summary.get(key)


def _wandb_time_to_ts(v: object) -> float:
    if hasattr(v, "timestamp") and callable(getattr(v, "timestamp")):
        try:
            return float(v.timestamp())  # type: ignore[no-any-return]
        except Exception:
            pass
    if isinstance(v, str) and v:
        s = v.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            pass
    return 0.0


def _run_recency_ts(run: object) -> float:
    """Latest known wall time for a W&B run (for ``last run per algo``)."""
    best = 0.0
    for attr in ("updated_at", "heartbeatAt", "heartbeat_at", "created_at"):
        t = _wandb_time_to_ts(getattr(run, attr, None))
        if t:
            best = max(best, t)
    return best


def _ensure_run_series(run) -> dict[str, list[float]]:
    """Populate per-run series from a **uniform sample** of W&B steps (``run.history``).

    ``scan_history`` is accurate but can take minutes on multi-million-step runs;
    peaks/minima here are **approximate** (best logged point in the sample).
    """
    rid = run.id
    if rid in _run_series:
        return _run_series[rid]
    keys = all_logged_test_metric_keys()
    series: dict[str, list[float]] = {k: [] for k in keys}
    q_ratios: list[float] = []
    try:
        key_list = sorted(keys)
        rows = run.history(samples=HISTORY_SAMPLES, keys=key_list, pandas=False)
        for row in rows:
            for k in keys:
                v = row.get(k)
                if isinstance(v, (int, float)) and v == v:
                    series[k].append(float(v))
            nq = row.get("test_navigation_quality_mean")
            bw = row.get("test_battle_won_mean")
            if (
                isinstance(nq, (int, float))
                and nq == nq
                and isinstance(bw, (int, float))
                and bw == bw
                and bw > 0
            ):
                q_ratios.append(float(nq) / float(bw))
    except Exception:
        pass
    _run_series[rid] = series
    _run_q_on_win_ratios[rid] = q_ratios
    return series


def _q_on_win_ratios(run) -> list[float]:
    _ensure_run_series(run)
    return _run_q_on_win_ratios.get(run.id, [])


def _peak_last_from_series(values: list[float], objective: str) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    last = values[-1]
    if objective == "min":
        peak = min(values)
    else:
        peak = max(values)
    return peak, last


def peak_last_for_key(run, key: str | None) -> tuple[float | None, float | None]:
    """(hist_peak, hist_last) from full W&B history for ``key``; empty → (None, None)."""
    if key is None:
        return None, None
    series = _ensure_run_series(run)
    vals = series.get(key) or []
    return _peak_last_from_series(vals, metric_objective(key))


def peak_last_q_on_win(run) -> tuple[float | None, float | None]:
    ratios = _q_on_win_ratios(run)
    if ratios:
        return _peak_last_from_series(ratios, "max")
    nq = val(run, "test_navigation_quality_mean")
    bw = val(run, "test_battle_won_mean")
    if nq is not None and bw and bw > 0:
        single = nq / bw
        return single, single
    return None, None


def fmt(v: float | None, w: int) -> str:
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>{w}.3f}"


def fmt_peak_last(peak: float | None, last: float | None, w: int) -> str:
    """Compact ``peak→last`` (W&B summary last); same values collapse to one number."""
    if peak is None and last is None:
        return f"{'—':>{w}}"
    if peak is None:
        return fmt(last, w)
    if last is None:
        return f"{peak:>{w}.3f}"[:w]
    if abs(peak - last) < 1e-9:
        return fmt(peak, w)
    s = f"{peak:.2f}→{last:.2f}"
    if len(s) <= w:
        return f"{s:>{w}}"
    return f"{s:>{w}}"[:w]


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


def metric_value(run, lbl: str, key: str | None, *, metrics_source: str = "history") -> float | None:
    """Scalar for sorting / filters (summary = W&B summary; history = best checkpoint in sampled curves).

    For ``metrics_source=history``, prefers the extremum (peak for max metrics, trough for lengths)
    over the final logged point, so late training collapse does not dominate scores.
    """
    if lbl == "q_on_win":
        if metrics_source == "summary":
            nq = val(run, "test_navigation_quality_mean")
            bw = val(run, "test_battle_won_mean")
            return (nq / bw) if (nq is not None and bw and bw > 0) else None
        peak, _last = peak_last_q_on_win(run)
        if peak is not None:
            return float(peak)
        nq = val(run, "test_navigation_quality_mean")
        bw = val(run, "test_battle_won_mean")
        return (nq / bw) if (nq is not None and bw and bw > 0) else None
    if metrics_source == "summary":
        return val(run, key)
    peak_h, last_h = peak_last_for_key(run, key)
    if peak_h is not None:
        return float(peak_h)
    last_summary = val(run, key)
    last = last_summary if last_summary is not None else last_h
    if last is not None:
        return last
    return val(run, key)


def metric_cell(
    run,
    lbl: str,
    key: str | None,
    *,
    metrics_source: str,
    val_w: int,
) -> str:
    """Table cell: either summary scalar or ``hist_peak→summary_last``."""
    if metrics_source == "summary":
        v = metric_value(run, lbl, key, metrics_source="summary")
        return fmt(v, val_w)
    if lbl == "q_on_win":
        peak, last = peak_last_q_on_win(run)
        if peak is None and last is None:
            nq = val(run, "test_navigation_quality_mean")
            bw = val(run, "test_battle_won_mean")
            if nq is not None and bw and bw > 0:
                last = nq / bw
                peak = last
        return fmt_peak_last(peak, last, val_w)
    if key is None:
        return f"{'—':>{val_w}}"
    peak_h, last_h = peak_last_for_key(run, key)
    last_summary = val(run, key)
    # Prefer W&B summary as "current" end state; fall back to history tail.
    last = last_summary if last_summary is not None else last_h
    peak = peak_h if peak_h is not None else last
    return fmt_peak_last(peak, last, val_w)


def print_section(
    title: str,
    runs: list,
    metrics: list[tuple[str, str | None]],
    top_n: int,
    *,
    metrics_source: str = "history",
) -> None:
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
    val_w = 12 if metrics_source == "history" else 10
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
    if metrics_source == "history":
        print(
            f"  Metrics: sampled best checkpoint (peak) → W&B summary (last); ~{HISTORY_SAMPLES} uniform "
            "``run.history`` steps per run (extrema are approximate; set ANALYSIS_HISTORY_SAMPLES for denser sampling). "
            "Length columns: lower peak is better. Use ``--metrics-source summary`` for final summary only."
        )
    print(f"{'═' * divider_w}")
    print(header)

    for algo_name, algo_runs in groups:
        alabel = algo_name.upper() if algo_name else "???"

        def sort_key(r: object) -> float:
            for lbl, key in metrics:
                if metrics_source == "history":
                    if lbl == "q_on_win":
                        pk, _ = peak_last_q_on_win(r)
                        if pk is not None:
                            return float(pk)
                        continue
                    if key is not None:
                        pk, _ = peak_last_for_key(r, key)
                        if pk is not None:
                            return float(pk)
                        continue
                else:
                    v = metric_value(r, lbl, key, metrics_source=metrics_source)
                    if v is not None:
                        return float(v)
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
                row += metric_cell(run, lbl, key, metrics_source=metrics_source, val_w=val_w)
            row += f"  {trend:<{trend_w}}  {icon}"
            print(row)

        if len(algo_runs) > 1:
            row = f"  {'':>{algo_w}} {'best':<{name_w}} {'':<{prog_w}}"
            for lbl, key in metrics:
                if lbl == "q_on_win":
                    vals = []
                    for r in algo_runs:
                        pk, _ = peak_last_q_on_win(r)
                        if pk is not None:
                            vals.append(pk)
                else:
                    vals = []
                    for r in algo_runs:
                        pk, _ = peak_last_for_key(r, key) if key else (None, None)
                        if pk is not None:
                            vals.append(pk)
                if not vals:
                    row += f"{'—':>{val_w}}"
                elif key and metric_objective(key) == "min":
                    row += fmt(min(vals), val_w)
                else:
                    row += fmt(max(vals), val_w)
            print(row)

    print()


def best_run_per_algo(
    runs: list,
    metrics: list[tuple[str, str | None]],
    *,
    metrics_source: str = "history",
) -> dict[str, object]:
    """One run per algorithm (best by first column: history peak or W&B summary)."""

    def primary_score(r: object) -> float:
        for lbl, key in metrics:
            if metrics_source == "history":
                if lbl == "q_on_win":
                    pk, _ = peak_last_q_on_win(r)
                    if pk is not None:
                        return float(pk)
                    continue
                if key is not None:
                    pk, _ = peak_last_for_key(r, key)
                    if pk is not None:
                        return float(pk)
                    continue
            else:
                if lbl == "q_on_win":
                    nq = val(r, "test_navigation_quality_mean")
                    bw = val(r, "test_battle_won_mean")
                    if nq is not None and bw and bw > 0:
                        return float(nq / bw)
                    continue
                if key is not None:
                    v = val(r, key)
                    if v is not None:
                        return float(v)
        return float("-inf")

    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        a = detect_algo(run)
        if a:
            by_algo[a].append(run)
    out: dict[str, object] = {}

    for a in ALGOS:
        if a not in by_algo:
            continue
        ranked = sorted(by_algo[a], key=primary_score, reverse=True)
        out[a] = ranked[0]
    return out


def last_run_per_algo(runs: list) -> dict[str, object]:
    """One run per algorithm: most recent by W&B ``updated_at`` / ``created_at`` (etc.)."""
    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        a = detect_algo(run)
        if a:
            by_algo[a].append(run)
    out: dict[str, object] = {}
    for a in ALGOS:
        if a not in by_algo:
            continue
        ranked = sorted(
            by_algo[a],
            key=lambda r: (_run_recency_ts(r), getattr(r, "id", "") or ""),
            reverse=True,
        )
        out[a] = ranked[0]
    return out
