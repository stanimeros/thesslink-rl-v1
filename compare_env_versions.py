#!/usr/bin/env python3
"""Compare ThessLink env versions on test metrics from Sacred ``metrics.json``.

Reads **AGR** and **GM** (golden-mean negotiation) from **negotiation-only** envs
(``*-neg``) and **REACH** from **navigation-only** envs (``*-nav``). **v2** is the
joint full task: all three metrics come from ``GridNegotiation-v2``.

**v3** / **v4** rows are **merged**: neg + nav (two specialist policies), one row
per algorithm.

Sacred layout::

    <results>/sacred/<algo>/thesslink_rl:thesslink/GridNegotiation-v<N>-…/…/metrics.json

Checks ``results/`` then ``epymarl/results/`` (same order as ``visualize.py``).

Examples::

    python compare_env_versions.py
    python compare_env_versions.py --csv comparison.csv
    python compare_env_versions.py --plots-dir plots/version_compare
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_constants_module() -> Any:
    """Load ``thesslink_rl/constants.py`` without importing ``thesslink_rl`` (no gymnasium)."""
    path = _SCRIPT_DIR / "thesslink_rl" / "constants.py"
    spec = importlib.util.spec_from_file_location("_thesslink_constants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load constants from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_c = _load_constants_module()
EPYMARL_DIR = _c.EPYMARL_DIR
RESULTS_DIR = _c.RESULTS_DIR
PLOTS_DIR: Path = _c.PLOTS_DIR
TRAINING_ALGOS: tuple[str, ...] = _c.TRAINING_ALGOS

VERSION_ORDER = ("v2_joint", "v3_merged", "v4_merged")
VERSION_LABEL = {
    "v2_joint": "v2 joint",
    "v3_merged": "v3 merged",
    "v4_merged": "v4 merged",
}
VERSION_COLOR = {
    "v2_joint": "#3498db",  # blue
    "v3_merged": "#f5b041",  # light orange
    "v4_merged": "#e67e22",  # orange
}

V2_MARKER = "GridNegotiation-v2"
V3_NEG = "GridNegotiation-v3-neg"
V3_NAV = "GridNegotiation-v3-nav"
V4_NEG = "GridNegotiation-v4-neg"
V4_NAV = "GridNegotiation-v4-nav"

KEY_AGR = "test_negotiation_agreed_mean"
KEY_GM = "test_negotiation_optimal_mean"
KEY_REACH = "test_battle_won_mean"

# Trailing mean window for version-compare curves (matches ``visualize.py``).
CURVE_SMOOTH_WINDOW = 5


def _last_scalar(metrics: dict[str, Any], key: str) -> float | None:
    block = metrics.get(key) or {}
    vals = block.get("values") or []
    if not vals:
        return None
    return float(vals[-1])


def _metric_series(metrics: dict[str, Any], key: str) -> tuple[Any, Any] | None:
    """Return ``(steps, values_pct)`` for a Sacred metric block, or ``None`` if missing."""
    import numpy as np

    block = metrics.get(key) or {}
    steps = block.get("steps") or []
    vals = block.get("values") or []
    if not vals:
        return None
    n = min(len(steps), len(vals)) if steps else len(vals)
    if n == 0:
        return None
    if steps and len(steps) >= n:
        s = np.asarray(steps[:n], dtype=float)
    else:
        s = np.arange(n, dtype=float)
    v = np.asarray(vals[:n], dtype=float) * 100.0
    return s, v


def _rolling_mean_expanding(values: Any, window: int) -> Any:
    """Trailing mean over at most *window* points; same length as *values* (numpy 1d)."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    w = max(1, int(window))
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - w + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i - lo + 1)
    return out


def _plot_smoothed_pct_line(ax: Any, ser: tuple[Any, Any], color: str, window: int) -> None:
    """Plot ``ser`` (steps, % values) with trailing-mean smoothing."""
    import numpy as np

    s, v = ser
    v_s = _rolling_mean_expanding(np.asarray(v, dtype=float), window)
    ax.plot(s, v_s, color=color, linewidth=1.75, alpha=0.92)


def _find_metrics_files(sacred_root: Path, algo: str, path_substring: str) -> list[Path]:
    algo_dir = sacred_root / algo
    if not algo_dir.is_dir():
        return []
    return sorted(
        p
        for p in algo_dir.rglob("metrics.json")
        if path_substring in str(p)
    )


def _load_latest_metrics(sacred_root: Path, algo: str, marker: str) -> dict[str, Any] | None:
    files = _find_metrics_files(sacred_root, algo, marker)
    if not files:
        return None
    path = files[-1]
    with path.open() as f:
        return json.load(f)


def _sacred_roots(extra: list[Path] | None) -> list[Path]:
    bases = [RESULTS_DIR, EPYMARL_DIR / "results"]
    if extra:
        bases = extra + bases
    out: list[Path] = []
    for b in bases:
        s = b / "sacred"
        if s.is_dir() and s not in out:
            out.append(s)
    return out


def _pick_metrics(
    roots: list[Path],
    algo: str,
    marker: str,
) -> dict[str, Any] | None:
    """Prefer the first root that has a run; latest file within that root."""
    found: list[Path] = []
    for root in roots:
        found = _find_metrics_files(root, algo, marker)
        if found:
            with found[-1].open() as f:
                return json.load(f)
    return None


def _triple_from_joint(m: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    return (
        _last_scalar(m, KEY_AGR),
        _last_scalar(m, KEY_GM),
        _last_scalar(m, KEY_REACH),
    )


def _triple_merged(
    neg_m: dict[str, Any] | None,
    nav_m: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    agr = _last_scalar(neg_m, KEY_AGR) if neg_m else None
    gm = _last_scalar(neg_m, KEY_GM) if neg_m else None
    reach = _last_scalar(nav_m, KEY_REACH) if nav_m else None
    return agr, gm, reach


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def collect_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algo in TRAINING_ALGOS:
        m2 = _pick_metrics(roots, algo, V2_MARKER)
        m3n = _pick_metrics(roots, algo, V3_NEG)
        m3v = _pick_metrics(roots, algo, V3_NAV)
        m4n = _pick_metrics(roots, algo, V4_NEG)
        m4v = _pick_metrics(roots, algo, V4_NAV)

        if m2:
            a, g, r = _triple_from_joint(m2)
            rows.append(
                {
                    "algo": algo,
                    "version": "v2_joint",
                    "agr": a,
                    "gm": g,
                    "reach": r,
                }
            )
        if m3n or m3v:
            a, g, r = _triple_merged(m3n, m3v)
            rows.append({"algo": algo, "version": "v3_merged", "agr": a, "gm": g, "reach": r})
        if m4n or m4v:
            a, g, r = _triple_merged(m4n, m4v)
            rows.append({"algo": algo, "version": "v4_merged", "agr": a, "gm": g, "reach": r})
    return rows


def plot_version_comparison_curves(
    rows: list[dict[str, Any]],
    roots: list[Path],
    out_dir: Path,
) -> Path | None:
    """Line plots per algorithm: GM and Reach vs timesteps (v2 joint; v3/v4 merged).

    Curves use a trailing mean (``CURVE_SMOOTH_WINDOW``); printed table / CSV still use
    the last raw Sacred scalar per version.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        smooth_w = CURVE_SMOOTH_WINDOW
    except ImportError:
        print(
            "matplotlib / numpy not installed; skipping plot. "
            "Install project requirements or pass --no-plot.",
            file=sys.stderr,
        )
        return None

    algos = [a for a in TRAINING_ALGOS if any(r["algo"] == a for r in rows)]
    if not algos:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "version_compare-curves.png"

    n_algos = len(algos)
    fig_h = max(4.0, 2.65 * n_algos)
    fig, axes = plt.subplots(n_algos, 2, figsize=(12.0, fig_h), squeeze=False)

    for i, algo in enumerate(algos):
        ax_gm = axes[i, 0]
        ax_r = axes[i, 1]
        m2 = _pick_metrics(roots, algo, V2_MARKER)
        m3n = _pick_metrics(roots, algo, V3_NEG)
        m3v = _pick_metrics(roots, algo, V3_NAV)
        m4n = _pick_metrics(roots, algo, V4_NEG)
        m4v = _pick_metrics(roots, algo, V4_NAV)

        if m2:
            ser = _metric_series(m2, KEY_GM)
            if ser is not None:
                _plot_smoothed_pct_line(ax_gm, ser, VERSION_COLOR["v2_joint"], smooth_w)
            ser = _metric_series(m2, KEY_REACH)
            if ser is not None:
                _plot_smoothed_pct_line(ax_r, ser, VERSION_COLOR["v2_joint"], smooth_w)
        if m3n:
            ser = _metric_series(m3n, KEY_GM)
            if ser is not None:
                _plot_smoothed_pct_line(ax_gm, ser, VERSION_COLOR["v3_merged"], smooth_w)
        if m3v:
            ser = _metric_series(m3v, KEY_REACH)
            if ser is not None:
                _plot_smoothed_pct_line(ax_r, ser, VERSION_COLOR["v3_merged"], smooth_w)
        if m4n:
            ser = _metric_series(m4n, KEY_GM)
            if ser is not None:
                _plot_smoothed_pct_line(ax_gm, ser, VERSION_COLOR["v4_merged"], smooth_w)
        if m4v:
            ser = _metric_series(m4v, KEY_REACH)
            if ser is not None:
                _plot_smoothed_pct_line(ax_r, ser, VERSION_COLOR["v4_merged"], smooth_w)

        ax_gm.set_ylabel("Test %")
        ax_gm.set_ylim(0.0, 105.0)
        ax_gm.grid(True, alpha=0.28)
        ax_gm.set_axisbelow(True)
        ax_gm.set_title(f"{algo.upper()} — GM (neg / joint)", fontsize=10)

        ax_r.set_ylim(0.0, 105.0)
        ax_r.grid(True, alpha=0.28)
        ax_r.set_axisbelow(True)
        ax_r.set_title(f"{algo.upper()} — Reach (nav / joint)", fontsize=10)

        if i == n_algos - 1:
            ax_gm.set_xlabel("Timesteps")
            ax_r.set_xlabel("Timesteps")

    legend_handles = [
        Line2D([0], [0], color=VERSION_COLOR[v], lw=2.2, label=VERSION_LABEL[v])
        for v in VERSION_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=9,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.99),
    )

    fig.suptitle(
        "Env versions: v2 = joint; v3/v4 = merged (GM from neg, REACH from nav; "
        f"x-axis = per-run timesteps; curves = {smooth_w}-pt trailing mean)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help="Extra base dir(s) containing sacred/ (can repeat). Repo results/ checked by default.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the same table as CSV to this path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=f"Directory for PNG plots (default: {PLOTS_DIR / 'version_compare'}).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write comparison curves PNG.",
    )
    args = parser.parse_args()
    roots = _sacred_roots(args.results_dir)

    if not roots:
        print("No sacred/ directory found under results or epymarl/results.", file=sys.stderr)
        return 1

    rows = collect_rows(roots)

    if not rows:
        print("No metrics.json found for any known env marker.", file=sys.stderr)
        return 1

    # Wide table: one block per algo
    hdr = (
        f"{'algo':<7} | {'version':<12} | "
        f"{'AGR (neg or joint)':>18} | {'GM (neg or joint)':>18} | {'REACH (nav or joint)':>20}"
    )
    print(hdr)
    print("-" * len(hdr))
    for algo in TRAINING_ALGOS:
        block = [x for x in rows if x["algo"] == algo]
        if not block:
            continue
        _ver_rank = {v: i for i, v in enumerate(VERSION_ORDER)}
        for x in sorted(block, key=lambda r: _ver_rank.get(r["version"], 99)):
            print(
                f"{x['algo']:<7} | {x['version']:<12} | "
                f"{_pct(x['agr']):>18} | {_pct(x['gm']):>18} | {_pct(x['reach']):>20}"
            )
        print()

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["algo", "version", "agr", "gm", "reach"],
                extrasaction="ignore",
            )
            w.writeheader()
            for row in rows:
                w.writerow(
                    {
                        "algo": row["algo"],
                        "version": row["version"],
                        "agr": row["agr"] if row["agr"] is not None else "",
                        "gm": row["gm"] if row["gm"] is not None else "",
                        "reach": row["reach"] if row["reach"] is not None else "",
                    }
                )
        print(f"Wrote {args.csv}")

    if not args.no_plot:
        plot_dir = args.plots_dir if args.plots_dir is not None else PLOTS_DIR / "version_compare"
        p = plot_version_comparison_curves(rows, roots, plot_dir)
        if p:
            print(f"Wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
