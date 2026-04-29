"""Defaults and metric definitions for W&B ThessLink reporting."""

from __future__ import annotations

import os

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "aid26006-university-of-macedonia")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "thesslink-rl")


def _parse_positive_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


# Uniform ``run.history`` samples per run; higher = closer to true best checkpoint on long runs.
HISTORY_SAMPLES = _parse_positive_int("ANALYSIS_HISTORY_SAMPLES", 2000)

ALGOS = ("iql", "qmix", "mappo")

NAV_METRICS = [
    ("nav_quality", "test_navigation_quality_mean"),
    ("q_on_win", None),
    ("nav_length", "test_navigation_length_mean"),
    ("battle_won", "test_battle_won_mean"),
]

NEG_METRICS = [
    ("neg_quality", "test_negotiation_quality_mean"),
    ("neg_agreed", "test_negotiation_agreed_mean"),
    ("neg_length", "test_negotiation_length_mean"),
    ("battle_won", "test_battle_won_mean"),
]

FULL_METRICS = [
    ("neg_q", "test_negotiation_quality_mean"),
    ("neg_agree", "test_negotiation_agreed_mean"),
    ("neg_len", "test_negotiation_length_mean"),
    ("nav_q", "test_navigation_quality_mean"),
    ("q_on_win", None),
    ("nav_len", "test_navigation_length_mean"),
    ("battle_won", "test_battle_won_mean"),
]

OTHER_METRICS = [
    ("return", "test_return_mean"),
    ("total_ret", "test_total_return_mean"),
    ("battle_won", "test_battle_won_mean"),
]

STATE_ICON = {"running": "⟳", "finished": "✓", "crashed": "✗", "failed": "✗"}

# History aggregation for ``scan_history`` peaks (lower is better for lengths).
_METRIC_OBJECTIVE_MIN = frozenset(
    {
        "test_navigation_length_mean",
        "test_negotiation_length_mean",
    }
)


def metric_objective(key: str | None) -> str:
    """Return ``"min"`` or ``"max"`` for scalar test metrics over logged history."""
    if key is None:
        return "max"
    return "min" if key in _METRIC_OBJECTIVE_MIN else "max"


def all_logged_test_metric_keys() -> frozenset[str]:
    """Union of W&B keys used by ``runs`` / compare tables (single ``scan_history`` per run)."""
    keys: set[str] = set()
    for block in (NAV_METRICS, NEG_METRICS, FULL_METRICS, OTHER_METRICS):
        for _, k in block:
            if k:
                keys.add(k)
    keys.update({"test_navigation_quality_mean", "test_battle_won_mean"})
    return frozenset(keys)
