"""Defaults and metric definitions for W&B ThessLink reporting."""

from __future__ import annotations

import os

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "aid26006-university-of-macedonia")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "thesslink-rl")

ALGOS = ("iql", "qmix", "mappo", "ippo", "maddpg")

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
