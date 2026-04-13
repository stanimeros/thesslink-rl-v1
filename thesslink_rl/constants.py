"""Shared constants: paths, grid, training algorithms, EPyMARL reward mode."""

from __future__ import annotations

from pathlib import Path

GRID_SIZE = 10
NUM_POIS = 3

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

EPYMARL_DIR = PROJECT_ROOT / "epymarl"
EPYMARL_SRC = EPYMARL_DIR / "src"
EPYMARL_RESULTS = EPYMARL_DIR / "results"

PLOTS_DIR = PROJECT_ROOT / "plots"
AGENT_CONFIG_YAMLS = PACKAGE_ROOT / "models"

# All algorithms launched together by ``train.sh`` / ``smoke_test.py``.
TRAINING_ALGOS: tuple[str, ...] = ("iql", "qmix", "vdn", "mappo", "coma")

# EPyMARL: per-agent rewards in the buffer only for these; others use aggregated reward.
PER_AGENT_REWARD_ALGOS: frozenset[str] = frozenset({"iql", "mappo"})


def uses_per_agent_epymarl_rewards(algo: str) -> bool:
    return algo.lower() in PER_AGENT_REWARD_ALGOS


def epymarl_common_reward_cli_flag(algo: str) -> str:
    """CLI fragment for ``main.py with ...`` (e.g. ``common_reward=False``)."""
    return (
        "common_reward=False"
        if uses_per_agent_epymarl_rewards(algo)
        else "common_reward=True"
    )
