"""Shared constants: paths, grid, training algorithms, EPyMARL reward mode."""

from __future__ import annotations

from pathlib import Path

import yaml

GRID_SIZE = 10
NUM_POIS = 3

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

EPYMARL_DIR = PROJECT_ROOT / "epymarl"
EPYMARL_SRC = EPYMARL_DIR / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

PLOTS_DIR = PROJECT_ROOT / "plots"
AGENT_CONFIG_YAMLS = PACKAGE_ROOT / "models"

# All algorithms launched together by ``train.sh`` / ``smoke_test.py``.
TRAINING_ALGOS: tuple[str, ...] = ("iql", "qmix", "vdn", "mappo", "coma")

# EPyMARL: per-agent rewards in the buffer only for these; others use aggregated reward.
PER_AGENT_REWARD_ALGOS: frozenset[str] = frozenset({"iql", "mappo"})

# Algorithms that cannot use ``common_reward=False`` (EPyMARL asserts / buffer layout).
REQUIRES_COMMON_REWARD_ALGOS: frozenset[str] = frozenset({"qmix", "vdn", "coma"})


def uses_per_agent_epymarl_rewards(algo: str) -> bool:
    return algo.lower() in PER_AGENT_REWARD_ALGOS


def env_yaml_common_reward(env_config_name: str) -> bool | None:
    """Read ``common_reward`` from ``epymarl_config/envs/<name>.yaml`` if the key exists."""
    path = PROJECT_ROOT / "epymarl_config" / "envs" / f"{env_config_name}.yaml"
    if not path.is_file():
        return None
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if "common_reward" not in data:
        return None
    return bool(data["common_reward"])


def resolve_common_reward(algo: str, env_config_name: str) -> bool:
    """Effective ``common_reward`` for EPyMARL given algorithm and env YAML.

    QMIX, VDN, and COMA always use ``True``. Otherwise the value comes from the env
    YAML key ``common_reward`` when present; if absent, IQL and MAPPO default to
    ``False`` and other algorithms to ``True`` (legacy behaviour).
    """
    al = algo.lower()
    if al in REQUIRES_COMMON_REWARD_ALGOS:
        return True
    explicit = env_yaml_common_reward(env_config_name)
    if explicit is not None:
        return explicit
    return not uses_per_agent_epymarl_rewards(algo)


def epymarl_common_reward_cli_flag(algo: str, env_config_name: str) -> str:
    """CLI fragment for ``main.py with ...`` (e.g. ``common_reward=False``)."""
    return f"common_reward={resolve_common_reward(algo, env_config_name)}"
