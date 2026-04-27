"""Shared constants: paths, grid, training algorithms, EPyMARL reward mode."""

from __future__ import annotations

from pathlib import Path

GRID_SIZE = 16
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
