#!/usr/bin/env python3
"""Smoke test: short training for all algorithms (IQL, QMIX, VDN, MAPPO, COMA).

Usage:
    source .venv/bin/activate
    THESSLINK_ENV=v3_neg python smoke_test.py   # non-interactive (try v4_neg / v4_nav)
    python smoke_test.py # prompts if stdin is a TTY

``train.sh`` exports THESSLINK_ENV before invoking this script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from thesslink_rl.env_catalog import prompt_help, resolve_env_choice, sacred_path_variants


def _ensure_env_selector_for_smoke() -> None:
    if "THESSLINK_ENV" in os.environ or "THESSLINK_ENV_VERSION" in os.environ:
        return
    if sys.stdin.isatty():
        while True:
            raw = input(f"ThessLink env selector [{prompt_help()}]: ").strip()
            try:
                choice = resolve_env_choice(raw)
                os.environ["THESSLINK_ENV"] = choice["env_config"]
                return
            except ValueError:
                print(f"  Enter one of: {prompt_help()}", file=sys.stderr)
    print(
        f"Error: set THESSLINK_ENV to one of {prompt_help()} or run via train.sh.",
        file=sys.stderr,
    )
    sys.exit(1)


_ensure_env_selector_for_smoke()

from config import ENV_CONFIG, ENV_SACRED_MARKER, ENV_SELECTOR
from thesslink_rl.constants import (
    EPYMARL_DIR,
    EPYMARL_SRC,
    PROJECT_ROOT,
    TRAINING_ALGOS,
    epymarl_common_reward_cli_flag,
)

T_MAX = 1000
LOG_INTERVAL = 50
TEST_INTERVAL = 250
SAVE_MODEL_INTERVAL = 500
# Match train.sh: stable Sacred / numpy / torch / env seed unless overridden.
SMOKE_SEED = int(os.environ.get("THESSLINK_SEED", "42"), 10)

SACRED_VERSION_MARKERS = sacred_path_variants(ENV_SACRED_MARKER)
SMOKE_RESULTS_DIR = EPYMARL_DIR / "results"


def _sacred_results_bases() -> tuple[Path, ...]:
    """Smoke/train runs write Sacred output under ``epymarl/results``."""
    return (SMOKE_RESULTS_DIR,)


def _latest_sacred_run_for_smoke_algo(algo: str) -> Path:
    """Newest Sacred run for *algo* and current env version."""
    algo_l = algo.lower()
    for base in _sacred_results_bases():
        sacred_algo = base / "sacred" / algo_l
        if not sacred_algo.is_dir():
            continue
        candidates = [
            p
            for p in sacred_algo.rglob("metrics.json")
            if any(m in str(p) for m in SACRED_VERSION_MARKERS)
        ]
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.parent
    bases = ", ".join(str(b) for b in _sacred_results_bases())
    raise FileNotFoundError(
        f"No Sacred tree for {algo_l!r} under sacred/{algo_l}/ in any of: {bases} "
        f"(need metrics.json with a path matching {SACRED_VERSION_MARKERS!r})",
    )


def run_training(algo: str) -> Path:
    """Launch a short training run for *algo*; return the Sacred run directory."""
    algo = algo.lower()
    cmd = [
        sys.executable, str(EPYMARL_SRC / "main.py"),
        f"--config={algo}", f"--env-config={ENV_CONFIG}",
        "with",
        f"local_results_path={SMOKE_RESULTS_DIR.resolve()}",
        f"seed={SMOKE_SEED}",
        f"t_max={T_MAX}",
        f"test_interval={TEST_INTERVAL}",
        f"log_interval={LOG_INTERVAL}",
        f"save_model=True",
        f"save_model_interval={SAVE_MODEL_INTERVAL}",
        f"test_nepisode=8",
        "use_wandb=False",
        epymarl_common_reward_cli_flag(algo, ENV_CONFIG),
    ]
    print(f"\n{'='*60}")
    print(f"STEP 1 — Smoke training: {algo.upper()}")
    print(
        f"  t_max={T_MAX}  test_interval={TEST_INTERVAL}  log_interval={LOG_INTERVAL}  "
        "use_wandb=False (smoke does not log to W&B)"
    )
    print(f"  {epymarl_common_reward_cli_flag(algo, ENV_CONFIG)}")
    print(f"  cmd: {' '.join(cmd[cmd.index('with'):])}")
    print(f"{'='*60}\n")

    # Match ``train.sh`` (repo root cwd). EPyMARL defaults ``local_results_path`` to
    # ``epymarl/results`` relative to cwd; running from ``epymarl/src`` breaks repo-root
    # ``results/sacred`` layout if the CLI override is not applied.
    smoke_env = {**os.environ, "WANDB_DISABLED": "true"}
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=smoke_env, capture_output=False,
    )
    if proc.returncode != 0:
        print(f"{algo} training failed with exit code {proc.returncode}")
        sys.exit(1)

    try:
        run_dir = _latest_sacred_run_for_smoke_algo(algo)
    except FileNotFoundError as e:
        print(f"No Sacred results found: {e}")
        sys.exit(1)
    print(f"\nSacred results at: {run_dir}")
    return run_dir


def load_sacred_metrics(run_dir: Path) -> dict:
    """Parse Sacred's metrics.json into {metric_name: {steps, values}}."""
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file) as f:
        raw = json.load(f)
    return raw


def print_results_table(metrics: dict, *, algo: str):
    """Print a summary table of training results."""
    print(f"\n{'='*60}")
    print(f"STEP 2 — Results summary ({algo.upper()})")
    print(f"{'='*60}")

    keys_of_interest = [
        "test_total_return_mean",
        "test_return_mean",
        "test_return_std",
        "test_negotiation_agreed_mean",
        "test_battle_won_mean",
        "test_ep_length_mean",
    ]

    header = f"{'Metric':<30} {'Last Value':>12} {'Steps':>8}"
    print(header)
    print("-" * len(header))

    for key in keys_of_interest:
        if key in metrics:
            values = metrics[key]["values"]
            steps = metrics[key]["steps"]
            if values:
                print(f"{key:<30} {values[-1]:>12.4f} {steps[-1]:>8}")
        else:
            print(f"{key:<30} {'(not found)':>12}")

    all_keys = sorted(metrics.keys())
    other_keys = [k for k in all_keys if k not in keys_of_interest and not k.endswith("_T")]
    if other_keys:
        print(f"\nOther logged metrics: {', '.join(other_keys)}")


def main():
    print("ThessLink RL — Smoke Test")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Environment: {ENV_SELECTOR} (env-config={ENV_CONFIG})")
    print(f"Algorithms: {', '.join(a.upper() for a in TRAINING_ALGOS)}")

    for algo in TRAINING_ALGOS:
        run_dir = run_training(algo)
        metrics = load_sacred_metrics(run_dir)
        print_results_table(metrics, algo=algo)

    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    print(f"{'='*60}")
    print(f"\nSacred / metrics under: {SMOKE_RESULTS_DIR}")

    models_root = SMOKE_RESULTS_DIR / "models"
    if models_root.exists():
        for algo in TRAINING_ALGOS:
            needle = f"/{algo}_"
            th_files = [
                p
                for p in models_root.rglob("*.th")
                if any(m in str(p) for m in SACRED_VERSION_MARKERS)
                and needle in str(p).replace("\\", "/").lower()
            ]
            if th_files:
                latest_th = max(th_files, key=lambda p: p.stat().st_mtime)
                print(
                    f"\nLatest checkpoint (smoke, {algo}, env path match): "
                    f"{latest_th.parent}",
                )
    print()


if __name__ == "__main__":
    main()
