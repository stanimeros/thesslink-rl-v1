#!/usr/bin/env python3
"""Smoke test: short training for all algorithms (IQL, QMIX, VDN, MAPPO, COMA) + plots.

Usage:
    source .venv/bin/activate
    THESSLINK_ENV_VERSION=3 python smoke_test.py   # non-interactive
    python smoke_test.py # prompts if stdin is a TTY

``train.sh`` exports THESSLINK_ENV_VERSION before invoking this script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def _ensure_env_version_for_smoke() -> None:
    if "THESSLINK_ENV_VERSION" in os.environ:
        return
    if sys.stdin.isatty():
        while True:
            raw = input("ThessLink env version [0-3]: ").strip()
            try:
                v = int(raw)
                if v in (0, 1, 2, 3):
                    os.environ["THESSLINK_ENV_VERSION"] = str(v)
                    return
            except ValueError:
                pass
            print("  Enter 0, 1, 2, or 3.", file=sys.stderr)
    print(
        "Error: set THESSLINK_ENV_VERSION=0..3 or run via train.sh.",
        file=sys.stderr,
    )
    sys.exit(1)


_ensure_env_version_for_smoke()

from config import ENV_CONFIG, ENV_TAG, ENV_VERSION, GridNegotiationEnv
from thesslink_rl.constants import (
    AGENT_CONFIG_YAMLS,
    EPYMARL_DIR,
    EPYMARL_SRC,
    PLOTS_DIR,
    PROJECT_ROOT,
    TRAINING_ALGOS,
    epymarl_common_reward_cli_flag,
)
from thesslink_rl.visualization import _make_filename

T_MAX = 1000
LOG_INTERVAL = 50
TEST_INTERVAL = 250
SAVE_MODEL_INTERVAL = 500

SACRED_VERSION_MARKER = f"GridNegotiation-v{ENV_VERSION}"
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
            p for p in sacred_algo.rglob("metrics.json")
            if SACRED_VERSION_MARKER in str(p)
        ]
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.parent
    bases = ", ".join(str(b) for b in _sacred_results_bases())
    raise FileNotFoundError(
        f"No Sacred tree for {algo_l!r} under sacred/{algo_l}/ in any of: {bases} "
        f"(need metrics.json with {SACRED_VERSION_MARKER!r} in path)",
    )


def run_training(algo: str) -> Path:
    """Launch a short training run for *algo*; return the Sacred run directory."""
    algo = algo.lower()
    cmd = [
        sys.executable, str(EPYMARL_SRC / "main.py"),
        f"--config={algo}", f"--env-config={ENV_CONFIG}",
        "with",
        f"local_results_path={SMOKE_RESULTS_DIR.resolve()}",
        f"t_max={T_MAX}",
        f"test_interval={TEST_INTERVAL}",
        f"log_interval={LOG_INTERVAL}",
        f"save_model=True",
        f"save_model_interval={SAVE_MODEL_INTERVAL}",
        f"test_nepisode=8",
        epymarl_common_reward_cli_flag(algo),
    ]
    print(f"\n{'='*60}")
    print(f"STEP 1 — Smoke training: {algo.upper()}")
    print(f"  t_max={T_MAX}  test_interval={TEST_INTERVAL}  log_interval={LOG_INTERVAL}")
    print(f"  {epymarl_common_reward_cli_flag(algo)}")
    print(f"  cmd: {' '.join(cmd[cmd.index('with'):])}")
    print(f"{'='*60}\n")

    # Match ``train.sh`` (repo root cwd). EPyMARL defaults ``local_results_path`` to
    # ``epymarl/results`` relative to cwd; running from ``epymarl/src`` breaks repo-root
    # ``results/sacred`` layout if the CLI override is not applied.
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=False,
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


def generate_plots(metrics: dict, algo: str):
    """Generate the same 3 plots the project already has."""
    from thesslink_rl.evaluation import AgentConfig, compute_poi_scores
    from thesslink_rl.visualization import (
        plot_training_curves,
        random_episode_frames,
        render_eval_heatmaps,
        replay_episode,
    )

    print(f"\n{'='*60}")
    print(f"STEP 3 — Plots ({algo.upper()})")
    print(f"{'='*60}")

    # --- 3a. Training curves from Sacred metrics ---
    from thesslink_rl.checkpoints import test_reward_series

    steps_arr, vals_arr = test_reward_series(metrics)
    steps = steps_arr.tolist() if steps_arr.size else []
    gm_vals = vals_arr.tolist() if vals_arr.size else []
    neg_vals = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
    reached_vals = metrics.get("test_battle_won_mean", {}).get("values", [])
    epl_vals = metrics.get("test_ep_length_mean", {}).get("values", [])

    stats = {
        "common_reward": gm_vals,
        "negotiate": [v * 100.0 for v in neg_vals],
        "reach": [v * 100.0 for v in reached_vals],
        "ep_len": epl_vals,
    }
    fname = _make_filename("training_curves", "png", algo)
    print(f"  [1/3] Training curves...")
    plot_training_curves(
        stats,
        window=min(5, max(1, len(gm_vals))),
        algo=algo,
        env_name=ENV_TAG,
        timesteps=steps if steps else None,
    )
    print(f"         -> plots/{ENV_TAG}/{fname}")

    # --- 3b. Evaluation heatmaps ---
    cfg_0 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

    env = GridNegotiationEnv(agent_configs=agent_configs, seed=42)
    env.reset(seed=42)

    agents = env.possible_agents
    for agent in agents:
        spawn = tuple(env.spawn_positions[agent])
        scores = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )
        env.poi_scores[agent] = scores

    fname = _make_filename("eval_heatmaps", "png", algo)
    print(f"  [2/3] Evaluation heatmaps...")
    render_eval_heatmaps(env, agent_configs, algo=algo, env_name=ENV_TAG)
    print(f"         -> plots/{ENV_TAG}/{fname}")

    # --- 3c. Episode replay GIF (random-action demo, same env/map) ---
    env.reset(seed=42)
    for agent in agents:
        spawn = tuple(env.spawn_positions[agent])
        scores = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )
        env.poi_scores[agent] = scores

    frames = random_episode_frames(env)

    fname = _make_filename("episode_replay", "gif", algo)
    print(f"  [3/3] Episode replay GIF...")
    replay_episode(
        frames,
        env,
        agent_configs=agent_configs,
        algo=algo,
        env_name=ENV_TAG,
    )
    print(f"         -> plots/{ENV_TAG}/{fname}")


def main():
    print("ThessLink RL — Smoke Test")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Environment version: v{ENV_VERSION} (env-config={ENV_CONFIG})")
    print(f"Algorithms: {', '.join(a.upper() for a in TRAINING_ALGOS)}")

    for algo in TRAINING_ALGOS:
        run_dir = run_training(algo)
        metrics = load_sacred_metrics(run_dir)
        print_results_table(metrics, algo=algo)
        generate_plots(metrics, algo=algo)

    env_plots = PLOTS_DIR / ENV_TAG
    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved in: {SMOKE_RESULTS_DIR}")
    print(f"Plots saved in:   {env_plots}")
    for algo in TRAINING_ALGOS:
        print(f"  [{algo}] {_make_filename('training_curves', 'png', algo)}")
        print(f"  [{algo}] {_make_filename('eval_heatmaps', 'png', algo)}")
        print(f"  [{algo}] {_make_filename('episode_replay', 'gif', algo)}")

    models_root = SMOKE_RESULTS_DIR / "models"
    if models_root.exists():
        for algo in TRAINING_ALGOS:
            needle = f"/{algo}_"
            th_files = [
                p for p in models_root.rglob("*.th")
                if SACRED_VERSION_MARKER in str(p)
                and needle in str(p).replace("\\", "/").lower()
            ]
            if th_files:
                latest_th = max(th_files, key=lambda p: p.stat().st_mtime)
                print(
                    f"\nLatest checkpoint (smoke, {algo}, {SACRED_VERSION_MARKER}): "
                    f"{latest_th.parent}",
                )
    print()


if __name__ == "__main__":
    main()
