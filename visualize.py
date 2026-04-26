#!/usr/bin/env python3
"""Generate episode replay GIFs from the best checkpoint under results/models.

With no Sacred results: writes one example GIF from a random-policy episode.
With results: one GIF per algorithm from its best checkpoint.

Usage:
    python visualize.py --env v3_neg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from thesslink_rl.env_catalog import prompt_help, resolve_env_choice
from thesslink_rl.checkpoints import (
    describe_models_dir_status,
    find_best_checkpoint_timestep_dir,
    load_epymarl_config_for_algo,
    rollout_episode_frames_for_gif,
    test_reward_series,
)
from thesslink_rl.evaluation import AgentConfig, compute_poi_scores
from thesslink_rl.constants import AGENT_CONFIG_YAMLS, EPYMARL_DIR, RESULTS_DIR
from thesslink_rl.visualization import (
    _env_out_dir,
    _make_filename,
    capture_frame,
    describe_actions,
    random_episode_frames,
    render_eval_heatmaps,
    replay_episode,
)

SEED = 105
EXAMPLE_TAG = "example"


def discover_runs(results_dir: Path) -> dict[str, dict]:
    """Find Sacred runs matching the active ENV_VERSION and parse their metrics."""
    from config import ENV_SACRED_MARKER
    from thesslink_rl.env_catalog import sacred_path_variants

    version_markers = sacred_path_variants(ENV_SACRED_MARKER)
    bases = (results_dir, EPYMARL_DIR / "results")
    if not any((b / "sacred").is_dir() for b in bases):
        print(
            "No Sacred results under any of: "
            + ", ".join(str(b / "sacred") for b in bases),
        )
        return {}

    runs: dict[str, dict] = {}
    for base in bases:
        sacred_dir = base / "sacred"
        if not sacred_dir.is_dir():
            continue
        for alg_dir in sorted(sacred_dir.iterdir()):
            if not alg_dir.is_dir() or alg_dir.name in runs:
                continue
            metrics_files = [
                f
                for f in alg_dir.rglob("metrics.json")
                if any(vm in str(f) for vm in version_markers)
            ]
            if not metrics_files:
                continue
            metrics_files.sort()
            with open(metrics_files[-1]) as f:
                metrics = json.load(f)
            runs[alg_dir.name] = metrics
    return runs


def _sync_poi_scores(env: Any, agent_configs: dict[str, AgentConfig]) -> None:
    """After ``reset``, fill ``env.poi_scores`` from spawn positions."""
    for agent in env.possible_agents:
        spawn = tuple(env.spawn_positions[agent])
        env.poi_scores[agent] = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )


def generate_example_outputs() -> None:
    """Write one example heatmap PNG and one example GIF from a random-policy episode."""
    from config import ENV_GRID_SIZE, ENV_TAG, GridNegotiationEnv

    cfg_0 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
    env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED, grid_size=ENV_GRID_SIZE)
    env.reset(seed=SEED)
    _sync_poi_scores(env, agent_configs)

    heatmap_fname = _make_filename("eval_heatmaps", "png", EXAMPLE_TAG)
    print(f"[1/2] Example eval heatmaps ({heatmap_fname})")
    render_eval_heatmaps(
        env, agent_configs,
        title="Example — add results/ (Sacred) to visualize real runs",
        algo=EXAMPLE_TAG,
        env_name=ENV_TAG,
    )
    print(f"  -> plots/{ENV_TAG}/{heatmap_fname}")

    frames = random_episode_frames(env)
    gif_fname = _make_filename("episode_replay", "gif", EXAMPLE_TAG)
    print(f"[2/2] Example episode replay ({gif_fname})")
    replay_episode(
        frames,
        env,
        agent_configs=agent_configs,
        algo=EXAMPLE_TAG,
        env_name=ENV_TAG,
    )
    print(f"  -> plots/{ENV_TAG}/{gif_fname}")


def generate_heatmaps_and_replays(
    algos: list[str],
    *,
    results_dir: Path | None = None,
    runs: dict[str, dict] | None = None,
    models_root: Path | None = None,
):
    """Generate one shared eval heatmap PNG and per-algorithm episode replay GIFs."""
    from config import ENV_CONFIG, ENV_GRID_SIZE, ENV_SACRED_MARKER, ENV_TAG, GridNegotiationEnv

    cfg_0 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
    env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED, grid_size=ENV_GRID_SIZE)
    env.reset(seed=SEED)
    _sync_poi_scores(env, agent_configs)

    heatmap_fname = _make_filename("eval_heatmaps", "png", None)
    render_eval_heatmaps(env, agent_configs, algo=None, env_name=ENV_TAG)
    print(f"  -> plots/{ENV_TAG}/{heatmap_fname}")

    for raw_algo in algos:
        algo = raw_algo.lower()
        env.reset(seed=SEED)
        _sync_poi_scores(env, agent_configs)

        frames: list[dict] | None = None
        metrics = runs.get(algo) if runs else None
        if results_dir is not None and metrics is not None:
            ckpt = find_best_checkpoint_timestep_dir(
                algo, results_dir, metrics, ENV_SACRED_MARKER,
                models_root=models_root,
            )
            if ckpt is not None:
                try:
                    cfg = load_epymarl_config_for_algo(algo, ENV_CONFIG, SEED)
                    frames = rollout_episode_frames_for_gif(
                        ckpt,
                        cfg,
                        SEED,
                        capture_frame,
                        describe_actions,
                    )
                    print(
                        f"  episode replay ({algo}): policy from {ckpt.name} "
                        f"({ckpt.parent.name})",
                    )
                except FileNotFoundError as e:
                    extra = getattr(e, "filename", None)
                    print(
                        f"  episode replay ({algo}): skipped — missing file"
                        f"{f' ({extra})' if extra else ''}: {e!r}",
                    )
                except Exception as e:
                    print(f"  episode replay ({algo}): skipped — load/rollout failed: {e!r}")
            else:
                hint = describe_models_dir_status(models_root, results_dir)
                print(f"  episode replay ({algo}): skipped — {hint}")
        elif results_dir is not None:
            print(f"  episode replay ({algo}): skipped — no Sacred metrics for this algo")

        if frames is not None:
            fname = _make_filename("episode_replay", "gif", algo)
            replay_episode(
                frames,
                env,
                agent_configs=agent_configs,
                algo=algo,
                env_name=ENV_TAG,
            )
            print(f"  -> plots/{ENV_TAG}/{fname}")


def print_summary(runs: dict[str, dict]):
    """Print a results table."""
    print()
    header = (
        f"  {'ALG':<7} {'T_ENV':>8} {'RETURN':>8} {'AGR%':>7} "
        f"{'GM%':>8} {'REACH%':>8} {'EP_LEN':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for algo, metrics in sorted(runs.items()):
        steps_arr, vals_arr = test_reward_series(metrics)
        ret = vals_arr.tolist() if vals_arr.size else []
        steps = steps_arr.tolist() if steps_arr.size else []
        neg = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
        neg_opt = metrics.get("test_negotiation_optimal_mean", {}).get("values", [])
        bw = metrics.get("test_battle_won_mean", {}).get("values", [])
        epl = metrics.get("test_ep_length_mean", {}).get("values", [])

        def _pct(vals):
            return f"{(vals[-1] * 100):>7.1f}%" if vals else "     —  "

        print(
            f"  {algo.upper():<7} {steps[-1] if steps else 0:>8} "
            f"{ret[-1] if ret else 0:>8.4f} "
            f"{_pct(neg):>7} "
            f"{_pct(neg_opt):>8} "
            f"{_pct(bw):>8} "
            f"{epl[-1] if epl else 0:>8.1f}"
        )
    print()


def _resolve_env_selector(cli_env: str | None) -> str:
    if cli_env is not None:
        try:
            return resolve_env_choice(cli_env)["env_config"]
        except ValueError:
            print(f"Invalid --env. Use one of: {prompt_help()}.", file=sys.stderr)
            sys.exit(1)
    if sys.stdin.isatty():
        while True:
            raw = input(f"ThessLink env selector [{prompt_help()}]: ").strip()
            try:
                return resolve_env_choice(raw)["env_config"]
            except ValueError:
                print(f"  Enter one of: {prompt_help()}.", file=sys.stderr)
    print(
        f"Error: pass --env one of {prompt_help()} (non-interactive).",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate episode replay GIFs for ThessLink RL")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="ENV",
        help="ThessLink env selector (see env_catalog), e.g. 0,1,2,v3_neg,v3_nav,v4_neg,v4_nav.",
    )
    args = parser.parse_args()

    env_selector = _resolve_env_selector(args.env)
    os.environ["THESSLINK_ENV"] = env_selector
    from config import ENV_SELECTOR, ENV_TAG

    print(f"Using environment: {ENV_TAG} (selector={ENV_SELECTOR})")

    runs = discover_runs(RESULTS_DIR)
    if not runs:
        print(
            f"No Sacred metrics for this selection — writing example outputs (*-{EXAMPLE_TAG}.*).",
        )
        generate_example_outputs()
        print(f"Done! Outputs saved to plots/{ENV_TAG}/")
        return

    algos = sorted(runs.keys())
    print(f"Found {len(algos)} algorithm(s): {', '.join(a.upper() for a in algos)}")
    print_summary(runs)

    print("Generating eval heatmap PNG + episode GIFs (best checkpoint per algorithm)...")
    generate_heatmaps_and_replays(
        algos,
        results_dir=RESULTS_DIR,
        runs=runs,
        models_root=None,
    )

    print(f"Done! Outputs saved to plots/{ENV_TAG}/")


if __name__ == "__main__":
    main()
