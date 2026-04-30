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
    rollout_two_phase_frames_for_gif,
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


def _build_vis_choices() -> tuple[list[dict], list[dict]]:
    """Return (full_entries, merged_pairs).

    full_entries: catalog entries where env_config contains '_full_' or ends with '_full'.
    merged_pairs: synthetic entries for neg+nav specialist pairs, keyed by a shared prefix.
    Each merged entry has keys: alias, neg_env_config, nav_env_config.
    """
    from thesslink_rl.env_catalog import available_env_catalog
    catalog = available_env_catalog()

    full_entries = [e for e in catalog if "_full" in e["env_config"]]

    # Find neg/nav pairs: group by stripping _neg_ / _nav_ to get shared key.
    import re
    neg_map: dict[str, str] = {}
    nav_map: dict[str, str] = {}
    for e in catalog:
        ec = e["env_config"]
        key = re.sub(r"_(neg|nav)_", "_", ec)
        if "_neg_" in ec:
            neg_map[key] = ec
        elif "_nav_" in ec:
            nav_map[key] = ec

    merged_pairs = []
    for key in sorted(neg_map.keys() & nav_map.keys()):
        # e.g. thesslink_e3_w6_v1_g64 → e3_w6_merged_v1_g64
        raw = re.sub(r"(_v\d+_)", r"_merged\1", key)
        alias = re.sub(r"^thesslink_", "", raw)
        merged_pairs.append({
            "alias": alias,
            "neg_env_config": neg_map[key],
            "nav_env_config": nav_map[key],
        })

    return full_entries, merged_pairs


def _all_choices() -> list[dict]:
    """Flat ordered list of all selectable options with a numeric index each."""
    full_entries, merged_pairs = _build_vis_choices()
    choices = []
    for e in full_entries:
        choices.append({"kind": "full", "alias": e["alias"], "env_config": e["env_config"]})
    for mp in merged_pairs:
        choices.append({"kind": "merged", "alias": mp["alias"],
                        "neg_env_config": mp["neg_env_config"],
                        "nav_env_config": mp["nav_env_config"]})
    for i, c in enumerate(choices):
        c["index"] = i
    return choices


def _vis_choices_prompt() -> str:
    return ", ".join(f"{c['index']}:{c['alias']}" for c in _all_choices())


def _resolve_env_selector(cli_env: str | None) -> str:
    """Return env_config string, or 'merged:<neg_env>:<nav_env>' sentinel."""
    choices = _all_choices()
    prompt = _vis_choices_prompt()

    def _try_resolve(raw: str) -> str | None:
        if raw.isdigit():
            idx = int(raw)
            for c in choices:
                if c["index"] == idx:
                    if c["kind"] == "full":
                        return c["env_config"]
                    return f"merged:{c['neg_env_config']}:{c['nav_env_config']}"
        for c in choices:
            if raw.lower() in (str(c["index"]), c["alias"].lower(), c.get("env_config", "").lower()):
                if c["kind"] == "full":
                    return c["env_config"]
                return f"merged:{c['neg_env_config']}:{c['nav_env_config']}"
        return None

    if cli_env is not None:
        result = _try_resolve(cli_env)
        if result:
            return result
        print(f"Invalid --env. Use one of: {prompt}.", file=sys.stderr)
        sys.exit(1)

    if sys.stdin.isatty():
        while True:
            raw = input(f"ThessLink env selector [{prompt}]: ").strip()
            result = _try_resolve(raw)
            if result:
                return result
            print(f"  Enter one of: {prompt}.", file=sys.stderr)

    print(f"Error: pass --env one of {prompt} (non-interactive).", file=sys.stderr)
    sys.exit(1)


def _discover_runs_for_env(env_config: str) -> tuple[dict, str]:
    """Switch THESSLINK_ENV, reload config, discover Sacred runs. Returns (runs, sacred_marker)."""
    import importlib
    os.environ["THESSLINK_ENV"] = env_config
    import config as _cfg_mod
    importlib.reload(_cfg_mod)
    marker = _cfg_mod.ENV_SACRED_MARKER
    runs = discover_runs(RESULTS_DIR)
    return runs, marker


def generate_merged_replays(algos: list[str], neg_env_config: str, nav_env_config: str) -> None:
    """Generate per-algo GIFs by chaining neg specialist → nav specialist."""
    from thesslink_rl.constants import AGENT_CONFIG_YAMLS

    neg_runs, neg_marker = _discover_runs_for_env(neg_env_config)
    nav_runs, nav_marker = _discover_runs_for_env(nav_env_config)

    # Reload neg env config for rendering context (GridNegotiationEnv + grid size).
    import importlib
    os.environ["THESSLINK_ENV"] = neg_env_config
    import config as _cfg_mod
    importlib.reload(_cfg_mod)
    from config import GridNegotiationEnv, ENV_GRID_SIZE  # type: ignore[import]

    cfg_0 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(AGENT_CONFIG_YAMLS / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
    render_env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED, grid_size=ENV_GRID_SIZE)
    render_env.reset(seed=SEED)

    for raw_algo in algos:
        algo = raw_algo.lower()
        neg_metrics = neg_runs.get(algo, {})
        nav_metrics = nav_runs.get(algo, {})
        neg_ckpt = find_best_checkpoint_timestep_dir(algo, RESULTS_DIR, neg_metrics, neg_marker)
        nav_ckpt = find_best_checkpoint_timestep_dir(algo, RESULTS_DIR, nav_metrics, nav_marker)
        if neg_ckpt is None or nav_ckpt is None:
            missing = []
            if neg_ckpt is None:
                missing.append(f"neg ({neg_marker})")
            if nav_ckpt is None:
                missing.append(f"nav ({nav_marker})")
            print(f"  merged replay ({algo}): skipped — missing checkpoint(s): {', '.join(missing)}")
            continue
        try:
            neg_cfg = load_epymarl_config_for_algo(algo, neg_env_config, SEED)
            nav_cfg = load_epymarl_config_for_algo(algo, nav_env_config, SEED)
            frames = rollout_two_phase_frames_for_gif(
                neg_ckpt, nav_ckpt, neg_cfg, nav_cfg, SEED, capture_frame, describe_actions
            )
            print(f"  merged replay ({algo}): {len(frames)} frames → neg={neg_ckpt.name}, nav={nav_ckpt.name}")
            tag = f"{algo}_merged"
            replay_episode(frames, render_env, agent_configs=agent_configs, algo=tag, env_name="v6_merged")
            print(f"  -> plots/merged/episode_replay-merged-{tag}.gif")
        except Exception as e:
            print(f"  merged replay ({algo}): failed — {e!r}")


def main():
    parser = argparse.ArgumentParser(description="Generate episode replay GIFs for ThessLink RL")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="ENV",
        help="Use index or alias from prompt (full envs + merged pairs).",
    )
    args = parser.parse_args()

    env_selector = _resolve_env_selector(args.env)

    # Merged mode: sentinel is "merged:<neg_env>:<nav_env>"
    if env_selector.startswith("merged:"):
        _, neg_env_config, nav_env_config = env_selector.split(":", 2)
        print(f"Merged mode: {neg_env_config} (neg) + {nav_env_config} (nav) → single GIF per algo")
        # Discover algos from neg Sacred results.
        runs_neg, _ = _discover_runs_for_env(neg_env_config)
        algos = sorted(runs_neg.keys()) if runs_neg else []
        if not algos:
            print("No Sacred results found for neg env — cannot generate merged GIFs.")
            return
        print(f"Found {len(algos)} algorithm(s): {', '.join(a.upper() for a in algos)}")
        generate_merged_replays(algos, neg_env_config, nav_env_config)
        print("Done! Outputs saved to plots/")
        return

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
