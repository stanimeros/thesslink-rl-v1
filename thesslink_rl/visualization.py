"""Visualization: grid rendering, training curves, and episode replay."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .environment import GRID_SIZE, GridNegotiationEnv

COLORS = {
    "empty": "#f0f0f0",
    "obstacle": "#2d2d2d",
    "poi_0": "#e74c3c",
    "poi_1": "#2ecc71",
    "poi_2": "#3498db",
    "agent_0": "#f39c12",
    "agent_1": "#9b59b6",
    "target": "#e74c3c",
}

OUT_DIR = Path("plots")


def _ensure_out_dir():
    OUT_DIR.mkdir(exist_ok=True)


# ── 1. Static grid snapshot ─────────────────────────────────────────────

def render_grid(
    env: GridNegotiationEnv,
    title: str = "",
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Axes:
    """Draw the current grid state with obstacles, POIs, and agents."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    grid_rgb = np.full((GRID_SIZE, GRID_SIZE, 3), mcolors.to_rgb(COLORS["empty"]))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if env.obstacle_map[r, c]:
                grid_rgb[r, c] = mcolors.to_rgb(COLORS["obstacle"])

    poi_keys = ["poi_0", "poi_1", "poi_2"]
    for i, (pr, pc) in enumerate(env.poi_positions):
        grid_rgb[pr, pc] = mcolors.to_rgb(COLORS[poi_keys[i]])

    ax.imshow(grid_rgb, origin="upper", extent=(0, GRID_SIZE, GRID_SIZE, 0))

    for i, (pr, pc) in enumerate(env.poi_positions):
        marker = "^" if env.agreed_poi == i else "D"
        size = 200 if env.agreed_poi == i else 120
        ax.scatter(pc + 0.5, pr + 0.5, marker=marker, s=size,
                   c=COLORS[poi_keys[i]], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(pc + 0.5, pr + 0.15, f"P{i}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    for agent in env.possible_agents:
        if agent not in env.agent_positions:
            continue
        r, c = env.agent_positions[agent]
        color = COLORS[agent]
        ax.scatter(c + 0.5, r + 0.5, marker="o", s=260,
                   c=color, edgecolors="white", linewidths=2, zorder=5)
        ax.text(c + 0.5, r + 0.5, agent[-1], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)

    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    phase_tag = f"  [{env.phase}]" if hasattr(env, "phase") else ""
    ax.set_title(title + phase_tag, fontsize=11)

    if save_path:
        _ensure_out_dir()
        ax.figure.savefig(OUT_DIR / save_path, dpi=150, bbox_inches="tight")
    if show and standalone:
        plt.show()
    return ax


# ── 2. Training curves ──────────────────────────────────────────────────

def plot_training_curves(
    stats: Dict[str, list],
    window: int = 20,
    save_path: str = "training_curves.png",
    show: bool = True,
):
    """Plot Golden Mean, reach rate, and PG loss with a rolling average."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("gm", "Golden Mean Reward", "#2ecc71"),
        ("reached", "Reach Rate", "#3498db"),
        ("pg_loss", "Policy Loss", "#e74c3c"),
    ]

    for ax, (key, label, color) in zip(axes, panels):
        data = np.array(stats.get(key, []))
        if len(data) == 0:
            ax.set_title(label)
            continue
        episodes = np.arange(1, len(data) + 1)
        ax.plot(episodes, data, alpha=0.25, color=color, linewidth=0.8)
        if len(data) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(data, kernel, mode="valid")
            ax.plot(np.arange(window, len(data) + 1), smoothed,
                    color=color, linewidth=2, label=f"{window}-ep avg")
            ax.legend(fontsize=8)
        ax.set_xlabel("Episode")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Progress", fontsize=13, y=1.02)
    plt.tight_layout()
    _ensure_out_dir()
    fig.savefig(OUT_DIR / save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── 3. Episode replay animation ─────────────────────────────────────────

def replay_episode(
    frames: list[dict],
    env: GridNegotiationEnv,
    save_path: str = "episode_replay.gif",
    interval_ms: int = 400,
    show: bool = True,
):
    """Animate an episode from a list of frame snapshots.

    Each frame dict: {"agent_positions": {name: [r,c]}, "phase": str,
                      "timestep": int, "agreed_poi": int|None}
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def _draw(idx):
        ax.clear()
        frame = frames[idx]
        env.agent_positions = frame["agent_positions"]
        env.phase = frame["phase"]
        env.agreed_poi = frame.get("agreed_poi")
        render_grid(env, title=f"Step {frame['timestep']}", ax=ax, show=False)

    anim = FuncAnimation(fig, _draw, frames=len(frames),
                         interval=interval_ms, repeat=False)
    _ensure_out_dir()
    anim.save(str(OUT_DIR / save_path), writer="pillow", dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def capture_frame(env: GridNegotiationEnv) -> dict:
    """Snapshot the env state for replay_episode."""
    return {
        "agent_positions": {a: list(pos) for a, pos in env.agent_positions.items()},
        "phase": env.phase,
        "timestep": env.timestep,
        "agreed_poi": getattr(env, "agreed_poi", None),
    }
