"""Gym wrapper for navigation-only training (w6 reward shaping on v3 core env).

Improvements over v5:

1. Minimal nav-specific observation — strips the three negotiation features
   that are constant noise in a nav-only episode:
     - self_scores (3): fixed preference scores, irrelevant once POI is chosen
     - peer_action (4): always [1,0,0,0] (no-action) since negotiation never runs
     - agreed_poi  (3): always the same one-hot since target is preset each reset
   Only the four actually informative features are kept.

2. 8-directional lidar — adds NE, SE, SW, NW diagonal rays alongside the
   existing N, S, E, W.  Agents can now see obstacles around corners and plan
   detours around clusters that were invisible with only 4 rays.

3. Per-agent shaping normalisation.  v5 normalised the potential by the global
   max BFS distance on the map (up to ~150+ on a 32x32 grid), making the
   shaping signal negligibly small.  v6 normalises by each agent's own initial
   BFS distance to the target, so the per-step gradient toward the goal is
   always O(1 / initial_dist) — much stronger and consistent regardless of
   episode difficulty.  Total accumulated shaping over an optimal episode ≈ 1.0,
   which pairs well with the arrival reward.

4. Per-agent rewards (``common_reward=False``): shaping, step penalty, timeout, and
   arrival credit use **only that agent's** state.  Arrival scale is multiplied by
   **that agent's** own POI preference vs their best POI, not joint negotiation_quality.

Observation layout (size 13):
  self_pos       (2)  (row, col) normalised to [0, 1]
  relative_pos   (2)  (target - self) / (grid_size - 1)
  lidar_card     (4)  N, S, E, W
  lidar_diag     (4)  NE, SE, SW, NW
  bfs_dist_norm  (1)  remaining shortest-path steps to target / initial BFS dist,
                      clipped to [0, 2] (inf unreachable → 1)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    optimal_poi,
)
from ..environments.v3 import (
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_POIS,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent

# Minimal nav obs: self_pos(2) + relative_pos(2) + lidar_card(4) + lidar_diag(4) + bfs_dist_norm(1)
OBS_FLAT_SIZE_V6 = 13

# Per-dim bounds: last channel is normalised remaining geodesic distance (may exceed 1 if off-path).
_OBS_LOW_V6 = np.array(
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
_OBS_HIGH_V6 = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    dtype=np.float32,
)


def _diag_lidar(obstacle_map: np.ndarray, grid_size: int, r: int, c: int) -> np.ndarray:
    """Cast rays in NE, SE, SW, NW; return normalised distance to nearest obstacle."""
    directions = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # NE, SE, SW, NW
    distances = np.ones(4, dtype=np.float32)
    max_dist = grid_size - 1
    for i, (dr, dc) in enumerate(directions):
        for step in range(1, grid_size):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size:
                distances[i] = step / max_dist
                break
            if obstacle_map[nr, nc]:
                distances[i] = step / max_dist
                break
        else:
            distances[i] = 1.0
    return distances


def _bfs_remaining_norm(
    target_bfs: np.ndarray,
    initial_dist: Dict[str, float],
    agent: str,
    r: int,
    c: int,
) -> float:
    """Normalised remaining geodesic distance to target (BFS steps / spawn BFS dist).

    Returns value in ``[0, 2]``: ``0`` at goal, ``1`` at spawn on a shortest path,
    ``>1`` if the agent moved away (clipped at ``2``).  ``inf`` / unreachable → ``1``.
    """
    d = float(target_bfs[r, c])
    idist = max(float(initial_dist.get(agent, 1.0)), 1e-6)
    if not np.isfinite(d) or np.isinf(d):
        return 1.0
    return float(np.clip(d / idist, 0.0, 2.0))


def _agent_poi_preference_quality(scores: np.ndarray, poi_idx: int) -> float:
    """How much *this* agent values the agreed POI vs their own best POI, in ``[0, 1]``.

    Used for arrival credit under ``common_reward=False`` so one agent's negotiated
    outcome does not scale the other's arrival bonus (unlike joint ``negotiation_quality``).
    """
    best = float(np.max(scores))
    if best < 1e-12:
        return 0.0
    return float(np.clip(float(scores[poi_idx]) / best, 0.0, 1.0))


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray, norm_dist: float) -> float:
    """Potential phi = -d / norm_dist, in [-1, 0].

    Using the agent's own initial BFS distance as norm_dist ensures a consistent
    shaping scale across episodes: phi = -1 at spawn, phi = 0 at target.
    """
    if norm_dist <= 0:
        return 0.0
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / norm_dist


class GridNegotiationGymEnv(gym.Env):
    """Navigation-only env: 8-dir lidar + per-agent shaping normalisation (w6).

    **Reward contract (individual learning):** ``step`` returns ``list[float]`` with
    one scalar per agent index.  For each agent ``a``, ``rewards[i]`` is built only
    from that agent's position history, own initial-distance normalisation, own
    POI-preference factor at arrival, and whether *that* agent timed out — there is
    no joint negotiation-quality multiplier and no shared arrival bonus.  Agents
    who have already reached the target receive zero reward on subsequent steps
    (no step penalty while frozen).  Pairwise interaction is only through the
    underlying grid dynamics.

    **Training:** set ``common_reward: false`` in EPyMARL so learners optimise
    their own returns instead of a team sum.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        grid_size: int = GRID_SIZE,
        time_limit: int = 512,
        shaping_gamma: float = 0.99,
        step_penalty: float = -0.001,
        arrival_scale: float = 25.0,
        timeout_penalty: float = -8.0,
        **kwargs: Any,
    ):
        if "first_arrival_bonus" in kwargs:
            warnings.warn(
                "first_arrival_bonus is not used by v6_nav (removed); remove it from config.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__()
        default_models = _PACKAGE_DIR / "models"
        cfg_0 = AgentConfig.from_yaml(agent0_config or str(default_models / "human.yaml"))
        cfg_1 = AgentConfig.from_yaml(agent1_config or str(default_models / "taxi.yaml"))
        self._agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

        self._env = GridNegotiationEnv(
            agent_configs=self._agent_configs,
            render_mode=render_mode,
            seed=seed,
            grid_size=grid_size,
        )
        self._grid_size = grid_size
        self._time_limit = time_limit
        self._shaping_gamma = shaping_gamma
        self._step_penalty = step_penalty
        self._arrival_scale = arrival_scale
        self._timeout_penalty = timeout_penalty
        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=np.copy(_OBS_LOW_V6),
                    high=np.copy(_OBS_HIGH_V6),
                    shape=(OBS_FLAT_SIZE_V6,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._target_bfs: np.ndarray | None = None
        self._prev_potentials: Dict[str, float] = {}
        self._individual_arrived: Dict[str, bool] = {}
        self._initial_dist: Dict[str, float] = {}
        self._nav_steps: int = 0

    def _get_nav_obs(self, agents: list[str]) -> tuple[np.ndarray, ...]:
        """Build minimal nav obs: self_pos + relative_pos + 8-dir lidar + BFS dist norm."""
        obs_list = []
        norm = self._grid_size - 1
        assert self._target_bfs is not None
        for a in agents:
            r, c = self._env.agent_positions[a]

            self_pos = np.array([r / norm, c / norm], dtype=np.float32)

            relative_pos = np.zeros(2, dtype=np.float32)
            if self._agreed_poi is not None:
                tr, tc = self._env.poi_positions[self._agreed_poi]
                relative_pos[0] = (tr - r) / norm
                relative_pos[1] = (tc - c) / norm

            card = self._env._lidar(r, c)
            diag = _diag_lidar(self._env.obstacle_map, self._grid_size, r, c)
            bfs_n = _bfs_remaining_norm(self._target_bfs, self._initial_dist, a, r, c)

            obs_list.append(
                np.concatenate(
                    [self_pos, relative_pos, card, diag, np.array([bfs_n], dtype=np.float32)]
                )
            )
        return tuple(obs_list)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)
        agents = self._env.possible_agents
        for agent in agents:
            spawn = tuple(self._env.spawn_positions[agent])
            cfg = self._agent_configs[agent]
            scores = compute_poi_scores(
                spawn, spawn, self._env.poi_positions, self._env.obstacle_map, cfg
            )
            self._poi_scores[agent] = scores
            self._env.poi_scores[agent] = scores

        self._optimal_poi = optimal_poi(self._poi_scores, agents)
        self._agreed_poi = self._optimal_poi

        self._env.agreed_poi = self._agreed_poi
        self._env.phase = "navigation"
        self._env.neg_turn = None
        self._env.last_suggestion = {}
        self._env.agents_reached = {a: False for a in agents}

        target = self._env.poi_positions[self._agreed_poi]
        self._target_bfs = bfs_distances(target, self._env.obstacle_map)

        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        self._initial_dist = {}
        self._nav_steps = 0

        for a in agents:
            pos = tuple(self._env.agent_positions[a])
            d = self._target_bfs[pos[0], pos[1]]
            # Per-agent normalisation: use own initial BFS dist (fall back to 1 if at target)
            self._initial_dist[a] = float(d) if (np.isfinite(d) and d > 0) else 1.0
            self._prev_potentials[a] = _potential(pos, self._target_bfs, self._initial_dist[a])

        obs_tuple = self._get_nav_obs(agents)
        info = {
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
            "agreed_poi": int(self._agreed_poi),
            "optimal_poi": int(self._optimal_poi),
        }
        return obs_tuple, info

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}
        _, _, terminated_d, truncated_d, _ = self._env.step(actions_dict)
        rewards = [0.0] * self.n_agents

        assert self._agreed_poi is not None
        assert self._target_bfs is not None
        self._nav_steps += 1

        for i, a in enumerate(agents):
            if self._individual_arrived.get(a, False):
                continue

            cur_pos = tuple(self._env.agent_positions[a])
            # Normalise by this agent's own initial distance for a strong, consistent signal
            cur_phi = _potential(cur_pos, self._target_bfs, self._initial_dist[a])
            prev_phi = self._prev_potentials.get(a, cur_phi)
            rewards[i] += self._shaping_gamma * cur_phi - prev_phi
            self._prev_potentials[a] = cur_phi
            rewards[i] += self._step_penalty

            if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                self._individual_arrived[a] = True
                aq = _agent_poi_preference_quality(
                    self._poi_scores[a], int(self._agreed_poi)
                )
                rewards[i] += aq * self._arrival_scale

        all_reached = all(self._env.agents_reached[a] for a in agents)

        done = all(terminated_d[a] for a in agents)
        truncated = self._nav_steps >= self._time_limit
        if truncated and not all_reached:
            for i, a in enumerate(agents):
                if not self._individual_arrived.get(a, False):
                    rewards[i] += self._timeout_penalty

        mean_opt = sum(self._initial_dist[a] for a in agents) / self.n_agents
        nav_eff = min(1.0, mean_opt / self._nav_steps) if (all_reached and self._nav_steps > 0) else 0.0
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "navigation_length": float(self._nav_steps),
            "navigation_quality": nav_eff,
        }
        # Per-agent list: rewards[i] only from agents[i]'s nav state (see class docstring).
        return self._get_nav_obs(agents), rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
