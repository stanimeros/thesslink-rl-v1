"""Gym wrapper for navigation-only training (v6 reward shaping on v3 core env).

Key fixes over v4_nav:
1. BUG FIX — _max_bfs_dist was grid_size^2 (e.g. 1024 for g32) instead of the
   actual max BFS distance from the target (~60-100 for a 32x32 map). This made
   the potential shaping signal ~15x too weak and the step penalty dominated,
   causing agents to learn to stay still.
   Fix: compute _max_bfs_dist from the finite entries of the BFS grid.

2. Step penalty reduced from -0.01 to -0.001; BFS shaping already discourages
   dawdling, so the penalty was over-penalising necessary moves in large grids.

3. Milestone rewards at 50% and 75% of initial distance eliminated to reduce
   sparse-reward delays; instead the arrival and team bonuses are scaled up so
   the terminal signal is stronger relative to the now-correct shaping.

4. Timeout penalty increased to create a stronger urgency signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    negotiation_quality,
    optimal_poi,
)
from ...environments.v3 import (
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_POIS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
_SHAPING_GAMMA = 0.99
_NAV_STEP_PENALTY = -0.001      # was -0.01; smaller so shaping dominates
_NAV_ARRIVAL_SCALE = 10.0       # was 6.0
_NAV_TEAM_SCALE = 30.0          # was 20.0
_NAV_TIMEOUT_PENALTY = -5.0     # was -2.0
_NAV_MILESTONE_SCALE = 3.0      # new: reward at 50% and 75% progress milestones
_MILESTONES = (0.5, 0.75)       # fraction of initial distance cleared


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray, max_bfs_dist: float) -> float:
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / max_bfs_dist


class GridNegotiationGymEnv(gym.Env):
    """Navigation-only env: navigate to a uniformly random agreed POI (v6 rewards)."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        grid_size: int = GRID_SIZE,
        **kwargs: Any,
    ):
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
        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._target_bfs: np.ndarray | None = None
        self._max_bfs_dist: float = float(grid_size * grid_size)
        self._prev_potentials: Dict[str, float] = {}
        self._individual_arrived: Dict[str, bool] = {}
        self._initial_dist: Dict[str, float] = {}
        self._milestones_hit: Dict[str, Set[float]] = {}
        self._nav_steps: int = 0
        self._first_arrival_step: int = 0

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
        self._agreed_poi = int(self._env._rng.randint(0, NUM_POIS))

        self._env.agreed_poi = self._agreed_poi
        self._env.phase = "navigation"
        self._env.neg_turn = None
        self._env.last_suggestion = {}
        self._env.agents_reached = {a: False for a in agents}

        target = self._env.poi_positions[self._agreed_poi]
        self._target_bfs = bfs_distances(target, self._env.obstacle_map)

        # FIX: use actual finite BFS max, not grid_size^2
        finite_dists = self._target_bfs[np.isfinite(self._target_bfs)]
        self._max_bfs_dist = float(finite_dists.max()) if len(finite_dists) > 0 else float(self._grid_size ** 2)

        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        self._initial_dist = {}
        self._milestones_hit = {a: set() for a in agents}
        self._nav_steps = 0
        self._first_arrival_step = 0

        for a in agents:
            pos = tuple(self._env.agent_positions[a])
            self._prev_potentials[a] = _potential(pos, self._target_bfs, self._max_bfs_dist)
            d = self._target_bfs[pos[0], pos[1]]
            self._initial_dist[a] = float(d) if np.isfinite(d) else self._max_bfs_dist

        obs_tuple = tuple(self._env._get_obs(a) for a in agents)
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
        obs_d, _, terminated_d, truncated_d, _ = self._env.step(actions_dict)
        obs_tuple = tuple(obs_d[a] for a in agents)
        rewards = [0.0] * self.n_agents

        assert self._agreed_poi is not None
        assert self._target_bfs is not None
        quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
        self._nav_steps += 1

        for i, a in enumerate(agents):
            if self._individual_arrived.get(a, False):
                continue

            cur_pos = tuple(self._env.agent_positions[a])
            cur_phi = _potential(cur_pos, self._target_bfs, self._max_bfs_dist)
            prev_phi = self._prev_potentials.get(a, cur_phi)
            rewards[i] += _SHAPING_GAMMA * cur_phi - prev_phi
            self._prev_potentials[a] = cur_phi
            rewards[i] += _NAV_STEP_PENALTY

            # Milestone rewards based on fraction of initial BFS distance cleared
            init_d = self._initial_dist.get(a, 0.0)
            if init_d > 0:
                cur_d = self._target_bfs[cur_pos[0], cur_pos[1]]
                if np.isfinite(cur_d):
                    progress = 1.0 - cur_d / init_d
                    for m in _MILESTONES:
                        if progress >= m and m not in self._milestones_hit[a]:
                            self._milestones_hit[a].add(m)
                            rewards[i] += _NAV_MILESTONE_SCALE * quality

            if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                self._individual_arrived[a] = True
                if self._first_arrival_step == 0:
                    self._first_arrival_step = self._nav_steps
                rewards[i] += quality * _NAV_ARRIVAL_SCALE

        all_reached = all(self._env.agents_reached[a] for a in agents)
        if all_reached:
            for i in range(self.n_agents):
                rewards[i] += quality * _NAV_TEAM_SCALE

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)
        if truncated and not all_reached:
            for i in range(self.n_agents):
                rewards[i] += _NAV_TIMEOUT_PENALTY

        mean_opt = sum(self._initial_dist[a] for a in agents) / self.n_agents
        nav_eff = min(1.0, mean_opt / self._nav_steps) if (all_reached and self._nav_steps > 0) else 0.0
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "battle_won_navigation": float(all_reached),
            "episode_length_navigation": float(self._nav_steps),
            "reached_poi": float(all_reached),
            "nav_efficiency": nav_eff,
            "first_arrival_step": float(self._first_arrival_step),
        }
        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
