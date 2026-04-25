"""Gym wrapper for navigation-only training on v3 core dynamics (v4 reward curriculum).

The agreed POI is sampled uniformly at random from the three available POIs
on every reset.  The nav policy's job is to reach whatever POI it is told —
optimality is the neg policy's concern, not navigation's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    negotiation_quality,
    optimal_poi,
)
from ..v3.environment import (
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_POIS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_SHAPING_GAMMA = 0.99
_NAV_STEP_PENALTY = -0.01
_NAV_ARRIVAL_SCALE = 6.0
_NAV_TEAM_SCALE = 20.0
_NAV_TIMEOUT_PENALTY = -2.0


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray, max_bfs_dist: float) -> float:
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / max_bfs_dist


class GridNegotiationGymEnv(gym.Env):
    """Navigation-only env: navigate to a uniformly random agreed POI."""

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
        self._max_bfs_dist = float(grid_size * grid_size)
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
        self._prev_potentials: Dict[str, float] = {}
        self._individual_arrived: Dict[str, bool] = {}

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
        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        for a in agents:
            pos = tuple(self._env.agent_positions[a])
            self._prev_potentials[a] = _potential(pos, self._target_bfs, self._max_bfs_dist)

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
        for i, a in enumerate(agents):
            if self._individual_arrived.get(a, False):
                continue
            cur_pos = tuple(self._env.agent_positions[a])
            cur_phi = _potential(cur_pos, self._target_bfs, self._max_bfs_dist)
            prev_phi = self._prev_potentials.get(a, cur_phi)
            rewards[i] += _SHAPING_GAMMA * cur_phi - prev_phi
            self._prev_potentials[a] = cur_phi
            rewards[i] += _NAV_STEP_PENALTY
            if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                self._individual_arrived[a] = True
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

        agreed_optimal = self._agreed_poi == self._optimal_poi
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "reached_poi": float(all_reached),
        }
        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
