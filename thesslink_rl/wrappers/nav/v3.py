"""Navigation-only wrapper — environment v3, reward shaping v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...evaluation import AgentConfig, bfs_distances, compute_poi_scores, negotiation_quality, optimal_poi
from ...environments.v3 import ACTION_DIM, GRID_SIZE, NUM_AGENTS, NUM_POIS, OBS_FLAT_SIZE, GridNegotiationEnv

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
_SHAPING_GAMMA = 0.99
_MAX_BFS_DIST = float(GRID_SIZE * GRID_SIZE)
_NAV_STEP_PENALTY = -0.01
_NAV_ARRIVAL_SCALE = 6.0
_NAV_TEAM_SCALE = 20.0


def _potential(pos: tuple[int, int], bfs_grid: np.ndarray) -> float:
    d = bfs_grid[pos[0], pos[1]]
    return -1.0 if np.isinf(d) else -d / _MAX_BFS_DIST


class GridNegotiationGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, agent0_config=None, agent1_config=None, render_mode=None, seed=0, **kwargs: Any):
        super().__init__()
        default_models = _PACKAGE_DIR / "models"
        cfg_0 = AgentConfig.from_yaml(agent0_config or str(default_models / "human.yaml"))
        cfg_1 = AgentConfig.from_yaml(agent1_config or str(default_models / "taxi.yaml"))
        self._agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
        self._env = GridNegotiationEnv(agent_configs=self._agent_configs, render_mode=render_mode, seed=seed)
        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents)))
        self.observation_space = spaces.Tuple(
            tuple(spaces.Box(low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32) for _ in range(self.n_agents))
        )
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._target_bfs: np.ndarray | None = None
        self._prev_potentials: Dict[str, float] = {}
        self._individual_arrived: Dict[str, bool] = {}
        self._initial_dist: Dict[str, float] = {}
        self._nav_steps: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)
        agents = self._env.possible_agents
        for agent in agents:
            spawn = tuple(self._env.spawn_positions[agent])
            scores = compute_poi_scores(spawn, spawn, self._env.poi_positions, self._env.obstacle_map, self._agent_configs[agent])
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
        self._initial_dist = {}
        self._nav_steps = 0
        for a in agents:
            pos = tuple(self._env.agent_positions[a])
            self._prev_potentials[a] = _potential(pos, self._target_bfs)
            d = self._target_bfs[pos[0], pos[1]]
            self._initial_dist[a] = float(d) if np.isfinite(d) else _MAX_BFS_DIST
        obs_tuple = tuple(self._env._get_obs(a) for a in agents)
        return obs_tuple, {"agreed_poi": int(self._agreed_poi), "optimal_poi": int(self._optimal_poi)}

    def step(self, actions):
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}
        obs_d, _, terminated_d, truncated_d, _ = self._env.step(actions_dict)
        rewards = [0.0] * self.n_agents
        quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
        self._nav_steps += 1
        for i, a in enumerate(agents):
            if self._individual_arrived.get(a, False):
                continue
            cur_pos = tuple(self._env.agent_positions[a])
            cur_phi = _potential(cur_pos, self._target_bfs)
            rewards[i] += _SHAPING_GAMMA * cur_phi - self._prev_potentials.get(a, cur_phi)
            self._prev_potentials[a] = cur_phi
            rewards[i] += _NAV_STEP_PENALTY
            if self._env.agents_reached.get(a, False):
                self._individual_arrived[a] = True
                rewards[i] += quality * _NAV_ARRIVAL_SCALE
        all_reached = all(self._env.agents_reached[a] for a in agents)
        if all_reached:
            for i in range(self.n_agents):
                rewards[i] += quality * _NAV_TEAM_SCALE
        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)
        mean_opt = sum(self._initial_dist[a] for a in agents) / self.n_agents
        nav_eff = min(1.0, mean_opt / self._nav_steps) if (all_reached and self._nav_steps > 0) else 0.0
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "navigation_length": float(self._nav_steps),
            "navigation_quality": nav_eff,
        }
        return tuple(obs_d[a] for a in agents), rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self): pass
    def close(self): pass
