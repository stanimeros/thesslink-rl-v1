"""Gym wrapper for navigation-only training (v7 reward shaping on v3 core env).

Key insight over v6: navigation is independent single-agent RL for each agent.
Both agents navigate to the same agreed POI, but each agent's reward is purely
a function of its own position — no team bonus, no cross-agent dependency.

Why the team bonus in v6 breaks IQL/VDN/COMA:
  The episode only terminates when BOTH agents arrive. The team bonus (30 * quality)
  is the largest reward signal but arrives only after the slower agent reaches the
  target. From an individual Q-learning perspective (IQL), agent A's Q-value becomes
  entangled with agent B's speed, which is outside agent A's control. This makes
  the credit-assignment problem much harder for weaker algorithms that cannot
  implicitly model other agents.

Changes over v6:
- Team bonus removed entirely (_NAV_TEAM_SCALE = 0).
- Individual arrival reward raised to 25.0 (was 10.0) to compensate for the
  missing team bonus and make the sparse terminal signal easier to bootstrap.
- Per-agent timeout penalty raised to -8.0 (vs -5.0 team-level in v6) and
  applied individually: each agent who has NOT reached pays the penalty, so
  fast agents are not penalised for a slow partner.
- Uses env v3 (obs size 18, same as v6) — no peer-position features.

Fixes applied vs. initial v7 run:
- Agreed POI now set to optimal_poi (was random). Quality is always 1.0, so the
  arrival reward is deterministic (25.0) and the BFS target is stable per episode.
  Random POI choice was making terminal rewards noisy and harder to bootstrap.
- Truncation is now managed internally: wrapper tracks nav_steps vs. time_limit
  so the timeout penalty fires correctly. Previously, the core v3 env never sets
  truncated=True, so the TimeLimit wrapper (applied by gymma above this wrapper)
  truncated episodes without the penalty ever being applied.
- First-arrival milestone added (5.0): gives value-based methods a bootstrapping
  anchor before the full team terminates, helping Q-values propagate toward goal
  states even in the early stages of training when joint success is rare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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
_NAV_STEP_PENALTY = -0.001      # small: BFS shaping dominates
_NAV_ARRIVAL_SCALE = 25.0       # raised from 10.0; compensates for no team bonus
_NAV_TIMEOUT_PENALTY = -8.0     # per-agent, only for agents who did not arrive
_NAV_FIRST_ARRIVAL_BONUS = 5.0  # milestone for first agent to reach the target


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray, max_bfs_dist: float) -> float:
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / max_bfs_dist


class GridNegotiationGymEnv(gym.Env):
    """Navigation-only env: each agent independently navigates to the agreed POI (v7)."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        grid_size: int = GRID_SIZE,
        time_limit: int = 480,
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
        self._time_limit = time_limit
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
        self._nav_steps: int = 0
        self._first_arrived: bool = False

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

        finite_dists = self._target_bfs[np.isfinite(self._target_bfs)]
        self._max_bfs_dist = float(finite_dists.max()) if len(finite_dists) > 0 else float(self._grid_size ** 2)

        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        self._initial_dist = {}
        self._nav_steps = 0
        self._first_arrived = False

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

            if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                self._individual_arrived[a] = True
                if not self._first_arrived:
                    self._first_arrived = True
                    rewards[i] += _NAV_FIRST_ARRIVAL_BONUS
                rewards[i] += quality * _NAV_ARRIVAL_SCALE

        all_reached = all(self._env.agents_reached[a] for a in agents)

        done = all(terminated_d[a] for a in agents)
        # v3 core env never sets truncated=True; track limit internally so the
        # timeout penalty fires before gymma's TimeLimit wrapper sees the step.
        truncated = self._nav_steps >= self._time_limit
        if truncated and not all_reached:
            for i, a in enumerate(agents):
                if not self._individual_arrived.get(a, False):
                    rewards[i] += _NAV_TIMEOUT_PENALTY

        mean_opt = sum(self._initial_dist[a] for a in agents) / self.n_agents
        nav_eff = min(1.0, mean_opt / self._nav_steps) if (all_reached and self._nav_steps > 0) else 0.0
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "navigation_length": float(self._nav_steps),
            "navigation_quality": nav_eff,
        }
        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
