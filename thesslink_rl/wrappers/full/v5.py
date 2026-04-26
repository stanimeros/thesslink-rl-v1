"""Two-phase env (v5): negotiation then navigation, phase flag in observation.

Like v2 but with stronger reward signals and the navigation BFS-shaping bug fixed.

Observation: 19 features = v3 obs (18) + phase_flag (1).
  phase_flag = 1.0 during negotiation, 0.0 during navigation.

Negotiation improvements over v2:
  - Suggest bonus is proportional to agent's own score (not binary argmax-only).
  - Accept bonus is proportional to agent's score for the peer's suggestion
    (was hardcoded 0.0 in v2's flexible/accept path for this env type).
  - Per-turn step penalty to discourage stalling.
  - Stronger timeout and wrong-agreement penalties.

Navigation improvements over v2:
  - BUG FIX: _max_bfs_dist uses actual finite BFS max from target (~60 for g32)
    instead of grid_size^2 (1024 for g32), which made shaping ~15x too weak.
  - Step penalty reduced from -0.01 to -0.001.
  - Milestone rewards at 50% and 75% of initial BFS distance.
  - Larger arrival and team bonuses.
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
    ACT_ACCEPT,
    ACT_SUGGEST_BASE,
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_SUGGEST_ACTIONS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent

# Observation size: v3 base + phase flag
OBS_SIZE = OBS_FLAT_SIZE + 1  # 19

_SHAPING_GAMMA = 0.99

# Negotiation
_NEG_STEP_PENALTY = -0.05
_NEG_SUGGEST_BONUS = 0.1        # proportional to own score
_NEG_ACCEPT_BONUS = 1.0         # proportional to own score for peer's suggestion
_NEG_ACCEPT_THRESHOLD = 0.5
_NEG_AGREEMENT_BONUS = 15.0     # quality * this on agreement
_NEG_WRONG_PENALTY = -5.0       # when agreed POI is not optimal

# Navigation
_NAV_STEP_PENALTY = -0.001
_NAV_ARRIVAL_SCALE = 10.0
_NAV_TEAM_SCALE = 30.0
_NAV_TIMEOUT_PENALTY = -5.0
_NAV_MILESTONE_SCALE = 3.0
_MILESTONES = (0.5, 0.75)


def _potential(pos: tuple[int, int], bfs_grid: np.ndarray, max_dist: float) -> float:
    d = bfs_grid[pos[0], pos[1]]
    return -1.0 if np.isinf(d) else -d / max_dist


class GridNegotiationGymEnv(gym.Env):
    """Two-phase (neg → nav) environment with phase flag in observation (v5)."""

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
                spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
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
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0
        self._nav_steps: int = 0

    def _obs(self, agent: str) -> np.ndarray:
        base = self._env._get_obs(agent)
        phase_flag = np.array([1.0 if self._env.phase == "negotiation" else 0.0], dtype=np.float32)
        return np.concatenate([base, phase_flag])

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

        self._agreed_poi = None
        self._optimal_poi = optimal_poi(self._poi_scores, agents)
        self._target_bfs = None
        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        self._initial_dist = {}
        self._milestones_hit = {a: set() for a in agents}
        self._agreement_quality = 0.0
        self._negotiation_length = 0
        self._nav_steps = 0

        obs_tuple = tuple(self._obs(a) for a in agents)
        info = {
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
            "optimal_poi": self._optimal_poi,
        }
        return obs_tuple, info

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}

        prev_phase = self._env.phase
        prev_neg_turn = self._env.neg_turn
        prev_suggestions = dict(self._env.last_suggestion)

        obs_d, _, terminated_d, truncated_d, _ = self._env.step(actions_dict)
        obs_tuple = tuple(self._obs(a) for a in agents)
        rewards = [0.0] * self.n_agents

        # ── Negotiation rewards ───────────────────────────────────────────
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            idx = agents.index(active)
            peer = agents[1 - idx]
            my_scores = self._poi_scores[active]

            self._negotiation_length += 1
            rewards[idx] += _NEG_STEP_PENALTY

            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                suggested = act - ACT_SUGGEST_BASE
                rewards[idx] += _NEG_SUGGEST_BONUS * float(my_scores[suggested])

            elif act == ACT_ACCEPT and peer in prev_suggestions:
                peer_suggested = prev_suggestions[peer]
                my_score_for_peer = float(my_scores[peer_suggested])
                if my_score_for_peer >= _NEG_ACCEPT_THRESHOLD:
                    rewards[idx] += _NEG_ACCEPT_BONUS * my_score_for_peer

            # Agreement just happened
            if self._agreed_poi is None and self._env.agreed_poi is not None:
                self._agreed_poi = self._env.agreed_poi
                quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
                self._agreement_quality = quality
                agreed_optimal_now = self._agreed_poi == self._optimal_poi
                for i in range(self.n_agents):
                    rewards[i] += quality * _NEG_AGREEMENT_BONUS
                    if not agreed_optimal_now:
                        rewards[i] += _NEG_WRONG_PENALTY

                # Set up BFS for navigation
                target = self._env.poi_positions[self._agreed_poi]
                self._target_bfs = bfs_distances(target, self._env.obstacle_map)
                finite_dists = self._target_bfs[np.isfinite(self._target_bfs)]
                self._max_bfs_dist = (
                    float(finite_dists.max()) if len(finite_dists) > 0
                    else float(self._grid_size ** 2)
                )
                for a in agents:
                    pos = tuple(self._env.agent_positions[a])
                    self._prev_potentials[a] = _potential(pos, self._target_bfs, self._max_bfs_dist)
                    d = self._target_bfs[pos[0], pos[1]]
                    self._initial_dist[a] = float(d) if np.isfinite(d) else self._max_bfs_dist
                    self._milestones_hit[a] = set()

        # ── Navigation rewards ────────────────────────────────────────────
        if prev_phase == "navigation" and self._agreed_poi is not None and self._target_bfs is not None:
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
                    rewards[i] += quality * _NAV_ARRIVAL_SCALE

            all_reached = all(self._env.agents_reached[a] for a in agents)
            if all_reached:
                for i in range(self.n_agents):
                    rewards[i] += quality * _NAV_TEAM_SCALE

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)
        if truncated and self._env.agreed_poi is None:
            for i in range(self.n_agents):
                rewards[i] += _NAV_TIMEOUT_PENALTY

        all_reached = all(self._env.agents_reached.get(a, False) for a in agents)
        negotiation_agreed = self._env.agreed_poi is not None
        agreed_optimal = negotiation_agreed and self._agreed_poi == self._optimal_poi
        mean_opt = sum(self._initial_dist[a] for a in agents) / self.n_agents if self._initial_dist else 0.0
        nav_eff = min(1.0, mean_opt / self._nav_steps) if (all_reached and self._nav_steps > 0) else 0.0
        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "negotiation_agreed": float(negotiation_agreed),
            "negotiation_optimal": float(agreed_optimal),
            "negotiation_quality": self._agreement_quality,
            "negotiation_length": float(self._negotiation_length),
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
