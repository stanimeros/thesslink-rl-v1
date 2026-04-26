"""Gymnasium multi-agent wrapper for EPyMARL -- v1 (cooperative meeting).

Symbolic 19-feature observation vector with GPS signal.
Reward: agreement bonus, step penalty (-0.05), terminal (+50 * quality).
Episode ends when ALL agents reach the agreed POI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..environments.v1 import (
    ACTION_DIM,
    NUM_AGENTS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)
from ..evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    negotiation_quality,
    optimal_poi,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent


class GridNegotiationGymEnv(gym.Env):
    """Gymnasium wrapper around GridNegotiationEnv (v1) for EPyMARL."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        neg_agreement_scale: float = 5.0,
        nav_step_penalty: float = 0.05,
        nav_team_scale: float = 50.0,
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
        )

        self.n_agents = NUM_AGENTS

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(OBS_FLAT_SIZE,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )

        self._neg_agreement_scale = neg_agreement_scale
        self._nav_step_penalty = nav_step_penalty
        self._nav_team_scale = nav_team_scale
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0
        self._initial_dist: Dict[str, float] = {}
        self._nav_steps: int = 0
        self._individual_arrived: Dict[str, bool] = {}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)

        agents = self._env.possible_agents
        for agent in agents:
            spawn = tuple(self._env.spawn_positions[agent])
            cfg = self._agent_configs.get(agent)
            scores = compute_poi_scores(
                spawn, spawn, self._env.poi_positions,
                self._env.obstacle_map, cfg,
            )
            self._poi_scores[agent] = scores
            self._env.poi_scores[agent] = scores

        self._agreed_poi = None
        self._optimal_poi = optimal_poi(self._poi_scores, agents)
        self._agreement_quality = 0.0
        self._negotiation_length = 0
        self._initial_dist = {}
        self._nav_steps = 0
        self._individual_arrived = {a: False for a in agents}

        obs_tuple = tuple(
            self._env._get_obs(a) for a in agents
        )

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

        obs_d, rewards_d, terminated_d, truncated_d, infos_d = self._env.step(actions_dict)

        if prev_phase == "negotiation" and prev_neg_turn is not None:
            self._negotiation_length += 1

        obs_tuple = tuple(obs_d[a] for a in agents)
        rewards = [rewards_d[a] for a in agents]

        just_agreed = (
            self._agreed_poi is None and self._env.agreed_poi is not None
        )
        if just_agreed:
            self._agreed_poi = self._env.agreed_poi
            quality = negotiation_quality(
                self._agreed_poi, self._poi_scores, agents,
            )
            self._agreement_quality = quality
            rewards = [quality * self._neg_agreement_scale] * self.n_agents
            target = self._env.poi_positions[self._agreed_poi]
            target_bfs = bfs_distances(target, self._env.obstacle_map)
            max_d = float(self._env.obstacle_map.size)
            for a in agents:
                pos = tuple(self._env.agent_positions[a])
                d = target_bfs[pos[0], pos[1]]
                self._initial_dist[a] = float(d) if np.isfinite(d) else max_d
                self._individual_arrived[a] = False

        if self._agreed_poi is not None and not just_agreed:
            self._nav_steps += 1
            for i, a in enumerate(agents):
                if self._env.agents_reached.get(a, False) and not self._individual_arrived.get(a, False):
                    self._individual_arrived[a] = True

            for i in range(self.n_agents):
                rewards[i] -= self._nav_step_penalty

            all_reached = all(self._env.agents_reached[a] for a in agents)
            if all_reached:
                quality = negotiation_quality(
                    self._agreed_poi, self._poi_scores, agents,
                )
                rewards = [quality * self._nav_team_scale] * self.n_agents

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)

        all_reached = all(
            "reached_poi" in infos_d.get(a, {}) for a in agents
        )
        negotiation_agreed = self._env.agreed_poi is not None
        agreed_optimal = (
            negotiation_agreed and self._agreed_poi == self._optimal_poi
        )
        mean_opt = (
            sum(self._initial_dist[a] for a in agents) / self.n_agents
            if self._initial_dist
            else 0.0
        )
        nav_eff = (
            min(1.0, mean_opt / self._nav_steps)
            if (all_reached and self._nav_steps > 0)
            else 0.0
        )

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
        """Return per-agent action masks for EPyMARL."""
        return [
            self._env.get_avail_actions(a) for a in self._env.possible_agents
        ]

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._env._seed = seed
            self._env._rng = np.random.RandomState(seed)
