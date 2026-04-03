"""Gymnasium multi-agent wrapper for EPyMARL compatibility.

EPyMARL's GymmaWrapper expects:
  - env.unwrapped.n_agents (int)
  - env.action_space = Tuple(Discrete, Discrete, ...)
  - env.observation_space = Tuple(Box, Box, ...) with flat 1D observations
  - reset() -> (tuple_of_obs, info)
  - step(list_of_actions) -> (tuple_of_obs, list_of_rewards, done, truncated, info)

Negotiation runs automatically inside reset(); EPyMARL agents only control
the navigation phase via discrete movement actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import (
    ACTION_DIM,
    NUM_AGENTS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)
from .evaluation import AgentConfig, golden_mean_reward
from .negotiation import run_negotiation

_PACKAGE_DIR = Path(__file__).resolve().parent


class GridNegotiationGymEnv(gym.Env):
    """Gymnasium wrapper around GridNegotiationEnv for EPyMARL."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
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
                    low=-np.inf,
                    high=np.inf,
                    shape=(OBS_FLAT_SIZE,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )

        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten the grid (C,H,W) + comm vector into a single 1D array."""
        grid_flat = obs_dict["grid"].flatten()
        return np.concatenate([grid_flat, obs_dict["comm"]])

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)

        self._agreed_poi, self._poi_scores = run_negotiation(self._env)

        obs_tuple = tuple(
            self._flatten_obs(self._env._get_obs(a))
            for a in self._env.possible_agents
        )

        sa = self._poi_scores[self._env.possible_agents[0]][self._agreed_poi]
        sb = self._poi_scores[self._env.possible_agents[1]][self._agreed_poi]
        gm = golden_mean_reward(float(sa), float(sb))

        info = {
            "agreed_poi": self._agreed_poi,
            "golden_mean": gm,
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
        }
        return obs_tuple, info

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}

        obs_d, rewards_d, terminated_d, truncated_d, infos_d = self._env.step(actions_dict)

        obs_tuple = tuple(
            self._flatten_obs(obs_d[a]) for a in agents
        )

        rewards = [rewards_d[a] for a in agents]

        if self._agreed_poi is not None:
            target = self._env.poi_positions[self._agreed_poi]
            for a in agents:
                if tuple(self._env.agent_positions.get(a, [-1, -1])) == target:
                    sa = self._poi_scores[agents[0]][self._agreed_poi]
                    sb = self._poi_scores[agents[1]][self._agreed_poi]
                    gm = golden_mean_reward(float(sa), float(sb))
                    rewards = [gm for _ in agents]
                    break

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)

        reached = any(
            "reached_poi" in infos_d.get(a, {}) for a in agents
        )
        info = {
            "battle_won": reached,
            "reached_poi": int(reached),
        }

        return obs_tuple, rewards, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._env._seed = seed
            self._env._rng = np.random.RandomState(seed)
