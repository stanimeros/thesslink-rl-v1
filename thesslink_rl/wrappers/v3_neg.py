"""Negotiation-only wrapper — environment v3, reward shaping v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..evaluation import AgentConfig, compute_poi_scores, negotiation_quality, optimal_poi
from ..environments.v3 import (
    ACT_ACCEPT,
    ACT_SUGGEST_BASE,
    ACTION_DIM,
    NUM_AGENTS,
    NUM_SUGGEST_ACTIONS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent


class GridNegotiationGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        suggest_bonus: float = 0.05,
        accept_bonus: float = 0.05,
        accept_threshold: float = 0.6,
        agreement_scale: float = 10.0,
        **kwargs: Any,
    ):
        super().__init__()
        default_models = _PACKAGE_DIR / "models"
        cfg_0 = AgentConfig.from_yaml(agent0_config or str(default_models / "human.yaml"))
        cfg_1 = AgentConfig.from_yaml(agent1_config or str(default_models / "taxi.yaml"))
        self._agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
        self._env = GridNegotiationEnv(agent_configs=self._agent_configs, render_mode=render_mode, seed=seed)
        self._suggest_bonus = suggest_bonus
        self._accept_bonus = accept_bonus
        self._accept_threshold = accept_threshold
        self._agreement_scale = agreement_scale
        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents)))
        self.observation_space = spaces.Tuple(
            tuple(spaces.Box(low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32) for _ in range(self.n_agents))
        )
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[tuple[np.ndarray, ...], dict]:
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)
        agents = self._env.possible_agents
        for agent in agents:
            spawn = tuple(self._env.spawn_positions[agent])
            scores = compute_poi_scores(spawn, spawn, self._env.poi_positions, self._env.obstacle_map, self._agent_configs[agent])
            self._poi_scores[agent] = scores
            self._env.poi_scores[agent] = scores
        self._agreed_poi = None
        self._optimal_poi = optimal_poi(self._poi_scores, agents)
        self._agreement_quality = 0.0
        self._negotiation_length = 0
        return tuple(self._env._get_obs(a) for a in agents), {"poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()}, "optimal_poi": self._optimal_poi}

    def step(self, actions):
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}
        prev_phase = self._env.phase
        prev_neg_turn = self._env.neg_turn
        prev_suggestions = dict(self._env.last_suggestion)
        obs_d, _, _, truncated_d, _ = self._env.step(actions_dict)
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            self._negotiation_length += 1
        rewards = [0.0] * self.n_agents

        if prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            idx = agents.index(active)
            peer = agents[1 - idx]
            my_scores = self._poi_scores[active]
            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                if act - ACT_SUGGEST_BASE == int(np.argmax(my_scores)):
                    rewards[idx] += self._suggest_bonus
            elif act == ACT_ACCEPT and peer in prev_suggestions:
                if my_scores[prev_suggestions[peer]] >= self._accept_threshold:
                    rewards[idx] += self._accept_bonus

        if self._agreed_poi is None and self._env.agreed_poi is not None:
            self._agreed_poi = self._env.agreed_poi
            quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
            self._agreement_quality = quality
            for i in range(self.n_agents):
                rewards[i] += quality * self._agreement_scale

        done = self._env.agreed_poi is not None
        truncated = all(truncated_d[a] for a in agents)
        agreed_optimal = done and (self._agreed_poi == self._optimal_poi)
        info: dict[str, Any] = {
            "battle_won": float(done),
            "negotiation_agreed": float(done),
            "negotiation_optimal": float(agreed_optimal),
            "negotiation_quality": self._agreement_quality,
            "negotiation_length": float(self._negotiation_length),
        }
        return tuple(obs_d[a] for a in agents), rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self): pass
    def close(self): pass
