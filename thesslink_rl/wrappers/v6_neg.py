"""Gym wrapper for negotiation-only training (w6: minimal obs on v3 core env).

Improvements over v5_neg:

- Minimal neg-specific observation — strips the three navigation features
  that are constant noise in a negotiation-only episode:
    - self_pos     (2): position irrelevant during negotiation
    - relative_pos (2): always zero (no agreed POI until acceptance)
    - lidar_4dir   (4): obstacle distances carry no negotiation signal
  Only the three actually informative features are kept.

Observation layout (size 10):
  self_scores   (3)  agent's preference scores for the 3 POIs
  peer_action   (4)  one-hot: [no_action, suggest_0, suggest_1, suggest_2]
  agreed_poi    (3)  one-hot: which POI was agreed (all 0 until agreement)

All reward logic is identical to v5_neg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..evaluation import (
    AgentConfig,
    compute_poi_scores,
    golden_mean_vector,
    negotiation_quality,
    optimal_poi,
)
from ..environments.v3 import (
    ACT_ACCEPT,
    ACT_SUGGEST_BASE,
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_POIS,
    NUM_SUGGEST_ACTIONS,
    PEER_ACTION_DIM,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent

# self_scores(3) + peer_action(4) + agreed_poi(3)
OBS_FLAT_SIZE_V6 = NUM_POIS + PEER_ACTION_DIM + NUM_POIS  # 10


class GridNegotiationGymEnv(gym.Env):
    """Negotiation-only env with minimal obs: scores + peer action + agreed POI (w6)."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        grid_size: int = GRID_SIZE,
        step_penalty: float = -0.05,
        suggest_bonus: float = 0.15,
        accept_bonus: float = 1.5,
        convergence_bonus: float = 0.3,
        optimal_agreement_bonus: float = 25.0,
        suboptimal_agreement_bonus: float = 0.5,
        wrong_agreement_penalty: float = -8.0,
        timeout_penalty: float = -8.0,
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
        self._step_penalty = step_penalty
        self._suggest_bonus = suggest_bonus
        self._accept_bonus = accept_bonus
        self._convergence_bonus = convergence_bonus
        self._optimal_agreement_bonus = optimal_agreement_bonus
        self._suboptimal_agreement_bonus = suboptimal_agreement_bonus
        self._wrong_agreement_penalty = wrong_agreement_penalty
        self._timeout_penalty = timeout_penalty
        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=0.0, high=1.0, shape=(OBS_FLAT_SIZE_V6,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._gm_vec: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._gm_norm: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0

    def _get_neg_obs(self, agents: list[str]) -> tuple[np.ndarray, ...]:
        """Build minimal neg obs: self_scores + peer_action + agreed_poi (size 10)."""
        obs_list = []
        for a in agents:
            own_scores = self._poi_scores.get(a, np.zeros(NUM_POIS, dtype=np.float32))

            peer = [x for x in agents if x != a][0]
            peer_action = np.zeros(PEER_ACTION_DIM, dtype=np.float32)
            if peer in self._env.last_suggestion:
                peer_action[self._env.last_suggestion[peer] + 1] = 1.0
            else:
                peer_action[0] = 1.0

            agreed_onehot = np.zeros(NUM_POIS, dtype=np.float32)
            if self._env.agreed_poi is not None:
                agreed_onehot[self._env.agreed_poi] = 1.0

            obs_list.append(np.concatenate([own_scores, peer_action, agreed_onehot]))
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
        self._agreed_poi = None
        self._optimal_poi = optimal_poi(self._poi_scores, agents)
        self._gm_vec = golden_mean_vector(self._poi_scores, agents)
        self._agreement_quality = 0.0
        self._negotiation_length = 0
        gm_best = float(self._gm_vec.max())
        self._gm_norm = self._gm_vec / gm_best if gm_best > 1e-12 else self._gm_vec.copy()
        obs_tuple = self._get_neg_obs(agents)
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

        _, _, _, truncated_d, _ = self._env.step(actions_dict)
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            self._negotiation_length += 1
        rewards = [0.0] * self.n_agents

        if prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            idx = agents.index(active)
            peer = agents[1 - idx]

            rewards[idx] += self._step_penalty

            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                suggested = act - ACT_SUGGEST_BASE
                rewards[idx] += self._suggest_bonus * float(self._gm_norm[suggested])

            elif act == ACT_ACCEPT and peer in prev_suggestions:
                peer_suggested = prev_suggestions[peer]
                rewards[idx] += self._accept_bonus * float(self._gm_norm[peer_suggested])

                if active in prev_suggestions and prev_suggestions[active] == peer_suggested:
                    rewards[idx] += self._convergence_bonus

        if self._agreed_poi is None and self._env.agreed_poi is not None:
            self._agreed_poi = self._env.agreed_poi
            quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
            self._agreement_quality = quality
            agreed_optimal_now = self._agreed_poi == self._optimal_poi
            for i in range(self.n_agents):
                if agreed_optimal_now:
                    rewards[i] += quality * self._optimal_agreement_bonus
                else:
                    rewards[i] += quality * self._suboptimal_agreement_bonus
                    rewards[i] += self._wrong_agreement_penalty

        done = self._env.agreed_poi is not None
        truncated = all(truncated_d[a] for a in agents)
        if truncated and not done:
            for i in range(self.n_agents):
                rewards[i] += self._timeout_penalty

        agreed_optimal = done and (self._agreed_poi == self._optimal_poi)
        info: dict[str, Any] = {
            "battle_won": float(done),
            "negotiation_agreed": float(done),
            "negotiation_optimal": float(agreed_optimal),
            "negotiation_quality": self._agreement_quality,
            "negotiation_length": float(self._negotiation_length),
        }
        return self._get_neg_obs(agents), rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
