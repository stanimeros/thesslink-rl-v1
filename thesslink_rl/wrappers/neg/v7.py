"""Gym wrapper for negotiation-only training (v7 reward shaping on v3 core env).

Key change over v6: step penalty removed entirely.

Diagnosis of v6 failure: the -0.05 per-turn step penalty caused agents to rush
into any agreement rather than negotiate for quality. Agreement rate was high
(~90%) but negotiation_optimal was only 50-80%. Removing the step penalty means
agents only care about WHAT they agree on, not HOW FAST. The timeout penalty
(-5.0) still provides urgency for the full episode horizon.

Other changes over v6:
- Suggest bonus slightly higher (0.2 vs 0.15) to give a clearer per-step
  learning signal without a step penalty to dominate it.
- Accept bonus unchanged (1.5 * GM quality).
- Convergence bonus raised (0.5 vs 0.3) — both suggesting the same POI is
  a stronger alignment signal now that agents aren't rushing.
- Optimal agreement bonus raised (35.0 vs 25.0).
- Suboptimal agreement bonus removed (0.0 vs 0.5) — bad deals give nothing.
- Wrong-agreement penalty raised (-15.0 vs -8.0) — stronger disincentive.
- Timeout penalty reduced (-5.0 vs -8.0) so agents aren't over-penalised
  when they correctly hold out but happen to hit the limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...evaluation import (
    AgentConfig,
    compute_poi_scores,
    golden_mean_vector,
    negotiation_quality,
    optimal_poi,
)
from ...environments.v3 import (
    ACT_ACCEPT,
    ACT_SUGGEST_BASE,
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_POIS,
    NUM_SUGGEST_ACTIONS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent

_NEG_STEP_PENALTY = 0.0              # removed — quality, not speed, drives learning
_NEG_SUGGEST_BONUS = 0.2             # scale for GM-proportional suggest reward
_NEG_ACCEPT_BONUS = 1.5              # scale for GM-proportional accept reward
_NEG_CONVERGENCE_BONUS = 0.5         # both suggested the same POI before accept
_NEG_OPTIMAL_AGREEMENT_BONUS = 35.0  # strong signal for reaching the best deal
_NEG_SUBOPTIMAL_AGREEMENT_BONUS = 0.0  # no reward for a bad deal
_NEG_WRONG_AGREEMENT_PENALTY = -15.0   # strong disincentive for suboptimal choice
_NEG_TIMEOUT_PENALTY = -5.0            # reduced vs v6: don't over-punish hold-outs


class GridNegotiationGymEnv(gym.Env):
    """Negotiation-only environment with terminal-on-agreement (v7 rewards)."""

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
        self._gm_vec: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._gm_norm: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0

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
        self._gm_vec = golden_mean_vector(self._poi_scores, agents)
        self._agreement_quality = 0.0
        self._negotiation_length = 0
        gm_best = float(self._gm_vec.max())
        self._gm_norm = self._gm_vec / gm_best if gm_best > 1e-12 else self._gm_vec.copy()
        obs_tuple = tuple(self._env._get_obs(a) for a in agents)
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

        obs_d, _, _, truncated_d, _ = self._env.step(actions_dict)
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            self._negotiation_length += 1
        obs_tuple = tuple(obs_d[a] for a in agents)
        rewards = [0.0] * self.n_agents

        if prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            idx = agents.index(active)
            peer = agents[1 - idx]

            # No step penalty in v7.

            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                suggested = act - ACT_SUGGEST_BASE
                rewards[idx] += _NEG_SUGGEST_BONUS * float(self._gm_norm[suggested])

            elif act == ACT_ACCEPT and peer in prev_suggestions:
                peer_suggested = prev_suggestions[peer]
                rewards[idx] += _NEG_ACCEPT_BONUS * float(self._gm_norm[peer_suggested])

                if active in prev_suggestions and prev_suggestions[active] == peer_suggested:
                    rewards[idx] += _NEG_CONVERGENCE_BONUS

        if self._agreed_poi is None and self._env.agreed_poi is not None:
            self._agreed_poi = self._env.agreed_poi
            quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
            self._agreement_quality = quality
            agreed_optimal_now = self._agreed_poi == self._optimal_poi
            for i in range(self.n_agents):
                if agreed_optimal_now:
                    rewards[i] += quality * _NEG_OPTIMAL_AGREEMENT_BONUS
                else:
                    rewards[i] += quality * _NEG_SUBOPTIMAL_AGREEMENT_BONUS
                    rewards[i] += _NEG_WRONG_AGREEMENT_PENALTY

        done = self._env.agreed_poi is not None
        truncated = all(truncated_d[a] for a in agents)
        if truncated and not done:
            for i in range(self.n_agents):
                rewards[i] += _NEG_TIMEOUT_PENALTY

        agreed_optimal = done and (self._agreed_poi == self._optimal_poi)
        info: dict[str, Any] = {
            "battle_won": float(done),
            "battle_won_negotiation": float(done),
            "battle_won_navigation": 0.0,
            "episode_length_negotiation": float(self._negotiation_length),
            "episode_length_navigation": 0.0,
            "negotiation_agreed": float(done),
            "negotiation_optimal": float(agreed_optimal),
            "agreement_quality": self._agreement_quality,
            "negotiation_length": float(self._negotiation_length),
        }
        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
