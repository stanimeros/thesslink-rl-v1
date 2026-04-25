"""Gym wrapper for negotiation-only training (v6 reward shaping on v3 core env).

Improvements over v5_neg:
- Convergence bonus: when both agents have already suggested the same POI,
  the accepting agent gets an extra reward, reinforcing mutual agreement signals.
- Suggest bonus scales with golden-mean score (both agents' product), not just
  the active agent's own score — aligns individual suggestions with cooperative optimum.
- Higher optimal agreement bonus and sharper suboptimal penalty to widen the
  gap between good and bad outcomes.
- Peer-suggestion quality visible in accept reward: uses golden-mean quality
  of the peer's suggestion rather than the agent's raw own-score, so both
  agents are pulled toward cooperative POI choice.
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

_NEG_STEP_PENALTY = -0.05            # per negotiation turn
_NEG_SUGGEST_BONUS = 0.15            # scale for GM-proportional suggest reward
_NEG_ACCEPT_BONUS = 1.5              # scale for GM-proportional accept reward
_NEG_CONVERGENCE_BONUS = 0.3         # both suggested the same POI before accept (new in v6)
_NEG_OPTIMAL_AGREEMENT_BONUS = 25.0  # was 22.0
_NEG_SUBOPTIMAL_AGREEMENT_BONUS = 0.5    # reduced: bad deal gives almost nothing
_NEG_WRONG_AGREEMENT_PENALTY = -8.0     # was -5.0: strong disincentive
_NEG_TIMEOUT_PENALTY = -8.0          # was -5.0


class GridNegotiationGymEnv(gym.Env):
    """Negotiation-only environment with terminal-on-agreement (v6 rewards)."""

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
        self._agreement_quality: float = 0.0

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
        prev_neg_turn = self._env.neg_turn
        prev_suggestions = dict(self._env.last_suggestion)

        obs_d, _, _, truncated_d, _ = self._env.step(actions_dict)
        obs_tuple = tuple(obs_d[a] for a in agents)
        rewards = [0.0] * self.n_agents

        if prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            idx = agents.index(active)
            peer = agents[1 - idx]

            rewards[idx] += _NEG_STEP_PENALTY

            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                suggested = act - ACT_SUGGEST_BASE
                # Reward proportional to golden-mean quality of this suggestion
                rewards[idx] += _NEG_SUGGEST_BONUS * float(self._gm_norm[suggested])

            elif act == ACT_ACCEPT and peer in prev_suggestions:
                peer_suggested = prev_suggestions[peer]
                # Reward proportional to golden-mean quality of the accepted POI
                rewards[idx] += _NEG_ACCEPT_BONUS * float(self._gm_norm[peer_suggested])

                # Convergence bonus: both previously suggested the same POI
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
            "negotiation_agreed": float(done),
            "negotiation_optimal": float(agreed_optimal),
            "agreement_quality": self._agreement_quality,
        }
        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
