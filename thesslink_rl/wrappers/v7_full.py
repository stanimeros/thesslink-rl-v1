"""Gymnasium multi-agent wrapper for EPyMARL — v7 full (negotiation → navigation).

Combines v6_neg and v6_nav reward logic into a single two-phase episode.
A ``phase_flag`` feature is prepended to every agent observation so that a
shared policy can distinguish the two phases without ambiguity.

Observation layout (size 23):
  phase_flag     (1)   0.0 = negotiation, 1.0 = navigation
  self_scores    (3)   agent's POI preference scores
  peer_action    (4)   one-hot last peer action [no_action, s0, s1, s2]
  agreed_poi     (3)   one-hot agreed POI (zeros until agreement)
  self_pos       (2)   (row, col) normalised to [0, 1]
  relative_pos   (2)   (target − self) / (grid_size − 1); zero before agreement
  lidar_card     (4)   N, S, E, W distances normalised to [0, 1]
  lidar_diag     (4)   NE, SE, SW, NW distances normalised to [0, 1]

Phase 1 — Negotiation (v6_neg reward logic):
  step_penalty per active-agent turn
  suggest_bonus  * gm_norm[suggested_poi]
  accept_bonus   * gm_norm[peer_poi]
  convergence_bonus if active also suggested the same POI
  On agreement:  optimal_agreement_bonus * quality
              or suboptimal_agreement_bonus * quality + wrong_agreement_penalty
  On timeout:    neg_timeout_penalty to all agents

Phase 2 — Navigation (v6_nav reward logic):
  Per-agent potential-based shaping normalised by own initial BFS distance
  nav_step_penalty per live (not-yet-arrived) agent per step
  first_arrival_bonus to the first agent that reaches the POI
  arrival_scale * quality when each individual agent reaches the POI
  nav_timeout_penalty to each agent that did not arrive within nav_time_limit
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

# phase_flag(1) + self_scores(3) + peer_action(4) + agreed_poi(3)
# + self_pos(2) + relative_pos(2) + lidar_card(4) + lidar_diag(4)
OBS_FLAT_SIZE_V8 = 1 + NUM_POIS + PEER_ACTION_DIM + NUM_POIS + 2 + 2 + 4 + 4  # 23


def _diag_lidar(obstacle_map: np.ndarray, grid_size: int, r: int, c: int) -> np.ndarray:
    """Cast rays in NE, SE, SW, NW; return normalised distance to nearest obstacle."""
    directions = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # NE, SE, SW, NW
    distances = np.ones(4, dtype=np.float32)
    max_dist = grid_size - 1
    for i, (dr, dc) in enumerate(directions):
        for step in range(1, grid_size):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size:
                distances[i] = step / max_dist
                break
            if obstacle_map[nr, nc]:
                distances[i] = step / max_dist
                break
        else:
            distances[i] = 1.0
    return distances


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray, norm_dist: float) -> float:
    """Phi = -d / norm_dist in [-1, 0].  0 at goal, -1 when unreachable.

    norm_dist is each agent's own initial BFS distance, giving a consistent
    per-step gradient regardless of episode difficulty.
    """
    if norm_dist <= 0:
        return 0.0
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / norm_dist


class GridNegotiationGymEnv(gym.Env):
    """Full two-phase env (negotiation → navigation) with v6 reward logic (w7)."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        grid_size: int = GRID_SIZE,
        # ── Negotiation params (v6_neg) ──────────────────────────────────
        step_penalty: float = -0.05,
        suggest_bonus: float = 0.2,
        accept_bonus: float = 1.5,
        convergence_bonus: float = 0.5,
        optimal_agreement_bonus: float = 35.0,
        suboptimal_agreement_bonus: float = 0.0,
        wrong_agreement_penalty: float = -12.0,
        neg_timeout_penalty: float = -5.0,
        neg_time_limit: int = 32,
        # ── Navigation params (v6_nav) ───────────────────────────────────
        shaping_gamma: float = 0.99,
        nav_step_penalty: float = -0.001,
        arrival_scale: float = 25.0,
        nav_timeout_penalty: float = -8.0,
        first_arrival_bonus: float = 5.0,
        nav_time_limit: int = 320,
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

        # Negotiation params
        self._step_penalty = step_penalty
        self._suggest_bonus = suggest_bonus
        self._accept_bonus = accept_bonus
        self._convergence_bonus = convergence_bonus
        self._optimal_agreement_bonus = optimal_agreement_bonus
        self._suboptimal_agreement_bonus = suboptimal_agreement_bonus
        self._wrong_agreement_penalty = wrong_agreement_penalty
        self._neg_timeout_penalty = neg_timeout_penalty
        self._neg_time_limit = neg_time_limit

        # Navigation params
        self._shaping_gamma = shaping_gamma
        self._nav_step_penalty = nav_step_penalty
        self._arrival_scale = arrival_scale
        self._nav_timeout_penalty = nav_timeout_penalty
        self._first_arrival_bonus = first_arrival_bonus
        self._nav_time_limit = nav_time_limit

        self.n_agents = NUM_AGENTS
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE_V8,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        # Episode state
        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0
        self._gm_vec: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._gm_norm: np.ndarray = np.zeros(NUM_POIS, dtype=np.float64)
        self._agreement_quality: float = 0.0
        self._negotiation_length: int = 0
        self._target_bfs: np.ndarray | None = None
        self._prev_potentials: Dict[str, float] = {}
        self._individual_arrived: Dict[str, bool] = {}
        self._initial_dist: Dict[str, float] = {}
        self._nav_steps: int = 0
        self._first_arrived: bool = False

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self, agents: list[str]) -> tuple[np.ndarray, ...]:
        """Build unified obs for both phases (size 23).

        All features are populated at every step.  The phase_flag guides the
        policy on which feature subset is meaningful this turn.
        """
        phase_flag = np.array(
            [1.0 if self._env.phase == "navigation" else 0.0], dtype=np.float32
        )
        norm = self._grid_size - 1
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

            r, c = self._env.agent_positions[a]
            self_pos = np.array([r / norm, c / norm], dtype=np.float32)

            relative_pos = np.zeros(2, dtype=np.float32)
            if self._agreed_poi is not None:
                tr, tc = self._env.poi_positions[self._agreed_poi]
                relative_pos[0] = (tr - r) / norm
                relative_pos[1] = (tc - c) / norm

            card = self._env._lidar(r, c)
            diag = _diag_lidar(self._env.obstacle_map, self._grid_size, r, c)

            obs_list.append(
                np.concatenate([phase_flag, own_scores, peer_action, agreed_onehot,
                                self_pos, relative_pos, card, diag])
            )
        return tuple(obs_list)

    # ── Reset ─────────────────────────────────────────────────────────────────

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
        gm_best = float(self._gm_vec.max())
        self._gm_norm = self._gm_vec / gm_best if gm_best > 1e-12 else self._gm_vec.copy()

        self._agreement_quality = 0.0
        self._negotiation_length = 0
        self._target_bfs = None
        self._prev_potentials = {}
        self._individual_arrived = {a: False for a in agents}
        self._initial_dist = {}
        self._nav_steps = 0
        self._first_arrived = False

        info = {
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
            "optimal_poi": self._optimal_poi,
        }
        return self._get_obs(agents), info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}

        prev_phase = self._env.phase
        prev_neg_turn = self._env.neg_turn
        prev_suggestions = dict(self._env.last_suggestion)

        self._env.step(actions_dict)
        rewards = [0.0] * self.n_agents

        # ── Negotiation rewards (v6_neg logic) ───────────────────────────
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            self._negotiation_length += 1
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

        # ── Agreement transition: initialise navigation state ─────────────
        if self._agreed_poi is None and self._env.agreed_poi is not None:
            self._agreed_poi = self._env.agreed_poi
            quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)
            self._agreement_quality = quality
            agreed_optimal = self._agreed_poi == self._optimal_poi
            for i in range(self.n_agents):
                if agreed_optimal:
                    rewards[i] += quality * self._optimal_agreement_bonus
                else:
                    rewards[i] += quality * self._suboptimal_agreement_bonus
                    rewards[i] += self._wrong_agreement_penalty

            target = self._env.poi_positions[self._agreed_poi]
            self._target_bfs = bfs_distances(target, self._env.obstacle_map)
            for a in agents:
                pos = tuple(self._env.agent_positions[a])
                d = self._target_bfs[pos[0], pos[1]]
                self._initial_dist[a] = float(d) if (np.isfinite(d) and d > 0) else 1.0
                self._prev_potentials[a] = _potential(
                    pos, self._target_bfs, self._initial_dist[a]
                )

        # ── Navigation rewards (v6_nav logic) ────────────────────────────
        if prev_phase == "navigation" and self._agreed_poi is not None and self._target_bfs is not None:
            self._nav_steps += 1
            quality = negotiation_quality(self._agreed_poi, self._poi_scores, agents)

            for i, a in enumerate(agents):
                if self._individual_arrived.get(a, False):
                    continue

                cur_pos = tuple(self._env.agent_positions[a])
                cur_phi = _potential(cur_pos, self._target_bfs, self._initial_dist[a])
                prev_phi = self._prev_potentials.get(a, cur_phi)
                rewards[i] += self._shaping_gamma * cur_phi - prev_phi
                self._prev_potentials[a] = cur_phi
                rewards[i] += self._nav_step_penalty

                if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                    self._individual_arrived[a] = True
                    if not self._first_arrived:
                        self._first_arrived = True
                        rewards[i] += self._first_arrival_bonus
                    rewards[i] += quality * self._arrival_scale

        # ── Termination ───────────────────────────────────────────────────
        all_reached = all(self._env.agents_reached.get(a, False) for a in agents)

        neg_timeout = (
            self._agreed_poi is None
            and self._negotiation_length >= self._neg_time_limit
        )
        nav_timeout = self._nav_steps >= self._nav_time_limit

        if neg_timeout:
            for i in range(self.n_agents):
                rewards[i] += self._neg_timeout_penalty

        if nav_timeout and not all_reached:
            for i, a in enumerate(agents):
                if not self._individual_arrived.get(a, False):
                    rewards[i] += self._nav_timeout_penalty

        done = all_reached or neg_timeout
        truncated = nav_timeout and not all_reached

        mean_opt = (
            sum(self._initial_dist[a] for a in agents) / self.n_agents
            if self._initial_dist else 0.0
        )
        nav_eff = (
            min(1.0, mean_opt / self._nav_steps)
            if (all_reached and self._nav_steps > 0) else 0.0
        )
        agreed = self._agreed_poi is not None
        agreed_optimal = agreed and (self._agreed_poi == self._optimal_poi)

        info: dict[str, Any] = {
            "battle_won": float(all_reached),
            "negotiation_agreed": float(agreed),
            "negotiation_optimal": float(agreed_optimal),
            "negotiation_quality": self._agreement_quality,
            "negotiation_length": float(self._negotiation_length),
            "navigation_length": float(self._nav_steps),
            "navigation_quality": nav_eff,
        }
        return self._get_obs(agents), rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        return [self._env.get_avail_actions(a) for a in self._env.possible_agents]

    def render(self):
        pass

    def close(self):
        pass
