"""Core grid environment: 10x10 grid with obstacles, POIs, and comms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .evaluation import AgentConfig

GRID_SIZE = 10
NUM_OBSTACLES = 10  # 10% of 100 cells
NUM_POIS = 3
NUM_AGENTS = 2
COMM_DIM = NUM_POIS  # each agent broadcasts its POI scores
MAX_EPISODE_STEPS = 60
ACTION_DIM = 5  # 0=stay, 1=up, 2=down, 3=left, 4=right

# Grid channel indices (C, H, W) with C=3
CH_OBSTACLE = 0
CH_POI = 1
CH_SELF = 2
NUM_CHANNELS = 3

# Flat observation size: grid (C*H*W) + comm (COMM_DIM)
OBS_FLAT_SIZE = NUM_CHANNELS * GRID_SIZE * GRID_SIZE + COMM_DIM


class GridNegotiationEnv:
    """Two agents negotiate over POIs then navigate to the agreed one."""

    metadata = {"name": "grid_negotiation_v0", "render_modes": ["human"]}

    def __init__(
        self,
        agent_configs: Dict[str, AgentConfig] | None = None,
        render_mode: Optional[str] = None,
        seed: int = 0,
    ):
        self.render_mode = render_mode
        self._seed = seed
        self.possible_agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.agent_configs = agent_configs or {}

        self._rng = np.random.RandomState(seed)
        self._build_static_map()

    def reset(self, seed=None, options=None) -> tuple[Dict, Dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._build_static_map()

        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.agent_positions: Dict[str, list[int]] = {}
        for agent in self.possible_agents:
            self.agent_positions[agent] = self._random_free_cell()
        self.spawn_positions = {a: list(p) for a, p in self.agent_positions.items()}

        self.comm_buffer: Dict[str, np.ndarray] = {
            a: np.zeros(COMM_DIM, dtype=np.float32) for a in self.agents
        }
        self.phase = "negotiation"
        self.agreed_poi: Optional[int] = None

        obs = {a: self._get_obs(a) for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, info

    def step(self, actions: Dict[str, int]):
        self.timestep += 1
        for agent, act in actions.items():
            self._apply_move(agent, act)

        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: self.timestep >= MAX_EPISODE_STEPS for a in self.agents}
        infos: Dict[str, dict] = {a: {"phase": self.phase} for a in self.agents}

        if self.phase == "navigation" and self.agreed_poi is not None:
            target = self.poi_positions[self.agreed_poi]
            for a in self.agents:
                if tuple(self.agent_positions[a]) == target:
                    terminated = {ag: True for ag in self.agents}
                    infos = {ag: {"reached_poi": self.agreed_poi} for ag in self.agents}
                    break

        if any(truncated.values()):
            self.agents = []
        elif any(terminated.values()):
            self.agents = []

        return obs, rewards, terminated, truncated, infos

    # --- Helpers ----------------------------------------------------------

    def _build_static_map(self):
        self.obstacle_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        placed = 0
        while placed < NUM_OBSTACLES:
            r, c = self._rng.randint(0, GRID_SIZE, size=2)
            if not self.obstacle_map[r, c]:
                self.obstacle_map[r, c] = True
                placed += 1

        self.poi_positions: list[tuple[int, int]] = []
        while len(self.poi_positions) < NUM_POIS:
            r, c = int(self._rng.randint(0, GRID_SIZE)), int(self._rng.randint(0, GRID_SIZE))
            if not self.obstacle_map[r, c] and (r, c) not in self.poi_positions:
                self.poi_positions.append((r, c))

    def _random_free_cell(self) -> list[int]:
        while True:
            r, c = int(self._rng.randint(0, GRID_SIZE)), int(self._rng.randint(0, GRID_SIZE))
            if not self.obstacle_map[r, c] and (r, c) not in self.poi_positions:
                return [r, c]

    def _apply_move(self, agent: str, action: int):
        dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r, c = self.agent_positions[agent]
        nr, nc = max(0, min(GRID_SIZE - 1, r + dr)), max(0, min(GRID_SIZE - 1, c + dc))
        if not self.obstacle_map[nr, nc]:
            self.agent_positions[agent] = [nr, nc]

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        grid = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[CH_OBSTACLE] = self.obstacle_map.astype(np.float32)
        for pr, pc in self.poi_positions:
            grid[CH_POI, pr, pc] = 1.0

        r, c = self.agent_positions[agent]
        grid[CH_SELF, r, c] = 1.0

        other = [a for a in self.possible_agents if a != agent][0]
        peer_comm = self.comm_buffer[other].copy()
        return {"grid": grid, "comm": peer_comm}

    def set_comm(self, agent: str, msg: np.ndarray):
        self.comm_buffer[agent] = np.clip(msg, 0.0, 1.0).astype(np.float32)

    def switch_to_navigation(self, poi_idx: int):
        self.phase = "navigation"
        self.agreed_poi = poi_idx
