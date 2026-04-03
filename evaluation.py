"""Preference scoring: each agent rates every POI based on energy and privacy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from environment import GRID_SIZE, NUM_POIS


@dataclass
class AgentConfig:
    """Parsed agent model from a YAML file."""
    name: str
    privacy_emphasis: float         # 0-1, higher = prefers POIs far from spawn
    energy_model: str               # "linear" or "exponential"
    energy_per_step: float
    energy_exponential_gamma: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            privacy_emphasis=d.get("privacy_emphasis", 0.5),
            energy_model=d.get("energy_model", "linear"),
            energy_per_step=d.get("energy_per_step", 1.0),
            energy_exponential_gamma=d.get("energy_exponential_gamma", 0.12),
        )


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _energy_cost(dist: int, cfg: AgentConfig) -> float:
    """Total energy to travel `dist` steps under the agent's energy model."""
    if cfg.energy_model == "exponential":
        gamma = cfg.energy_exponential_gamma
        return cfg.energy_per_step * (1 - np.exp(-gamma * dist)) / gamma
    return cfg.energy_per_step * dist


def _energy_score(
    spawn: tuple[int, int],
    poi: tuple[int, int],
    cfg: AgentConfig,
) -> float:
    """How cheap it is to reach the POI under the agent's energy model.

    Normalised against the maximum possible travel cost on the grid so
    the result is in [0, 1]: 1 = very cheap, 0 = maximally expensive.
    """
    dist = manhattan(spawn, poi)
    max_dist = 2 * (GRID_SIZE - 1)
    cost = _energy_cost(dist, cfg)
    max_cost = _energy_cost(max_dist, cfg)
    if max_cost == 0:
        return 1.0
    return float(np.clip(1.0 - cost / max_cost, 0.0, 1.0))


def _privacy_score(
    spawn: tuple[int, int],
    poi: tuple[int, int],
) -> float:
    """How well visiting this POI conceals the agent's spawn location.

    A POI that is close to spawn is *low* privacy — an observer seeing the
    agent at that POI can narrow down where it started.  A distant POI is
    high privacy because many spawn locations could explain the visit.

    Returns 0.0 (POI is right at spawn) to 1.0 (POI is maximally far).
    """
    dist = manhattan(spawn, poi)
    max_dist = 2 * (GRID_SIZE - 1)
    return float(dist / max_dist)


def compute_poi_scores(
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig | None = None,
) -> np.ndarray:
    """Return an array of shape (NUM_POIS,) with scores in [0, 1].

    Only two factors:
        energy_score  — can the agent afford to reach the POI?
        privacy_score — does visiting the POI reveal the spawn location?

    Formula:
        score = (1 - p) * energy + p * privacy
    where p = cfg.privacy_emphasis (0 = pure energy, 1 = pure privacy).
    """
    assert len(poi_positions) == NUM_POIS

    if cfg is None:
        cfg = AgentConfig(
            name="default", privacy_emphasis=0.0,
            energy_model="linear", energy_per_step=1.0,
            energy_exponential_gamma=0.12,
        )

    p = cfg.privacy_emphasis
    scores = np.zeros(NUM_POIS, dtype=np.float32)
    for i, poi in enumerate(poi_positions):
        e = _energy_score(spawn, poi, cfg)
        priv = _privacy_score(spawn, poi)
        scores[i] = (1.0 - p) * e + p * priv
    return np.clip(scores, 0.0, 1.0)


def golden_mean_reward(score_a: float, score_b: float) -> float:
    """Golden Mean reward: product of both agents' scores for the reached POI."""
    return score_a * score_b
