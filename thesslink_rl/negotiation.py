"""Negotiation phase: agents exchange POI scores over fixed rounds."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .environment import COMM_DIM, NUM_POIS, GridNegotiationEnv
from .evaluation import compute_poi_scores

NEGOTIATION_ROUNDS = 5
AGREEMENT_THRESHOLD = 0.8


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / max(norm, 1e-8)


def run_negotiation(env: GridNegotiationEnv) -> tuple[int, Dict[str, np.ndarray]]:
    """Execute the heuristic negotiation phase and return the agreed POI index.

    Each agent scores all POIs based on its config (energy + privacy),
    exchanges scores for several rounds with small noise perturbations,
    then the POI maximising the product of both agents' scores is chosen.

    Returns:
        agreed_poi: index of the chosen POI
        poi_scores: {agent_name: final scores array}
    """
    poi_scores: Dict[str, np.ndarray] = {}
    for agent in env.agents:
        spawn = tuple(env.spawn_positions[agent])
        cfg = env.agent_configs.get(agent)
        scores = compute_poi_scores(spawn, env.poi_positions, env.obstacle_map, cfg)
        poi_scores[agent] = scores
        env.set_comm(agent, scores)

    for _ in range(NEGOTIATION_ROUNDS):
        for agent in env.agents:
            updated = poi_scores[agent] + 0.05 * np.random.randn(NUM_POIS)
            poi_scores[agent] = np.clip(updated, 0.0, 1.0).astype(np.float32)
            env.set_comm(agent, poi_scores[agent])

    agreed_poi = _select_poi(poi_scores)
    env.switch_to_navigation(agreed_poi)
    return agreed_poi, poi_scores


def _select_poi(poi_scores: Dict[str, np.ndarray]) -> int:
    """Pick the POI that maximises the Golden Mean (product of scores)."""
    agents = list(poi_scores.keys())
    sa = poi_scores[agents[0]]
    sb = poi_scores[agents[1]]
    products = sa * sb
    return int(np.argmax(products))
