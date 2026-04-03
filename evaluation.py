"""Preference scoring: each agent rates every POI from 0.0 to 1.0."""

from __future__ import annotations

import numpy as np

from environment import GRID_SIZE, NUM_POIS


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reachability_score(
    pos: tuple[int, int],
    poi: tuple[int, int],
    obstacle_map: np.ndarray,
) -> float:
    """Inverse-distance score penalised by nearby obstacle density."""
    dist = manhattan(pos, poi)
    max_dist = 2 * (GRID_SIZE - 1)
    proximity = 1.0 - dist / max_dist

    r, c = poi
    neighbours = 0
    obstacle_count = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                neighbours += 1
                if obstacle_map[nr, nc]:
                    obstacle_count += 1
    clearance = 1.0 - obstacle_count / max(neighbours, 1)

    return float(np.clip(proximity * clearance, 0.0, 1.0))


def _centrality_score(poi: tuple[int, int]) -> float:
    """How central the POI is on the grid (centre = 1, corners = low)."""
    centre = (GRID_SIZE - 1) / 2.0
    dist = abs(poi[0] - centre) + abs(poi[1] - centre)
    max_dist = 2 * centre
    return float(1.0 - dist / max_dist)


# NOTE: No peer-distance factor — agents cannot observe each other.
# Cooperation must emerge through the negotiation comm channel only.

def compute_poi_scores(
    agent_pos: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    w_reach: float = 0.7,
    w_central: float = 0.3,
) -> np.ndarray:
    """Return an array of shape (NUM_POIS,) with scores in [0, 1].

    Each agent scores POIs based only on its own view — no knowledge of
    the peer's position. Cooperation emerges through negotiation only.

    Formula per POI:
        score = w_reach * reachability + w_central * centrality
    """
    assert len(poi_positions) == NUM_POIS
    scores = np.zeros(NUM_POIS, dtype=np.float32)
    for i, poi in enumerate(poi_positions):
        reach = _reachability_score(agent_pos, poi, obstacle_map)
        central = _centrality_score(poi)
        scores[i] = w_reach * reach + w_central * central
    return np.clip(scores, 0.0, 1.0)


def golden_mean_reward(score_a: float, score_b: float) -> float:
    """Golden Mean reward: product of both agents' scores for the reached POI."""
    return score_a * score_b
