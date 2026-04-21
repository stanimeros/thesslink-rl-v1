"""Preference scoring: each agent rates every POI based on energy and privacy.

Energy cost is dynamic — BFS steps ``d`` from the caller’s *current_pos* to each
POI (see ``compute_poi_scores``).  ``linear``: cost ``energy_step * d``.
``exponential``: total ``energy_step * (gamma**d - 1) / (gamma - 1)`` for
``gamma != 1`` (geometric step weights ``1, gamma, gamma**2, ...`` times
``energy_step``).  YAML: ``energy_exponential_gamma`` is ``gamma``;
``energy_step`` scales the whole trip (e.g. gas vs electric).

Privacy value is static — BFS distance from spawn to each POI, divided by the
maximum BFS distance reachable from spawn on the map (not min-max across only
the three POIs).

Raw costs are **min-max normalised across the three POIs** and flipped to an
energy score; privacy is already in ``[0, 1]``.  Blend:
``(1 - privacy_emphasis) * energy + privacy_emphasis * privacy``.

**Golden-mean negotiation (same idea everywhere in logs/plots).**  For each POI
``k``, define ``g_k = ∏_a s_k^{(a)}`` (product of both agents’ POI scores) via
``golden_mean_vector``.  That is the *cooperative* compromise: no agent’s
preference dominates a single dimension; the best *joint* meeting point is
``argmax_k g_k`` (``optimal_poi``).  **Good negotiation** here means locking in
that POI; wrappers expose it as ``negotiation_optimal`` and Sacred as
``negotiation_optimal_mean`` / ``test_negotiation_optimal_mean`` (often labelled
“golden-mean” in plots).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .constants import GRID_SIZE, NUM_POIS


@dataclass
class AgentConfig:
    """Parsed agent model from a YAML file."""
    name: str
    privacy_emphasis: float         # 0-1, higher = prefers POIs far from spawn
    energy_model: str               # "linear" or "exponential"
    energy_exponential_gamma: float  # γ: geometric ratio between step costs (ignored if linear)
    energy_step: float              # scales total energy cost (e.g. gas vs electric)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            privacy_emphasis=d.get("privacy_emphasis", 0.5),
            energy_model=d.get("energy_model", "linear"),
            energy_exponential_gamma=d.get("energy_exponential_gamma", 2.0),
            energy_step=float(d.get("energy_step", 1.0)),
        )


def bfs_distances(
    origin: tuple[int, int],
    obstacle_map: np.ndarray,
) -> np.ndarray:
    """BFS shortest-path distance from *origin* to every reachable cell.

    Returns a float grid where unreachable / obstacle cells are ``np.inf``.
    """
    dist = np.full((GRID_SIZE, GRID_SIZE), np.inf, dtype=np.float64)
    dist[origin[0], origin[1]] = 0.0
    queue: deque[tuple[int, int]] = deque([origin])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if not obstacle_map[nr, nc] and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
    return dist


def _energy_cost(dist: float, cfg: AgentConfig) -> float:
    """Total energy to travel *dist* steps under the agent's energy model.

    ``linear``:      ``energy_step * dist``
    ``exponential``: ``energy_step * (γ^d - 1) / (γ - 1)`` for ``γ != 1``,
    else ``energy_step * dist`` (``γ ≤ 0`` or ``γ ≈ 1``).
    """
    scale = float(cfg.energy_step)
    if scale <= 0:
        scale = 1.0
    if not np.isfinite(dist):
        return 1e9
    if dist <= 0:
        return 0.0
    d = int(round(dist))
    if d <= 0:
        return 0.0
    if cfg.energy_model == "exponential":
        gamma = float(cfg.energy_exponential_gamma)
        if gamma <= 0:
            return scale * float(d)
        if abs(gamma - 1.0) < 1e-12:
            return scale * float(d)
        return scale * float((gamma**d - 1.0) / (gamma - 1.0))
    return scale * float(dist)


def _minmax(arr: np.ndarray, *, equal_fill: float = 0.5) -> np.ndarray:
    """Min-max normalise *arr* to [0, 1].

    When all values are equal (zero range), returns ``equal_fill`` for each
    entry.  Used for **travel costs** with ``equal_fill=0`` so ties map to
    zero before ``1 - _minmax(...)`` turns them into “all equally cheap.”
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, equal_fill, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def compute_poi_scores(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig | None = None,
) -> np.ndarray:
    """Return shape ``(NUM_POIS,)`` scores in ``[0, 1]``.

    *current_pos* is the origin for travel cost (heatmap and tooling vary it;
    the v2 RL wrapper fixes it to *spawn* for the whole episode so preferences
    stay static).

    1. Raw travel cost from *current_pos* to each POI via ``_energy_cost``.
    2. Privacy: spawn-to-POI BFS length divided by max spawn-to-cell BFS on the
       map, clipped to ``[0, 1]`` (not min-max across POIs).
    3. Energy score: min-max raw costs across the three POIs, then flip so
       cheapest → ``1``.
    4. ``(1 - α) * energy + α * privacy`` with ``α = cfg.privacy_emphasis``.
    """
    assert len(poi_positions) == NUM_POIS

    if cfg is None:
        cfg = AgentConfig(
            name="default", privacy_emphasis=0.0,
            energy_model="linear",
            energy_exponential_gamma=2.0,
            energy_step=1.0,
        )

    bfs_cur = bfs_distances(current_pos, obstacle_map)
    bfs_spn = bfs_distances(spawn, obstacle_map)

    finite_spawn = bfs_spn[np.isfinite(bfs_spn)]
    max_spawn_dist = float(finite_spawn.max()) if len(finite_spawn) > 0 else 1.0
    if max_spawn_dist < 1e-12:
        max_spawn_dist = 1.0

    raw_energy = np.zeros(NUM_POIS, dtype=np.float64)
    privacy_val = np.zeros(NUM_POIS, dtype=np.float64)

    for i, (pr, pc) in enumerate(poi_positions):
        d_cur = bfs_cur[pr, pc]
        d_spn = bfs_spn[pr, pc]
        raw_energy[i] = _energy_cost(d_cur, cfg) if np.isfinite(d_cur) else 1e9
        if np.isfinite(d_spn) and d_spn >= 0:
            privacy_val[i] = min(1.0, float(d_spn) / max_spawn_dist)
        else:
            privacy_val[i] = 0.0

    # equal_fill: ties → minmax raw cost = 0 → energy_norm = 1 (all equally cheap)
    energy_norm = 1.0 - _minmax(raw_energy, equal_fill=0.0)

    p = float(cfg.privacy_emphasis)
    scores = (1.0 - p) * energy_norm + p * privacy_val
    return scores.astype(np.float32)


def compute_eval_heatmap(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig,
) -> np.ndarray:
    """Compute a 2-D evaluation heatmap for visualization.

    Each cell's value reflects how desirable it is as a destination,
    based on proximity to the POIs weighted by their scores.

    The best-scored POI cell = 1.0; values fall off with BFS distance
    to each POI, normalised per-POI so every POI's influence fades at
    the same rate.  Obstacles = 0.
    """
    poi_scores = compute_poi_scores(
        current_pos, spawn, poi_positions, obstacle_map, cfg,
    )

    poi_bfs: list[np.ndarray] = []
    poi_max_bfs: list[float] = []
    for poi in poi_positions:
        b = bfs_distances(poi, obstacle_map)
        poi_bfs.append(b)
        finite = b[np.isfinite(b)]
        poi_max_bfs.append(float(finite.max()) if len(finite) > 0 else 1.0)

    best_poi_score = float(poi_scores.max())
    if best_poi_score == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obstacle_map[r, c]:
                continue
            best_val = 0.0
            for i, poi in enumerate(poi_positions):
                d_to_poi = poi_bfs[i][r, c]
                if np.isinf(d_to_poi):
                    continue
                max_d = poi_max_bfs[i] if poi_max_bfs[i] > 0 else 1.0
                falloff = max(1.0 - d_to_poi / max_d, 0.0)
                val = (poi_scores[i] / best_poi_score) * falloff
                best_val = max(best_val, val)
            heatmap[r, c] = best_val

    # Pin each POI cell to *that* POI's relative score.  Otherwise max_i
    # (score_i * falloff_i) lets a nearby high-scoring POI "paint" another POI's
    # cell bright green even when that POI has low preference (matches labels).
    for k, (pr, pc) in enumerate(poi_positions):
        if not obstacle_map[pr, pc]:
            heatmap[pr, pc] = float(poi_scores[k] / best_poi_score)

    return np.clip(heatmap, 0.0, 1.0)


def golden_mean_vector(
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> np.ndarray:
    """Per-POI cooperative criterion ``g_k = ∏_a s_k^{(a)}`` — shape ``(NUM_POIS,)``.

    High ``g_k`` means both agents like POI ``k`` in a balanced (multiplicative)
    way; ``optimal_poi`` picks the best joint choice.  This is what we call
    **golden-mean negotiation** in docs and plots (agreement *on that POI* =
    high-quality mutual deal, not merely any agreement).
    """
    gm = np.ones(NUM_POIS, dtype=np.float64)
    for a in agents:
        gm *= scores[a].astype(np.float64)
    return gm


def optimal_poi(
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> int:
    """POI index that maximises ``golden_mean_vector`` — the golden-mean-optimal meeting point.

    When negotiators settle on this POI, ``negotiation_optimal`` is true: they
    achieved the best *mutual* outcome under the product criterion.
    """
    return int(np.argmax(golden_mean_vector(scores, agents)))


def negotiation_quality(
    poi_idx: int,
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> float:
    """How good is the agreed POI relative to the golden-mean optimum?

    Returns a value in ``[0, 1]``: ``1.0`` = agreed on ``optimal_poi`` (best
    golden-mean joint choice), lower values = worse mutual compromise.  The
    ratio ``gm_chosen / gm_best`` makes rewards smooth near the optimum.
    """
    gm = golden_mean_vector(scores, agents)
    best = float(gm.max())
    if best < 1e-12:
        return 0.0
    return float(np.clip(gm[poi_idx] / best, 0.0, 1.0))
