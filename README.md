# ThessLink RL v2 — Modular MARL with Negotiation

Multi-Agent Reinforcement Learning system where two agents **negotiate** over Points of Interest (POIs) on a grid, then **navigate** to the agreed target.

## Architecture

| File | Purpose | Lines |
|---|---|---|
| `environment.py` | PettingZoo Parallel env — 10×10 grid, obstacles, POIs, comms | ~130 |
| `evaluation.py` | POI preference scoring (reachability, centrality, peer proximity) | ~85 |
| `models.py` | Hybrid CNN (spatial) + GRU (negotiation history) → Policy + Value | ~115 |
| `negotiation.py` | Negotiation phase — agents exchange scores, GRU hidden persists | ~130 |
| `navigation.py` | Navigation phase — move to agreed POI, obstacle avoidance | ~105 |
| `train.py` | CleanRL-style PPO loop coordinating both phases | ~155 |
| `visualization.py` | Grid rendering, training curves, episode replay GIF | ~165 |

## Key Design Decisions

- **State machine**: A simple `phase` flag (`"negotiation"` / `"navigation"`) in the environment switches between phases.
- **Golden Mean reward**: `Score_A × Score_B` for the reached POI — incentivises agents to find mutually beneficial targets rather than selfish ones.
- **Shared weights**: Both agents use the same `HybridAgent` network (parameter sharing).
- **GRU persistence**: The RNN hidden state carries context from negotiation into navigation.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --total-episodes 500 --seed 42

# With visualization (saves plots/ directory)
python train.py --total-episodes 200 --seed 42 --visualize
```

## Observation Space

Each agent observes:
- **grid** `(3, 10, 10)` float32 — channels: obstacles, POIs, self position (no visibility of the other agent)
- **comm** `(3,)` float32 — the peer agent's latest POI score broadcast (the only inter-agent channel)

## Action Space

Discrete(5): stay, up, down, left, right.

## POI Scoring Formula

```
score = w_reach × reachability + w_central × centrality + w_privacy × privacy
```

Each factor is derived from the agent's YAML config (`models/*.yaml`):
- **Reachability** — energy cost (linear or exponential) vs budget (`max_steps`), obstacle clearance, and operational range (`l_inf_from_spawn` or `full_grid`). POIs outside a drone's radius score 0.
- **Centrality** — distance from grid centre.
- **Privacy** — how secluded the POI is (weighted by `privacy_emphasis` from config).

Agents have **no visibility of each other** — cooperation must emerge through negotiation only.

## Agent Configs

Define agent types as YAML files in `models/`:

```yaml
# models/drone.yaml
name: Drone
privacy_emphasis: 1.0
max_steps: 60
energy_model: linear
energy_per_step: 1.0
energy_exponential_gamma: 0.12
operational_type: l_inf_from_spawn
max_radius_cells: 6
```

Pass configs to training:

```bash
python train.py --agent0-config models/human.yaml --agent1-config models/taxi.yaml
```

## Visualization

Pass `--visualize` to `train.py` to generate:

- **`plots/training_curves.png`** — Golden Mean reward, reach rate, and policy loss over episodes
- **`plots/episode_replay.gif`** — Animated step-by-step replay of the last (or specified) episode

Use `--replay-episode N` to record a specific episode number instead of the last one.
