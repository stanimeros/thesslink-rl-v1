# ThessLink RL -- Multi-Agent Grid Negotiation for EPyMARL

Two agents **negotiate** a meeting spot on a 10×10 grid, then **navigate** there. Both phases are learned with RL. Works with [EPyMARL](https://github.com/uoe-agents/epymarl) (QMIX, MAPPO, IQL, VDN, etc.).

## The task

Each episode: two agents spawn on a map with obstacles and **three POIs** (candidate meeting places). From YAML profiles they each prefer some POIs over others—**easier to reach** and/or **better for privacy** from where they started. They take turns suggesting or accepting until they **agree on one POI**, then both **walk to it**. The run ends successfully only when **both** have reached the agreed cell (whoever arrives first waits).

## Environment versions

| Version | Observation | Notes |
|--------|-------------|--------|
| **v0** | Full grid flattened (311 values) | Both agents suggest every step; agreement when they pick the **same** POI on the **same** step. |
| **v1** | Compact vector (19 values) | Turn-based suggest / accept; “GPS” toward target + obstacle sense; no full grid in the observation. |
| **v2** | Same as v1 | Same rules as v1, **plus** extra reward shaping for negotiation and navigation (the default setup in this repo). |

Local scripts use **`config.py` → `ENV_VERSION`** (default **2**). For EPyMARL, pick the matching config name, e.g. **`thesslink_v2`**.

## POI preference (evaluation)

Each POI gets a score in \([0, 1]\): a mix of **energy** (how costly to reach from where you are) and **privacy** (how far it is from your spawn), controlled by **`privacy_emphasis`** in the agent YAML. When rewards depend on “how good the deal was for everyone,” that uses the **product** of both agents’ scores on the chosen POI compared to the best possible common choice—so almost-optimal agreements still pay off.

## v2 — observations, actions, rewards

**What agents see (19 numbers):** negotiation vs navigation; three POI scores; what the other agent last suggested; which POI is locked in; own position; offset toward the target in navigation; short “lidar” distances to walls in four directions.

**Actions:**

| Action | Meaning |
|--------|---------|
| 0–4 | Stay, up, down, left, right |
| 5–7 | Suggest POI 0, 1, or 2 |
| 8 | Accept the other agent’s last suggestion |

In negotiation, **one agent moves at a time** (suggest or accept); the other is idle. In navigation, both move until both are at the POI.

**Rewards (v2 wrapper):** small bonuses for sensible negotiation moves; a larger shared bonus when they agree (scaled by how good that POI is for both); during navigation, shaping toward the goal, a small per-step cost, rewards when each agent arrives, and a final bonus when **everyone** has arrived.

## Plots and replays

With Sacred results under `epymarl/results` (or pass `--results`):

```bash
python visualize.py
```

Outputs go to `plots/<env_tag>/` (e.g. `plots/v2/`). Comparison chart and example MAPPO / IQL replays:

![Training curves — all algorithms](plots/v2/training_curves-all.png)

| MAPPO | IQL |
|-------|-----|
| ![Episode replay — MAPPO](plots/v2/episode_replay-mappo.gif) | ![Episode replay — IQL](plots/v2/episode_replay-iql.gif) |

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. EPyMARL

```bash
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl
pip install -r requirements.txt
```

### 3. Run (v2)

Copy env configs from this repo into EPyMARL:

```bash
cp ../epymarl_config/envs/*.yaml src/config/envs/
```

Examples:

```bash
python src/main.py --config=qmix --env-config=thesslink_v2
python src/main.py --config=mappo --env-config=thesslink_v2 with common_reward=False
python src/main.py --config=iql --env-config=thesslink_v2
```

Or register the gym env directly:

```bash
python src/main.py --config=qmix --env-config=gymma \
  with env_args.time_limit=60 env_args.key="thesslink_rl:thesslink/GridNegotiation-v2"
```

## Agent configs

YAML files in `thesslink_rl/models/` define each agent type, for example:

```yaml
name: Drone
privacy_emphasis: 1.0
energy_model: linear
energy_exponential_gamma: 0.12
```

## Algorithms

EPyMARL supports value-decomposition methods (QMIX, VDN, …), actor–critic and policy-gradient variants (MAPPO, IPPO, COMA, …), and independent learners (IQL, IA2C, …); use **`common_reward`** where the algorithm distinguishes team vs per-agent rewards.

## Rendering (optional)

```python
from config import GridNegotiationEnv
from thesslink_rl.visualization import render_grid, capture_frame

env = GridNegotiationEnv(seed=42)
env.reset()
render_grid(env, title="Initial State")
```
