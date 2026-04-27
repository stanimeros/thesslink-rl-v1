# ThessLink RL — Multi-Agent Grid Negotiation

Two agents **score POIs**, **negotiate** a meeting spot, then **navigate** there together. All three phases are evaluated analytically; the negotiation and navigation are learned end-to-end with RL using [EPyMARL](https://github.com/uoe-agents/epymarl).

## What the agents do

Each episode unfolds in two phases:

1. **Negotiation.** The agents have different profiles (energy model + privacy emphasis). They evaluate each of the three candidate POIs on the grid, then take turns suggesting or accepting until they agree on one. The goal is to converge on the *golden-mean* POI — the one that maximises the product of both agents' scores.

2. **Navigation.** Both agents walk to the agreed POI. The first to arrive waits; the episode ends only when **both** stand on the agreed cell.

---

## Scoring and evaluation

Each agent loads a YAML profile from `thesslink_rl/models/`:
- **privacy emphasis** α ∈ [0, 1]
- **energy model** — linear or exponential, with optional γ and step weight *w*

**BFS distances.** For POI *k*: *d_k* = steps from current cell, *s_k* = steps from spawn. *D_max* = largest reachable spawn distance on the map.

**Travel cost:**

$$
C_k =
\begin{cases}
w\,d_k & \text{linear} \\[0.4em]
w\,\dfrac{\gamma^{d_k}-1}{\gamma-1} & \text{exponential},\ \gamma\neq 1
\end{cases}
$$

**Privacy** (farther from spawn = higher):

$$
P_k = \min\!\left(1,\ \frac{s_k}{D_{\max}}\right)
$$

**Energy value** $\tilde{E}_k$: costs min–maxed across POIs, then flipped (cheapest = 1).

**POI score:**

$$
s_k = (1-\alpha)\,\tilde{E}_k + \alpha\,P_k
$$

**Negotiation quality** after agreement on POI *k*★:

$$
g_k = \prod_a s_k^{(a)}, \qquad
\text{quality} = \frac{g_{k^\star}}{\max_\ell g_\ell} \in [0,1]
$$

This scalar multiplies all agreement and navigation bonuses. **Golden-mean (GM)** means both agents agreed on `argmax_k g_k` — the strongest possible cooperative outcome.

---

## Codebase structure

The project is organised so that each improvement to the task, observation, or reward lives in its own versioned file. This makes it easy to run ablations, compare versions, and roll back.

```
thesslink_rl/
  environments/       Core gym envs (v0–v3): grid dynamics, actions, observations
    v0.py             Full 10×10 grid observation, simultaneous suggestions
    v1.py             Compact 19-dim obs, turn-based suggest/accept
    v2.py             v1 + reward shaping
    v3.py             v2 without phase flag in obs; configurable grid size

  wrappers/           EPyMARL-compatible Gymnasium wrappers
    v0_full.py  …     Early full-episode wrappers
    v3_neg.py         Negotiation-only wrapper on v3 core env
    v3_nav.py         Navigation-only wrapper on v3 core env (pretrained neg policy)
    v4_neg.py   …     Incremental reward improvements per wrapper version
    v5_nav.py   …
    v6_neg.py         Minimal 10-dim obs (scores + peer action + agreed POI only)
    v6_nav.py         Navigation wrapper with diagonal lidar + per-agent shaping
    v7_full.py        Full two-phase episode (neg → nav) with v6 reward logic + phase flag

  evaluation.py       Scoring, BFS, golden-mean, negotiation quality
  env_catalog.py      Auto-discovers YAML configs; resolves env selectors by index/alias

epymarl_config/
  envs/               One YAML per registered environment variant
    thesslink_e3_w3_neg_v1_g32.yaml   neg-only, wrapper w3, env v3, grid 32
    thesslink_e3_w7_full_v1_g32.yaml  full episode, wrapper w7, env v3, grid 32
    …
  algs/               Algorithm overrides (QMIX, MAPPO, IPPO, IQL)

config.py             Reads THESSLINK_ENV and imports the right env class
```

**Naming convention** for env YAMLs: `thesslink_e{env_version}_w{wrapper_version}_{phase}_{obs_version}_g{grid_size}.yaml`
- `phase` = `neg` (negotiation only), `nav` (navigation only), or `full` (both)
- `obs_version` = `v1` or `v2` depending on the observation schema used inside the wrapper

**Adding a new version:**
1. Add or extend a core env in `thesslink_rl/environments/` if dynamics change.
2. Add a wrapper in `thesslink_rl/wrappers/` with the new reward/obs logic.
3. Add a YAML in `epymarl_config/envs/` — `env_catalog.py` picks it up automatically.
4. Run with `THESSLINK_ENV=<alias>` or `THESSLINK_ENV=<index>`.

---

## Environment versions at a glance

| Env file | Grid | Obs size | Phase flag | Notes |
|----------|------|----------|-----------|-------|
| v0 | 10 | 311 | yes | Full grid flattened; simultaneous suggestions |
| v1 | 10 | 19 | yes | Compact vector; turn-based |
| v2 | 10 | 19 | yes | v1 + reward shaping |
| v3 | 16/32 | 18 | **no** | No phase flag (wrapper adds it if needed); configurable grid |

## Wrapper versions at a glance

| Wrapper | Phase | Obs size | Key improvement |
|---------|-------|----------|-----------------|
| v0–v2 | full | 311/19 | Early baselines |
| v3_neg | neg | 18 | First neg-only split |
| v3_nav | nav | 18 | Nav-only with pretrained neg policy |
| v4_neg/nav | neg/nav | 18/18 | Refined shaping |
| v5_neg/nav | neg/nav | 18/18 | Golden-mean-weighted suggest bonus; convergence bonus |
| v6_neg | neg | **10** | Minimal obs (strips pos/lidar — irrelevant in neg) |
| v6_nav | nav | 22 | + diagonal lidar (8-dir), per-agent potential shaping |
| **v7_full** | **full** | **23** | Phase flag prepended; v6 rewards; first successful learning on grid-32 |

---

## Observation — v7_full (current best)

Flat vector length **23**.

| Block | Size | Content |
|-------|------|---------|
| Phase flag | 1 | 0 = negotiation, 1 = navigation |
| POI scores | 3 | Agent's preference score for POI 0, 1, 2 |
| Peer action | 4 | One-hot: no suggestion, or peer suggested POI 0/1/2 |
| Agreed POI | 3 | One-hot of locked-in POI (zeros until agreement) |
| Self position | 2 | (row, col) normalised by grid extent |
| Relative offset | 2 | Toward agreed POI during navigation (zeros in negotiation) |
| Lidar cardinal | 4 | N, S, E, W distance to nearest obstacle |
| Lidar diagonal | 4 | NE, SE, SW, NW distance to nearest obstacle |

## Actions

| ID | Meaning |
|----|---------|
| 0 | Stay |
| 1 | Up |
| 2 | Down |
| 3 | Left |
| 4 | Right |
| 5 | Suggest POI 0 |
| 6 | Suggest POI 1 |
| 7 | Suggest POI 2 |
| 8 | Accept peer's last suggestion |

In **negotiation**, only the active agent may use actions 5–8; the other is restricted to Stay. In **navigation**, both use 0–4.

## Rewards — v7_full

| Phase | Event | Reward |
|-------|-------|--------|
| Negotiation | Active agent suggests | `suggest_bonus × gm_norm[poi]` |
| Negotiation | Accept peer's suggestion | `accept_bonus × gm_quality` |
| Negotiation | Accept same POI as own suggestion | + `convergence_bonus` |
| Negotiation | Agreement (optimal POI) | `optimal_agreement_bonus × quality` |
| Negotiation | Agreement (suboptimal) | `suboptimal_agreement_bonus × quality − wrong_agreement_penalty` |
| Negotiation | Timeout | `neg_timeout_penalty` to all |
| Navigation | Each step (live agent) | Potential-based shaping + `nav_step_penalty` |
| Navigation | First agent reaches POI | `first_arrival_bonus` |
| Navigation | Agent reaches POI | `arrival_scale × quality` |
| Navigation | Timeout | `nav_timeout_penalty` per agent that didn't arrive |

---

## Running

```bash
# Local visualisation / smoke test
THESSLINK_ENV=e3_w7_full_v1_g32 python smoke_test.py
THESSLINK_ENV=e3_w7_full_v1_g32 python visualize.py

# EPyMARL training (IQL recommended for grid-32)
python epymarl/src/main.py \
  --config=iql \
  --env-config=thesslink_e3_w7_full_v1_g32

# List all registered environments
python -c "from thesslink_rl.env_catalog import available_env_catalog; [print(e['index'], e['alias']) for e in available_env_catalog()]"
```

## Algorithms

EPyMARL supports QMIX, VDN, MAPPO, IPPO, COMA, IQL, IA2C. **`common_reward`** is set per env YAML; QMIX/VDN/COMA always use `common_reward=True`. Override on the command line: `with common_reward=false`.

**Recommended per phase:**
- Negotiation-only (`neg`): IQL or MAPPO with `common_reward=false`
- Navigation-only (`nav`): IQL
- Full episode (`full`) on grid-32: **IQL** (first confirmed learning; see breakthrough notes)

## Plots and replays

```bash
THESSLINK_ENV=e3_w7_full_v1_g32 python visualize.py
```

Outputs under `plots/<env_tag>/`.

![Agent evaluation heatmaps](plots/v2/eval_heatmaps.png)
