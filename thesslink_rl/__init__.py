"""ThessLink RL -- Multi-agent grid negotiation environment for EPyMARL."""

from gymnasium.envs.registration import register

# ── Legacy full-episode envs (e0/e1/e2, g16 default grid) ────────────────────
register(id="thesslink/ThessLink-e0-w0-full-v1-g16", entry_point="thesslink_rl.wrappers.v0_full:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e1-w1-full-v1-g16", entry_point="thesslink_rl.wrappers.v1_full:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e2-w2-full-v1-g16", entry_point="thesslink_rl.wrappers.v2_full:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e2-w2-full-v1-g32", entry_point="thesslink_rl.wrappers.v2_full:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — negotiation-only wrappers (w3/w4/w5) ────────────────────
register(id="thesslink/ThessLink-e3-w3-neg-v1", entry_point="thesslink_rl.wrappers.v3_neg:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-w4-neg-v1", entry_point="thesslink_rl.wrappers.v4_neg:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-w5-neg-v1", entry_point="thesslink_rl.wrappers.v5_neg:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-w5-neg-v2", entry_point="thesslink_rl.wrappers.v5_neg:GridNegotiationGymEnv")

register(id="thesslink/ThessLink-e3-w3-neg-v1-g32", entry_point="thesslink_rl.wrappers.v3_neg:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-w4-neg-v1-g32", entry_point="thesslink_rl.wrappers.v4_neg:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-w5-neg-v1-g32", entry_point="thesslink_rl.wrappers.v5_neg:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-w5-neg-v2-g32", entry_point="thesslink_rl.wrappers.v5_neg:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — navigation-only wrappers (w3/w4/w5) ─────────────────────
register(id="thesslink/ThessLink-e3-w3-nav-v1", entry_point="thesslink_rl.wrappers.v3_nav:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-w4-nav-v1", entry_point="thesslink_rl.wrappers.v4_nav:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-w5-nav-v1", entry_point="thesslink_rl.wrappers.v5_nav:GridNegotiationGymEnv")

register(id="thesslink/ThessLink-e3-w3-nav-v1-g32", entry_point="thesslink_rl.wrappers.v3_nav:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-w4-nav-v1-g32", entry_point="thesslink_rl.wrappers.v4_nav:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-w5-nav-v1-g32", entry_point="thesslink_rl.wrappers.v5_nav:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — navigation w6: 8-dir lidar + per-agent shaping ──────────
register(id="thesslink/ThessLink-e3-w6-nav-v1-g32", entry_point="thesslink_rl.wrappers.v6_nav:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — negotiation w6: minimal obs (no nav noise) ──────────────
register(id="thesslink/ThessLink-e3-w6-neg-v1-g32", entry_point="thesslink_rl.wrappers.v6_neg:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — w6 @ 64×64 (same wrappers as g32) ───────────────────────
register(id="thesslink/ThessLink-e3-w6-nav-v1-g64", entry_point="thesslink_rl.wrappers.v6_nav:GridNegotiationGymEnv", kwargs={"grid_size": 64})
register(id="thesslink/ThessLink-e3-w6-neg-v1-g64", entry_point="thesslink_rl.wrappers.v6_neg:GridNegotiationGymEnv", kwargs={"grid_size": 64})

# ── Environment v3 — full episode w7: phase_flag + v6_neg + v6_nav rewards ───
register(id="thesslink/ThessLink-e3-w7-full-v1-g32", entry_point="thesslink_rl.wrappers.v7_full:GridNegotiationGymEnv", kwargs={"grid_size": 32})
