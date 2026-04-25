"""ThessLink RL -- Multi-agent grid negotiation environment for EPyMARL."""

from gymnasium.envs.registration import register

# ── Legacy full-episode envs (e0/e1/e2, g10 default grid) ────────────────────
register(id="thesslink/ThessLink-e0-full-v0-g10", entry_point="thesslink_rl.wrappers.full.v0:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e1-full-v1-g10", entry_point="thesslink_rl.wrappers.full.v1:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e2-full-v2-g10", entry_point="thesslink_rl.wrappers.full.v2:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e2-full-v2-g32", entry_point="thesslink_rl.wrappers.full.v2:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — negotiation-only wrappers ────────────────────────────────
register(id="thesslink/ThessLink-e3-neg-v3", entry_point="thesslink_rl.wrappers.neg.v3:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-neg-v4", entry_point="thesslink_rl.wrappers.neg.v4:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-neg-v6", entry_point="thesslink_rl.wrappers.neg.v6:GridNegotiationGymEnv")

register(id="thesslink/ThessLink-e3-neg-v3-g32", entry_point="thesslink_rl.wrappers.neg.v3:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-neg-v4-g32", entry_point="thesslink_rl.wrappers.neg.v4:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-neg-v6-g32", entry_point="thesslink_rl.wrappers.neg.v6:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — navigation-only wrappers ─────────────────────────────────
register(id="thesslink/ThessLink-e3-nav-v3", entry_point="thesslink_rl.wrappers.nav.v3:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-nav-v4", entry_point="thesslink_rl.wrappers.nav.v4:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-nav-v6", entry_point="thesslink_rl.wrappers.nav.v6:GridNegotiationGymEnv")

register(id="thesslink/ThessLink-e3-nav-v3-g32", entry_point="thesslink_rl.wrappers.nav.v3:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-nav-v4-g32", entry_point="thesslink_rl.wrappers.nav.v4:GridNegotiationGymEnv", kwargs={"grid_size": 32})
register(id="thesslink/ThessLink-e3-nav-v6-g32", entry_point="thesslink_rl.wrappers.nav.v6:GridNegotiationGymEnv", kwargs={"grid_size": 32})

# ── Environment v3 — full two-phase wrappers ──────────────────────────────────
register(id="thesslink/ThessLink-e3-full-v5", entry_point="thesslink_rl.wrappers.full.v5:GridNegotiationGymEnv")
register(id="thesslink/ThessLink-e3-full-v5-g32", entry_point="thesslink_rl.wrappers.full.v5:GridNegotiationGymEnv", kwargs={"grid_size": 32})
