"""ThessLink RL -- Multi-agent grid negotiation environment for EPyMARL."""

from gymnasium.envs.registration import register

register(
    id="thesslink/ThessLink-v0",
    entry_point="thesslink_rl.v0.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v1",
    entry_point="thesslink_rl.v1.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v2",
    entry_point="thesslink_rl.v2.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v3-neg",
    entry_point="thesslink_rl.v3_neg.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v3-nav",
    entry_point="thesslink_rl.v3_nav.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v4-neg",
    entry_point="thesslink_rl.v4_neg.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/ThessLink-v4-nav",
    entry_point="thesslink_rl.v4_nav.gym_wrapper:GridNegotiationGymEnv",
)

# 32×32 grid variants
register(
    id="thesslink/ThessLink-v2-g32",
    entry_point="thesslink_rl.v2.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)
register(
    id="thesslink/ThessLink-v4-neg-g32",
    entry_point="thesslink_rl.v4_neg.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)
register(
    id="thesslink/ThessLink-v4-nav-g32",
    entry_point="thesslink_rl.v4_nav.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)
register(
    id="thesslink/ThessLink-v5-g32",
    entry_point="thesslink_rl.v5.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)
register(
    id="thesslink/ThessLink-v6-neg-g32",
    entry_point="thesslink_rl.v6_neg.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)
register(
    id="thesslink/ThessLink-v6-nav-g32",
    entry_point="thesslink_rl.v6_nav.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 32},
)

# 64×64 grid variants
register(
    id="thesslink/ThessLink-v2-g64",
    entry_point="thesslink_rl.v2.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 64},
)
register(
    id="thesslink/ThessLink-v4-neg-g64",
    entry_point="thesslink_rl.v4_neg.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 64},
)
register(
    id="thesslink/ThessLink-v4-nav-g64",
    entry_point="thesslink_rl.v4_nav.gym_wrapper:GridNegotiationGymEnv",
    kwargs={"grid_size": 64},
)