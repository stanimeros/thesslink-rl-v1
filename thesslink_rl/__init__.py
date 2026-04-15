"""ThessLink RL -- Multi-agent grid negotiation environment for EPyMARL."""

from gymnasium.envs.registration import register

register(
    id="thesslink/GridNegotiation-v0",
    entry_point="thesslink_rl.v0.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/GridNegotiation-v1",
    entry_point="thesslink_rl.v1.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/GridNegotiation-v2",
    entry_point="thesslink_rl.v2.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/GridNegotiation-v3",
    entry_point="thesslink_rl.v3.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/GridNegotiation-v3-neg",
    entry_point="thesslink_rl.v3_neg.gym_wrapper:GridNegotiationGymEnv",
)

register(
    id="thesslink/GridNegotiation-v3-nav",
    entry_point="thesslink_rl.v3_nav.gym_wrapper:GridNegotiationGymEnv",
)
