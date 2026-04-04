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
