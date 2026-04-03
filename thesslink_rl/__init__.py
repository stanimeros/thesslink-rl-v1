"""ThessLink RL -- Multi-agent grid negotiation environment for EPyMARL."""

from gymnasium.envs.registration import register

register(
    id="thesslink/GridNegotiation-v0",
    entry_point="thesslink_rl.gym_wrapper:GridNegotiationGymEnv",
)
