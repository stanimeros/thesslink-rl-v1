"""Single source of truth for the active training/eval environment selector."""

import os

_RAW_SELECTOR = os.environ.get("THESSLINK_ENV", os.environ.get("THESSLINK_ENV_VERSION", "")).strip()
if not _RAW_SELECTOR:
    raise RuntimeError(
        "Environment selector is not set. Use one of: 0, 1, 2, v3_neg, v3_nav "
        "(via THESSLINK_ENV or THESSLINK_ENV_VERSION).",
    )
if _RAW_SELECTOR not in {"0", "1", "2", "v3_neg", "v3_nav"}:
    raise ValueError(
        f"Invalid environment selector: {_RAW_SELECTOR!r}. "
        "Expected one of: 0, 1, 2, v3_neg, v3_nav.",
    )

# --- Derived (do not edit) ------------------------------------------------

ENV_SELECTOR = _RAW_SELECTOR
ENV_VERSION = int(_RAW_SELECTOR) if _RAW_SELECTOR.isdigit() else 3
ENV_LABEL = _RAW_SELECTOR

if _RAW_SELECTOR == "2":
    from thesslink_rl.v2 import ENV_TAG, GridNegotiationEnv
elif _RAW_SELECTOR == "1":
    from thesslink_rl.v1 import ENV_TAG, GridNegotiationEnv
elif _RAW_SELECTOR == "v3_neg":
    from thesslink_rl.v3.environment import GridNegotiationEnv
    ENV_TAG = "v3_neg"
elif _RAW_SELECTOR == "v3_nav":
    from thesslink_rl.v3.environment import GridNegotiationEnv
    ENV_TAG = "v3_nav"
else:
    from thesslink_rl.v0 import ENV_TAG, GridNegotiationEnv

_ENV_CONFIG_MAP = {
    "0": "thesslink",
    "1": "thesslink_v1",
    "2": "thesslink_v2",
    "v3_neg": "thesslink_v3_neg",
    "v3_nav": "thesslink_v3_nav",
}
_ENV_MARKER_MAP = {
    "0": "GridNegotiation-v0",
    "1": "GridNegotiation-v1",
    "2": "GridNegotiation-v2",
    "v3_neg": "GridNegotiation-v3-neg",
    "v3_nav": "GridNegotiation-v3-nav",
}

ENV_CONFIG = _ENV_CONFIG_MAP[ENV_SELECTOR]
ENV_SACRED_MARKER = _ENV_MARKER_MAP[ENV_SELECTOR]
