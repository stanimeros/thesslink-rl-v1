"""Single source of truth for active training/eval environment selection."""

import os
import re

from thesslink_rl.env_catalog import prompt_help, resolve_env_choice

_RAW_SELECTOR = os.environ.get("THESSLINK_ENV", os.environ.get("THESSLINK_ENV_VERSION", "")).strip()
if not _RAW_SELECTOR:
    raise RuntimeError(
        "Environment selector is not set. Use one of: "
        f"{prompt_help()} "
        "(via THESSLINK_ENV or THESSLINK_ENV_VERSION).",
    )
_choice = resolve_env_choice(_RAW_SELECTOR)

# --- Derived (do not edit) ------------------------------------------------

ENV_SELECTOR = _choice["env_config"]
ENV_LABEL = _choice["alias"]
ENV_INDEX = _choice["index"]
ENV_SACRED_MARKER = _choice["marker"]
_BASE_VERSION = _choice["base_version"]
ENV_VERSION = _BASE_VERSION

if _BASE_VERSION == 2:
    from thesslink_rl.v2 import GridNegotiationEnv
elif _BASE_VERSION == 1:
    from thesslink_rl.v1 import GridNegotiationEnv
else:
    from thesslink_rl.v0 import GridNegotiationEnv

ENV_CONFIG = ENV_SELECTOR
ENV_TAG = ENV_LABEL if re.fullmatch(r"v\d+_.+", ENV_LABEL) else f"v{_BASE_VERSION}"
