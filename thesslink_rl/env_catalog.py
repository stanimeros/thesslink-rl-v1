"""Auto-discovery for ThessLink env selectors from YAML configs."""

from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_env_key(yaml_path: Path) -> str | None:
    try:
        for line in yaml_path.read_text().splitlines():
            s = line.strip()
            if not s.startswith("key:"):
                continue
            return s.split(":", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _parse_env_grid_size(yaml_path: Path) -> int:
    """Parse grid_size from env_args block; returns 10 if not present."""
    try:
        in_env_args = False
        for line in yaml_path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "env_args:":
                in_env_args = True
                continue
            if in_env_args:
                if stripped.startswith("grid_size:"):
                    return int(stripped.split(":", 1)[1].strip())
                if stripped and not stripped.startswith("#") and not line[0:1] in (" ", "\t"):
                    in_env_args = False
    except (OSError, ValueError):
        pass
    return 10


def _alias_from_env_config(env_config: str) -> str:
    if env_config == "thesslink":
        return "0"
    if env_config.startswith("thesslink_"):
        return env_config[len("thesslink_") :]
    return env_config


def _sort_key(env_config: str) -> tuple[int, str]:
    if env_config == "thesslink":
        return (0, env_config)
    m = re.fullmatch(r"thesslink_v(\d+)", env_config)
    if m:
        return (1, f"{int(m.group(1)):06d}")
    return (2, env_config)


def available_env_catalog() -> list[dict]:
    """Return sorted discovered env entries with numeric index aliases."""
    env_dir = _project_root() / "epymarl_config" / "envs"
    candidates = sorted(env_dir.glob("thesslink*.yaml"))
    entries: list[dict] = []
    for path in candidates:
        env_config = path.stem
        key = _parse_env_key(path)
        if not key:
            continue
        marker = key.split("/")[-1]  # GridNegotiation-vX...
        alias = _alias_from_env_config(env_config)
        m = re.search(r"-v(\d+)", marker)
        base_version = int(m.group(1)) if m else 0
        grid_size = _parse_env_grid_size(path)
        entries.append(
            {
                "env_config": env_config,
                "alias": alias,
                "marker": marker,
                "base_version": base_version,
                "grid_size": grid_size,
                "yaml_path": str(path),
            }
        )

    entries.sort(key=lambda e: _sort_key(e["env_config"]))
    for i, e in enumerate(entries):
        e["index"] = str(i)
        e["aliases"] = {e["index"], e["alias"], e["env_config"]}
    return entries


def resolve_env_choice(raw: str) -> dict:
    catalog = available_env_catalog()
    choice = (raw or "").strip()
    if not choice:
        raise ValueError("Empty environment selector.")
    for e in catalog:
        if choice in e["aliases"]:
            return e
    allowed = ", ".join(f"{e['index']}:{e['alias']}" for e in catalog)
    raise ValueError(f"Invalid env selector {choice!r}. Available: {allowed}")


def prompt_help() -> str:
    catalog = available_env_catalog()
    if not catalog:
        return "(none)"
    return ", ".join(f"{e['index']}:{e['alias']}" for e in catalog)
