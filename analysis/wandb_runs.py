"""Fetch runs from Weights & Biases and classify them."""

from __future__ import annotations

from .config import ALGOS


def fetch_runs(api, entity: str, project: str, version: str, algo: str | None, state: str):
    filters: dict = {}
    if state not in ("all", "active"):
        filters["state"] = state
    all_runs = api.runs(f"{entity}/{project}", filters=filters)

    matched = []
    vl = version.lower()
    for run in all_runs:
        tags_l = [t.lower() for t in (run.tags or [])]
        group_l = (run.group or "").lower()
        name_l = run.name.lower()
        config_env = (run.config.get("env_config") or run.config.get("env", "") or "").lower()
        combined = " ".join([name_l, group_l, config_env] + tags_l)

        if vl not in combined:
            continue

        if algo:
            al = algo.lower()
            algo_match = (
                al in (run.config.get("name") or "").lower()
                or al in name_l
                or any(al == t for t in tags_l)
                or al in group_l
            )
            if not algo_match:
                continue

        matched.append(run)

    if state == "active":
        matched = [r for r in matched if getattr(r, "state", None) in ("running", "finished")]
    return matched


def detect_algo(run) -> str | None:
    for a in ALGOS:
        if a in (run.name or "").lower() or a in (run.group or "").lower():
            return a
    return None


def is_full_episode_run(run) -> bool:
    """Full neg→nav (e.g. *-w7-full-*, env *w7_full*), not nav-only / neg-only."""
    name = (run.name or "").lower()
    group = (run.group or "").lower()
    cfg = (run.config.get("env_config") or run.config.get("env", "") or "").lower()
    tag_blob = " ".join((t or "").lower() for t in (run.tags or []))
    for blob in (name, group, cfg, tag_blob):
        if "-full-" in blob or "_full_" in blob:
            return True
        if "w7_full" in blob or "w7-full" in blob:
            return True
    return False
