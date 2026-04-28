"""Split runs into full / nav / neg / other buckets."""

from __future__ import annotations

from dataclasses import dataclass

from .wandb_runs import is_full_episode_run


@dataclass
class RunPartition:
    full: list
    nav: list
    neg: list
    other: list


def partition_runs(runs: list) -> RunPartition:
    full_runs = [r for r in runs if is_full_episode_run(r)]
    rest = [r for r in runs if r not in full_runs]
    nav_runs = [r for r in rest if "nav" in r.name.lower()]
    neg_runs = [r for r in rest if "neg" in r.name.lower()]
    other = [r for r in rest if r not in nav_runs and r not in neg_runs]
    return RunPartition(full=full_runs, nav=nav_runs, neg=neg_runs, other=other)
