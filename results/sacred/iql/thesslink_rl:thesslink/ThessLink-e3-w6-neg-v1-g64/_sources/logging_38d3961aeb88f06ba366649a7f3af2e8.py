from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import logging
import os
import re

import numpy as np


def _wandb_env_tag(config):
    """Short label for W&B run/group names (avoids full gym registration keys)."""
    ea = config.get("env_args") or {}
    if "map_name" in ea:
        return str(ea["map_name"])
    key = ea.get("key")
    if isinstance(key, str):
        tail = key.rsplit("/", 1)[-1] if "/" in key else key.rsplit(":", 1)[-1]
        for prefix in ("ThessLink-", "GridNegotiation-"):
            if tail.startswith(prefix):
                return tail[len(prefix):]
        return tail.replace(":", "_")[:48]
    return str(config.get("env", "env"))


def _wandb_phase_tag(env_tag: str) -> str | None:
    """Coarse phase label from ThessLink-style env tail (e.g. e3_full_v5_g32, e0-full-v0-g10)."""
    et = env_tag.lower().replace("-", "_")
    if "_neg_" in et or "neg_v" in et:
        return "phase_neg"
    if "_nav_" in et or "nav_v" in et:
        return "phase_nav"
    if "_full_" in et or "full_v" in et or re.match(r"^v\d", env_tag.lower()):
        return "phase_full"
    return None


def _wandb_tags(config, alg_name: str, env_tag: str, config_hash: str) -> list[str]:
    """Short filterable tags for W&B (keep each tag concise; W&B limits tag length)."""
    tags = [
        f"algo-{alg_name}",
        f"env-{env_tag}",
        f"seed-{config['seed']}",
        f"cfg-{config_hash[-8:]}",
    ]
    phase = _wandb_phase_tag(env_tag)
    if phase:
        tags.append(phase)
    ea = config.get("env_args") or {}
    if isinstance(ea.get("grid_size"), (int, float)):
        tags.append(f"grid-{int(ea['grid_size'])}")
    if isinstance(ea.get("time_limit"), int):
        tags.append(f"tlimit-{ea['time_limit']}")
    # Dedupe and clamp length (W&B recommends short tags)
    out: list[str] = []
    seen = set()
    for t in tags:
        t = str(t).replace(" ", "_")[:64]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

        self.console_logger.info("*******************")
        self.console_logger.info("Tensorboard logging dir:")
        self.console_logger.info(f"{directory_name}")
        self.console_logger.info("*******************")

    def setup_wandb(self, config, team_name, project_name, mode):
        import wandb

        assert (
            team_name is not None and project_name is not None
        ), "W&B logging requires specification of both `wandb_team` and `wandb_project`."
        assert (
            mode in ["offline", "online"]
        ), f"Invalid value for `wandb_mode`. Received {mode} but only 'online' and 'offline' are supported."

        self.use_wandb = True

        alg_name = config["name"]
        env_tag = _wandb_env_tag(config)

        non_hash_keys = ["seed"]
        full_hash = sha256(
            json.dumps(
                {k: v for k, v in config.items() if k not in non_hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()
        self.config_hash = full_hash[-10:]

        # Group: same algo + env so different seeds appear as one group in W&B (easy compare).
        # Full hyperparameters remain in ``config``. Optional override: THESSLINK_WANDB_GROUP.
        default_group = f"{alg_name}_{env_tag}"
        group_name = os.environ.get("THESSLINK_WANDB_GROUP", default_group)
        if len(group_name) > 128:
            group_name = group_name[:128]

        # Run name: algo, env tail, seed (unique per run; config hash as tag for disambiguation).
        run_name = f"{alg_name}_{env_tag}_{config['seed']}"
        if len(run_name) > 128:
            run_name = run_name[:128]

        tags = _wandb_tags(config, alg_name, env_tag, full_hash)

        self.wandb = wandb.init(
            entity=team_name,
            project=project_name,
            name=run_name,
            group=group_name,
            tags=tags,
            config=config,
            mode=mode,
        )

        self.console_logger.info("*******************")
        self.console_logger.info("WANDB RUN ID:")
        self.console_logger.info(f"{self.wandb.id}")
        self.console_logger.info(f"WANDB group={group_name!r} name={run_name!r}")
        self.console_logger.info(f"WANDB tags: {tags}")
        self.console_logger.info("*******************")

        # accumulate data at same timestep and only log in one batch once
        # all data has been gathered
        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            if self.wandb_current_t != t and self.wandb_current_data:
                # self.console_logger.info(
                #     f"Logging to WANDB: {self.wandb_current_data} at t={self.wandb_current_t}"
                # )
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
                self.wandb_current_data = {}
            self.wandb_current_t = t
            self.wandb_current_data[key] = value

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def finish(self):
        if self.use_wandb:
            if self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
            self.wandb.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
