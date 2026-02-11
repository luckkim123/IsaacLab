# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared config resolution for Hero Agent workflow scripts.

Extracts environment and agent configs from gymnasium task registry,
eliminating duplication between Phase 2 training and Phase 3 evaluation scripts.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def resolve_task_configs(
    task_name: str,
    agent_cfg_key: str = "rsl_rl_cfg_entry_point",
) -> tuple[DirectRLEnvCfg, RslRlBaseRunnerCfg]:
    """Resolve env_cfg and agent_cfg from gymnasium task entry points.

    Args:
        task_name: Registered gymnasium task ID (e.g. "Isaac-HeroAgent-Adapt-TDC-v0").
        agent_cfg_key: Key for agent config entry point in gym.spec kwargs.
            Default "rsl_rl_cfg_entry_point" for RSL-RL; future RL libraries
            can use e.g. "skrl_cfg_entry_point".

    Returns:
        Tuple of (env_cfg, agent_cfg) instantiated from entry points.
    """
    spec = gym.spec(task_name)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    agent_cfg_entry = spec.kwargs[agent_cfg_key]

    env_cfg = _import_and_instantiate(env_cfg_entry)
    agent_cfg = _import_and_instantiate(agent_cfg_entry)

    return env_cfg, agent_cfg


def apply_cli_overrides(
    env_cfg: DirectRLEnvCfg,
    num_envs: int,
    seed: int = 42,
    device: str | None = None,
) -> str:
    """Apply CLI overrides to env_cfg and return resolved device string.

    Args:
        env_cfg: Environment config to modify in-place.
        num_envs: Number of parallel environments.
        seed: Random seed.
        device: Device string. If None, defaults to "cuda:0".

    Returns:
        Resolved device string (e.g. "cuda:0").
    """
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    resolved_device = device if device is not None else "cuda:0"
    env_cfg.sim.device = resolved_device
    return resolved_device


def _import_and_instantiate(entry_point: str) -> object:
    """Import a class from 'module.path:ClassName' and return an instance."""
    module_path, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
