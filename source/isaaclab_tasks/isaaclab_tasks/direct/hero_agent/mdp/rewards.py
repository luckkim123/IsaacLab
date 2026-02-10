# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

Provides reward configuration, a lightweight reward manager, and reward functions
for ALBC (joint-based attitude control) training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..hero_agent_env import HeroAgentEnv


# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class ALBCRewardCfg:
    """ALBC reward weights.

    reward = potential_weight * scale * exp(-e) + progress_weight * (prev_e - e) + action_cost_weight * ||a||^2
    """

    potential_weight: float = 1.0
    """Weight for potential reward: scale * exp(-potential)."""

    potential_scale: float = 8.0
    """Scale factor for potential reward."""

    progress_weight: float = 1.0
    """Weight for progress reward: prev_potential - current_potential."""

    action_cost_weight: float = -1.0
    """Weight for action cost (negative = penalty)."""


@configclass
class RewardTermCfg:
    """Configuration for a single reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """Reward function: func(robot, **params) -> (num_envs,)."""

    weight: float = MISSING
    """Weight multiplier applied to function output."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters passed to the function."""

    scale_by_dt: bool = True
    """Whether to scale reward by dt. Set False for progress-style rewards."""


# =============================================================================
# Reward Manager
# =============================================================================


class RewardManager:
    """Lightweight reward manager for DirectRLEnv UUV environments.

    Computes total reward as a weighted sum of individual terms, with automatic
    dt scaling and episode sum tracking for logging.
    """

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        num_envs: int,
        device: str,
    ) -> None:
        self.num_envs = num_envs
        self.device = device

        # Parse configurations (skip terms with zero weight)
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        for name, term_cfg in cfg.items():
            if term_cfg.weight != 0.0:
                self._term_names.append(name)
                self._term_cfgs.append(term_cfg)

        # Initialize buffers
        self._reward_buf = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(num_envs, dtype=torch.float32, device=device) for name in self._term_names
        }

    @property
    def active_terms(self) -> list[str]:
        """List of active reward term names."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episode sums for each reward term (for logging)."""
        return self._episode_sums

    def compute(
        self,
        robot: Articulation,
        dt: float,
        **context: Any,
    ) -> torch.Tensor:
        """Compute total reward as weighted sum of terms."""
        self._reward_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            merged_params = {**term_cfg.params, **context}
            term_value = term_cfg.func(robot, **merged_params)

            if term_cfg.scale_by_dt:
                scaled_value = term_value * term_cfg.weight * dt
            else:
                scaled_value = term_value * term_cfg.weight

            self._reward_buf += scaled_value
            self._episode_sums[name] += scaled_value

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset episode sums and return mean values before reset."""
        sums_before_reset = {name: self._episode_sums[name][env_ids].mean().item() for name in self._term_names}

        for name in self._term_names:
            self._episode_sums[name][env_ids] = 0.0

        return sums_before_reset


# =============================================================================
# Shared Reward Functions
# =============================================================================


def albc_potential_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    scale: float = 8.0,
    **_kwargs,
) -> torch.Tensor:
    """scale * exp(-potential). Maximized at zero attitude error."""
    return scale * torch.exp(-env._potentials)


def albc_progress_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """prev_potential - current_potential. Positive when error decreases."""
    return env._prev_potentials - env._potentials


def squared_action_cost(
    _robot: Articulation,
    actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """sum(actions^2). Use with negative weight to penalize large actions/gains."""
    return torch.sum(actions**2, dim=-1)
