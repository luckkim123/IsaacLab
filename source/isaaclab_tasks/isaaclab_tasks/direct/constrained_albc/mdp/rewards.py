# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for ALBC environments.

Provides reward configuration, a lightweight reward manager, and reward
functions for ALBC (joint-based attitude control) training.

2-term reward architecture:
    1. command    (+): attitude tracking (quadratic or laplacian), dt-scaled
    2. smoothness (-): mean(da^2) + mean(d2a^2), dt-scaled

Plus: termination_penalty (one-time, NOT dt-scaled).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..albc_env import ALBCEnv


# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class ALBCRewardCfg:
    """ALBC reward configuration: 2-term architecture.

    Active terms (all dt-scaled):
        command    (+5.0): attitude tracking (quadratic or laplacian)
        smoothness (-0.1): mean(da^2) + mean(d2a^2) action smoothness
    """

    # Command tracking reward
    command_weight: float = 5.0
    command_type: str = "quadratic"
    """Reward type: "quadratic" = -(e_r^2 + e_p^2),
    "laplacian" = exp(-|e|/sigma) per axis summed,
    "min_laplacian" = exp(-|e|/sigma) per axis min (worst-axis drives reward),
    "smooth_min_laplacian" = soft-min via LogSumExp (alpha=5, differentiable).
    smooth_min_laplacian gives ~70% gradient to worse axis, ~30% to better axis,
    preventing gradient oscillation when axes are close while maintaining
    worst-axis focus at large disparity."""

    command_sigma: float = 0.15
    """Laplacian scale parameter (rad). Controls reward sharpness near zero.
    Smaller sigma = sharper peak = stronger near-zero gradient."""

    # Action smoothness: first + second order action difference
    smoothness_weight: float = -0.5

    # -- Meta fields (not reward terms) --
    termination_penalty: float = -10.0


# =============================================================================
# Reward Term Configuration
# =============================================================================


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
    """Whether to scale reward by dt. Default True: ensures time-invariant
    reward accumulation across different simulation timesteps."""


# =============================================================================
# Reward Manager
# =============================================================================


class RewardManager:
    """Lightweight reward manager for DirectRLEnv UUV environments.

    Computes total reward as a weighted sum of individual terms, with automatic
    dt scaling and episode sum tracking for logging. All weights are fixed from
    construction.
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

    def set_max_iterations(self, _max_iterations: int) -> None:
        """No-op: penalty curriculum removed. Kept for runner compatibility."""

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
            weight = term_cfg.weight

            if term_cfg.scale_by_dt:
                scaled_value = term_value * weight * dt
            else:
                scaled_value = term_value * weight

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
# Reward Functions (2-term architecture)
# =============================================================================


def command_reward(
    _robot: Articulation,
    env: ALBCEnv,
    command_type: str = "quadratic",
    sigma: float = 0.15,
    **_kwargs,
) -> torch.Tensor:
    """Attitude tracking reward with selectable kernel.

    Quadratic: -(e_r^2 + e_p^2). Gradient = -2e, weakens near zero.
    Laplacian: exp(-|e|/sigma) per axis, summed. Gradient increases near zero.
    Min-Laplacian: exp(-|e|/sigma) per axis, min. Worst axis drives reward,
        preventing the better axis from dominating gradient signal.

    Args:
        env: Environment instance (provides _attitude_error).
        command_type: "quadratic", "laplacian", or "min_laplacian".
        sigma: Laplacian scale parameter (rad). Only used for laplacian types.
    """
    err_rp = env._attitude_error[:, :2]
    if command_type == "smooth_min_laplacian":
        per_axis = torch.exp(-err_rp.abs() / sigma)
        # Soft-min via negative LogSumExp: differentiable min approximation.
        # alpha=5: ~70% gradient to worse axis, ~30% to better axis.
        # Preserves min_laplacian's worst-axis focus at large disparity while
        # providing smooth gradient when axes are close (prevents oscillation).
        alpha = 5.0
        return -torch.logsumexp(-alpha * per_axis, dim=-1) / alpha
    if command_type == "min_laplacian":
        per_axis = torch.exp(-err_rp.abs() / sigma)
        return per_axis.min(dim=-1).values
    if command_type == "laplacian":
        return torch.exp(-err_rp.abs() / sigma).sum(dim=-1)
    return -(err_rp[:, 0] ** 2 + err_rp[:, 1] ** 2)


def action_smoothness_penalty(
    _robot: Articulation,
    env: ALBCEnv,
    **_kwargs,
) -> torch.Tensor:
    """First-order + second-order action difference penalty.

    da = a_t - a_{t-1} (jerk), d2a = a_t - 2*a_{t-1} + a_{t-2} (oscillation).
    Penalizes rapid action changes and action-space acceleration.
    dt-scaled. Use with negative weight.

    Requires: env._prev_prev_actions buffer.
    """
    da = env._actions - env._prev_actions
    d2a = env._actions - 2.0 * env._prev_actions + env._prev_prev_actions
    return torch.mean(da**2, dim=-1) + torch.mean(d2a**2, dim=-1)
