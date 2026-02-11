# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

Provides reward configuration, a lightweight reward manager with curriculum
support, and reward functions for ALBC (joint-based attitude control) training.

Reward design principles:
    - Gaussian kernel normalization: positive rewards use exp(-err^2/sigma^2)
      for natural [0,1] bounding and intuitive weight interpretation.
    - dt-scaling: "instantaneous state quality" terms are dt-scaled;
      "telescoping difference" terms are NOT dt-scaled.
    - Environment-specific configs: Base RL and Encoder-TDC have different
      action semantics, so reward weights are separated.
    - Curriculum: penalty terms start small and increase over training to
      preserve early exploration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv


# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class ALBCRewardCfg:
    """ALBC reward configuration with Gaussian tracking + multi-term penalties.

    Reward = tracking * w_t * dt
           + progress * w_p
           + angular_velocity * w_av * dt
           + action_magnitude * w_am * dt
           + action_rate * w_ar
    """

    # Tracking (Gaussian kernel)
    tracking_weight: float = 1.0
    tracking_sigma: float = 0.25

    # Progress (telescoping, NOT dt-scaled)
    progress_weight: float = 1.0

    # Angular velocity penalty (curriculum: starts at 1/10)
    angular_velocity_weight: float = -1.0

    # Action magnitude penalty
    action_magnitude_weight: float = -1.0

    # Action rate penalty (NOT dt-scaled, curriculum: starts at 1/10)
    action_rate_weight: float = -0.005

    # Curriculum
    curriculum_end_iter: int = 200


@configclass
class EncoderTDCRewardCfg(ALBCRewardCfg):
    """Encoder-TDC reward config with adjusted weights and TDE residual penalty.

    Inherits tracking/progress/angular_velocity from ALBCRewardCfg.
    Overrides action weights for gain-tuning semantics.
    Adds TDE residual penalty to encourage accurate M_hat.
    """

    # Lighter action_magnitude: sigmoid midpoint is a reasonable default
    action_magnitude_weight: float = -0.5

    # Heavier action_rate: gain stability is critical for TDC performance
    action_rate_weight: float = -0.01

    tde_residual_weight: float = -0.05


@configclass
class ConstrainedEncoderTDCRewardCfg(EncoderTDCRewardCfg):
    """Reward config for constrained Encoder-TDC training.

    Disables safety-related penalties that are now handled by IPO constraints:
        - action_magnitude -> smoothness constraint
        - action_rate -> smoothness constraint
        - angular_velocity -> implicit in tracking reward

    Retains pure performance signals (tracking, progress) and tde_residual
    for M_hat accuracy (no constraint equivalent).
    """

    angular_velocity_weight: float = 0.0
    action_magnitude_weight: float = 0.0
    action_rate_weight: float = 0.0


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
    """Whether to scale reward by dt. Set False for progress-style rewards."""

    curriculum_start_weight: float | None = None
    """If set, weight starts at this value and linearly ramps to ``weight``
    over curriculum_end_iter iterations. None means no curriculum (constant weight)."""


# =============================================================================
# Reward Manager
# =============================================================================


class RewardManager:
    """Lightweight reward manager for DirectRLEnv UUV environments.

    Computes total reward as a weighted sum of individual terms, with automatic
    dt scaling, episode sum tracking for logging, and curriculum support for
    gradually increasing penalty weights during training.
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

        # Active weights (modified by curriculum).
        # Initialize to curriculum_start_weight when set, so iteration 0
        # uses the correct starting weight (not full weight).
        self._active_weights = [
            cfg.curriculum_start_weight if cfg.curriculum_start_weight is not None else cfg.weight
            for cfg in self._term_cfgs
        ]

        # Per-step raw (unweighted, un-dt-scaled) mean values for diagnostics.
        # Updated each compute() call; read by _collect_episode_metrics().
        self._step_raw_means: dict[str, float] = {name: 0.0 for name in self._term_names}

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

    @property
    def step_raw_means(self) -> dict[str, float]:
        """Last step's unweighted, un-dt-scaled raw mean per term."""
        return self._step_raw_means

    @property
    def active_weights(self) -> dict[str, float]:
        """Current active weight per term (curriculum-adjusted)."""
        return {name: self._active_weights[i] for i, name in enumerate(self._term_names)}

    def update_curriculum(self, iteration: int, end_iter: int) -> None:
        """Update penalty weights based on training progress.

        Linear ramp from curriculum_start_weight to weight over end_iter iterations.

        Args:
            iteration: Current training iteration (0-based).
            end_iter: Iteration at which curriculum reaches full weight.
        """
        if end_iter <= 0:
            return

        progress = min(1.0, iteration / end_iter)
        for i, term_cfg in enumerate(self._term_cfgs):
            if term_cfg.curriculum_start_weight is not None:
                start = term_cfg.curriculum_start_weight
                full = term_cfg.weight
                self._active_weights[i] = start + (full - start) * progress

    def compute(
        self,
        robot: Articulation,
        dt: float,
        **context: Any,
    ) -> torch.Tensor:
        """Compute total reward as weighted sum of terms."""
        self._reward_buf.zero_()

        for i, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            merged_params = {**term_cfg.params, **context}
            term_value = term_cfg.func(robot, **merged_params)
            weight = self._active_weights[i]

            # Store raw (unweighted, un-dt-scaled) mean for diagnostics
            self._step_raw_means[name] = term_value.mean().item()

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
# Shared Reward Functions (Base RL + Encoder-TDC)
# =============================================================================


def tracking_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    sigma: float = 0.25,
    **_kwargs,
) -> torch.Tensor:
    """Gaussian kernel tracking reward: exp(-||e||^2 / sigma^2).

    Output in [0, 1]. error=0 -> 1.0, error=sigma -> 1/e (~0.37).
    Uses L2 squared norm of roll/pitch errors for smoother gradient near zero.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        sigma: Gaussian kernel width in radians. Default 0.25 rad (~14.3 deg).
    """
    err_sq = env._potentials**2
    return torch.exp(-err_sq / (sigma**2))


def progress_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Telescoping progress reward: prev_potential - current_potential.

    Positive when error decreases. Episode sum = phi_0 - phi_T (initial - final error).
    NOT dt-scaled because the telescoping sum is naturally frequency-invariant.
    """
    return env._prev_potentials - env._potentials


def angular_velocity_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Sum of squared body angular velocities (roll rate, pitch rate).

    Penalizes fast rotations to suppress oscillation after reaching the target.
    Use with negative weight. dt-scaled (instantaneous quality measure).
    """
    ang_vel = env._robot.data.root_ang_vel_b[:, :2]  # [p, q]
    return torch.sum(ang_vel**2, dim=-1)


def action_rate_penalty(
    _robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Sum of squared action differences between consecutive steps.

    Penalizes abrupt action changes to encourage smooth control.
    NOT dt-scaled: per-step delta naturally scales with frequency
    (halving dt -> halving delta -> quartering squared delta, but doubling steps).
    Use with negative weight.
    """
    return torch.sum((actions - prev_actions) ** 2, dim=-1)


def action_magnitude_penalty(
    _robot: Articulation,
    actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Sum of squared actions. Penalizes large control effort.

    dt-scaled (instantaneous quality measure). Use with negative weight.
    """
    return torch.sum(actions**2, dim=-1)


# =============================================================================
# Encoder-TDC Exclusive Reward Functions
# =============================================================================


def tde_residual_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """TDE residual ratio: ||U_hat|| / (||M_hat * u_pd|| + eps).

    Measures how large the TDE compensation torque is relative to the PD torque.
    A small ratio indicates accurate M_hat and appropriate gains.
    Good values < 0.5, problematic > 1.0.

    dt-scaled (instantaneous quality measure). Use with negative weight.

    Requires env._tdc to be available (TDC environments only).
    """
    u_hat_norm = env._tdc.u_hat.norm(dim=-1)
    pd_norm = env._tdc.pd_torque.norm(dim=-1) + 1e-6
    return u_hat_norm / pd_norm
