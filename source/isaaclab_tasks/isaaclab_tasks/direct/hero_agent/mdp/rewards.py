# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

This module provides a complete reward system for ALBC (Active Linear Buoyancy
Controller) environments, including configuration, manager, and reward functions.

ALBC uses joint-based attitude control without thrusters, so rewards focus on
orientation (roll/pitch) error and joint action costs.
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
    """Configuration for ALBC reward weights.

    Total reward = potential_reward + progress_reward + action_cost

    - potential_reward: scale * exp(-attitude_error) - encourages low error
    - progress_reward: prev_error - current_error - encourages improvement
    - action_cost: -weight * sum(actions^2) - penalizes large actions
    """

    potential_weight: float = 1.0
    """Weight for potential reward (scale * exp(-potential))."""

    potential_scale: float = 8.0
    """Scale factor for potential reward. Higher = stronger reward at target."""

    progress_weight: float = 1.0
    """Weight for progress reward (prev_potential - current_potential)."""

    action_cost_weight: float = -1.0
    """Weight for action cost. Negative = penalty for large actions."""


@configclass
class TDCRewardCfg(ALBCRewardCfg):
    """Configuration for TDC-specific reward weights.

    Extends ALBCRewardCfg with additional TDC-specific rewards:
        - attitude_penalty_weight: Direct linear penalty for attitude error
        - stability_weight: Reward for accurate inertia estimation
        - gain_smoothness_weight: Penalty for rapid gain changes
        - gain_magnitude_weight: Penalty for large gain values

    Design Rationale (v2):
        The original weights let potential_reward dominate (~91% of total reward),
        causing the agent to optimize for "survive long" rather than "minimize
        attitude error". The rebalanced weights add a direct linear attitude
        penalty and reduce potential dominance to encourage precise stabilization.
    """

    # Override base ALBC weights for better balance
    potential_weight: float = 0.3
    """Weight for potential reward. Reduced from 1.0 to limit dominance."""

    potential_scale: float = 4.0
    """Scale factor for potential reward. Reduced from 8.0."""

    progress_weight: float = 3.0
    """Weight for progress reward. Increased from 1.0 for stronger improvement signal."""

    # TDC-specific rewards
    attitude_penalty_weight: float = 1.0
    """Weight for direct linear attitude error penalty (-||e||).
    Provides constant gradient at large errors, complementing exp-based potential."""

    stability_weight: float = 0.5
    """Weight for TDC stability reward (exp(-||1 - M/M_hat||))."""

    stability_scale: float = 2.0
    """Scale factor for stability reward."""

    gain_smoothness_weight: float = 0.1
    """Weight for gain smoothness penalty. Positive weight * negative func = penalty."""

    gain_magnitude_weight: float = -0.1
    """Weight for gain magnitude penalty."""


@configclass
class BaseTDCRewardCfg(TDCRewardCfg):
    """TDC reward configuration for encoder-free training (fixed M_hat).

    Removes stability reward (free lunch with fixed M_hat) and
    attitude penalty (cancels out potential reward gradient).
    Relies on potential + progress for attitude control signal.
    """

    stability_weight: float = 0.0
    """Disabled: M_hat is fixed, so stability reward is constant (free lunch)."""

    attitude_penalty_weight: float = 0.0
    """Disabled: linear penalty cancels potential reward gradient signal."""


@configclass
class RewardTermCfg:
    """Configuration for a single reward term.

    Each term consists of a callable function, a weight, and optional parameters.
    """

    func: Callable[..., torch.Tensor] = MISSING
    """Reward function: func(robot, **params) -> Tensor of shape (num_envs,)."""

    weight: float = MISSING
    """Weight multiplier. Final value = weight * dt * func(robot, **params)."""

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
        """Initialize the reward manager.

        Args:
            cfg: Dictionary mapping term names to RewardTermCfg.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
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
        """Compute total reward as weighted sum of terms.

        Args:
            robot: Robot articulation for state access.
            dt: Time step duration for scaling.
            **context: Additional context (actions, task, etc.).

        Returns:
            Total reward tensor of shape (num_envs,).
        """
        self._reward_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Compute term value
            merged_params = {**term_cfg.params, **context}
            term_value = term_cfg.func(robot, **merged_params)

            # Apply weight and dt scaling (dt optional for progress-style rewards)
            if term_cfg.scale_by_dt:
                scaled_value = term_value * term_cfg.weight * dt
            else:
                scaled_value = term_value * term_cfg.weight

            # Accumulate
            self._reward_buf += scaled_value
            self._episode_sums[name] += scaled_value

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset episode sums and return values before reset (for logging).

        Args:
            env_ids: Environment indices to reset.

        Returns:
            Dictionary of mean episode sums before reset.
        """
        sums_before_reset = {name: self._episode_sums[name][env_ids].mean().item() for name in self._term_names}

        for name in self._term_names:
            self._episode_sums[name][env_ids] = 0.0

        return sums_before_reset


# =============================================================================
# Reward Functions
# =============================================================================


def albc_potential_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    scale: float = 8.0,
    **_kwargs,
) -> torch.Tensor:
    """Potential-based pose reward for ALBC.

    Computes: scale * exp(-potential)

    At zero error, reward is maximized at 'scale'.

    Args:
        robot: Robot articulation (unused, required for signature).
        env: HeroAgentEnv with potential tracking.
        scale: Maximum reward at zero error.

    Returns:
        Reward values of shape (num_envs,).
    """
    return scale * torch.exp(-env._potentials)


def albc_progress_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Progress reward based on potential difference.

    Computes: prev_potential - current_potential

    Positive when error decreases, negative when error increases.

    Args:
        robot: Robot articulation (unused, required for signature).
        env: HeroAgentEnv with potential tracking.

    Returns:
        Reward values of shape (num_envs,).
    """
    return env._prev_potentials - env._potentials


def albc_action_cost(
    _robot: Articulation,
    actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Action magnitude cost for ALBC joint control.

    Computes: sum(actions^2, dim=-1)

    Use with negative weight to penalize large actions.

    Args:
        robot: Robot articulation (unused, required for signature).
        actions: Joint velocity commands of shape (num_envs, 2).

    Returns:
        Cost values (positive) of shape (num_envs,).
    """
    return torch.sum(actions**2, dim=-1)


# =============================================================================
# TDC-Specific Reward Functions
# =============================================================================


def tdc_attitude_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Direct linear attitude error penalty for TDC.

    Computes: -||[e_roll, e_pitch]||

    Unlike potential_reward (exp-based, saturates at large errors), this
    provides a linear penalty proportional to error magnitude, creating
    stronger gradient signal when the attitude is far from target.

    The combination of exp potential + linear penalty gives:
        - Linear penalty: "reduce large errors quickly" (constant gradient)
        - Exp potential: "align precisely near target" (increasing gradient near 0)

    Args:
        robot: Robot articulation (unused, required for signature).
        env: HeroAgentEnv with potential tracking.

    Returns:
        Negative error magnitude of shape (num_envs,).
    """
    return -env._potentials


def tdc_stability_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    scale: float = 1.0,
    **_kwargs,
) -> torch.Tensor:
    """TDC stability reward based on inertia estimation accuracy.

    Computes: scale * exp(-||1 - M/M_hat||)

    The TDC stability condition requires M_hat to be close to true M.
    This reward encourages accurate inertia estimation by the encoder.

    Args:
        robot: Robot articulation (unused, required for signature).
        env: HeroAgentTDCEnv with TDC controller and true inertia info.
        scale: Maximum reward at perfect estimation.

    Returns:
        Reward values of shape (num_envs,).
    """
    # Check if env has TDC controller (HeroAgentTDCEnv)
    if not hasattr(env, "_tdc_controller"):
        return torch.zeros(env.num_envs, device=env.device)

    # Get stability metric from TDC controller
    # Note: true_inertia is obtained from privileged info during training
    true_inertia = getattr(env, "_true_inertia", None)
    stability_metric = env._tdc_controller.get_stability_metric(true_inertia)

    return scale * torch.exp(-stability_metric)


def tdc_gain_smoothness_reward(
    _robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Gain smoothness reward penalizing rapid gain changes.

    Computes: -sum((actions - prev_actions)^2, dim=-1)

    Encourages smooth gain trajectories, preventing oscillations and
    improving control stability.

    Args:
        robot: Robot articulation (unused, required for signature).
        actions: Current gain commands of shape (num_envs, 4).
        prev_actions: Previous gain commands of shape (num_envs, 4).

    Returns:
        Penalty values (negative for use with positive weight) of shape (num_envs,).
    """
    gain_change = actions - prev_actions
    return -torch.sum(gain_change**2, dim=-1)


def tdc_workspace_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    threshold: float = 0.8,
    excess_cap: float = 2.0,
    **_kwargs,
) -> torch.Tensor:
    """Workspace saturation penalty based on pre-clamp radius.

    Uses the TDC controller's pre-clamp workspace utilization ratio (r/r_max)
    which changes continuously with K_p even in the saturated region (r > r_max).

    Penalty function (Huber-like with cap, C1 continuous):
        excess in [0, 1]:       -excess^2         (quadratic near boundary)
        excess in [1, cap]:     -(2*excess - 1)   (linear growth)
        excess > cap:           -(2*cap - 1)      (saturated)

    The cap prevents reward domination when Lambda_inv amplifies torques
    at large tilt angles (lf = cos(theta)*cos(phi)*F_bu -> 0).
    With default cap=2.0 and weight=1.5: max episode penalty ~ -67.

    Args:
        robot: Robot articulation (unused, required for signature).
        env: HeroAgentTDCEnv with TDC controller.
        threshold: Utilization ratio above which penalty applies (default: 0.8).
        excess_cap: Maximum excess value before saturation (default: 2.0).

    Returns:
        Penalty values (<= 0) of shape (num_envs,).
    """
    if not hasattr(env, "_tdc_controller"):
        return torch.zeros(env.num_envs, device=env.device)

    utilization = env._tdc_controller.workspace_utilization
    excess = torch.clamp(utilization - threshold, min=0.0, max=excess_cap)
    penalty = torch.where(
        excess <= 1.0,
        excess**2,
        2.0 * excess - 1.0,
    )
    return -penalty


def tdc_gain_magnitude_cost(
    _robot: Articulation,
    actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Gain magnitude cost for TDC control.

    Computes: sum(actions^2, dim=-1)

    Encourages using moderate gains instead of extreme values.
    Use with negative weight to penalize large gains.

    Args:
        robot: Robot articulation (unused, required for signature).
        actions: Gain commands of shape (num_envs, 4).

    Returns:
        Cost values (positive) of shape (num_envs,).
    """
    return torch.sum(actions**2, dim=-1)
