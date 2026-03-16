# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

Provides reward configuration, a lightweight reward manager, and reward
functions for ALBC (joint-based attitude control) training.

4-term reward architecture:
    1. command  (+): composite Laplacian + linear ramp [0,1], dt-scaled
    2. settling (+): binary threshold bonus {0,1}, dt-scaled
    3. energy   (-): mean(joint_vel^2), dt-scaled
    4. smoothness (-): mean(da^2) + mean(d2a^2), dt-scaled

Plus: termination_penalty (one-time, NOT dt-scaled).
TDC-specific terms (mhat_accuracy, tdc_torque) disabled by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

from ..controllers.tdc import compute_M_bb

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv


# =============================================================================
# Configuration Classes
# =============================================================================


@configclass
class ALBCRewardCfg:
    """ALBC reward configuration: 4-term architecture.

    Active terms (all dt-scaled):
        command    (+5.0): composite Laplacian + linear ramp [0,1]
        settling   (+3.0): binary threshold bonus {0,1}
        energy     (-0.3): mean(joint_vel^2) kinetic energy proxy
        smoothness (-0.5): mean(da^2) + mean(d2a^2) action smoothness
    """

    # Command: composite Laplacian + linear ramp
    # Laplacian (alpha fraction): exp(-||e||/sigma), max gradient at e=0
    # Linear ramp (1-alpha fraction): max(0, 1-||e||/e_max), constant gradient
    command_weight: float = 5.0
    command_sigma: float = 0.35  # Laplacian 1/e point (rad)
    command_e_max: float = 1.0  # Linear ramp saturation (rad)
    command_alpha: float = 0.6  # Laplacian fraction

    # Settling: binary threshold bonus (1 if ||e|| < threshold, 0 otherwise)
    settling_weight: float = 3.0
    settling_threshold: float = 0.087  # ~5 deg

    # Energy: joint velocity squared (kinetic energy proxy)
    energy_weight: float = -0.3

    # Action smoothness: first + second order action difference
    smoothness_weight: float = -0.5

    # -- Meta fields (not reward terms) --
    termination_penalty: float = -10.0
    penalty_curriculum_ratio: float = 0.0

    # EMA alpha for constraint system (not used by rewards directly)
    ema_joint_vel_alpha: float = 0.2

    # -- PBRS progress reward --
    progress_weight: float = 0.0
    progress_gamma: float = 0.99

    # -- TDC-specific (disabled by default) --
    stability_gate_enable: bool = False

    mhat_accuracy_weight: float = 0.0
    mhat_accuracy_sigma: float = 0.5
    mhat_accuracy_kernel: str = "cauchy"

    tdc_torque_weight: float = 0.0


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
    construction (DORAEMON manages DR difficulty, not reward weights).
    """

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        num_envs: int,
        device: str,
        penalty_curriculum_ratio: float = 0.0,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self._penalty_curriculum_ratio = penalty_curriculum_ratio
        self._penalty_curriculum_end_iter = 0
        self._penalty_scale = 1.0 if penalty_curriculum_ratio <= 0 else 0.0

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
        # Last step's per-term per-env scaled values (for post-hoc gate correction)
        self._last_step_terms: dict[str, torch.Tensor] = {
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
    def penalty_scale(self) -> float:
        """Current penalty curriculum scale [0, 1]."""
        return self._penalty_scale

    def set_max_iterations(self, max_iterations: int) -> None:
        """Compute penalty curriculum end iteration from ratio and max_iterations."""
        if self._penalty_curriculum_ratio <= 0:
            return
        self._penalty_curriculum_end_iter = int(self._penalty_curriculum_ratio * max_iterations)

    def update_curriculum(self, iteration: int) -> None:
        """Update penalty scale based on training iteration (linear ramp)."""
        if self._penalty_curriculum_end_iter <= 0:
            return
        self._penalty_scale = min(1.0, iteration / self._penalty_curriculum_end_iter)

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

            # Penalty curriculum: scale negative-weight terms
            if weight < 0:
                scaled_value = scaled_value * self._penalty_scale

            self._last_step_terms[name] = scaled_value
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
# Reward Functions (4-term architecture)
# =============================================================================


def command_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    sigma: float = 0.35,
    e_max: float = 1.0,
    alpha: float = 0.6,
    **_kwargs,
) -> torch.Tensor:
    """Composite Laplacian + linear ramp command tracking reward.

    Laplacian (alpha fraction): exp(-||e||/sigma) -- max gradient at e=0,
    strong near-target signal. Linear ramp (1-alpha): max(0, 1-||e||/e_max)
    -- constant gradient for large-error recovery.

    Output in [0, 1]. dt-scaled.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        sigma: Laplacian 1/e width in radians.
        e_max: Linear ramp saturation in radians.
        alpha: Laplacian fraction [0, 1].
    """
    e = env._potentials
    laplacian = torch.exp(-e / sigma)
    linear = torch.clamp(1.0 - e / e_max, min=0.0)
    return alpha * laplacian + (1.0 - alpha) * linear


def settling_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    threshold: float = 0.087,
    **_kwargs,
) -> torch.Tensor:
    """Binary per-step settling bonus: 1.0 if error < threshold, 0.0 otherwise.

    Incentivizes fast settling: episode sum = weight * dt * (steps inside threshold).
    dt-scaled.

    Args:
        env: Environment instance (provides _potentials).
        threshold: Settling zone boundary in radians.
    """
    return (env._potentials < threshold).float()


def progress_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    gamma: float = 0.99,
    **_kwargs,
) -> torch.Tensor:
    """PBRS (Potential-Based Reward Shaping) progress reward.

    Returns prev_potential - gamma * potential. Positive when error decreases
    (making progress), negative when error increases (regressing).
    Theoretically safe: does not change the optimal policy (Ng et al. 1999).

    NOT dt-scaled: the temporal difference already reflects per-step change.

    Args:
        env: Environment instance (provides _prev_potentials, _potentials).
        gamma: Discount factor for shaping (should match training gamma).
    """
    return env._prev_potentials - gamma * env._potentials


def energy_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Joint kinetic energy proxy: mean(joint_vel^2).

    Zero at equilibrium, penalizes fast joint motion. Provides soft preference
    for energy-efficient control within hard velocity bounds (constraint system).
    dt-scaled. Use with negative weight.
    """
    return torch.mean(_robot.data.joint_vel[:, env._albc_joint_ids] ** 2, dim=-1)


def action_smoothness_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
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


def tdc_torque_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Mean of squared TDC desired torque (tau_desired = U_hat + delta_T_b + M_hat*u_pd).

    Penalizes large control effort from the TDC controller. Unlike action_size
    which penalizes raw gain logits (meaningless for TDC), this penalizes the actual
    physical torque command.

    dt-scaled (instantaneous quality measure). Use with negative weight.
    Requires: env._tdc (TDCController with pd_torque, u_hat, delta_T_b properties).

    Returns:
        (num_envs,) penalty value (positive; apply negative weight).
    """
    tau = env._tdc.pd_torque + env._tdc.u_hat + env._tdc.delta_T_b
    return torch.mean(tau**2, dim=-1)


# =============================================================================
# TDC-Specific Reward Functions
# =============================================================================


def _compute_M_true(env: HeroAgentEnv) -> torch.Tensor:
    """Compute configuration-dependent true inertia M_bb via parallel axis theorem.

    Returns:
        M_true [roll, pitch]. Shape: (num_envs, 2).
    """
    joint_pos = env._robot.data.joint_pos[:, env._albc_joint_ids]
    p_EE = env._kinematics.forward(joint_pos)
    return compute_M_bb(
        I_ROV=env._hydro.rigid_body_inertia[:, :2],
        m_A=env._buoy_hydro.added_mass_matrix[:, 1, 1],
        x_bu=p_EE[:, 0],
        y_bu=p_EE[:, 1],
        h=env.cfg.tdc.h,
        m_body=env._buoy_hydro.body_mass,
    )


def mhat_accuracy_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    sigma: float = 10.0,
    kernel: str = "cauchy",
    **_kwargs,
) -> torch.Tensor:
    """Reward for M_hat accuracy relative to true M_bb(gamma).

    Computes configuration-dependent true inertia M_bb(gamma) using the
    parallel axis theorem, then rewards M_hat closeness via configurable kernel
    on relative error.

    Kernels:
        cauchy:   1 / (1 + sum(rel_err^2) / sigma^2)  -- heavy tail, never saturates
        gaussian: exp(-sum(rel_err^2) / sigma^2)       -- fast decay at large errors

    M_true formula (DYNAMICS_ANALYSIS.md Section 3.7):
        I_p = I_ROV_roll + m_A * (y_bu^2 + h^2)
        I_q = I_ROV_pitch + m_A * (x_bu^2 + h^2)
    where (x_bu, y_bu) = FK(gamma_1, gamma_2).

    Output [0, 1]. Use with positive weight. dt-scaled.
    Requires: env._tdc, env._kinematics, env._hydro, env._buoy_hydro.

    Args:
        env: TDC environment instance (requires _tdc attribute).
        sigma: Kernel width for relative error.
        kernel: "cauchy" (default) or "gaussian".
    """
    M_true = _compute_M_true(env)
    M_hat = env._tdc._m_hat
    rel_error_sq = ((M_hat - M_true) / M_true.clamp(min=1e-4)) ** 2
    score = rel_error_sq.sum(dim=-1) / (sigma**2)

    if kernel == "cauchy":
        return 1.0 / (1.0 + score)
    else:
        return torch.exp(-score)


def compute_stability_gate(env: HeroAgentEnv) -> torch.Tensor:
    """Compute TDC stability gate: 1.0 if stable, 0.0 if violated.

    TDC stability condition (Hsia & Gao 1990, diagonal form):
        rho(I - M_hat^{-1} M_true) < 1
        => |1 - M_true_i / M_hat_i| < 1  for each axis i in {roll, pitch}
        => M_hat > M_true / 2  (underestimation is dangerous, not overestimation)

    Requires: env._tdc, env._kinematics, env._hydro, env._buoy_hydro.

    Args:
        env: TDC environment instance.

    Returns:
        Gate mask (num_envs,): 1.0 if all axes stable, 0.0 if any axis violated.
    """
    M_true = _compute_M_true(env)
    M_hat = env._tdc._m_hat
    ratio = M_true / M_hat.clamp(min=1e-6)
    stability_norm = (1.0 - ratio).abs().max(dim=-1).values
    return (stability_norm < 1.0).float()
