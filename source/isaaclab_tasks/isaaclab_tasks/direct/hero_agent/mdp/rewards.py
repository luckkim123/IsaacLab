# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

Provides reward configuration, a lightweight reward manager, and reward
functions for ALBC (joint-based attitude control) training.

Reward design principles:
    - Gaussian kernel tracking + settling bonus dominate the positive signal
      ([0,1] dense gradient). Penalties (joint_oscillation, joint_velocity)
      provide directional regularization, ramped via penalty curriculum.
    - dt-scaling: state-quality terms (tracking, settling, joint_oscillation,
      joint_velocity, angular_velocity) are dt-scaled.
    - PBRS progress shaping (Ng 1999): preserves optimal policy guarantee.
    - Joint oscillation: EMA high-pass filter isolates high-frequency
      joint velocity oscillation while allowing smooth movement.
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
    """ALBC reward configuration with Laplacian tracking + regularization penalties.

    Active terms (5, all raw per-step — NOT dt-scaled):
        tracking          * w_t   (Laplacian kernel, positive)
        settling          * w_s   (binary near-target bonus, positive)
        joint_oscillation * w_jo  (EMA high-pass filtered, negative)
        joint_velocity    * w_jv  (joint speed penalty, negative)
        progress          * w_p   (PBRS, positive)

    Design: tracking + settling dominate the positive signal. Two
    penalties (joint_oscillation, joint_velocity) provide directional
    regularization. Raw per-step rewards (no dt-scaling) ensure PPO
    advantage estimates have sufficient signal-to-noise ratio.
    """

    # Tracking (Laplacian kernel): exp(-||e|| / sigma)
    # Gradient = 1/sigma at all errors (maximum at e=0, unlike Gaussian which is 0 at e=0).
    tracking_weight: float = 5.0
    tracking_sigma: float = 0.35  # 20.1 deg 1/e point

    # Joint oscillation penalty (EMA high-pass filtered joint velocity).
    # Penalizes high-frequency oscillation while allowing smooth movement.
    # dt-scaled. Use with negative weight.
    joint_oscillation_weight: float = -2.5
    joint_oscillation_alpha: float = 0.2  # EMA smoothing factor (cutoff ~1.6Hz at 50Hz)

    # Joint velocity penalty: mean(joint_vel^2). Penalizes fast joint movement,
    # improving control stability and energy efficiency.
    # dt-scaled. Use with negative weight.
    joint_velocity_weight: float = -0.5

    # Linear error penalty: -min(||err||, max_err) / max_err.
    # Provides constant gradient at ALL error levels (unlike Gaussian which
    # vanishes at large errors). Clamped to [-1, 0]. dt-scaled.
    linear_error_weight: float = -3.0
    linear_error_max: float = 1.0  # clamp at ~57 degrees

    # Progress (potential-based shaping): PBRS (Ng 1999) preserves optimal policy.
    # NOT dt-scaled. Provides immediate reward for error reduction at all levels.
    progress_weight: float = 0.3
    progress_scale: float = 0.01
    progress_mode: str = "pbrs"  # "tanh" or "pbrs" (SAC-safe, policy-preserving)
    progress_gamma: float = 0.99  # discount factor for PBRS (match PPO gamma)

    # Settling bonus: binary per-step (1 if error < threshold, 0 otherwise).
    # Incentivizes fast response time: sooner you enter threshold, more steps
    # you accumulate reward. dt-scaled episode sum = weight * dt * (steps inside).
    settling_weight: float = 3.0
    settling_threshold: float = 0.087  # radians (~5 deg)

    # Angular velocity penalty (dt-scaled, discourages oscillation under DR)
    angular_velocity_weight: float = 0.0

    # Penalty curriculum: linearly ramp penalty scale from 0 to 1 over this
    # ratio of max_iterations. 0 = disabled (penalties always at full weight).
    # Applies to all terms with negative weight.
    penalty_curriculum_ratio: float = 0.0

    # TDC stability gate: multiply total reward by 0 when |1 - M_hat/M_true| >= 1
    # Only effective in TDC envs. Based on Baek et al. (ACC 2022).
    stability_gate_enable: bool = False

    # -- TDC-specific rewards (defaults 0.0: inactive in base RL envs) --

    # M_hat accuracy reward (Cauchy/Gaussian kernel on relative error, dt-scaled)
    mhat_accuracy_weight: float = 0.0
    mhat_accuracy_sigma: float = 0.5
    mhat_accuracy_kernel: str = "cauchy"

    # TDC torque penalty: ||tau_desired||^2 (actual control effort, dt-scaled)
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

    scale_by_dt: bool = False
    """Whether to scale reward by dt. Default False: raw per-step values for
    stronger PPO advantage signal. Set True only for rate-based quantities
    that must be time-invariant (e.g., power consumption in W)."""


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
# Shared Reward Functions
# =============================================================================


def tracking_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    sigma: float = 1.0,
    **_kwargs,
) -> torch.Tensor:
    """Laplacian kernel tracking reward: exp(-||e|| / sigma).

    Output in [0, 1]. error=0 -> 1.0, error=sigma -> 1/e (~0.37).
    Gradient = (1/sigma) * exp(-||e||/sigma), maximum at error=0.
    Unlike Gaussian (gradient=0 at target), Laplacian provides the strongest
    corrective signal exactly at the target, eliminating steady-state error.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        sigma: Kernel width in radians. Default 1.0 rad (~57.3 deg).
    """
    return torch.exp(-env._potentials / sigma)


def joint_oscillation_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """EMA high-pass filtered joint velocity penalty.

    Penalizes high-frequency oscillation while allowing smooth movement.
    The EMA tracks the low-frequency component; the difference is the
    high-frequency residual that gets penalized.

    dt-scaled. Use with negative weight.

    Requires: env._ema_joint_vel (updated before reward computation each step).
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    hf = joint_vel - env._ema_joint_vel
    return torch.mean(hf**2, dim=-1)


def joint_velocity_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    **_kwargs,
) -> torch.Tensor:
    """Quadratic joint velocity penalty: mean(joint_vel^2).

    Penalizes fast joint movement, improving control stability and energy
    efficiency. Unlike joint_oscillation (EMA high-pass, high-freq only),
    this penalizes all joint velocity magnitudes.

    dt-scaled. Use with negative weight.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    return torch.mean(joint_vel**2, dim=-1)


def linear_error_penalty(
    _robot: Articulation,
    env: HeroAgentEnv,
    max_err: float = 1.0,
    **_kwargs,
) -> torch.Tensor:
    """Linear error penalty: min(||err||, max_err) / max_err.

    Provides constant gradient at all error levels (unlike Gaussian which
    vanishes at large errors). Output in [0, 1], clamped at max_err.
    dt-scaled. Use with negative weight.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        max_err: Clamp threshold in radians. Default 1.0 rad (~57 deg).
    """
    return torch.clamp(env._potentials / max_err, max=1.0)


def progress_reward(
    _robot: Articulation,
    env: HeroAgentEnv,
    scale: float = 0.01,
    **_kwargs,
) -> torch.Tensor:
    """Tanh-wrapped progress reward: tanh((prev - curr) / scale).

    tanh breaks the telescoping property of raw delta (which sums to ~0 over
    an episode as positive/negative steps cancel). With tanh, each convergence
    step contributes ~0.3-0.5 regardless of cancellation.

    NOT dt-scaled. Use weight to control relative importance vs tracking.

    Args:
        env: Environment instance (provides _prev_potentials, _potentials).
        scale: Normalization scale for tanh input.
    """
    delta = env._prev_potentials - env._potentials
    return torch.tanh(delta / scale)


def progress_reward_pbrs(
    _robot: Articulation,
    env: HeroAgentEnv,
    gamma: float = 0.99,
    **_kwargs,
) -> torch.Tensor:
    """Proper PBRS: Phi(s) - gamma * Phi(s').

    Preserves optimal policy (Ng et al. 1999).
    Safe for off-policy (SAC) replay buffer.
    NOT dt-scaled.

    Args:
        env: Environment instance (provides _prev_potentials, _potentials).
        gamma: Discount factor matching the RL algorithm (e.g. SAC gamma).
    """
    return env._prev_potentials - gamma * env._potentials


def settling_bonus(
    _robot: Articulation,
    env: HeroAgentEnv,
    threshold: float = 0.10,
    **_kwargs,
) -> torch.Tensor:
    """Binary per-step settling bonus: 1.0 if error < threshold, 0.0 otherwise.

    Incentivizes fast response time: the sooner the agent enters the threshold
    zone, the more steps it accumulates reward for. With dt-scaling, episode sum
    equals weight * dt * (number of steps within threshold).

    Role separation from tracking/linear_error:
        - tracking/linear_error: "reduce error magnitude" (continuous gradient)
        - settling: "reach threshold quickly and stay there" (time-based)

    Markov-safe: depends only on current state. Compatible with SAC replay buffer.
    dt-scaled. Use with positive weight.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        threshold: Settling zone boundary in radians. Default 0.10 rad (~5.7 deg).
    """
    return (env._potentials < threshold).float()


def angular_velocity_penalty(
    robot: Articulation,
    **_kwargs,
) -> torch.Tensor:
    """Sum of squared roll/pitch angular velocities (body frame).

    Only penalizes controllable axes (roll, pitch). Yaw is excluded because
    buoyancy-based control cannot generate Z-axis torque.
    dt-scaled (instantaneous quality measure). Use with negative weight.

    Note: Uses ``sum`` (not ``mean``) over the 2 axes so the penalty scales
    with total angular velocity magnitude. The axis count is fixed at 2.

    Returns:
        (num_envs,) penalty value (positive; apply negative weight).
    """
    return torch.sum(robot.data.root_ang_vel_b[:, :2] ** 2, dim=-1)


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
    joint_pos = env._robot.data.joint_pos[:, env._albc_joint_ids]
    p_EE = env._kinematics.forward(joint_pos)

    M_true = compute_M_bb(
        I_ROV=env._hydro.rigid_body_inertia[:, :2],
        m_A=env._buoy_hydro.added_mass_matrix[:, 1, 1],
        x_bu=p_EE[:, 0],
        y_bu=p_EE[:, 1],
        h=env.cfg.tdc.h,
        m_body=env._buoy_hydro.body_mass,
    )

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
    joint_pos = env._robot.data.joint_pos[:, env._albc_joint_ids]
    p_EE = env._kinematics.forward(joint_pos)

    M_true = compute_M_bb(
        I_ROV=env._hydro.rigid_body_inertia[:, :2],
        m_A=env._buoy_hydro.added_mass_matrix[:, 1, 1],
        x_bu=p_EE[:, 0],
        y_bu=p_EE[:, 1],
        h=env.cfg.tdc.h,
        m_body=env._buoy_hydro.body_mass,
    )

    M_hat = env._tdc._m_hat
    ratio = M_true / M_hat.clamp(min=1e-6)
    stability_norm = (1.0 - ratio).abs().max(dim=-1).values
    return (stability_norm < 1.0).float()
