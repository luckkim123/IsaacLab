# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

Provides reward configuration, a lightweight reward manager, and reward
functions for ALBC (joint-based attitude control) training.

Reward design principles:
    - Gaussian kernel tracking + settling bonus dominate: positive rewards in
      [0,1] provide dense gradient. Penalties are tiny regularizers to avoid
      suppressing exploration.
    - dt-scaling: state-quality terms (tracking, settling, linear_error) are
      dt-scaled; action_rate and action_oscillation are NOT dt-scaled
      (per-step values naturally scale with frequency).
    - PBRS progress shaping (Ng 1999): preserves optimal policy guarantee.
    - Sigma annealing: starts wide (0.5 rad, safe at large errors) and
      tightens to 0.25 rad for fine-tuning gradient. Prevents reward
      death spiral during early training while improving precision later.
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
    """ALBC reward configuration with Gaussian tracking + regularization penalties.

    Reward = tracking * w_t * dt
           + settling * w_s * dt
           + linear_error * w_le * dt (strong tail gradient)
           + action_rate * w_ar  (NOT dt-scaled, penalizes action changes)
           + action_oscillation * w_ao  (NOT dt-scaled)
           + progress (PBRS)

    Design: tracking + settling dominate (positive, [0,1]). linear_error
    ensures gradient at ALL error levels (Gaussian vanishes at large errors).
    """

    # Tracking (Gaussian kernel) with sigma annealing
    tracking_weight: float = 3.0
    tracking_sigma: float = 0.5  # initial sigma (28.6 deg 1/e point, safe for large errors)
    tracking_sigma_final: float = 0.25  # final sigma (14.3 deg, stronger fine-tuning gradient)
    sigma_anneal_iters: int = 1000  # anneal sigma linearly over this many runner iterations

    # Action rate penalty: ||a_t - a_{t-1}||^2 (penalizes large changes).
    # NOT dt-scaled (per-step delta naturally scales with frequency).
    action_rate_weight: float = -0.01

    # Action oscillation penalty: ||a_t - 2*a_{t-1} + a_{t-2}||^2 (2nd derivative)
    # Specifically targets direction reversals (high-frequency oscillation).
    # NOT dt-scaled.
    action_oscillation_weight: float = -0.01

    # Linear error penalty: -min(||err||, max_err) / max_err.
    # Provides constant gradient at ALL error levels (unlike Gaussian which
    # vanishes at large errors). Clamped to [-1, 0]. dt-scaled.
    linear_error_weight: float = -0.3
    linear_error_max: float = 1.0  # clamp at ~57 degrees

    # Progress (potential-based shaping): PBRS (Ng 1999) preserves optimal policy.
    # NOT dt-scaled. Provides immediate reward for error reduction at all levels.
    progress_weight: float = 0.3
    progress_scale: float = 0.01
    progress_mode: str = "pbrs"  # "tanh" or "pbrs" (SAC-safe, policy-preserving)
    progress_gamma: float = 0.99  # discount factor for PBRS (match PPO gamma)

    # Settling bonus: sigmoid(sharpness * (threshold - error)), dt-scaled.
    # Dense gradient near target where Gaussian tracking has flat top.
    settling_weight: float = 2.0
    settling_threshold: float = 0.10  # radians (~5.7 deg)
    settling_sharpness: float = 30.0  # 1/radians

    # Angular velocity penalty (dt-scaled, discourages oscillation under DR)
    # Keep at 0.0 for attitude control: robot needs angular velocity to correct errors.
    angular_velocity_weight: float = 0.0

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

    scale_by_dt: bool = True
    """Whether to scale reward by dt. Set False for progress-style rewards."""


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
    def current_sigma(self) -> float | None:
        """Current tracking sigma value."""
        for name, cfg in zip(self._term_names, self._term_cfgs):
            if name == "tracking" and "sigma" in cfg.params:
                return cfg.params["sigma"]
        return None

    def update_sigma(self, iteration: int, reward_cfg: ALBCRewardCfg) -> float | None:
        """Anneal tracking sigma linearly from start to final over configured iterations.

        Args:
            iteration: Current runner iteration (0-indexed).
            reward_cfg: Reward config with sigma_anneal_iters and tracking_sigma_final.

        Returns:
            New sigma value, or None if no tracking term.
        """
        total = reward_cfg.sigma_anneal_iters
        if total <= 0:
            return self.current_sigma
        t = min(1.0, iteration / total)
        new_sigma = reward_cfg.tracking_sigma + t * (reward_cfg.tracking_sigma_final - reward_cfg.tracking_sigma)
        for name, cfg in zip(self._term_names, self._term_cfgs):
            if name == "tracking" and "sigma" in cfg.params:
                cfg.params["sigma"] = new_sigma
                return new_sigma
        return None

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


def action_rate_penalty(
    _robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Mean of squared action differences: ||a_t - a_{t-1}||^2.

    Penalizes large action changes (rate of change).
    Uses mean (not sum) so the penalty magnitude is independent of action_space dim.
    NOT dt-scaled: per-step delta naturally scales with frequency.
    Use with negative weight.
    """
    return torch.mean((actions - prev_actions) ** 2, dim=-1)


def action_oscillation_penalty(
    _robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_prev_actions: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Mean of squared second derivative: ||a_t - 2*a_{t-1} + a_{t-2}||^2.

    Specifically targets oscillation (direction reversals). A smooth ramp
    has zero second derivative; oscillation produces large values.
    Uses mean (not sum) so the penalty magnitude is independent of action_space dim.
    NOT dt-scaled. Use with negative weight.
    """
    second_diff = actions - 2.0 * prev_actions + prev_prev_actions
    return torch.mean(second_diff**2, dim=-1)


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
    """Potential-based progress reward: tanh((prev - curr) / scale).

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
    gamma: float = 0.997,
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
    sharpness: float = 30.0,
    **_kwargs,
) -> torch.Tensor:
    """Sigmoid-gated settling bonus: sigmoid(k * (threshold - error)).

    Provides dense gradient in the near-target zone where Gaussian tracking
    reward has a flat top (gradient -> 0 as error -> 0).

    Output [0, 1]. sigmoid(0)=0.5 at error=threshold.
    Markov-safe: depends only on current state. Compatible with SAC replay buffer.
    dt-scaled (instantaneous state quality). Use with positive weight.

    Args:
        env: Environment instance (provides _potentials = ||[roll_err, pitch_err]||).
        threshold: Settling zone boundary in radians. Default 0.10 rad (~5.7 deg).
        sharpness: Sigmoid slope coefficient (1/rad). Higher = sharper transition.
    """
    return torch.sigmoid(sharpness * (threshold - env._potentials))


def angular_velocity_penalty(
    robot: Articulation,
    **_kwargs,
) -> torch.Tensor:
    """Sum of squared body-frame angular velocities.

    Penalizes oscillatory motion to encourage smooth settling.
    dt-scaled (instantaneous quality measure). Use with negative weight.

    Returns:
        (num_envs,) penalty value (positive; apply negative weight).
    """
    return torch.sum(robot.data.root_ang_vel_b**2, dim=-1)


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
# Encoder-TDC Exclusive Reward Functions
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
        env: Encoder-TDC environment instance.
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
    )

    M_hat = env._tdc._m_hat
    ratio = M_true / M_hat.clamp(min=1e-6)
    stability_norm = (1.0 - ratio).abs().max(dim=-1).values
    return (stability_norm < 1.0).float()
