# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time Delay Control (TDC) implementation for Hero Agent attitude control.

This module implements the TDC controller from the IROS 2026 paper (RL-ALBC).
TDC uses time-delayed control inputs to compensate for model uncertainty,
combined with learned PD gains from the RL policy.

Control Law (from 05_derivation.md, Step 9):
    p_EE,t = Lambda_t^{-1} * [
        Lambda_{t-L} * p_EE_{t-L}        -- TDE: delayed control contribution
        - M_hat * nu_dot_{t-L}            -- TDE: angular acceleration correction
        + M_hat * (K_d * e_dot + K_p * e) -- PD term
        + Delta_T_b                       -- passive restoring change
    ]

Where:
    - Lambda: Anti-diagonal coupling matrix [[0, -lf], [lf, 0]]
      with lf = cos(theta) * cos(phi) * F_bu
    - T_b: Passive restoring torque [cos(theta)*sin(phi)*F_bu*h, sin(theta)*F_bu*h]
    - Delta_T_b = T_b_{t-L} - T_b_t
    - nu_dot: Angular acceleration [p_dot, q_dot] via finite difference
    - M_hat: Estimated inertia (diagonal, from encoder z)
    - K_p, K_d: PD gains (from RL actor)

Reference:
    - Hsia & Gao (1990), "An explanation of the stability of model-based
      control using time delay estimation"
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


@configclass
class TDCControllerCfg:
    """Configuration for the TDC (Time Delay Control) controller.

    The TDC controller outputs desired end-effector positions based on
    attitude error and learned PD gains.
    """

    # Gain bounds (RL actor output is scaled to these ranges)
    k_p_min: float = 1.0
    """Minimum proportional gain."""

    k_p_max: float = 50.0
    """Maximum proportional gain."""

    k_d_min: float = 0.1
    """Minimum derivative gain."""

    k_d_max: float = 10.0
    """Maximum derivative gain."""

    # TDE (Time Delay Estimation) parameters
    tde_delay_steps: int = 1
    """Number of steps for time delay estimation (L in the paper)."""

    # Workspace radius (circular limit, computed from link lengths at runtime)
    workspace_radius_min: float = 0.01
    """Minimum reachable radius (|L1 - L2| + epsilon)."""

    workspace_radius_max: float = 0.466
    """Maximum reachable radius (L1 + L2 - epsilon)."""

    # Initial M_hat (diagonal inertia estimate)
    default_m_hat: tuple[float, float] = (1.0, 1.0)
    """Default diagonal inertia estimate for roll and pitch."""

    # ALBC geometry
    height_offset: float = 0.1625
    """Height offset h (joint1 z-offset from base) for T_b computation."""


class TDCController:
    """GPU-parallel Time Delay Control for ALBC attitude stabilization.

    This controller computes desired end-effector positions based on:
    1. Attitude error (roll, pitch) from target orientation
    2. PD gains from the RL policy (learned adaptive control)
    3. TDE (Time Delay Estimation) for model uncertainty compensation
    4. Estimated inertia M_hat from the encoder network

    The controller operates on the roll and pitch axes only, as yaw control
    is not possible with buoyancy-based actuation.
    """

    def __init__(
        self,
        cfg: TDCControllerCfg,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the TDC controller.

        Args:
            cfg: TDC controller configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Gain ranges for action scaling
        self._k_p_range = cfg.k_p_max - cfg.k_p_min
        self._k_d_range = cfg.k_d_max - cfg.k_d_min

        # Cache TDE delay steps and geometry
        self._delay = cfg.tde_delay_steps
        self._height_offset = cfg.height_offset

        # Initialize state buffers
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Initialize internal state buffers for TDE computation."""
        # Current gains (2D: roll, pitch)
        self._k_p = torch.ones(self.num_envs, 2, device=self.device) * self.cfg.k_p_min
        self._k_d = torch.ones(self.num_envs, 2, device=self.device) * self.cfg.k_d_min

        # Inertia estimate (2D diagonal: roll, pitch)
        default_m = torch.tensor(self.cfg.default_m_hat, device=self.device)
        self._m_hat = default_m.unsqueeze(0).expand(self.num_envs, -1).clone()

        # TDE history buffers: +2 extra for angular velocity finite difference
        # Need idx and idx-1 for nu_dot, plus delay offset
        self._buffer_size = self._delay + 3
        self._p_ee_history = torch.zeros(self.num_envs, self._buffer_size, 2, device=self.device)
        self._angular_vel_history = torch.zeros(self.num_envs, self._buffer_size, 2, device=self.device)
        self._roll_history = torch.zeros(self.num_envs, self._buffer_size, device=self.device)
        self._pitch_history = torch.zeros(self.num_envs, self._buffer_size, device=self.device)
        self._lambda_factor_history = torch.zeros(self.num_envs, self._buffer_size, device=self.device)
        self._history_idx = 0

        # Per-env warmup counter to suppress TDE spikes after reset
        self._warmup_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._min_warmup = 3  # Need enough steps for valid finite difference

        # Last output for continuity
        self._last_output = torch.zeros(self.num_envs, 2, device=self.device)

    def set_gains(self, actions: torch.Tensor) -> None:
        """Set PD gains from RL actor output.

        The actor outputs 4D actions in [-1, 1] that are scaled to the
        configured gain ranges.

        Args:
            actions: Actor output [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch].
                Shape: (num_envs, 4).
        """
        # Scale from [-1, 1] to [0, 1]
        actions_01 = (actions + 1.0) * 0.5

        # Scale to gain ranges
        self._k_p[:, 0] = self.cfg.k_p_min + actions_01[:, 0] * self._k_p_range  # roll
        self._k_d[:, 0] = self.cfg.k_d_min + actions_01[:, 1] * self._k_d_range  # roll
        self._k_p[:, 1] = self.cfg.k_p_min + actions_01[:, 2] * self._k_p_range  # pitch
        self._k_d[:, 1] = self.cfg.k_d_min + actions_01[:, 3] * self._k_d_range  # pitch

    def set_inertia_estimate(self, z: torch.Tensor) -> None:
        """Set inertia estimate from encoder latent output.

        M_hat = diag(z) where z is the encoder output (positive via softplus).
        Only uses the first 2 components of z for roll/pitch.

        Args:
            z: Encoder latent output (positive values).
                Shape: (num_envs, latent_dim) where latent_dim >= 2.
        """
        # Use first 2 components for roll/pitch inertia
        self._m_hat = torch.clamp(z[:, :2], min=0.1)

    # ------------------------------------------------------------------
    # Lambda and T_b helpers (from 05_derivation.md, Step 2)
    # ------------------------------------------------------------------

    def _compute_lambda_factor(self, roll: torch.Tensor, pitch: torch.Tensor, f_bu: torch.Tensor) -> torch.Tensor:
        """Compute the scalar factor for the Lambda coupling matrix.

        Lambda = [[0, -lf], [lf, 0]] where lf = cos(theta)*cos(phi)*F_bu.

        Args:
            roll: Roll angle phi in radians. Shape: (num_envs,).
            pitch: Pitch angle theta in radians. Shape: (num_envs,).
            f_bu: Buoyancy force magnitude. Shape: (num_envs,).

        Returns:
            Lambda factor lf = cos(pitch)*cos(roll)*F_bu. Shape: (num_envs,).
        """
        return torch.cos(pitch) * torch.cos(roll) * f_bu

    def _apply_lambda(self, p_ee: torch.Tensor, lambda_factor: torch.Tensor) -> torch.Tensor:
        """Apply Lambda matrix: tau = Lambda * p_EE.

        Lambda = [[0, -lf], [lf, 0]], so:
            tau_roll  = -lf * y_EE
            tau_pitch =  lf * x_EE

        Args:
            p_ee: End-effector position [x, y]. Shape: (num_envs, 2).
            lambda_factor: Lambda factor lf. Shape: (num_envs,).

        Returns:
            Torque [tau_roll, tau_pitch]. Shape: (num_envs, 2).
        """
        tau_roll = -lambda_factor * p_ee[:, 1]  # -lf * y
        tau_pitch = lambda_factor * p_ee[:, 0]  # lf * x
        return torch.stack([tau_roll, tau_pitch], dim=-1)

    def _apply_lambda_inv(self, torque: torch.Tensor, lambda_factor: torch.Tensor) -> torch.Tensor:
        """Apply Lambda inverse: p_EE = Lambda^{-1} * tau.

        Lambda^{-1} = [[0, 1/lf], [-1/lf, 0]], so:
            x_EE =  tau_pitch / lf
            y_EE = -tau_roll / lf

        Args:
            torque: Torque [tau_roll, tau_pitch]. Shape: (num_envs, 2).
            lambda_factor: Lambda factor lf. Shape: (num_envs,).

        Returns:
            End-effector position [x, y]. Shape: (num_envs, 2).
        """
        # Numerical safety: lf = cos(theta)*cos(phi)*F_bu must be positive.
        # Small-angle assumption guarantees this in normal operation (+/-45 deg DR range).
        # Clamp handles transient exceedance during exploration.
        lf_safe = torch.clamp(lambda_factor, min=1e-8)  # (N,)
        x_ee = torque[:, 1] / lf_safe  #  tau_pitch / lf
        y_ee = -torque[:, 0] / lf_safe  # -tau_roll / lf
        return torch.stack([x_ee, y_ee], dim=-1)

    def _compute_t_b(self, roll: torch.Tensor, pitch: torch.Tensor, f_bu: torch.Tensor) -> torch.Tensor:
        """Compute passive restoring torque T_b.

        T_b = [cos(theta)*sin(phi)*F_bu*h, sin(theta)*F_bu*h]

        Args:
            roll: Roll angle phi in radians. Shape: (num_envs,).
            pitch: Pitch angle theta in radians. Shape: (num_envs,).
            f_bu: Buoyancy force magnitude. Shape: (num_envs,).

        Returns:
            Restoring torque [T_b_roll, T_b_pitch]. Shape: (num_envs, 2).
        """
        h = self._height_offset
        t_b_roll = torch.cos(pitch) * torch.sin(roll) * f_bu * h
        t_b_pitch = torch.sin(pitch) * f_bu * h
        return torch.stack([t_b_roll, t_b_pitch], dim=-1)

    # ------------------------------------------------------------------

    def compute(
        self,
        attitude_error: torch.Tensor,
        angular_velocity: torch.Tensor,
        dt: float,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        f_bu: torch.Tensor,
    ) -> torch.Tensor:
        """Compute desired end-effector position using full TDC control law.

        Implements the control law from 05_derivation.md (Step 9):
            p_EE,t = Lambda_t^{-1} * [
                Lambda_{t-L} * p_EE_{t-L}
                - M_hat * nu_dot_{t-L}
                + M_hat * (K_d * e_dot + K_p * e)
                + T_b_{t-L} - T_b_t
            ]

        Args:
            attitude_error: Attitude error [roll_error, pitch_error] in radians.
                Shape: (num_envs, 2).
            angular_velocity: Body angular velocity [omega_x, omega_y] in rad/s.
                Shape: (num_envs, 2). Roll and pitch rates.
            dt: Time step in seconds.
            roll: Current roll angle phi in radians. Shape: (num_envs,).
            pitch: Current pitch angle theta in radians. Shape: (num_envs,).
            f_bu: Buoyancy force magnitude. Shape: (num_envs,).

        Returns:
            Desired end-effector position [x, y] in meters.
                Shape: (num_envs, 2).
        """
        # error_dot: d(error)/dt = -angular_velocity (target = 0 is constant)
        error_dot = -angular_velocity

        # PD term in torque space: M_hat * (K_p * e + K_d * e_dot)
        pd_term = self._m_hat * (self._k_p * attitude_error + self._k_d * error_dot)

        # Current Lambda factor and T_b
        lf_current = self._compute_lambda_factor(roll, pitch, f_bu)
        t_b_current = self._compute_t_b(roll, pitch, f_bu)

        # Store current state in history
        self._update_tde_history(self._last_output, angular_velocity, roll, pitch, lf_current)

        # Retrieve delayed quantities (t - L)
        delayed_idx = (self._history_idx - self._delay) % self._buffer_size
        p_ee_delayed = self._p_ee_history[:, delayed_idx, :]
        lf_delayed = self._lambda_factor_history[:, delayed_idx]
        roll_delayed = self._roll_history[:, delayed_idx]
        pitch_delayed = self._pitch_history[:, delayed_idx]
        omega_delayed = self._angular_vel_history[:, delayed_idx, :]

        # Angular acceleration via first-order finite difference: nu_dot = (omega_t - omega_{t-1}) / dt
        prev_delayed_idx = (delayed_idx - 1) % self._buffer_size
        omega_prev_delayed = self._angular_vel_history[:, prev_delayed_idx, :]
        nu_dot_delayed = (omega_delayed - omega_prev_delayed) / (dt + 1e-8)

        # T_b at delayed time.
        # Note: f_bu is assumed constant within an episode (set at reset via DR).
        # If time-varying buoyancy is added, f_bu must also be stored in history.
        t_b_delayed = self._compute_t_b(roll_delayed, pitch_delayed, f_bu)

        # Assemble control law terms (all in torque space)
        # term1: Lambda_{t-L} * p_EE_{t-L}
        term1 = self._apply_lambda(p_ee_delayed, lf_delayed)
        # term2: -M_hat * nu_dot_{t-L}
        term2 = -self._m_hat * nu_dot_delayed
        # term4: Delta_T_b = T_b_{t-L} - T_b_t
        term4 = t_b_delayed - t_b_current

        # During warmup (first 3 steps after reset), use PD only through Lambda_inv
        # This prevents TDE spikes from zero/stale history
        warmup_mask = (self._warmup_steps < self._min_warmup).unsqueeze(-1)  # (N, 1)
        tde_terms = term1 + term2 + term4  # All TDE-dependent terms
        tde_terms = torch.where(warmup_mask, torch.zeros_like(tde_terms), tde_terms)
        self._warmup_steps += 1

        # Total torque command
        tau_total = tde_terms + pd_term

        # Convert torque to EE position: p_EE = Lambda_t^{-1} * tau
        p_ee_desired = self._apply_lambda_inv(tau_total, lf_current)

        # Clamp to circular workspace (2-link arm reachable region)
        r = torch.linalg.norm(p_ee_desired, dim=-1, keepdim=True)
        too_small = r < 1e-6
        r_clamped = torch.clamp(r, self.cfg.workspace_radius_min, self.cfg.workspace_radius_max)
        scale = r_clamped / (r + 1e-8)
        p_ee_desired = torch.where(too_small, torch.zeros_like(p_ee_desired), p_ee_desired * scale)

        self._last_output = p_ee_desired.clone()
        return p_ee_desired

    def _update_tde_history(
        self,
        p_ee: torch.Tensor,
        angular_vel: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        lambda_factor: torch.Tensor,
    ) -> None:
        """Update all TDE history buffers with current state.

        Args:
            p_ee: Current end-effector position [x, y]. Shape: (num_envs, 2).
            angular_vel: Angular velocity [p, q]. Shape: (num_envs, 2).
            roll: Roll angle phi. Shape: (num_envs,).
            pitch: Pitch angle theta. Shape: (num_envs,).
            lambda_factor: Precomputed cos(theta)*cos(phi)*F_bu. Shape: (num_envs,).
        """
        # Advance circular buffer index
        self._history_idx = (self._history_idx + 1) % self._buffer_size
        idx = self._history_idx

        self._p_ee_history[:, idx, :] = p_ee
        self._angular_vel_history[:, idx, :] = angular_vel
        self._roll_history[:, idx] = roll
        self._pitch_history[:, idx] = pitch
        self._lambda_factor_history[:, idx] = lambda_factor

    def get_stability_metric(self, true_inertia: torch.Tensor | None = None) -> torch.Tensor:
        """Compute TDC stability metric for reward computation.

        The stability condition requires that the inertia estimate is close
        to the true inertia: ||1 - M/M_hat|| should be small.

        Args:
            true_inertia: True diagonal inertia values (if known).
                Shape: (num_envs, 2). If None, returns zeros.

        Returns:
            Stability metric (lower is better, 0 = perfect estimate).
                Shape: (num_envs,).
        """
        if true_inertia is None:
            return torch.zeros(self.num_envs, device=self.device)

        # Compute ||1 - M/M_hat||
        ratio = true_inertia / (self._m_hat + 1e-8)
        deviation = torch.abs(1.0 - ratio)
        return torch.linalg.norm(deviation, dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset controller state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        # Reset gains to minimum
        self._k_p[env_ids] = self.cfg.k_p_min
        self._k_d[env_ids] = self.cfg.k_d_min

        # Reset M_hat to default (broadcasts from (2,) to (len(env_ids), 2))
        self._m_hat[env_ids] = torch.tensor(self.cfg.default_m_hat, device=self.device)

        # Reset all history buffers
        self._p_ee_history[env_ids] = 0.0
        self._angular_vel_history[env_ids] = 0.0
        self._roll_history[env_ids] = 0.0
        self._pitch_history[env_ids] = 0.0
        self._lambda_factor_history[env_ids] = 0.0
        self._last_output[env_ids] = 0.0
        self._warmup_steps[env_ids] = 0

    @property
    def current_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current PD gains.

        Returns:
            Tuple of (K_p, K_d) tensors, each of shape (num_envs, 2).
        """
        return self._k_p, self._k_d

    @property
    def inertia_estimate(self) -> torch.Tensor:
        """Get current inertia estimate M_hat.

        Returns:
            M_hat tensor of shape (num_envs, 2).
        """
        return self._m_hat
