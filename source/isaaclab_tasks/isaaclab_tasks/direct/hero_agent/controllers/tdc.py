# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time Delay Controller (TDC) for Hero Agent roll/pitch attitude stabilization.

TDC uses Time Delay Estimation (TDE) to approximate uncertain nonlinear dynamics
without explicit modeling. The controller positions the buoyancy element (end-effector)
to generate restoring torques for attitude control.

Control Law (from IROS 2026 derivation):
    p_EE(t) = Lambda_inv(t) @ [Lambda(t-L) @ p_EE(t-L)
              - M_hat @ nu_dot(t-L) + M_hat @ (Kd @ e_dot + Kp @ e)
              + T_b(t-L) - T_b(t)]

where:
    Lambda:   Anti-diagonal coupling matrix (roll/pitch <-> EE position)
    T_b:      Passive restoring torque (known, computed explicitly)
    M_hat:    Design inertia matrix (constant diagonal)
    H_hat:    TDE estimate of uncertain dynamics
    e:        Attitude error [phi_d - phi, theta_d - theta]
    e_dot:    Error rate (small-angle approx: [-p, -q])

References:
    - 05_derivation.md (IROS 2026 notes)
    - T.C. Hsia & L.S. Lasky, "Robust independent joint controller design
      for industrial robot manipulators," IEEE Trans. Ind. Electron., 1991.
"""

from __future__ import annotations

import torch


class TDCController:
    """GPU-parallel TDC for Hero Agent roll/pitch attitude stabilization.

    Computes desired end-effector position to stabilize roll/pitch angles
    using Time Delay Estimation for uncertain dynamics compensation.

    The controller operates in a 2D task space [phi, theta] and outputs
    2D end-effector positions [x_EE, y_EE] for the ALBC arm.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        m_hat: tuple[float, float] = (0.15, 0.15),
        kp: float = 4.0,
        kd: float = 3.0,
        F_bu: float = 26.24,
        h: float = 0.230,
        dls_damping: float = 0.01,
        dt: float = 0.01,
        workspace_radius: float = 0.45,
        nu_dot_ema_alpha: float = 0.3,
        tde_gain: float = 1.0,
        h_hat_filter_alpha: float = 0.05,
    ) -> None:
        """Initialize TDC controller.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device (e.g., "cuda:0").
            m_hat: Design inertia for (roll, pitch) in kg*m^2.
            kp: Proportional gain (omega_n^2 for second-order dynamics).
            kd: Derivative gain (2*zeta*omega_n for second-order dynamics).
            F_bu: Buoyancy force magnitude in N (from buoy hydrodynamics).
            h: Height offset from CoG to CoB in meters.
            dls_damping: Damped Least Squares regularization for Lambda inverse.
            dt: Control timestep in seconds (= TDE delay L).
            workspace_radius: Maximum EE distance from origin (m). Must be less
                than l1+l2 to avoid IK singularity at full extension.
            nu_dot_ema_alpha: EMA smoothing factor for angular acceleration
                finite difference. Lower = smoother (0 < alpha <= 1).
            tde_gain: Scale factor for TDE contribution (0.0 = pure PD,
                1.0 = full TDC). Useful for diagnosing TDE instability.
            h_hat_filter_alpha: EMA filter for H_hat (TDE estimate).
                Low values = strong filtering (cuts high-freq noise in TDE).
                alpha = dt / (tau + dt) where tau is the filter time constant.
                Default 0.05 corresponds to tau ~ 0.19s (cutoff ~ 0.8 Hz).
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.dls_damping = dls_damping
        self.h = h
        self.workspace_radius = workspace_radius
        self.nu_dot_ema_alpha = nu_dot_ema_alpha
        self.tde_gain = tde_gain
        self.h_hat_filter_alpha = h_hat_filter_alpha

        # Design inertia (diagonal 2x2)
        self._m_hat = torch.tensor(m_hat, device=device, dtype=torch.float32)

        # PD gains (scalar, broadcast to 2D)
        self._kp = kp
        self._kd = kd

        # Buoyancy force (per-env, updated at reset from hydrodynamics model)
        self._F_bu = torch.full((num_envs,), F_bu, device=device, dtype=torch.float32)

        # --- History buffers for TDE ---
        self._nu_prev = torch.zeros(num_envs, 2, device=device)  # [p, q] at t-L
        self._nu_dot_filtered = torch.zeros(num_envs, 2, device=device)  # EMA-filtered angular accel
        self._p_EE_prev = torch.zeros(num_envs, 2, device=device)  # EE position at t-L
        self._Lambda_prev = torch.zeros(num_envs, 2, 2, device=device)  # Lambda at t-L
        self._T_b_prev = torch.zeros(num_envs, 2, device=device)  # restoring torque at t-L
        self._U_hat_filtered = torch.zeros(num_envs, 2, device=device)  # filtered pure uncertainty
        self._is_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update_buoyancy_force(self, F_bu: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Update buoyancy force from hydrodynamics model.

        Args:
            F_bu: Buoyancy force magnitude. Shape: (N,) or scalar.
            env_ids: Environment indices to update. None = all.
        """
        if env_ids is None:
            self._F_bu[:] = F_bu
        else:
            self._F_bu[env_ids] = F_bu[env_ids] if F_bu.dim() > 0 else F_bu

    def _compute_lambda(self, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """Compute Lambda coupling matrix.

        Lambda = [[0, -lf], [lf, 0]]
        where lf = cos(theta) * cos(phi) * F_bu

        Args:
            roll: Roll angle (phi) in radians. Shape: (num_envs,).
            pitch: Pitch angle (theta) in radians. Shape: (num_envs,).

        Returns:
            Lambda matrix. Shape: (num_envs, 2, 2).
        """
        lf = torch.cos(pitch) * torch.cos(roll) * self._F_bu  # (num_envs,)

        Lambda = torch.zeros(self.num_envs, 2, 2, device=self.device)
        Lambda[:, 0, 1] = -lf
        Lambda[:, 1, 0] = lf
        return Lambda

    def _compute_lambda_inv(self, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """Compute DLS-regularized inverse of Lambda.

        Lambda_inv = [[0, lf_inv], [-lf_inv, 0]]
        where lf_inv = lf / (lf^2 + damping^2)

        Args:
            roll: Roll angle (phi) in radians. Shape: (num_envs,).
            pitch: Pitch angle (theta) in radians. Shape: (num_envs,).

        Returns:
            Lambda inverse matrix. Shape: (num_envs, 2, 2).
        """
        lf = torch.cos(pitch) * torch.cos(roll) * self._F_bu
        lf_inv = lf / (lf**2 + self.dls_damping**2)

        Lambda_inv = torch.zeros(self.num_envs, 2, 2, device=self.device)
        Lambda_inv[:, 0, 1] = lf_inv
        Lambda_inv[:, 1, 0] = -lf_inv
        return Lambda_inv

    def _compute_restoring_torque(self, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """Compute passive restoring torque T_b.

        T_b = [cos(theta)*sin(phi)*F_bu*h, sin(theta)*F_bu*h]

        Args:
            roll: Roll angle (phi) in radians. Shape: (num_envs,).
            pitch: Pitch angle (theta) in radians. Shape: (num_envs,).

        Returns:
            Restoring torque vector. Shape: (num_envs, 2).
        """
        T_b = torch.stack([
            torch.cos(pitch) * torch.sin(roll) * self._F_bu * self.h,
            torch.sin(pitch) * self._F_bu * self.h,
        ], dim=-1)
        return T_b

    def compute(
        self,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        ang_vel_body: torch.Tensor,
        target_euler: torch.Tensor,
        joint_pos: torch.Tensor,
        kinematics,
    ) -> torch.Tensor:
        """Compute desired end-effector position using TDC law.

        Args:
            roll: Current roll angle (phi) in radians. Shape: (num_envs,).
            pitch: Current pitch angle (theta) in radians. Shape: (num_envs,).
            ang_vel_body: Body angular velocity [p, q, r]. Shape: (num_envs, 3).
            target_euler: Target [roll, pitch, yaw] in radians. Shape: (num_envs, 3).
            joint_pos: Current joint angles [gamma1, gamma2]. Shape: (num_envs, 2).
            kinematics: ALBCKinematics instance for FK (joint_pos -> p_EE).

        Returns:
            Desired EE position [x, y] in meters. Shape: (num_envs, 2).
        """
        # Current angular velocity (roll/pitch only)
        nu = ang_vel_body[:, :2]  # [p, q]

        # Current EE position from FK
        p_EE_current = kinematics.forward(joint_pos)

        # Angular acceleration: finite difference + EMA low-pass filter
        # Raw finite difference is noisy at 100Hz; EMA smooths TDE input
        nu_dot_raw = (nu - self._nu_prev) / self.dt
        alpha = self.nu_dot_ema_alpha
        nu_dot = alpha * nu_dot_raw + (1.0 - alpha) * self._nu_dot_filtered

        # Current Lambda, Lambda_inv, T_b
        Lambda = self._compute_lambda(roll, pitch)
        Lambda_inv = self._compute_lambda_inv(roll, pitch)
        T_b = self._compute_restoring_torque(roll, pitch)

        # --- PD error dynamics ---
        # e = [phi_d - phi, theta_d - theta]
        e = torch.stack([target_euler[:, 0] - roll, target_euler[:, 1] - pitch], dim=-1)
        # e_dot = [-p, -q] (small angle approximation: d/dt(phi_d - phi) ~ -p)
        e_dot = -nu

        # u_pd = Kd * e_dot + Kp * e
        u_pd = self._kd * e_dot + self._kp * e

        # --- TDC control law (derivation Step 9) ---
        # p_EE = Lambda_inv @ [Lambda_prev @ p_EE_prev - M_hat*nu_dot_prev
        #                      + M_hat*u_pd + (T_b_prev - T_b)]
        #
        # Split into: U_hat (pure uncertainty, filtered) + delta_T_b (not filtered)
        #   U_hat_raw = Lambda_prev @ p_EE_prev - M_hat * nu_dot_prev
        #   delta_T_b = T_b_prev - T_b
        #   tau = tde_gain * (filter(U_hat_raw) + delta_T_b) + M_hat*u_pd

        # TDE term: Lambda_prev @ p_EE_prev (batched matmul)
        tde_lambda_p = torch.bmm(self._Lambda_prev, self._p_EE_prev.unsqueeze(-1)).squeeze(-1)

        # TDE components
        tde_m_nu_dot = self._m_hat * self._nu_dot_filtered
        tde_delta_T_b = self._T_b_prev - T_b
        m_hat_u_pd = self._m_hat * u_pd

        # Pure uncertainty (T_b excluded): U_hat = Lambda_prev @ p_EE_prev - M_hat*nu_dot
        U_hat_raw = tde_lambda_p - tde_m_nu_dot

        # Low-pass filter on pure uncertainty only.
        # T_b is excluded so that delta_T_b = T_b_prev - T_b cancels exactly.
        beta = self.h_hat_filter_alpha
        U_hat = beta * U_hat_raw + (1.0 - beta) * self._U_hat_filtered

        # Full H_hat for debug logging: H_hat = U_hat + T_b_prev (per derivation Step 6)
        tde_H_hat = U_hat + self._T_b_prev

        # tau_desired for initialized envs (full TDC)
        tau_tdc = (
            self.tde_gain * (U_hat + tde_delta_T_b)
            + m_hat_u_pd
        )

        # tau_desired for uninitialized envs (pure PD, no TDE)
        tau_pd = m_hat_u_pd

        # Select between TDC and pure PD based on initialization
        init_mask = self._is_initialized.unsqueeze(-1)  # (num_envs, 1)
        tau_desired = torch.where(init_mask, tau_tdc, tau_pd)

        # p_EE = Lambda_inv @ tau_desired
        p_EE = torch.bmm(Lambda_inv, tau_desired.unsqueeze(-1)).squeeze(-1)

        # --- Workspace clamping ---
        # Clamp p_EE to workspace radius to prevent IK saturation.
        # Without this, TDE feedback diverges when commands exceed workspace.
        p_EE_raw = p_EE.clone()
        r = torch.norm(p_EE, dim=-1, keepdim=True)  # (num_envs, 1)
        scale = torch.clamp(self.workspace_radius / (r + 1e-8), max=1.0)
        p_EE = p_EE * scale

        # --- Store debug info for logging ---
        self._debug = {
            "tde_lambda_p": tde_lambda_p,
            "tde_m_nu_dot": tde_m_nu_dot,
            "tde_H_hat": tde_H_hat,
            "m_hat_u_pd": m_hat_u_pd,
            "tde_delta_T_b": tde_delta_T_b,
            "tau_desired": tau_desired,
            "p_EE_raw": p_EE_raw,
            "p_EE_raw_norm": torch.norm(p_EE_raw, dim=-1),
            "nu_dot_raw": nu_dot_raw,
            "nu_dot_filtered": nu_dot,
        }

        # --- Update history buffers ---
        self._nu_prev = nu.clone()
        self._nu_dot_filtered = nu_dot.clone()
        self._p_EE_prev = p_EE_current.clone()
        self._Lambda_prev = Lambda.clone()
        self._T_b_prev = T_b.clone()
        self._U_hat_filtered = U_hat.clone()
        self._is_initialized[:] = True

        return p_EE

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset controller state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._nu_prev[env_ids] = 0.0
        self._nu_dot_filtered[env_ids] = 0.0
        self._p_EE_prev[env_ids] = 0.0
        self._Lambda_prev[env_ids] = 0.0
        self._T_b_prev[env_ids] = 0.0
        self._U_hat_filtered[env_ids] = 0.0
        self._is_initialized[env_ids] = False
