# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""2-Link planar arm kinematics for ALBC (Active Linear Buoyancy Controller).

This module implements forward and inverse kinematics for the Hero Agent's
2-link planar arm that positions a buoyancy element for attitude control.

ALBC Arm Geometry (from IROS 2026 paper):
    l1 = l2 = 0.233 m  (link lengths, from isaaclab_assets constants)

    Joint configuration: Two revolute joints (gamma1, gamma2) operating in
    the XY plane, with the end-effector (buoyancy element) position computed
    as a function of joint angles.

Inverse Kinematics uses DLS (Damped Least Squares) Jacobian pseudo-inverse
with Yoshikawa-style adaptive damping for smooth singularity handling.

Note: This kinematic model is defined in the robot's local frame, where:
    - X points forward (along roll axis)
    - Y points left (along pitch axis)
    - Z points up
"""

from __future__ import annotations

import torch

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
)

# Small epsilon to prevent division by zero in 2x2 determinant inversion
_EPSILON_DET = 1e-10


class ALBCKinematics:
    """GPU-parallel kinematics for 2-link ALBC arm.

    This class provides efficient batch computation of FK and IK for the ALBC
    arm across multiple parallel environments.

    IK uses DLS Jacobian pseudo-inverse (closed-form 2x2 inversion) with
    Yoshikawa-style adaptive damping. This replaces analytical cosine-law IK,
    providing smooth behavior at workspace boundaries without clamping margins.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        link1_length: float = HERO_AGENT_ALBC_LINK1_LENGTH,
        link2_length: float = HERO_AGENT_ALBC_LINK2_LENGTH,
    ) -> None:
        """Initialize ALBC kinematics model.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device (e.g., "cuda:0", "cpu").
            link1_length: Length of first link in meters.
            link2_length: Length of second link in meters.
        """
        self.num_envs = num_envs
        self.device = device

        # Arm parameters
        self.l1 = link1_length
        self.l2 = link2_length

        # Maximum manipulability (for adaptive lambda normalization)
        # sqrt(|l1 * l2|) is the max of sqrt(|l1 * l2 * sin(g2)|)
        self._sqrt_l1l2 = (abs(self.l1 * self.l2)) ** 0.5

    def forward(
        self,
        joint_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute end-effector position from joint angles (forward kinematics).

        Standard 2-link planar FK:
            x = l1*cos(g1) + l2*cos(g1+g2)
            y = l1*sin(g1) + l2*sin(g1+g2)

        Args:
            joint_angles: Joint angles [gamma1, gamma2] in radians.
                Shape: (num_envs, 2).

        Returns:
            End-effector position [x, y] in meters. Shape: (num_envs, 2).
        """
        g1 = joint_angles[:, 0]
        g12 = joint_angles[:, 0] + joint_angles[:, 1]

        x = self.l1 * torch.cos(g1) + self.l2 * torch.cos(g12)
        y = self.l1 * torch.sin(g1) + self.l2 * torch.sin(g12)

        return torch.stack([x, y], dim=-1)

    def inverse(
        self,
        target_position: torch.Tensor,
        current_joint_angles: torch.Tensor,
        lambda_dls: float = 0.15,
    ) -> torch.Tensor:
        """Compute joint angle update from desired EE position using DLS IK.

        Uses Jacobian DLS (Damped Least Squares) pseudo-inverse with
        Yoshikawa-style adaptive damping. For the 2x2 Jacobian, JJ^T is
        inverted analytically (closed-form, no iteration).

        DLS formula:
            J_dls = J^T (J J^T + lambda^2 I)^{-1}
            delta_q = J_dls @ (p_target - p_current)
            q_new = q_current + delta_q

        Adaptive lambda (Yoshikawa):
            lambda_adaptive = lambda_base * (1 - sqrt(|l1*l2*sin(g2)|) / sqrt(|l1*l2|))
            Near singularity (g2~0 or pi): lambda -> lambda_base (max damping)
            Far from singularity: lambda -> 0 (no damping, full accuracy)

        Args:
            target_position: Desired EE position [x, y] in meters.
                Shape: (num_envs, 2).
            current_joint_angles: Current joint angles [gamma1, gamma2] in radians.
                Shape: (num_envs, 2).
            lambda_dls: Base DLS damping factor. Larger = more damping near
                singularity. C++ reference uses 0.15.

        Returns:
            New joint angles [gamma1, gamma2] in radians.
                Shape: (num_envs, 2).
        """
        g1 = current_joint_angles[:, 0]
        g2 = current_joint_angles[:, 1]
        g12 = g1 + g2

        # Current EE position via FK
        p_current = self.forward(current_joint_angles)

        # Position error
        dp = target_position - p_current  # (num_envs, 2)

        # 2x2 Jacobian
        # J = [[-l1*sin(g1) - l2*sin(g1+g2),  -l2*sin(g1+g2)],
        #      [ l1*cos(g1) + l2*cos(g1+g2),   l2*cos(g1+g2)]]
        sin_g1 = torch.sin(g1)
        cos_g1 = torch.cos(g1)
        sin_g12 = torch.sin(g12)
        cos_g12 = torch.cos(g12)

        j11 = -self.l1 * sin_g1 - self.l2 * sin_g12
        j12 = -self.l2 * sin_g12
        j21 = self.l1 * cos_g1 + self.l2 * cos_g12
        j22 = self.l2 * cos_g12

        # Yoshikawa-style adaptive lambda
        # manipulability ~ |l1 * l2 * sin(g2)|
        manipulability = torch.sqrt(torch.abs(self.l1 * self.l2 * torch.sin(g2)))
        lambda_adaptive = lambda_dls * (1.0 - manipulability / self._sqrt_l1l2)
        lam2 = lambda_adaptive**2  # (num_envs,)

        # A = J @ J^T + lambda^2 * I  (2x2, computed element-wise)
        # A11 = j11^2 + j12^2 + lam^2
        # A12 = j11*j21 + j12*j22
        # A21 = A12
        # A22 = j21^2 + j22^2 + lam^2
        a11 = j11**2 + j12**2 + lam2
        a12 = j11 * j21 + j12 * j22
        a22 = j21**2 + j22**2 + lam2

        # Analytical 2x2 inverse: A_inv = (1/det) * [[a22, -a12], [-a12, a11]]
        det_A = a11 * a22 - a12**2
        inv_det = 1.0 / (det_A + _EPSILON_DET)

        # A_inv columns
        ai11 = a22 * inv_det
        ai12 = -a12 * inv_det
        ai22 = a11 * inv_det

        # J_dls = J^T @ A_inv  (2x2 @ 2x2 -> 2x2, done element-wise)
        # J^T = [[j11, j21], [j12, j22]]
        # J_dls row 0 = [j11*ai11 + j21*ai12, j11*ai12 + j21*ai22]
        # J_dls row 1 = [j12*ai11 + j22*ai12, j12*ai12 + j22*ai22]
        jd11 = j11 * ai11 + j21 * ai12
        jd12 = j11 * ai12 + j21 * ai22
        jd21 = j12 * ai11 + j22 * ai12
        jd22 = j12 * ai12 + j22 * ai22

        # delta_q = J_dls @ dp
        dq1 = jd11 * dp[:, 0] + jd12 * dp[:, 1]
        dq2 = jd21 * dp[:, 0] + jd22 * dp[:, 1]

        q_new = current_joint_angles + torch.stack([dq1, dq2], dim=-1)
        return q_new
