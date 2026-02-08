# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""2-Link planar arm inverse kinematics for ALBC (Active Linear Buoyancy Controller).

This module implements inverse kinematics for the Hero Agent's 2-link planar arm
that positions a buoyancy element for attitude control.

ALBC Arm Geometry (from IROS 2026 paper):
    l1 = l2 = 0.233 m  (link lengths)
    h = 0.230 m        (constant height offset)

    Joint configuration: Two revolute joints (gamma1, gamma2) operating in
    the XY plane, with the end-effector (buoyancy element) position computed
    as a function of joint angles.

Inverse Kinematics (2-link planar, elbow-up solution):
    r = sqrt(x^2 + y^2)
    cos(gamma2) = (r^2 - l1^2 - l2^2) / (2*l1*l2)
    gamma2 = atan2(sqrt(1 - cos^2(gamma2)), cos(gamma2))
    gamma1 = atan2(y, x) - atan2(l2*sin(gamma2), l1 + l2*cos(gamma2))

Note: This kinematic model is defined in the robot's local frame, where:
    - X points forward (along roll axis)
    - Y points left (along pitch axis)
    - Z points up
"""

from __future__ import annotations

import torch


class ALBCKinematics:
    """GPU-parallel inverse kinematics for 2-link ALBC arm.

    This class provides efficient batch computation of IK for the ALBC
    arm across multiple parallel environments. The arm operates in a 2D
    plane (XY) with a constant Z offset.

    All computations are performed on GPU tensors for parallel simulation.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        link1_length: float = 0.233,
        link2_length: float = 0.233,
        height_offset: float = 0.230,
    ) -> None:
        """Initialize ALBC kinematics model.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device (e.g., "cuda:0", "cpu").
            link1_length: Length of first link in meters (default: 0.233).
            link2_length: Length of second link in meters (default: 0.233).
            height_offset: Constant Z offset in meters (default: 0.230).
        """
        self.num_envs = num_envs
        self.device = device

        # Arm parameters
        self.l1 = link1_length
        self.l2 = link2_length
        self.h = height_offset

        # Workspace limits (reachable range)
        self.r_min = abs(self.l1 - self.l2) + 1e-4  # Avoid singularity
        self.r_max = self.l1 + self.l2 - 1e-4  # Avoid singularity

        # Pre-computed constants for IK
        self._l1_sq = self.l1**2
        self._l2_sq = self.l2**2
        self._2l1l2 = 2.0 * self.l1 * self.l2

    def inverse(
        self,
        target_position: torch.Tensor,
        elbow_up: bool = True,
    ) -> torch.Tensor:
        """Compute joint angles from desired end-effector position.

        Uses analytical 2-link planar IK with elbow-up (default) or elbow-down
        configuration. Handles workspace boundary clamping to prevent
        unreachable targets.

        Args:
            target_position: Desired end-effector position [x, y] in meters.
                Shape: (num_envs, 2).
            elbow_up: If True, use elbow-up solution (gamma2 > 0).
                If False, use elbow-down solution (gamma2 < 0).

        Returns:
            Joint angles [gamma1, gamma2] in radians.
                Shape: (num_envs, 2).
        """
        x = target_position[:, 0]
        y = target_position[:, 1]

        # Compute distance from origin to target
        r_sq = x**2 + y**2
        r = torch.sqrt(r_sq)

        # Clamp to reachable workspace
        r_clamped = torch.clamp(r, self.r_min, self.r_max)

        # Scale target position to clamped radius
        scale = r_clamped / (r + 1e-8)
        x_clamped = x * scale
        y_clamped = y * scale
        r_sq_clamped = r_clamped**2

        # Compute gamma2 using law of cosines
        cos_gamma2 = (r_sq_clamped - self._l1_sq - self._l2_sq) / self._2l1l2
        cos_gamma2 = torch.clamp(cos_gamma2, -1.0, 1.0)
        sin_gamma2_abs = torch.sqrt(1.0 - cos_gamma2**2)

        # Compute gamma2 using atan2 with pre-computed sin value
        # For elbow_up: sin_gamma2 is positive
        # For elbow_down: sin_gamma2 is negative
        if elbow_up:
            gamma2 = torch.atan2(sin_gamma2_abs, cos_gamma2)
            sin_gamma2 = sin_gamma2_abs
        else:
            gamma2 = torch.atan2(-sin_gamma2_abs, cos_gamma2)
            sin_gamma2 = -sin_gamma2_abs

        # Compute gamma1
        k1 = self.l1 + self.l2 * cos_gamma2
        k2 = self.l2 * sin_gamma2
        gamma1 = torch.atan2(y_clamped, x_clamped) - torch.atan2(k2, k1)

        return torch.stack([gamma1, gamma2], dim=-1)

    def forward(
        self,
        joint_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute end-effector position from joint angles (forward kinematics).

        Standard 2-link planar FK:
            x = l1*cos(g1) + l2*cos(g1+g2)
            y = l1*sin(g1) + l2*sin(g1+g2)

        Note: Z coordinate is constant (height_offset) and not returned.

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

    def jacobian(
        self,
        joint_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 2x2 planar Jacobian matrix.

        J = [[-l1*sin(g1) - l2*sin(g1+g2), -l2*sin(g1+g2)],
             [ l1*cos(g1) + l2*cos(g1+g2),  l2*cos(g1+g2)]]

        Usage: v_EE = J @ gamma_dot (end-effector velocity from joint velocities)

        Args:
            joint_angles: Joint angles [gamma1, gamma2] in radians.
                Shape: (num_envs, 2).

        Returns:
            Jacobian matrix. Shape: (num_envs, 2, 2).
        """
        g1 = joint_angles[:, 0]
        g12 = joint_angles[:, 0] + joint_angles[:, 1]

        s1 = torch.sin(g1)
        c1 = torch.cos(g1)
        s12 = torch.sin(g12)
        c12 = torch.cos(g12)

        j11 = -self.l1 * s1 - self.l2 * s12   # dx/dg1
        j12 = -self.l2 * s12                    # dx/dg2
        j21 = self.l1 * c1 + self.l2 * c12     # dy/dg1
        j22 = self.l2 * c12                      # dy/dg2

        row1 = torch.stack([j11, j12], dim=-1)
        row2 = torch.stack([j21, j22], dim=-1)
        return torch.stack([row1, row2], dim=-2)
