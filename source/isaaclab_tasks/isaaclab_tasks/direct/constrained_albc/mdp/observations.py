# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for ALBC environments.

Teacher policy observation design following the paper's framework:
    o_t (14D): Proprioceptive observation (measurable on real robot)
    p_t (23D): Privileged information (simulator-only DR parameters)

The encoder receives only p_t to compress physical unknowns into latent z.
The actor receives o_t + z. The critic receives o_t + p_t (asymmetric).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from isaaclab_tasks.models import HydrodynamicsModel

    from ..albc_env import ALBCEnv


def compute_policy_obs(
    env: ALBCEnv,
    robot: Articulation,
) -> torch.Tensor:
    """Compute policy observation o_t (14D).

    Measurable on the real robot via IMU and motor encoders:
        [0:3]   euler angles (roll, pitch, yaw)
        [3:6]   angular velocity in body frame (p, q, r)
        [6:8]   attitude error roll, pitch (command signal)
        [8:10]  joint positions (normalized to [-1, 1])
        [10:12] joint velocities
        [12:14] previous actions
    """
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    joint_pos = robot.data.joint_pos[:, env._albc_joint_ids]
    joint_pos_norm = 2.0 * (joint_pos - env._joint_limits_lower) / env._joint_limits_range - 1.0
    joint_vel = robot.data.joint_vel[:, env._albc_joint_ids]

    att_err = env._get_attitude_error()

    return torch.cat(
        [
            torch.stack([roll, pitch, yaw], dim=-1),  # 3D: euler
            robot.data.root_ang_vel_b,  # 3D: angular velocity
            att_err[:, :2],  # 2D: attitude error (roll, pitch only)
            joint_pos_norm,  # 2D: joint positions
            joint_vel,  # 2D: joint velocities
            env._prev_actions_obs,  # 2D: previous actions
        ],
        dim=-1,
    )


def _hydro_privileged(hydro: HydrodynamicsModel) -> torch.Tensor:
    """Pack hydrodynamic parameters: [volume, CoG_z, CoB_z] (3D)."""
    return torch.cat(
        [
            hydro.volume.unsqueeze(-1),
            hydro.center_of_gravity[:, 2:3],
            hydro.center_of_buoyancy[:, 2:3],
        ],
        dim=-1,
    )


def compute_privileged_obs(
    env: ALBCEnv,
) -> torch.Tensor:
    """Compute privileged information p_t (23D).

    Simulator-only DR parameters that the encoder must compress into latent z.
    Organized by physical category:

        Hydrodynamics (6D):
            [0:3]   main body: volume, CoG_z, CoB_z
            [3:6]   buoy body: volume, CoG_z, CoB_z
        Inertia (4D):
            [6:8]   main body Ixx, Iyy
            [8:10]  buoy body Ixx, Iyy
        Damping (4D):
            [10:12] linear damping roll, pitch
            [12:14] quadratic damping roll, pitch
        Body properties (2D):
            [14]    body mass
            [15]    added mass surge
        Payload (4D):
            [16]    payload mass
            [17:20] payload CoG offset (x, y, z)
        Actuator (2D):
            [20]    joint stiffness Kp
            [21]    joint damping Kd
        Environment (1D):
            [22]    water density
    """
    jid = env._albc_joint_ids[0]

    return torch.cat(
        [
            # Hydrodynamics (6D)
            _hydro_privileged(env._hydro),
            _hydro_privileged(env._buoy_hydro),
            # Inertia (4D)
            env._hydro.rigid_body_inertia[:, :2],
            env._buoy_hydro.rigid_body_inertia[:, :2],
            # Damping (4D)
            env._hydro.linear_damping[:, 3:5],
            env._hydro.quadratic_damping[:, 3:5],
            # Body properties (2D)
            env._hydro.body_mass.unsqueeze(-1),
            env._hydro.added_mass_matrix[:, 0, 0].unsqueeze(-1),
            # Payload (4D)
            env._payload_mass.unsqueeze(-1),
            env._payload_cog_offset,
            # Actuator (2D)
            env._robot.data.joint_stiffness[:, jid : jid + 1],
            env._robot.data.joint_damping[:, jid : jid + 1],
            # Environment (1D)
            env._hydro.water_density.unsqueeze(-1),
        ],
        dim=-1,
    )
