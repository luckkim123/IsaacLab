# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for ALBC environments.

This module provides observation computation functions that can be called
from the ALBCEnv class. Separating observation logic enables:
- Cleaner environment code
- Easier testing of observation components
- Consistent structure with Isaac Lab mdp conventions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from isaaclab_tasks.models import HydrodynamicsModel

    from ..albc_env import ALBCEnv


def _hydro_privileged_info(hydro: HydrodynamicsModel) -> torch.Tensor:
    """Pack hydrodynamic parameters into a 3D privileged observation vector.

    Only z-components of CoG/CoB: x/y offsets are +-0.01m (negligible torque
    contribution vs 6-10Nm effort limits). z-component dominates roll/pitch dynamics.

    Returns: (num_envs, 3) = [volume(1), r_cg_z(1), r_cb_z(1)].
    """
    return torch.cat(
        [
            hydro.volume.unsqueeze(-1),
            hydro.center_of_gravity[:, 2:3],  # z-component only
            hydro.center_of_buoyancy[:, 2:3],  # z-component only
        ],
        dim=-1,
    )


def compute_policy_obs(
    env: ALBCEnv,
    robot: Articulation,
) -> torch.Tensor:
    """Compute policy observations for ALBC control.

    Returns 13-dim observation:
        [0:3]   roll, pitch, yaw (euler angles)
        [3:6]   angular velocity in body frame
        [6:9]   attitude errors from task
        [9:11]  joint positions (normalized to [-1, 1])
        [11:13] previous actions (joint velocities)

    Args:
        env: The ALBC environment instance.
        robot: The robot articulation.

    Returns:
        Policy observation tensor of shape (num_envs, 13).
    """
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    # Normalize joint positions to [-1, 1]
    joint_pos_normalized = (
        2.0 * (robot.data.joint_pos[:, env._albc_joint_ids] - env._joint_limits_lower) / env._joint_limits_range - 1.0
    )

    return torch.cat(
        [
            torch.stack([roll, pitch, yaw], dim=-1),  # 3: euler angles
            robot.data.root_ang_vel_b,  # 3: angular velocity
            env._get_attitude_error(),  # 3: attitude errors
            joint_pos_normalized,  # 2: joint positions
            env._prev_actions_obs,  # 2: previous actions
        ],
        dim=-1,
    )


def compute_privileged_obs(
    env: ALBCEnv,
) -> torch.Tensor:
    """Compute privileged observations for asymmetric training.

    Returns privileged info containing hydrostatic + dynamics + actuator parameters:
        - Main body hydro (3D): volume, r_cg_z, r_cb_z
        - Buoy body hydro (3D): volume, r_cg_z, r_cb_z
        - Main body inertia (2D): Ixx, Iyy (roll/pitch only)
        - Buoy inertia (2D): Ixx, Iyy
        - Payload (4D): mass, cog_offset (3)
        - Main body added mass surge (1D)
        - Joint stiffness (1D): PD Kp (DR range 40-120, 3x)
        - Joint damping (1D): PD Kd (DR range 0.5-5.0, 10x)
        - Joint effort limit (1D): max torque (DR range 0.7-1.0 x 9.5Nm)

    Total: 18D.
    Changes from 19D: removed CoG x/y (+-0.01m, negligible), added joint actuator
    params (stiffness/damping/effort_limit) which have the largest DR ranges and
    directly determine control response but were previously invisible to the encoder.

    Args:
        env: The ALBC environment instance.

    Returns:
        Privileged observation tensor of shape (num_envs, state_space).
    """
    priv_obs = [
        _hydro_privileged_info(env._hydro),  # 3D: volume, CoG_z, CoB_z
        _hydro_privileged_info(env._buoy_hydro),  # 3D: volume, CoG_z, CoB_z
    ]

    # Inertia: Ixx/Iyy only (Izz irrelevant for roll/pitch attitude control).
    # Independently randomized per body by _randomize_hydro_model().
    priv_obs.append(env._hydro.rigid_body_inertia[:, :2])  # 2D: main Ixx, Iyy
    priv_obs.append(env._buoy_hydro.rigid_body_inertia[:, :2])  # 2D: buoy Ixx, Iyy

    # Payload: mass + CoG offset (4D)
    payload_priv = torch.cat(
        [env._payload_mass.unsqueeze(-1), env._payload_cog_offset],
        dim=-1,
    )
    priv_obs.append(payload_priv)  # 4D: mass, cog_offset_xyz

    # Main body surge added mass: effective inertia = I_rigid + M_added (1D).
    priv_obs.append(env._hydro.added_mass_matrix[:, 0, 0].unsqueeze(-1))  # 1D: main M_a surge

    # Joint actuator parameters: critical DR params previously missing from privileged obs.
    # Both ALBC joints share the same DR'd value (scalar per env, broadcast to 2 joints).
    # Read from Isaac Lab's ArticulationData (updated by write_joint_*_to_sim).
    jid = env._albc_joint_ids[0]
    priv_obs.append(env._robot.data.joint_stiffness[:, jid : jid + 1])  # 1D: Kp
    priv_obs.append(env._robot.data.joint_damping[:, jid : jid + 1])  # 1D: Kd
    priv_obs.append(env._robot.data.joint_effort_limits[:, jid : jid + 1])  # 1D: effort limit

    return torch.cat(priv_obs, dim=-1)
