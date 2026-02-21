# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Hero Agent ALBC environments.

This module provides observation computation functions that can be called
from the HeroAgentEnv class. Separating observation logic enables:
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

    from ..base_env import HeroAgentEnv


def _hydro_privileged_info(hydro: HydrodynamicsModel) -> torch.Tensor:
    """Pack hydrodynamic parameters into a 7D privileged observation vector.

    Returns: (num_envs, 7) = [volume(1), r_cg(3), r_cb(3)].
    """
    return torch.cat(
        [
            hydro.volume.unsqueeze(-1),
            hydro.center_of_gravity,
            hydro.center_of_buoyancy,
        ],
        dim=-1,
    )


def compute_policy_obs(
    env: HeroAgentEnv,
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
        env: The Hero Agent environment instance.
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
    env: HeroAgentEnv,
) -> torch.Tensor:
    """Compute privileged observations for asymmetric training.

    Returns privileged info containing hydrostatic + dynamics parameters:
        - Main body (7D): volume, r_cg (3), r_cb (3)
        - Buoy body (7D): volume, r_cg (3), r_cb (3)
        - Main body dynamics (4D): inertia Ixx/Iyy/Izz (3), m_A surge (1)
        - Buoy dynamics (4D): inertia Ixx/Iyy/Izz (3), m_A surge (1)
        - Payload (4D, optional): mass, cog_offset (3)

    Total: 26D when payload is included, 22D otherwise.

    Args:
        env: The Hero Agent environment instance.

    Returns:
        Privileged observation tensor of shape (num_envs, state_space).
    """
    priv_obs = [
        _hydro_privileged_info(env._hydro),  # 7D: volume, CoG, CoB
        _hydro_privileged_info(env._buoy_hydro),  # 7D: volume, CoG, CoB
    ]

    # Dynamics parameters affected by DR (inertia_scale, added_mass_scale)
    # Each body is independently randomized by _randomize_hydro_model()
    priv_obs.append(env._hydro.rigid_body_inertia)  # 3D: main Ixx, Iyy, Izz
    priv_obs.append(env._hydro.added_mass_matrix[:, 0, 0].unsqueeze(-1))  # 1D: main m_A
    priv_obs.append(env._buoy_hydro.rigid_body_inertia)  # 3D: buoy Ixx, Iyy, Izz
    priv_obs.append(env._buoy_hydro.added_mass_matrix[:, 1, 1].unsqueeze(-1))  # 1D: buoy m_A (sway)

    # Include payload info if enabled and state_space is large enough
    if env._payload_mass is not None and env._payload_cog_offset is not None and env.cfg.state_space >= 26:
        payload_priv = torch.cat(
            [env._payload_mass.unsqueeze(-1), env._payload_cog_offset],
            dim=-1,
        )
        priv_obs.append(payload_priv)  # 4D: mass, cog_offset_xyz

    return torch.cat(priv_obs, dim=-1)
