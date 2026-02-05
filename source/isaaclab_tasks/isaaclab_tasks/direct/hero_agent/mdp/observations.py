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

    from ..hero_agent_env import HeroAgentEnv


def compute_policy_obs(
    env: HeroAgentEnv,
    robot: Articulation,
) -> torch.Tensor:
    """Compute policy observations for ALBC control.

    Returns 13-dim observation (base env) or 15-dim (TDC env):
        [0:3]   roll, pitch, yaw (euler angles)
        [3:6]   angular velocity in body frame
        [6:9]   attitude errors from task
        [9:11]  joint positions (normalized to [-1, 1])
        [11:]   previous actions (2D for base env, 4D for TDC env)

    The output dimension depends on ``env._prev_actions_obs`` shape:
    base env stores 2D (joint velocities), TDC env stores 4D (PD gains).

    Args:
        env: The Hero Agent environment instance.
        robot: The robot articulation.

    Returns:
        Policy observation tensor of shape (num_envs, 13) or (num_envs, 15).
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

    Returns compact privileged info containing hydrostatic parameters:
        - Main body (10D): volume, r_cg (3), r_cb (3), inertia (3)
        - Buoy body (10D): volume, r_cg (3), r_cb (3), inertia (3)
        - Payload (2D, optional): mass, z-offset

    Total: 22D when payload is included, 20D otherwise.

    Args:
        env: The Hero Agent environment instance.

    Returns:
        Privileged observation tensor of shape (num_envs, state_space).
    """
    # Main body privileged info (10D)
    main_priv = torch.cat(
        [
            env._hydro.volume.unsqueeze(-1),
            env._hydro.center_of_gravity,
            env._hydro.center_of_buoyancy,
            env._hydro.rigid_body_inertia,
        ],
        dim=-1,
    )

    # Buoy privileged info (10D)
    buoy_priv = torch.cat(
        [
            env._buoy_hydro.volume.unsqueeze(-1),
            env._buoy_hydro.center_of_gravity,
            env._buoy_hydro.center_of_buoyancy,
            env._buoy_hydro.rigid_body_inertia,
        ],
        dim=-1,
    )

    priv_obs = [main_priv, buoy_priv]

    # Include payload info if enabled and state_space is large enough
    if env._payload_mass is not None and env.cfg.state_space >= 22:
        payload_priv = torch.stack(
            [env._payload_mass, env._payload_attachment_offset[:, 2]],
            dim=-1,
        )
        priv_obs.append(payload_priv)

    return torch.cat(priv_obs, dim=-1)
