# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for BlueROV domain randomization.

These functions follow Isaac Lab's EventTerm pattern and can be used with
EventCfg for configuring domain randomization in BlueROV environments.

The functions are designed to work with the BlueROV environment's hydrodynamics
model, thruster model, and robot articulation.

Example usage with EventCfg:
    @configclass
    class EventCfg:
        randomize_hydro = EventTerm(
            func=randomize_hydrodynamics,
            mode="reset",
            params={
                "added_mass_scale": (0.5, 1.0),
                "linear_damping_scale": (0.5, 1.0),
                "quadratic_damping_scale": (0.5, 1.0),
            },
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..bluerov_env import BlueROVEnv


def randomize_hydrodynamics(
    env: BlueROVEnv,
    env_ids: torch.Tensor | None,
    added_mass_scale: tuple[float, float] = (0.8, 1.2),
    linear_damping_scale: tuple[float, float] = (0.8, 1.2),
    quadratic_damping_scale: tuple[float, float] = (0.8, 1.2),
    volume_scale: tuple[float, float] = (0.9, 1.1),
    mass_scale: tuple[float, float] = (0.9, 1.1),
    cob_offset_scale: tuple[float, float] = (1.0, 1.0),
    inertia_scale: tuple[float, float] = (1.0, 1.0),
    payload_mass_ratio: tuple[float, float] = (0.0, 0.0),
    payload_cog_offset_z: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Randomize hydrodynamic parameters for specified environments.

    This function applies scale-based randomization to the hydrodynamics model
    following MarineGym's domain randomization pattern.

    Args:
        env: The BlueROV environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        added_mass_scale: Scale range for added mass coefficients.
        linear_damping_scale: Scale range for linear damping coefficients.
        quadratic_damping_scale: Scale range for quadratic damping coefficients.
        volume_scale: Scale range for vehicle volume (affects buoyancy).
        mass_scale: Scale range for robot mass.
        cob_offset_scale: Scale range for center of buoyancy offset.
        inertia_scale: Scale range for rigid body inertia.
        payload_mass_ratio: Range for additional payload mass as ratio of base mass.
        payload_cog_offset_z: Range for payload-induced CoG offset in Z direction.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "_hydro"):
        return

    env._hydro.randomize_parameters(
        env_ids=env_ids,
        added_mass_scale=added_mass_scale,
        linear_damping_scale=linear_damping_scale,
        quadratic_damping_scale=quadratic_damping_scale,
        volume_scale=volume_scale,
        mass_scale=mass_scale,
        cob_offset_scale=cob_offset_scale,
        inertia_scale=inertia_scale,
        payload_mass_ratio=payload_mass_ratio,
        payload_cog_offset_z=payload_cog_offset_z,
    )


def randomize_thruster_params(
    env: BlueROVEnv,
    env_ids: torch.Tensor | None,
    thrust_coeff_scale: tuple[float, float] = (0.8, 1.2),
    time_constant_scale: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize thruster parameters for specified environments.

    This function applies scale-based randomization to the thruster model
    to simulate manufacturing variations and wear.

    Args:
        env: The BlueROV environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        thrust_coeff_scale: Scale range for thrust coefficient.
        time_constant_scale: Scale range for thruster time constant.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "_thruster"):
        return

    env._thruster.randomize_parameters(
        env_ids=env_ids,
        thrust_coeff_scale=thrust_coeff_scale,
        time_constant_scale=time_constant_scale,
    )


def randomize_ocean_current(
    env: BlueROVEnv,
    env_ids: torch.Tensor | None,
    max_velocity: tuple[float, float, float] = (0.5, 0.5, 0.1),
) -> None:
    """Randomize ocean current for specified environments.

    This function sets random initial ocean current velocities within
    the specified bounds.

    Args:
        env: The BlueROV environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        max_velocity: Maximum current velocity in (x, y, z) directions.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "_hydro"):
        return

    env._hydro.randomize_current(env_ids)


def randomize_robot_pose(
    env: BlueROVEnv,
    env_ids: torch.Tensor | None,
    position_x_range: tuple[float, float] = (-2.5, 2.5),
    position_y_range: tuple[float, float] = (-2.5, 2.5),
    position_z_range: tuple[float, float] = (1.5, 2.5),
    roll_range: tuple[float, float] = (-0.628, 0.628),
    pitch_range: tuple[float, float] = (-0.628, 0.628),
    yaw_range: tuple[float, float] = (0.0, 6.283),
) -> None:
    """Randomize robot initial pose for specified environments.

    This function sets random initial positions and orientations for the
    underwater vehicle.

    Args:
        env: The BlueROV environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        position_x_range: Range for X position offset from origin.
        position_y_range: Range for Y position offset from origin.
        position_z_range: Range for Z position (depth).
        roll_range: Range for roll angle in radians.
        pitch_range: Range for pitch angle in radians.
        yaw_range: Range for yaw angle in radians.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_reset = len(env_ids)
    device = env.device

    # Get default state
    default_root_state = env._robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env._terrain.env_origins[env_ids]

    # Randomize position
    pos_x = (
        torch.rand(num_reset, device=device) * (position_x_range[1] - position_x_range[0])
        + position_x_range[0]
    )
    pos_y = (
        torch.rand(num_reset, device=device) * (position_y_range[1] - position_y_range[0])
        + position_y_range[0]
    )
    pos_z = (
        torch.rand(num_reset, device=device) * (position_z_range[1] - position_z_range[0])
        + position_z_range[0]
    )

    default_root_state[:, 0] += pos_x
    default_root_state[:, 1] += pos_y
    default_root_state[:, 2] = pos_z

    # Randomize orientation
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    roll = (
        torch.rand(num_reset, device=device) * (roll_range[1] - roll_range[0])
        + roll_range[0]
    )
    pitch = (
        torch.rand(num_reset, device=device) * (pitch_range[1] - pitch_range[0])
        + pitch_range[0]
    )
    yaw = (
        torch.rand(num_reset, device=device) * (yaw_range[1] - yaw_range[0])
        + yaw_range[0]
    )

    random_quat = quat_from_euler_xyz(roll, pitch, yaw)
    default_root_state[:, 3:7] = quat_mul(default_root_state[:, 3:7], random_quat)

    # Apply to robot
    env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
