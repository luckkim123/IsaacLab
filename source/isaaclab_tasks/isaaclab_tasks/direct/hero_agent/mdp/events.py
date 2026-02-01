# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for Hero Agent ALBC domain randomization.

These functions follow Isaac Lab's EventTerm pattern and can be used with
EventCfg for configuring domain randomization in Hero Agent ALBC environments.

The functions are designed to work with the Hero Agent environment's hydrodynamics
model and robot articulation. Hero Agent uses joint-based control (ALBC) without
thrusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab_tasks.models import HydrodynamicsModel

    from ..hero_agent_env import HeroAgentEnv
    from ..hero_agent_env_cfg import DomainRandomizationCfg


def _randomize_hydro_model(
    hydro: HydrodynamicsModel,
    env_ids: torch.Tensor,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Apply domain randomization to a hydrodynamics model.

    Args:
        hydro: The hydrodynamics model to randomize.
        env_ids: Environment indices to randomize.
        rand_cfg: Domain randomization configuration with scale ranges.
    """
    hydro.randomize_parameters(
        env_ids=env_ids,
        added_mass_scale=rand_cfg.added_mass_scale,
        linear_damping_scale=rand_cfg.linear_damping_scale,
        quadratic_damping_scale=rand_cfg.quadratic_damping_scale,
        volume_scale=rand_cfg.volume_scale,
        mass_scale=rand_cfg.mass_scale,
        cob_offset_scale=rand_cfg.cob_offset_scale,
        inertia_scale=rand_cfg.inertia_scale,
        payload_mass_ratio=rand_cfg.payload_mass_ratio,
        payload_cog_offset_z=rand_cfg.payload_cog_offset_z,
    )


def randomize_hydrodynamics(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize main body hydrodynamic parameters.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration with scale ranges.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_hydro"):
        return
    _randomize_hydro_model(env._hydro, env_ids, rand_cfg)


def randomize_buoy_hydrodynamics(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize buoy (link3) hydrodynamic parameters.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration with scale ranges.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_buoy_hydro"):
        return
    _randomize_hydro_model(env._buoy_hydro, env_ids, rand_cfg)


def randomize_ocean_current(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Randomize ocean current for specified environments.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_hydro"):
        return
    env._hydro.randomize_current(env_ids)


def randomize_robot_pose(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize robot initial pose for specified environments.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration with pose ranges.
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
        torch.rand(num_reset, device=device) * (rand_cfg.position_x_range[1] - rand_cfg.position_x_range[0])
        + rand_cfg.position_x_range[0]
    )
    pos_y = (
        torch.rand(num_reset, device=device) * (rand_cfg.position_y_range[1] - rand_cfg.position_y_range[0])
        + rand_cfg.position_y_range[0]
    )
    pos_z = (
        torch.rand(num_reset, device=device) * (rand_cfg.position_z_range[1] - rand_cfg.position_z_range[0])
        + rand_cfg.position_z_range[0]
    )

    default_root_state[:, 0] += pos_x
    default_root_state[:, 1] += pos_y
    # Z uses terrain origin as base, not default_root_state Z (which already includes origin)
    default_root_state[:, 2] = env._terrain.env_origins[env_ids, 2] + pos_z

    # Randomize orientation
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    roll = (
        torch.rand(num_reset, device=device) * (rand_cfg.roll_range[1] - rand_cfg.roll_range[0])
        + rand_cfg.roll_range[0]
    )
    pitch = (
        torch.rand(num_reset, device=device) * (rand_cfg.pitch_range[1] - rand_cfg.pitch_range[0])
        + rand_cfg.pitch_range[0]
    )
    yaw = (
        torch.rand(num_reset, device=device) * (rand_cfg.yaw_range[1] - rand_cfg.yaw_range[0])
        + rand_cfg.yaw_range[0]
    )

    random_quat = quat_from_euler_xyz(roll, pitch, yaw)
    default_root_state[:, 3:7] = quat_mul(default_root_state[:, 3:7], random_quat)

    # Apply to robot
    env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


def reset_robot_pose_default(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    initial_height: float = 4.5,
) -> None:
    """Reset robot to default pose (no randomization).

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to reset. If None, resets all.
        initial_height: Default height above terrain origin.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    default_root_state = env._robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env._terrain.env_origins[env_ids]
    default_root_state[:, 2] = env._terrain.env_origins[env_ids, 2] + initial_height

    env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


def randomize_joint_positions(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    joint_pos_range: tuple[float, float] = (-6.0, 6.0),
) -> None:
    """Randomize ALBC joint positions for specified environments.

    Randomizes joint positions, clamps to limits, and synchronizes
    the joint position target buffer.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        joint_pos_range: Range for joint position in radians.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_reset = len(env_ids)
    device = env.device

    # Get default joint state
    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    # Randomize ALBC joint positions
    random_pos = (
        torch.rand(num_reset, len(env._albc_joint_ids), device=device)
        * (joint_pos_range[1] - joint_pos_range[0])
        + joint_pos_range[0]
    )

    # Clamp to joint limits
    random_pos = torch.clamp(
        random_pos,
        env._joint_limits_lower,
        env._joint_limits_upper,
    )

    default_joint_pos[:, env._albc_joint_ids] = random_pos

    # Synchronize joint position target buffer
    env._joint_pos_targets[env_ids] = random_pos

    # Apply to robot
    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_joint_positions_default(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Reset joints to default positions (no randomization).

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to reset. If None, resets all.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    # Reset joint position targets to zero (default position for ALBC joints)
    env._joint_pos_targets[env_ids] = 0.0

    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
