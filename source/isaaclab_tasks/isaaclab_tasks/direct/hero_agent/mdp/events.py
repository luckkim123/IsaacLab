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


# -----------------------------------------------------------------------------
# Helper Functions (Private)
# -----------------------------------------------------------------------------


def _ensure_env_ids(env: HeroAgentEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Ensure env_ids is a valid tensor, defaulting to all environments if None.

    Args:
        env: The environment instance.
        env_ids: Environment indices or None.

    Returns:
        A tensor of environment indices.
    """
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _rand_uniform(
    shape: tuple | int,
    low: float,
    high: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Generate uniform random values in [low, high].

    Args:
        shape: Output tensor shape.
        low: Lower bound of the range.
        high: Upper bound of the range.
        device: Torch device.

    Returns:
        Tensor of random values uniformly distributed in [low, high].
    """
    if isinstance(shape, int):
        shape = (shape,)
    return torch.rand(shape, device=device) * (high - low) + low


def _rand_uniform_range(
    shape: tuple | int,
    range_tuple: tuple[float, float],
    device: str | torch.device,
) -> torch.Tensor:
    """Generate uniform random values from a (low, high) tuple.

    Args:
        shape: Output tensor shape.
        range_tuple: Tuple of (low, high) bounds.
        device: Torch device.

    Returns:
        Tensor of random values uniformly distributed in [low, high].
    """
    return _rand_uniform(shape, range_tuple[0], range_tuple[1], device)


def _apply_xyz_offset(
    target: torch.Tensor,
    env_ids: torch.Tensor,
    base: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    device: str | torch.device,
    z_absolute: bool = False,
) -> None:
    """Apply random XYZ offsets to a target tensor.

    Args:
        target: Target tensor to modify (shape: [num_envs, 3]).
        env_ids: Environment indices to modify.
        base: Base XYZ values to offset from (shape: [3]).
        x_range: Random offset range for X.
        y_range: Random offset range for Y.
        z_range: Random offset range for Z.
        device: Torch device.
        z_absolute: If True, Z uses absolute value from range instead of offset.
    """
    num_envs = len(env_ids)
    target[env_ids, 0] = base[0] + _rand_uniform_range(num_envs, x_range, device)
    target[env_ids, 1] = base[1] + _rand_uniform_range(num_envs, y_range, device)
    if z_absolute:
        target[env_ids, 2] = _rand_uniform_range(num_envs, z_range, device)
    else:
        target[env_ids, 2] = base[2] + _rand_uniform_range(num_envs, z_range, device)


# -----------------------------------------------------------------------------
# Hydrodynamics Randomization
# -----------------------------------------------------------------------------


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
    num_envs = len(env_ids)
    device = hydro.device

    # Added mass (scale each of 6 DOF)
    am_scales = _rand_uniform_range((num_envs, 6), rand_cfg.added_mass_scale, device)
    base_am = torch.tensor(hydro.cfg.added_mass, dtype=torch.float32, device=device)
    hydro.added_mass_matrix[env_ids] = torch.diag_embed(base_am.unsqueeze(0) * am_scales)

    # Linear damping
    ld_scales = _rand_uniform_range((num_envs, 6), rand_cfg.linear_damping_scale, device)
    base_ld = torch.tensor(hydro.cfg.linear_damping, dtype=torch.float32, device=device)
    hydro.linear_damping[env_ids] = base_ld.unsqueeze(0) * ld_scales

    # Quadratic damping
    qd_scales = _rand_uniform_range((num_envs, 6), rand_cfg.quadratic_damping_scale, device)
    base_qd = torch.tensor(hydro.cfg.quadratic_damping, dtype=torch.float32, device=device)
    hydro.quadratic_damping[env_ids] = base_qd.unsqueeze(0) * qd_scales

    # Volume: read base from config to avoid using already-randomized live tensor values
    if hydro.cfg.volume is not None:
        base_volume = hydro.cfg.volume
    else:
        base_volume = hydro.volume[0].item()
    hydro.volume[env_ids] = base_volume * _rand_uniform_range(num_envs, rand_cfg.volume_scale, device)
    hydro.update_buoyancy_force(env_ids)

    # Center of Buoyancy (offset from base)
    base_cob = torch.tensor(hydro.cfg.center_of_buoyancy, dtype=torch.float32, device=device)
    _apply_xyz_offset(
        hydro.center_of_buoyancy,
        env_ids,
        base_cob,
        rand_cfg.cob_offset_x,
        rand_cfg.cob_offset_y,
        rand_cfg.cob_offset_z,
        device,
    )

    # Center of Gravity (offset from base)
    base_cog = torch.tensor(hydro.cfg.center_of_gravity, dtype=torch.float32, device=device)
    _apply_xyz_offset(
        hydro.center_of_gravity,
        env_ids,
        base_cog,
        rand_cfg.cog_offset_x,
        rand_cfg.cog_offset_y,
        rand_cfg.cog_offset_z,
        device,
    )

    # Rigid body inertia
    inertia_scales = _rand_uniform_range((num_envs, 3), rand_cfg.inertia_scale, device)
    if hydro.cfg.rigid_body_inertia is not None:
        base_inertia = torch.tensor(hydro.cfg.rigid_body_inertia, dtype=torch.float32, device=device)
    else:
        base_inertia = torch.tensor(hydro.cfg.added_mass[3:6], dtype=torch.float32, device=device) * 0.5
    hydro.rigid_body_inertia[env_ids] = base_inertia.unsqueeze(0) * inertia_scales


def randomize_hydrodynamics(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize hydrodynamic parameters for main body and buoy.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration with scale ranges.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    # Main body hydrodynamics
    if hasattr(env, "_hydro"):
        _randomize_hydro_model(env._hydro, env_ids, rand_cfg)

    # Buoy (link3) hydrodynamics
    if hasattr(env, "_buoy_hydro"):
        _randomize_hydro_model(env._buoy_hydro, env_ids, rand_cfg)


def randomize_ocean_current(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Randomize ocean current for specified environments.

    Sets the same ocean current for both main body and buoy, since both
    bodies are in the same water volume.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
    """
    env_ids = _ensure_env_ids(env, env_ids)
    if not hasattr(env, "_hydro"):
        return
    env._hydro.set_ocean_current(env_ids)

    # Share the same current with buoy (same water volume)
    if hasattr(env, "_buoy_hydro"):
        env._buoy_hydro.set_ocean_current(env_ids, velocity=env._hydro._current_velocity[env_ids])


# -----------------------------------------------------------------------------
# Robot Pose Randomization
# -----------------------------------------------------------------------------


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
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    env_ids = _ensure_env_ids(env, env_ids)
    num_reset = len(env_ids)
    device = env.device

    # Get default state
    default_root_state = env._robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]

    # Randomize position
    pos_x = _rand_uniform_range(num_reset, rand_cfg.position_x_range, device)
    pos_y = _rand_uniform_range(num_reset, rand_cfg.position_y_range, device)
    pos_z = _rand_uniform_range(num_reset, rand_cfg.position_z_range, device)

    default_root_state[:, 0] += pos_x
    default_root_state[:, 1] += pos_y
    # Z uses terrain origin as base, not default_root_state Z (which already includes origin)
    default_root_state[:, 2] = env.scene.env_origins[env_ids, 2] + pos_z

    # Randomize orientation
    roll = _rand_uniform_range(num_reset, rand_cfg.roll_range, device)
    pitch = _rand_uniform_range(num_reset, rand_cfg.pitch_range, device)
    yaw = _rand_uniform_range(num_reset, rand_cfg.yaw_range, device)

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
    env_ids = _ensure_env_ids(env, env_ids)

    default_root_state = env._robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    default_root_state[:, 2] = env.scene.env_origins[env_ids, 2] + initial_height

    env._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


# -----------------------------------------------------------------------------
# Joint Randomization
# -----------------------------------------------------------------------------


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
    env_ids = _ensure_env_ids(env, env_ids)
    num_reset = len(env_ids)
    device = env.device

    # Get default joint state
    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    # Randomize ALBC joint positions
    random_pos = _rand_uniform_range((num_reset, len(env._albc_joint_ids)), joint_pos_range, device)

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
    env_ids = _ensure_env_ids(env, env_ids)

    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    # Reset joint position targets to zero (default position for ALBC joints)
    env._joint_pos_targets[env_ids] = 0.0

    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Payload Randomization
# -----------------------------------------------------------------------------


def randomize_payload(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize simple payload parameters (mass and attachment offset).

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    if env._payload_mass is None or env._payload_attachment_offset is None:
        return

    num_reset = len(env_ids)
    device = env.device

    # Randomize mass
    env._payload_mass[env_ids] = _rand_uniform_range(num_reset, rand_cfg.payload_mass_range, device)

    # Randomize attachment offset (x, y, z) - Z is absolute, not relative
    base_offset = torch.tensor(env.cfg.payload_attachment_offset, device=device, dtype=torch.float32)
    _apply_xyz_offset(
        env._payload_attachment_offset,
        env_ids,
        base_offset,
        rand_cfg.payload_attachment_x_range,
        rand_cfg.payload_attachment_y_range,
        rand_cfg.payload_attachment_z_range,
        device,
        z_absolute=True,
    )
