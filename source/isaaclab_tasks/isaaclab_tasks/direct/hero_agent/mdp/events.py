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

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab_tasks.models import HydrodynamicsModel

    from ..base_env import HeroAgentEnv
    from ..config import DomainRandomizationCfg


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
) -> None:
    """Apply random XYZ offsets to a target tensor.

    Each axis is set to ``base[i] + uniform(range[i])``.

    Args:
        target: Target tensor to modify (shape: [num_envs, 3]).
        env_ids: Environment indices to modify.
        base: Base XYZ values to offset from (shape: [3]).
        x_range: Random offset range for X.
        y_range: Random offset range for Y.
        z_range: Random offset range for Z.
        device: Torch device.
    """
    num_envs = len(env_ids)
    target[env_ids, 0] = base[0] + _rand_uniform_range(num_envs, x_range, device)
    target[env_ids, 1] = base[1] + _rand_uniform_range(num_envs, y_range, device)
    target[env_ids, 2] = base[2] + _rand_uniform_range(num_envs, z_range, device)


# -----------------------------------------------------------------------------
# Hydrodynamics Randomization
# -----------------------------------------------------------------------------


class _HydroBaseCache:
    """Cached base tensors from a HydrodynamicsModel config for DR scaling.

    Avoids recreating identical tensors from config lists on every reset call.
    """

    __slots__ = (
        "added_mass",
        "linear_damping",
        "quadratic_damping",
        "volume",
        "cob",
        "cog",
        "inertia",
        "water_density",
    )

    def __init__(self, hydro: HydrodynamicsModel) -> None:
        kw = {"dtype": torch.float32, "device": hydro.device}
        self.added_mass = torch.tensor(hydro.cfg.added_mass, **kw)
        self.linear_damping = torch.tensor(hydro.cfg.linear_damping, **kw)
        self.quadratic_damping = torch.tensor(hydro.cfg.quadratic_damping, **kw)
        self.volume: float = hydro.cfg.volume if hydro.cfg.volume is not None else hydro.volume[0].item()
        self.cob = torch.tensor(hydro.cfg.center_of_buoyancy, **kw)
        self.cog = torch.tensor(hydro.cfg.center_of_gravity, **kw)
        self.inertia = (
            torch.tensor(hydro.cfg.rigid_body_inertia, **kw)
            if hydro.cfg.rigid_body_inertia is not None
            else torch.tensor(hydro.cfg.added_mass[3:6], **kw) * 0.5
        )
        self.water_density: float = hydro.cfg.water_density


def _get_hydro_base(hydro: HydrodynamicsModel) -> _HydroBaseCache:
    """Get or create cached base tensors for a hydrodynamics model."""
    if not hasattr(hydro, "_dr_base_cache"):
        hydro._dr_base_cache = _HydroBaseCache(hydro)  # type: ignore[attr-defined]
    return hydro._dr_base_cache  # type: ignore[attr-defined]


def _randomize_hydro_model(
    hydro: HydrodynamicsModel,
    env_ids: torch.Tensor,
    rand_cfg: DomainRandomizationCfg,
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Apply domain randomization to a hydrodynamics model.

    Args:
        hydro: The hydrodynamics model to randomize.
        env_ids: Environment indices to randomize.
        rand_cfg: Domain randomization configuration with scale ranges.
        sampled: Optional DORAEMON-sampled values. When provided, uses these
            instead of uniform random for the corresponding parameters.
    """
    num_envs = len(env_ids)
    device = hydro.device
    base = _get_hydro_base(hydro)

    # Added mass (scale each of 6 DOF)
    if sampled and "added_mass_scale" in sampled:
        am_scales = sampled["added_mass_scale"].unsqueeze(-1).expand(-1, 6)
    else:
        am_scales = _rand_uniform_range((num_envs, 6), rand_cfg.added_mass_scale, device)
    hydro.added_mass_matrix[env_ids] = torch.diag_embed(base.added_mass.unsqueeze(0) * am_scales)

    # Linear damping
    if sampled and "linear_damping_scale" in sampled:
        ld_scales = sampled["linear_damping_scale"].unsqueeze(-1).expand(-1, 6)
    else:
        ld_scales = _rand_uniform_range((num_envs, 6), rand_cfg.linear_damping_scale, device)
    hydro.linear_damping[env_ids] = base.linear_damping.unsqueeze(0) * ld_scales

    # Quadratic damping
    if sampled and "quadratic_damping_scale" in sampled:
        qd_scales = sampled["quadratic_damping_scale"].unsqueeze(-1).expand(-1, 6)
    else:
        qd_scales = _rand_uniform_range((num_envs, 6), rand_cfg.quadratic_damping_scale, device)
    hydro.quadratic_damping[env_ids] = base.quadratic_damping.unsqueeze(0) * qd_scales

    # Volume
    if sampled and "volume_scale" in sampled:
        vol_scales = sampled["volume_scale"]
    else:
        vol_scales = _rand_uniform_range(num_envs, rand_cfg.volume_scale, device)
    hydro.volume[env_ids] = base.volume * vol_scales

    # Water density
    if sampled and "water_density" in sampled:
        hydro.water_density[env_ids] = sampled["water_density"]
    else:
        hydro.water_density[env_ids] = _rand_uniform_range(num_envs, rand_cfg.water_density_range, device)

    hydro.update_buoyancy_force(env_ids)

    # Center of Buoyancy (offset from base)
    # CoB XY always uniform (not DORAEMON-managed); Z can be overridden.
    if sampled and "cob_offset_z" in sampled:
        num = len(env_ids)
        hydro.center_of_buoyancy[env_ids, 0] = base.cob[0] + _rand_uniform_range(num, rand_cfg.cob_offset_x, device)
        hydro.center_of_buoyancy[env_ids, 1] = base.cob[1] + _rand_uniform_range(num, rand_cfg.cob_offset_y, device)
        hydro.center_of_buoyancy[env_ids, 2] = base.cob[2] + sampled["cob_offset_z"]
    else:
        _apply_xyz_offset(
            hydro.center_of_buoyancy,
            env_ids,
            base.cob,
            rand_cfg.cob_offset_x,
            rand_cfg.cob_offset_y,
            rand_cfg.cob_offset_z,
            device,
        )

    # Center of Gravity (offset from base)
    if sampled and "cog_offset_z" in sampled:
        num = len(env_ids)
        hydro.center_of_gravity[env_ids, 0] = base.cog[0] + _rand_uniform_range(num, rand_cfg.cog_offset_x, device)
        hydro.center_of_gravity[env_ids, 1] = base.cog[1] + _rand_uniform_range(num, rand_cfg.cog_offset_y, device)
        hydro.center_of_gravity[env_ids, 2] = base.cog[2] + sampled["cog_offset_z"]
    else:
        _apply_xyz_offset(
            hydro.center_of_gravity,
            env_ids,
            base.cog,
            rand_cfg.cog_offset_x,
            rand_cfg.cog_offset_y,
            rand_cfg.cog_offset_z,
            device,
        )

    # Rigid body inertia
    if sampled and "inertia_scale" in sampled:
        inertia_scales = sampled["inertia_scale"].unsqueeze(-1).expand(-1, 3)
    else:
        inertia_scales = _rand_uniform_range((num_envs, 3), rand_cfg.inertia_scale, device)
    hydro.rigid_body_inertia[env_ids] = base.inertia.unsqueeze(0) * inertia_scales


def randomize_hydrodynamics(
    env: HeroAgentEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Randomize hydrodynamic parameters for main body and buoy.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration with scale ranges.
        sampled: Optional DORAEMON-sampled values.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    # Main body hydrodynamics
    if hasattr(env, "_hydro"):
        _randomize_hydro_model(env._hydro, env_ids, rand_cfg, sampled)

    # Buoy (link3) hydrodynamics
    if hasattr(env, "_buoy_hydro"):
        _randomize_hydro_model(env._buoy_hydro, env_ids, rand_cfg, sampled)


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
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Randomize payload parameters (mass, attachment offset, CoG offset).

    Payload is applied to the gripper body (fixed to base). Offsets are in gripper frame.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        rand_cfg: Domain randomization configuration.
        sampled: Optional DORAEMON-sampled values.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    if env._payload_mass is None or env._payload_attachment_offset is None:
        return

    num_reset = len(env_ids)
    device = env.device

    # Randomize mass
    if sampled and "payload_mass" in sampled:
        env._payload_mass[env_ids] = sampled["payload_mass"]
    else:
        env._payload_mass[env_ids] = _rand_uniform_range(num_reset, rand_cfg.payload_mass_range, device)

    # Reset attachment offset to fixed default (no randomization)
    base_offset = torch.tensor(env.cfg.payload_attachment_offset, device=device, dtype=torch.float32)
    env._payload_attachment_offset[env_ids] = base_offset.unsqueeze(0)

    # Randomize CoG offset (relative to attachment point)
    # XY: uniform in disk of radius payload_cog_offset_xy_radius
    # Z: uniform in payload_cog_offset_z range
    if env._payload_cog_offset is not None:
        r_max = rand_cfg.payload_cog_offset_xy_radius
        if r_max > 0:
            angle = torch.rand(num_reset, device=device) * 2.0 * torch.pi
            radius = r_max * torch.sqrt(torch.rand(num_reset, device=device))
            env._payload_cog_offset[env_ids, 0] = radius * torch.cos(angle)
            env._payload_cog_offset[env_ids, 1] = radius * torch.sin(angle)
        else:
            env._payload_cog_offset[env_ids, 0] = 0.0
            env._payload_cog_offset[env_ids, 1] = 0.0
        z_lo, z_hi = rand_cfg.payload_cog_offset_z
        env._payload_cog_offset[env_ids, 2] = _rand_uniform_range(num_reset, (z_lo, z_hi), device)

        # Clamp effective offset so max payload moment <= buoy restoring moment.
        # Constraint: m * g * |r_eff| <= F_bu * h
        # => |r_eff| <= F_bu * h / (m * g)
        if hasattr(env, "_buoy_hydro"):
            F_bu = env._buoy_hydro.buoyancy_force[env_ids]  # (N,)
            h = rand_cfg.buoy_moment_arm  # scalar
            mass = env._payload_mass[env_ids]  # (N,)
            g = 9.81

            # effective_offset = attachment_offset + cog_offset
            effective = env._payload_attachment_offset[env_ids] + env._payload_cog_offset[env_ids]  # (N, 3)
            current_norm = effective.norm(dim=-1)  # (N,)

            # max offset magnitude per-env (inf when mass=0)
            max_norm = torch.where(
                mass > 1e-6,
                (F_bu * h) / (mass * g),
                torch.full_like(mass, float("inf")),
            )

            # Scale down effective_offset if exceeding max, keep direction
            scale = torch.clamp(max_norm / current_norm.clamp(min=1e-8), max=1.0)  # (N,)
            clamped_effective = effective * scale.unsqueeze(-1)  # (N, 3)

            # Write back cog_offset = clamped_effective - attachment_offset
            env._payload_cog_offset[env_ids] = clamped_effective - env._payload_attachment_offset[env_ids]


# -----------------------------------------------------------------------------
# Joint Actuator Gain Randomization
# -----------------------------------------------------------------------------


def randomize_joint_gains(
    env: HeroAgentEnv,
    env_ids: torch.Tensor,
    rand_cfg: DomainRandomizationCfg,
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Randomize ALBC joint actuator stiffness and damping with absolute values.

    Draws stiffness and damping from uniform distributions defined by
    ``rand_cfg.joint_stiffness_range`` and ``rand_cfg.joint_damping_range``.
    The same gain value is applied to both ALBC joints per environment.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize.
        rand_cfg: Domain randomization configuration with gain ranges.
        sampled: Optional DORAEMON-sampled values.
    """
    num_reset = len(env_ids)
    device = env.device

    if sampled and "joint_stiffness" in sampled:
        stiffness = sampled["joint_stiffness"]
    else:
        stiffness = _rand_uniform_range(num_reset, rand_cfg.joint_stiffness_range, device)

    if sampled and "joint_damping" in sampled:
        damping = sampled["joint_damping"]
    else:
        damping = _rand_uniform_range(num_reset, rand_cfg.joint_damping_range, device)

    # unsqueeze for broadcasting: (num_reset,) -> (num_reset, 1) -> (num_reset, num_joints)
    env._robot.write_joint_stiffness_to_sim(stiffness.unsqueeze(-1), joint_ids=env._albc_joint_ids, env_ids=env_ids)
    env._robot.write_joint_damping_to_sim(damping.unsqueeze(-1), joint_ids=env._albc_joint_ids, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Body Mass Randomization
# -----------------------------------------------------------------------------


def randomize_body_mass(
    env: HeroAgentEnv,
    env_ids: torch.Tensor,
    rand_cfg: DomainRandomizationCfg,
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Randomize rigid body masses for specified environments.

    Applies a single scale factor to all bodies per environment (manufacturing
    tolerance model). Inertia is separately randomized via ``inertia_scale``
    in the hydrodynamics DR.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize.
        rand_cfg: Domain randomization configuration with mass scale range.
        sampled: Optional DORAEMON-sampled values.
    """
    num_reset = len(env_ids)
    env_ids_cpu = env_ids.cpu()

    # Read current masses and reset to defaults
    masses = env._robot.root_physx_view.get_masses()
    masses[env_ids_cpu] = env._robot.data.default_mass[env_ids_cpu].clone()

    # Single scale per env, broadcast to all bodies
    if sampled and "body_mass_scale" in sampled:
        scales = sampled["body_mass_scale"].cpu()
    else:
        scales = _rand_uniform_range(num_reset, rand_cfg.body_mass_scale, "cpu")
    masses[env_ids_cpu] *= scales.unsqueeze(-1)
    masses = torch.clamp(masses, min=1e-6)

    env._robot.root_physx_view.set_masses(masses, env_ids_cpu)

    # Sync hydrodynamics model body_mass tensors with PhysX (for privileged obs)
    body_idx = env._body_id[0]
    buoy_idx = env._buoy_body_id[0]
    device = env.device
    if env._hydro.body_mass is not None:
        env._hydro.body_mass[env_ids] = masses[env_ids_cpu, body_idx].to(device)
    if env._buoy_hydro.body_mass is not None:
        env._buoy_hydro.body_mass[env_ids] = masses[env_ids_cpu, buoy_idx].to(device)


# -----------------------------------------------------------------------------
# Joint Friction Randomization
# -----------------------------------------------------------------------------


def randomize_joint_friction(
    env: HeroAgentEnv,
    env_ids: torch.Tensor,
    rand_cfg: DomainRandomizationCfg,
    sampled: dict[str, torch.Tensor] | None = None,
) -> None:
    """Randomize ALBC joint friction coefficients.

    Applies static (Coulomb) and viscous friction to the ALBC joints.
    The same friction value is applied to both joints per environment.
    Uses Isaac Sim 5.0+ friction model: static + viscous.

    Args:
        env: The Hero Agent environment instance.
        env_ids: Environment indices to randomize.
        rand_cfg: Domain randomization configuration with friction ranges.
        sampled: Optional DORAEMON-sampled values.
    """
    num_reset = len(env_ids)
    device = env.device

    if sampled and "joint_static_friction" in sampled:
        static = sampled["joint_static_friction"]
    else:
        static = _rand_uniform_range(num_reset, rand_cfg.joint_static_friction_range, device)

    if sampled and "joint_viscous_friction" in sampled:
        viscous = sampled["joint_viscous_friction"]
    else:
        viscous = _rand_uniform_range(num_reset, rand_cfg.joint_viscous_friction_range, device)

    # unsqueeze for broadcasting: (num_reset,) -> (num_reset, 1) -> (num_reset, num_joints)
    env._robot.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static.unsqueeze(-1),
        joint_viscous_friction_coeff=viscous.unsqueeze(-1),
        joint_ids=env._albc_joint_ids,
        env_ids=env_ids,
    )
