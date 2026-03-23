# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for ALBC domain randomization.

These functions follow Isaac Lab's EventTerm pattern and can be used with
EventCfg for configuring domain randomization in ALBC environments.

The functions are designed to work with the ALBC environment's hydrodynamics
model and robot articulation. ALBC uses joint-based control without
thrusters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
)

if TYPE_CHECKING:
    from isaaclab_tasks.models import HydrodynamicsModel

    from ..albc_env import ALBCEnv
    from ..config import DomainRandomizationCfg


# -----------------------------------------------------------------------------
# Helper Functions (Private)
# -----------------------------------------------------------------------------


def _ensure_env_ids(env: ALBCEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Ensure env_ids is a valid tensor, defaulting to all environments if None."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _rand_uniform_range(
    shape: tuple | int,
    range_tuple: tuple[float, float],
    device: str | torch.device,
) -> torch.Tensor:
    """Generate uniform random values in [low, high] from a (low, high) tuple."""
    if isinstance(shape, int):
        shape = (shape,)
    lo, hi = range_tuple
    return torch.rand(shape, device=device) * (hi - lo) + lo


# -----------------------------------------------------------------------------
# DRSampler: bundles DR config for domain randomization
# -----------------------------------------------------------------------------


class DRSampler:
    """Bundles DR config for domain randomization.

    Encapsulates the ``rand_cfg + num_envs + device`` parameter pattern
    that every DR function requires. Provides ``get()`` for scalar/vector
    sampling and ``sample_xyz_offset()`` for XYZ offset randomization.
    """

    def __init__(
        self,
        cfg: DomainRandomizationCfg,
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    def get(self, range_tuple: tuple[float, float], shape: tuple | int | None = None) -> torch.Tensor:
        """Sample uniform random values for DR parameter.

        Args:
            range_tuple: (low, high) for uniform sampling.
            shape: Output tensor shape. Defaults to num_envs.

        Returns:
            Tensor of random values.
        """
        if shape is None:
            shape = self.num_envs
        return _rand_uniform_range(shape, range_tuple, self.device)

    def sample_xyz_offset(
        self,
        target: torch.Tensor,
        env_ids: torch.Tensor,
        base: torch.Tensor,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> None:
        """Apply uniform random XYZ offsets to target tensor."""
        num = len(env_ids)
        target[env_ids, 0] = base[0] + _rand_uniform_range(num, x_range, self.device)
        target[env_ids, 1] = base[1] + _rand_uniform_range(num, y_range, self.device)
        target[env_ids, 2] = base[2] + _rand_uniform_range(num, z_range, self.device)


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
        if hydro.cfg.rigid_body_inertia is not None:
            self.inertia = torch.tensor(hydro.cfg.rigid_body_inertia, **kw)
        else:
            logger.warning(
                "rigid_body_inertia not set for %s; falling back to 0.5 * added_mass[3:6]. "
                "This heuristic may not match the actual rigid body inertia.",
                type(hydro.cfg).__name__,
            )
            self.inertia = torch.tensor(hydro.cfg.added_mass[3:6], **kw) * 0.5
        self.water_density: float = hydro.cfg.water_density


def _get_hydro_base(hydro: HydrodynamicsModel) -> _HydroBaseCache:
    """Get or create cached base tensors for a hydrodynamics model."""
    if not hasattr(hydro, "_dr_base_cache"):
        hydro._dr_base_cache = _HydroBaseCache(hydro)  # type: ignore[attr-defined]
    return hydro._dr_base_cache  # type: ignore[attr-defined]


def _randomize_hydro_model(
    hydro: HydrodynamicsModel,
    env_ids: torch.Tensor,
    dr: DRSampler,
) -> None:
    """Apply domain randomization to a hydrodynamics model.

    Args:
        hydro: The hydrodynamics model to randomize.
        env_ids: Environment indices to randomize.
        dr: DRSampler with config.
    """
    n = dr.num_envs
    cfg = dr.cfg
    base = _get_hydro_base(hydro)

    # Added mass (scale each of 6 DOF)
    am_scales = dr.get(cfg.added_mass_scale, shape=(n, 6))
    hydro.added_mass_matrix[env_ids] = torch.diag_embed(base.added_mass.unsqueeze(0) * am_scales)

    # Linear damping
    ld_scales = dr.get(cfg.linear_damping_scale, shape=(n, 6))
    hydro.linear_damping[env_ids] = base.linear_damping.unsqueeze(0) * ld_scales

    # Quadratic damping
    qd_scales = dr.get(cfg.quadratic_damping_scale, shape=(n, 6))
    hydro.quadratic_damping[env_ids] = base.quadratic_damping.unsqueeze(0) * qd_scales

    # Yaw-specific quadratic damping override (index 5 = yaw)
    if hasattr(cfg, "yaw_damping_scale"):
        yaw_scales = dr.get(cfg.yaw_damping_scale)
        hydro.quadratic_damping[env_ids, 5] = base.quadratic_damping[5] * yaw_scales

    # Volume
    vol_scales = dr.get(cfg.volume_scale)
    hydro.volume[env_ids] = base.volume * vol_scales

    # Water density
    hydro.water_density[env_ids] = dr.get(cfg.water_density_range)

    hydro.update_buoyancy_force(env_ids)

    # Center of Buoyancy (offset from base)
    dr.sample_xyz_offset(
        hydro.center_of_buoyancy,
        env_ids,
        base.cob,
        cfg.cob_offset_x,
        cfg.cob_offset_y,
        cfg.cob_offset_z,
    )

    # Center of Gravity (offset from base)
    dr.sample_xyz_offset(
        hydro.center_of_gravity,
        env_ids,
        base.cog,
        cfg.cog_offset_x,
        cfg.cog_offset_y,
        cfg.cog_offset_z,
    )

    # Rigid body inertia
    inertia_scales = dr.get(cfg.inertia_scale, shape=(n, 3))
    hydro.rigid_body_inertia[env_ids] = base.inertia.unsqueeze(0) * inertia_scales

    # Enforce added mass stability constraint after DR.
    # Per-axis clamp: M_a[i] < threshold * I_rigid[i] to prevent exponential instability
    # in explicit integration (forward Euler). MarineGym equivalent: _enforce_stability_constraint().
    if hydro.cfg.apply_added_mass_force and hydro.body_mass is not None:
        threshold = 0.95
        am_diag = torch.diagonal(hydro.added_mass_matrix[env_ids], dim1=-2, dim2=-1)  # (N, 6)
        body_mass = hydro.body_mass[env_ids]  # (N,)
        rot_inertia = hydro.rigid_body_inertia[env_ids]  # (N, 3)
        gen_inertia = torch.cat([body_mass.unsqueeze(-1).expand(-1, 3), rot_inertia], dim=-1)  # (N, 6)
        max_am = threshold * gen_inertia
        exceeded = am_diag > max_am
        if exceeded.any():
            clamped = torch.where(exceeded, max_am, am_diag)
            hydro.added_mass_matrix[env_ids] = torch.diag_embed(clamped)


def randomize_hydrodynamics(
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    dr: DRSampler,
) -> None:
    """Randomize hydrodynamic parameters for main body and buoy.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        dr: DRSampler with config.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    # Main body hydrodynamics
    if hasattr(env, "_hydro"):
        _randomize_hydro_model(env._hydro, env_ids, dr)

    # Buoy (link3) hydrodynamics
    if hasattr(env, "_buoy_hydro"):
        _randomize_hydro_model(env._buoy_hydro, env_ids, dr)


def randomize_ocean_current(
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Randomize ocean current for specified environments.

    Sets the same ocean current for both main body and buoy, since both
    bodies are in the same water volume.

    Args:
        env: The ALBC environment instance.
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
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    rand_cfg: DomainRandomizationCfg,
) -> None:
    """Randomize robot initial pose for specified environments.

    Args:
        env: The ALBC environment instance.
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
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    initial_height: float = 4.5,
) -> None:
    """Reset robot to default pose (no randomization).

    Args:
        env: The ALBC environment instance.
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
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    joint_pos_range: tuple[float, float] = (-6.0, 6.0),
) -> None:
    """Randomize ALBC joint positions for specified environments.

    Randomizes joint positions, clamps to limits, and synchronizes
    the joint position target buffer.

    Args:
        env: The ALBC environment instance.
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
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Reset joints to default positions (no randomization).

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to reset. If None, resets all.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    # Reset joint position targets to zero (default position for ALBC joints)
    env._joint_pos_targets[env_ids] = 0.0

    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def compute_equilibrium_joint_positions(
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    noise_range: tuple[float, float] = (-0.3, 0.3),
) -> None:
    """Compute joint positions from gravity-buoyancy equilibrium at current attitude.

    Zero-torque equilibrium: tau = Lambda @ p_EE + T_b = 0, which gives:
        x_eq = -h * tan(pitch) / cos(roll)
        y_eq =  h * tan(roll)
    Then 2-link analytical IK computes (g1, g2) from (x_eq, y_eq).

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to reset. If None, resets all.
        noise_range: Uniform noise range (rad) added around equilibrium.
    """
    env_ids = _ensure_env_ids(env, env_ids)
    num_reset = len(env_ids)
    device = env.device

    # Get current roll/pitch from already-set robot pose
    root_quat = env._robot.data.root_quat_w[env_ids]
    roll, pitch, _ = euler_xyz_from_quat(root_quat)

    # Equilibrium EE position (h = buoy pivot height above CoG)
    h = 0.180  # meters, same as TDCControllerCfg.h
    x_eq = -h * torch.tan(pitch) / torch.cos(roll)
    y_eq = h * torch.tan(roll)

    # 2-link analytical IK
    l1 = HERO_AGENT_ALBC_LINK1_LENGTH  # 0.233m
    l2 = HERO_AGENT_ALBC_LINK2_LENGTH  # 0.233m
    r_sq = x_eq**2 + y_eq**2
    cos_g2 = (r_sq - l1**2 - l2**2) / (2.0 * l1 * l2)
    cos_g2 = torch.clamp(cos_g2, -1.0, 1.0)
    g2 = torch.acos(cos_g2)
    g1 = torch.atan2(y_eq, x_eq) - torch.atan2(l2 * torch.sin(g2), l1 + l2 * cos_g2)

    # Add noise
    noise = _rand_uniform_range((num_reset,), noise_range, device)
    g1 = g1 + noise
    noise = _rand_uniform_range((num_reset,), noise_range, device)
    g2 = g2 + noise

    # Clamp to joint limits
    g1 = torch.clamp(g1, env._joint_limits_lower[0], env._joint_limits_upper[0])
    g2 = torch.clamp(g2, env._joint_limits_lower[1], env._joint_limits_upper[1])

    # Build full joint state
    default_joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)

    joint_pos = torch.stack([g1, g2], dim=-1)
    default_joint_pos[:, env._albc_joint_ids] = joint_pos
    env._joint_pos_targets[env_ids] = joint_pos

    env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Payload Randomization
# -----------------------------------------------------------------------------


def randomize_payload(
    env: ALBCEnv,
    env_ids: torch.Tensor | None,
    dr: DRSampler,
) -> None:
    """Randomize payload parameters (mass, attachment offset, CoG offset).

    Payload is applied to the gripper body (fixed to base). Offsets are in gripper frame.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        dr: DRSampler with config.
    """
    env_ids = _ensure_env_ids(env, env_ids)

    if env._payload_mass is None or env._payload_attachment_offset is None:
        return

    num_reset = len(env_ids)
    device = env.device
    cfg = dr.cfg

    # Randomize mass
    env._payload_mass[env_ids] = dr.get(cfg.payload_mass_range)

    # Reset attachment offset to fixed default (no randomization)
    base_offset = torch.tensor(env.cfg.payload_attachment_offset, device=device, dtype=torch.float32)
    env._payload_attachment_offset[env_ids] = base_offset.unsqueeze(0)

    # Randomize CoG offset (relative to attachment point)
    # XY: uniform in disk of radius payload_cog_offset_xy_radius
    # Z: uniform in payload_cog_offset_z range
    if env._payload_cog_offset is not None:
        r_max = cfg.payload_cog_offset_xy_radius
        if r_max > 0:
            angle = torch.rand(num_reset, device=device) * 2.0 * torch.pi
            radius = r_max * torch.sqrt(torch.rand(num_reset, device=device))
            env._payload_cog_offset[env_ids, 0] = radius * torch.cos(angle)
            env._payload_cog_offset[env_ids, 1] = radius * torch.sin(angle)
        else:
            env._payload_cog_offset[env_ids, 0] = 0.0
            env._payload_cog_offset[env_ids, 1] = 0.0
        z_lo, z_hi = cfg.payload_cog_offset_z
        env._payload_cog_offset[env_ids, 2] = _rand_uniform_range(num_reset, (z_lo, z_hi), device)

        # Clamp effective offset so max payload moment <= buoy restoring moment.
        # Constraint: m * g * |r_eff_xy| <= F_bu * h
        # => |r_eff_xy| <= F_bu * h / (m * g)
        # Only horizontal (xy) offset contributes to roll/pitch restoring moment.
        if hasattr(env, "_buoy_hydro"):
            F_bu = env._buoy_hydro.buoyancy_force[env_ids]  # (N,)
            h = cfg.buoy_moment_arm  # scalar
            mass = env._payload_mass[env_ids]  # (N,)
            g = 9.81

            # effective_offset = attachment_offset + cog_offset
            effective = env._payload_attachment_offset[env_ids] + env._payload_cog_offset[env_ids]  # (N, 3)
            current_norm = effective[:, :2].norm(dim=-1)  # (N,) horizontal only

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
    env: ALBCEnv,
    env_ids: torch.Tensor,
    dr: DRSampler,
) -> None:
    """Randomize ALBC joint actuator stiffness and damping with absolute values.

    Draws stiffness and damping from uniform distributions defined by
    DR config. The same gain value is applied to both ALBC joints per environment.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize.
        dr: DRSampler with config.
    """
    cfg = dr.cfg

    stiffness = dr.get(cfg.joint_stiffness_range)
    damping = dr.get(cfg.joint_damping_range)

    # unsqueeze for broadcasting: (num_reset,) -> (num_reset, 1) -> (num_reset, num_joints)
    env._robot.write_joint_stiffness_to_sim(stiffness.unsqueeze(-1), joint_ids=env._albc_joint_ids, env_ids=env_ids)
    env._robot.write_joint_damping_to_sim(damping.unsqueeze(-1), joint_ids=env._albc_joint_ids, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Joint Effort Limit Randomization
# -----------------------------------------------------------------------------


def randomize_joint_effort_limit(
    env: ALBCEnv,
    env_ids: torch.Tensor,
    dr: DRSampler,
) -> None:
    """Randomize ALBC joint effort limits (max torque).

    Draws a scale factor and applies it to the asset default effort limit.
    The same scale is applied to both ALBC joints per environment.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize.
        dr: DRSampler with config.
    """
    scale = dr.get(dr.cfg.joint_effort_limit_range)

    # Cache default effort limit on first call (no default_joint_effort_limit in Isaac Lab API)
    if not hasattr(env, "_default_effort_limit"):
        env._default_effort_limit = env._robot.data.joint_effort_limits[0, env._albc_joint_ids[0]].item()
    num_joints = len(env._albc_joint_ids)
    effort = (env._default_effort_limit * scale).unsqueeze(-1).expand(-1, num_joints)

    env._robot.write_joint_effort_limit_to_sim(effort, joint_ids=env._albc_joint_ids, env_ids=env_ids)

    # Sync actuator internal buffers (Isaac Lab #128 workaround).
    # write_joint_effort_limit_to_sim updates _data.joint_effort_limits + PhysX,
    # but actuator.effort_limit / effort_limit_sim remain at init-time values.
    albc_ids_t = torch.tensor(env._albc_joint_ids, device=env.device)
    for actuator in env._robot.actuators.values():
        jids = actuator.joint_indices
        if isinstance(jids, torch.Tensor):
            mask = torch.isin(jids, albc_ids_t)
            if mask.any():
                local_ids = mask.nonzero(as_tuple=True)[0]
                actuator.effort_limit[env_ids[:, None], local_ids] = effort[..., : local_ids.shape[0]]
                actuator.effort_limit_sim[env_ids[:, None], local_ids] = effort[..., : local_ids.shape[0]]
        elif jids == slice(None):
            actuator.effort_limit[env_ids[:, None], albc_ids_t] = effort
            actuator.effort_limit_sim[env_ids[:, None], albc_ids_t] = effort


# -----------------------------------------------------------------------------
# Body Mass Randomization
# -----------------------------------------------------------------------------


def randomize_body_mass(
    env: ALBCEnv,
    env_ids: torch.Tensor,
    dr: DRSampler,
) -> None:
    """Randomize rigid body masses for specified environments.

    Applies a single scale factor to all bodies per environment (manufacturing
    tolerance model). Inertia is separately randomized via ``inertia_scale``
    in the hydrodynamics DR.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize.
        dr: DRSampler with config.
    """
    env_ids_cpu = env_ids.cpu()

    # Read current masses and reset to defaults
    masses = env._robot.root_physx_view.get_masses()
    masses[env_ids_cpu] = env._robot.data.default_mass[env_ids_cpu].clone()

    # Single scale per env, broadcast to all bodies (always cpu for PhysX API)
    scales = dr.get(dr.cfg.body_mass_scale).cpu()
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
    env: ALBCEnv,
    env_ids: torch.Tensor,
    dr: DRSampler,
) -> None:
    """Randomize ALBC joint friction coefficients.

    Applies static (Coulomb) and viscous friction to the ALBC joints.
    The same friction value is applied to both joints per environment.
    Uses Isaac Sim 5.0+ friction model: static + viscous.

    Args:
        env: The ALBC environment instance.
        env_ids: Environment indices to randomize.
        dr: DRSampler with config.
    """
    cfg = dr.cfg

    static = dr.get(cfg.joint_static_friction_range)
    viscous = dr.get(cfg.joint_viscous_friction_range)

    # unsqueeze for broadcasting: (num_reset,) -> (num_reset, 1) -> (num_reset, num_joints)
    env._robot.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static.unsqueeze(-1),
        joint_viscous_friction_coeff=viscous.unsqueeze(-1),
        joint_ids=env._albc_joint_ids,
        env_ids=env_ids,
    )
