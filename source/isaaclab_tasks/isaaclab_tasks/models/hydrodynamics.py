# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fossen model hydrodynamics for underwater vehicles.

This module implements the 6-DOF hydrodynamic forces and torques based on
the Fossen model for marine craft dynamics.

Reference:
    Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

# Import configuration classes from isaaclab_assets
from isaaclab_assets.robots.uuv import HydrodynamicsCfg, OceanCurrentCfg

if TYPE_CHECKING:
    from collections.abc import Sequence

# Re-export for backward compatibility
__all__ = ["HydrodynamicsModel", "HydrodynamicsCfg", "OceanCurrentCfg"]


class HydrodynamicsModel:
    """Fossen model hydrodynamics calculator for underwater vehicles.

    This class computes hydrodynamic forces and torques acting on an underwater
    vehicle based on the Fossen model. The forces include:
        - Added mass effects (inertia from accelerating surrounding fluid)
        - Linear and quadratic damping (drag forces)
        - Coriolis and centripetal forces (coupling between linear/angular motion)
        - Buoyancy and restoring forces (orientation-dependent)

    The model assumes:
        - 6-DOF rigid body dynamics
        - Diagonal added mass and damping matrices (simplified)
        - Constant density fluid (incompressible)
        - No wave or surface effects

    Attributes:
        num_envs: Number of parallel environments.
        device: Computation device (cpu or cuda).
        cfg: Hydrodynamics configuration.
        current_cfg: Ocean current configuration.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        cfg: HydrodynamicsCfg,
        current_cfg: OceanCurrentCfg | None = None,
        dt: float = 0.01,
        robot_mass: float | None = None,
    ) -> None:
        """Initialize the hydrodynamics model.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Hydrodynamics configuration.
            current_cfg: Ocean current configuration. Defaults to no current.
            dt: Simulation timestep for acceleration calculation.
            robot_mass: Robot mass from physics engine. Used if cfg.vehicle_mass is None.
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.current_cfg = current_cfg or OceanCurrentCfg()
        self.dt = dt

        # Build hydrodynamic matrices (num_envs, 6, 6)
        self._added_mass_matrix = (
            torch.diag(torch.tensor(cfg.added_mass, dtype=torch.float32, device=device))
            .unsqueeze(0)
            .repeat(num_envs, 1, 1)
        )

        self._linear_damping_diag = (
            torch.tensor(cfg.linear_damping, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        )

        self._quadratic_damping_diag = (
            torch.tensor(cfg.quadratic_damping, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        )

        # Vehicle mass for weight-buoyancy difference
        if cfg.vehicle_mass is not None:
            self._vehicle_mass = torch.full((num_envs,), cfg.vehicle_mass, dtype=torch.float32, device=device)
        elif robot_mass is not None:
            self._vehicle_mass = torch.full((num_envs,), robot_mass, dtype=torch.float32, device=device)
        else:
            # Neutral buoyancy: mass = volume * water_density
            self._vehicle_mass = torch.full(
                (num_envs,), cfg.volume * cfg.water_density, dtype=torch.float32, device=device
            )

        # Buoyancy parameters
        self._volume = torch.full((num_envs,), cfg.volume, dtype=torch.float32, device=device)
        self._water_density = cfg.water_density
        self._gravity = 9.81

        # Weight = m * g
        self._weight = self._vehicle_mass * self._gravity

        # Buoyancy force magnitude = rho * V * g
        self._buoyancy_force_base = self._water_density * self._gravity * self._volume

        # Center of buoyancy and gravity in body frame
        self._r_cb = (
            torch.tensor(cfg.center_of_buoyancy, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        )
        self._r_cg = (
            torch.tensor(cfg.center_of_gravity, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        )

        # Keep legacy _cob_offset for compatibility
        self._cob_offset = self._r_cb[:, 2]

        # Rigid body inertia for full Coriolis matrix
        if cfg.rigid_body_inertia is not None:
            self._rigid_body_inertia = (
                torch.tensor(cfg.rigid_body_inertia, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
            )
        else:
            # Estimate from added mass rotational terms
            self._rigid_body_inertia = (
                torch.tensor(cfg.added_mass[3:6], dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
                * 0.5
            )

        self._use_full_coriolis = cfg.use_full_coriolis

        # State buffers for acceleration filtering
        self._prev_body_vel = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._prev_body_acc = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._alpha = cfg.acceleration_filter_alpha

        # Ocean current state (world frame, 6-DOF)
        self._current_velocity = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._max_current_vel = torch.tensor(self.current_cfg.max_velocity, dtype=torch.float32, device=device)
        self._current_noise_scale = torch.tensor(self.current_cfg.noise_scale, dtype=torch.float32, device=device)

    def compute_forces(
        self,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hydrodynamic forces and torques in body frame.

        Args:
            root_lin_vel_w: Root linear velocity in world frame. Shape: (num_envs, 3).
            root_ang_vel_w: Root angular velocity in world frame. Shape: (num_envs, 3).
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Tuple of (forces_b, torques_b) in body frame. Each has shape (num_envs, 3).
        """
        # Transform world velocities to body frame
        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_vel = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

        # Transform ocean current to body frame and compute relative velocity
        current_w = self._current_velocity
        current_lin_b = quat_apply_inverse(root_quat_w, current_w[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, current_w[:, 3:])
        current_b = torch.cat([current_lin_b, current_ang_b], dim=-1)

        # Relative velocity (vehicle velocity - current velocity)
        relative_vel = body_vel - current_b

        # Compute body acceleration (filtered)
        body_acc = self._compute_acceleration(relative_vel)

        # Compute individual hydrodynamic force components
        damping = self._compute_damping(relative_vel)

        # NOTE: We do NOT apply added mass (M_A * v_dot) as external force.
        # The physics engine already handles M_RB * v_dot internally.
        # Adding M_A * v_dot as external force causes double counting and instability.
        # The Coriolis matrix C(v) = C_RB(v) + C_A(v) includes the added mass coupling effects.

        if self._use_full_coriolis:
            coriolis = self._compute_coriolis_full(relative_vel)
        else:
            coriolis = self._compute_coriolis(relative_vel)

        buoyancy = self._compute_buoyancy_quat(root_quat_w)

        # Total hydrodynamic wrench (negative because these forces oppose motion)
        # Per Fossen formulation: τ_hydro = -C(v)*v - D(v)*v + g(η)
        hydro = -(coriolis + damping)

        # Split into forces and torques
        forces_b = hydro[:, :3] + buoyancy[:, :3]
        torques_b = hydro[:, 3:] + buoyancy[:, 3:]

        return forces_b, torques_b

    def _compute_acceleration(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute filtered body acceleration."""
        raw_acc = (body_vel - self._prev_body_vel) / self.dt
        filtered_acc = (1.0 - self._alpha) * self._prev_body_acc + self._alpha * raw_acc
        self._prev_body_vel = body_vel.clone()
        self._prev_body_acc = filtered_acc.clone()
        return filtered_acc

    def _compute_damping(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute damping forces (linear + quadratic).

        Uses simple diagonal damping: D_l * v + D_q * |v| * v
        This avoids non-physical asymmetric force-torque coupling from off-diagonal terms.
        """
        linear_term = self._linear_damping_diag * body_vel
        quadratic_term = self._quadratic_damping_diag * torch.abs(body_vel) * body_vel
        return linear_term + quadratic_term

    def _compute_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        """Compute added mass forces."""
        added_mass = torch.bmm(self._added_mass_matrix, body_acc.unsqueeze(-1)).squeeze(-1)
        return added_mass

    def _compute_coriolis(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis and centripetal forces (C_A only, legacy method)."""
        ma_v = torch.bmm(self._added_mass_matrix, body_vel.unsqueeze(-1)).squeeze(-1)
        lin_vel = body_vel[:, :3]
        ang_vel = body_vel[:, 3:]
        ma_lin = ma_v[:, :3]
        ma_ang = ma_v[:, 3:]

        coriolis_force = -torch.cross(ma_lin, ang_vel, dim=-1)
        coriolis_torque = -(torch.cross(ma_lin, lin_vel, dim=-1) + torch.cross(ma_ang, ang_vel, dim=-1))

        return torch.cat([coriolis_force, coriolis_torque], dim=-1)

    def _compute_coriolis_full(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute full Coriolis: C(v) = C_RB(v) + C_A(v)."""
        lin_vel = body_vel[:, :3]
        ang_vel = body_vel[:, 3:]

        # C_RB: Rigid body Coriolis
        h_rb = self._rigid_body_inertia * ang_vel
        c_rb_force = torch.zeros_like(lin_vel)
        c_rb_torque = -torch.cross(ang_vel, h_rb, dim=-1)

        # C_A: Added mass Coriolis
        ma_v = torch.bmm(self._added_mass_matrix, body_vel.unsqueeze(-1)).squeeze(-1)
        ma_lin = ma_v[:, :3]
        ma_ang = ma_v[:, 3:]

        c_a_force = -torch.cross(ang_vel, ma_lin, dim=-1)
        c_a_torque = -(torch.cross(lin_vel, ma_lin, dim=-1) + torch.cross(ang_vel, ma_ang, dim=-1))

        total_force = c_rb_force + c_a_force
        total_torque = c_rb_torque + c_a_torque

        return torch.cat([total_force, total_torque], dim=-1)

    def _compute_buoyancy_quat(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute net buoyancy force and restoring moment."""
        up_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        up_dir_w[:, 2] = 1.0

        down_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        down_dir_w[:, 2] = -1.0

        up_dir_b = quat_apply_inverse(root_quat_w, up_dir_w)
        down_dir_b = quat_apply_inverse(root_quat_w, down_dir_w)

        buoyancy_force_b = self._buoyancy_force_base.unsqueeze(-1) * up_dir_b
        weight_force_b = self._weight.unsqueeze(-1) * down_dir_b

        net_force_b = buoyancy_force_b + weight_force_b

        buoyancy_moment_b = torch.cross(self._r_cb, buoyancy_force_b, dim=-1)
        weight_moment_b = torch.cross(self._r_cg, weight_force_b, dim=-1)
        net_moment_b = buoyancy_moment_b + weight_moment_b

        wrench = torch.cat([net_force_b, net_moment_b], dim=-1)
        return wrench

    def set_ocean_current(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Set ocean current velocity for specified environments."""
        if isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        if velocity is None:
            rand_vel = torch.rand(len(env_ids), 6, device=self.device)
            velocity = rand_vel * self._max_current_vel * 2 - self._max_current_vel

            if self._current_noise_scale.any():
                noise = torch.randn(len(env_ids), 6, device=self.device) * self._current_noise_scale
                velocity = velocity + noise

        self._current_velocity[env_ids] = velocity

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset hydrodynamics state for specified environments."""
        if env_ids is None:
            self._prev_body_vel.zero_()
            self._prev_body_acc.zero_()
            self._current_velocity.zero_()
        else:
            if isinstance(env_ids, (list, tuple)):
                env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            self._prev_body_vel[env_ids] = 0.0
            self._prev_body_acc[env_ids] = 0.0
            self._current_velocity[env_ids] = 0.0

    def randomize_current(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Randomize ocean current for specified environments."""
        self.set_ocean_current(env_ids, velocity=None)

    def randomize_parameters(
        self,
        env_ids: torch.Tensor | Sequence[int],
        added_mass_scale: tuple[float, float] = (1.0, 1.0),
        linear_damping_scale: tuple[float, float] = (1.0, 1.0),
        quadratic_damping_scale: tuple[float, float] = (1.0, 1.0),
        volume_scale: tuple[float, float] = (1.0, 1.0),
        mass_scale: tuple[float, float] = (1.0, 1.0),
        cob_offset_scale: tuple[float, float] = (1.0, 1.0),
        inertia_scale: tuple[float, float] = (1.0, 1.0),
        payload_mass_ratio: tuple[float, float] = (0.0, 0.0),
        payload_cog_offset_z: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Randomize hydrodynamic parameters for specified environments.

        Args:
            env_ids: Environment indices to randomize.
            added_mass_scale: Scale range for added mass coefficients.
            linear_damping_scale: Scale range for linear damping coefficients.
            quadratic_damping_scale: Scale range for quadratic damping coefficients.
            volume_scale: Scale range for vehicle volume (affects buoyancy).
            mass_scale: Scale range for vehicle mass.
            cob_offset_scale: Scale range for center of buoyancy offset from base value.
                Applied to the Z-component of center_of_buoyancy. Affects restoring moments.
            inertia_scale: Scale range for rigid body inertia (Ixx, Iyy, Izz).
                Affects Coriolis forces when use_full_coriolis is enabled.
            payload_mass_ratio: Range for additional payload mass as ratio of base mass.
                E.g., (0.0, 0.2) adds 0-20% of base mass as payload.
            payload_cog_offset_z: Range for payload-induced CoG offset in Z direction (meters).
                Simulates payload attached above/below the vehicle center.
        """
        if isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)

        # Added mass
        am_scales = (
            torch.rand(num_envs, 6, device=self.device) * (added_mass_scale[1] - added_mass_scale[0])
            + added_mass_scale[0]
        )
        base_added_mass = torch.tensor(self.cfg.added_mass, dtype=torch.float32, device=self.device)
        randomized_am = base_added_mass.unsqueeze(0) * am_scales
        self._added_mass_matrix[env_ids] = torch.diag_embed(randomized_am)

        # Linear damping
        ld_scales = (
            torch.rand(num_envs, 6, device=self.device) * (linear_damping_scale[1] - linear_damping_scale[0])
            + linear_damping_scale[0]
        )
        base_linear_damping = torch.tensor(self.cfg.linear_damping, dtype=torch.float32, device=self.device)
        self._linear_damping_diag[env_ids] = base_linear_damping.unsqueeze(0) * ld_scales

        # Quadratic damping
        qd_scales = (
            torch.rand(num_envs, 6, device=self.device) * (quadratic_damping_scale[1] - quadratic_damping_scale[0])
            + quadratic_damping_scale[0]
        )
        base_quadratic_damping = torch.tensor(self.cfg.quadratic_damping, dtype=torch.float32, device=self.device)
        self._quadratic_damping_diag[env_ids] = base_quadratic_damping.unsqueeze(0) * qd_scales

        # Volume
        vol_scales = torch.rand(num_envs, device=self.device) * (volume_scale[1] - volume_scale[0]) + volume_scale[0]
        self._volume[env_ids] = self.cfg.volume * vol_scales

        # Mass (base mass before payload)
        mass_scales = torch.rand(num_envs, device=self.device) * (mass_scale[1] - mass_scale[0]) + mass_scale[0]
        if self.cfg.vehicle_mass is not None:
            base_mass = self.cfg.vehicle_mass
        else:
            base_mass = self.cfg.volume * self.cfg.water_density
        scaled_mass = base_mass * mass_scales

        # Payload mass (added on top of scaled mass)
        payload_ratios = (
            torch.rand(num_envs, device=self.device) * (payload_mass_ratio[1] - payload_mass_ratio[0])
            + payload_mass_ratio[0]
        )
        payload_mass = scaled_mass * payload_ratios
        total_mass = scaled_mass + payload_mass
        self._vehicle_mass[env_ids] = total_mass
        self._weight[env_ids] = total_mass * self._gravity

        # Update buoyancy force base
        self._buoyancy_force_base[env_ids] = self._water_density * self._gravity * self._volume[env_ids]

        # Center of Buoyancy offset (coBM) randomization
        # Scale the Z-component of center_of_buoyancy relative to base value
        cob_scales = (
            torch.rand(num_envs, device=self.device) * (cob_offset_scale[1] - cob_offset_scale[0])
            + cob_offset_scale[0]
        )
        base_cob = torch.tensor(self.cfg.center_of_buoyancy, dtype=torch.float32, device=self.device)
        self._r_cb[env_ids, 0] = base_cob[0]
        self._r_cb[env_ids, 1] = base_cob[1]
        self._r_cb[env_ids, 2] = base_cob[2] * cob_scales
        self._cob_offset[env_ids] = self._r_cb[env_ids, 2]

        # Center of Gravity offset from payload
        # Payload shifts CoG in Z direction (positive = upward, negative = downward)
        cog_offsets = (
            torch.rand(num_envs, device=self.device) * (payload_cog_offset_z[1] - payload_cog_offset_z[0])
            + payload_cog_offset_z[0]
        )
        base_cog = torch.tensor(self.cfg.center_of_gravity, dtype=torch.float32, device=self.device)
        self._r_cg[env_ids, 0] = base_cog[0]
        self._r_cg[env_ids, 1] = base_cog[1]
        self._r_cg[env_ids, 2] = base_cog[2] + cog_offsets

        # Inertia randomization
        # Scale all three diagonal components (Ixx, Iyy, Izz) independently
        inertia_scales = (
            torch.rand(num_envs, 3, device=self.device) * (inertia_scale[1] - inertia_scale[0])
            + inertia_scale[0]
        )
        if self.cfg.rigid_body_inertia is not None:
            base_inertia = torch.tensor(self.cfg.rigid_body_inertia, dtype=torch.float32, device=self.device)
        else:
            base_inertia = torch.tensor(self.cfg.added_mass[3:6], dtype=torch.float32, device=self.device) * 0.5
        self._rigid_body_inertia[env_ids] = base_inertia.unsqueeze(0) * inertia_scales
