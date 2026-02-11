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

import warnings
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
        articulation_prim_path: str | None = None,
    ) -> None:
        """Initialize the hydrodynamics model.

        This model computes hydrodynamic forces for underwater vehicles.
        Weight (gravity) is handled by PhysX, so this model only applies
        buoyancy as an external upward force.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Hydrodynamics configuration.
            current_cfg: Ocean current configuration. Defaults to no current.
            dt: Simulation timestep for acceleration calculation.
            articulation_prim_path: USD path to articulation root for auto volume calculation.
                Only used if cfg.volume is None.
        """
        # Store basic parameters
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.current_cfg = current_cfg or OceanCurrentCfg()
        self.dt = dt

        # Common tensor creation kwargs
        self._tensor_kwargs = {"dtype": torch.float32, "device": device}

        # Initialize components
        self._init_hydrodynamic_matrices(cfg)
        self._init_buoyancy_params(cfg, articulation_prim_path)
        self._init_state_buffers(cfg)

    def _init_hydrodynamic_matrices(self, cfg: HydrodynamicsCfg) -> None:
        """Initialize added mass and damping matrices.

        Args:
            cfg: Hydrodynamics configuration.
        """
        # Added mass matrix (6x6 diagonal)
        added_mass_diag = torch.diag(torch.tensor(cfg.added_mass, **self._tensor_kwargs))
        self._added_mass_matrix = added_mass_diag.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Damping coefficients (linear and quadratic)
        self._linear_damping_diag = (
            torch.tensor(cfg.linear_damping, **self._tensor_kwargs).expand(self.num_envs, -1).clone()
        )
        self._quadratic_damping_diag = (
            torch.tensor(cfg.quadratic_damping, **self._tensor_kwargs).expand(self.num_envs, -1).clone()
        )

        # Rigid body inertia for Coriolis matrix
        if cfg.rigid_body_inertia is not None:
            inertia = torch.tensor(cfg.rigid_body_inertia, **self._tensor_kwargs)
        else:
            # Fallback: estimate from added mass rotational terms (heuristic)
            inertia = torch.tensor(cfg.added_mass[3:6], **self._tensor_kwargs) * 0.5
        self._rigid_body_inertia = inertia.expand(self.num_envs, -1).clone()
        self._use_full_coriolis = cfg.use_full_coriolis
        if self._use_full_coriolis:
            warnings.warn(
                f"HydrodynamicsModel({cfg.body_name}): use_full_coriolis=True computes C_RB internally. "
                "Ensure enable_gyroscopic_forces=False in RigidBodyPropertiesCfg to avoid double-counting "
                "rigid body gyroscopic effects.",
                stacklevel=2,
            )

        # Added mass force settings
        self._apply_added_mass = cfg.apply_added_mass_force
        self._am_stability_factor = cfg.added_mass_stability_factor

        # Off-diagonal damping cross-coupling
        self._damping_cross_coupling = cfg.damping_cross_coupling

    def _init_buoyancy_params(self, cfg: HydrodynamicsCfg, articulation_prim_path: str | None) -> None:
        """Initialize buoyancy related parameters.

        Args:
            cfg: Hydrodynamics configuration.
            articulation_prim_path: USD path to articulation root for auto volume calculation.
        """
        self._water_density = torch.full((self.num_envs,), cfg.water_density, **self._tensor_kwargs)
        self._gravity = 9.81

        # Volume and buoyancy force
        volume_value = self._resolve_volume(cfg, articulation_prim_path)
        self._volume = torch.full((self.num_envs,), volume_value, **self._tensor_kwargs)
        self._buoyancy_force_base = self._water_density * self._gravity * self._volume

        # Center of buoyancy/gravity in body frame
        self._r_cb = torch.tensor(cfg.center_of_buoyancy, **self._tensor_kwargs).expand(self.num_envs, -1).clone()
        self._r_cg = torch.tensor(cfg.center_of_gravity, **self._tensor_kwargs).expand(self.num_envs, -1).clone()
        self._cob_offset = self._r_cb[:, 2]  # Legacy compatibility

        # Nominal CoG and body mass for gravity restoring moment correction.
        # When CoG is randomized away from nominal, a correction torque is applied:
        #   M_correction = (r_cg - r_cg_nominal) x F_weight_body
        # This is zero when r_cg equals the nominal (URDF/PhysX) value.
        self._r_cg_nominal = torch.tensor(cfg.center_of_gravity, **self._tensor_kwargs)
        if cfg.body_mass is not None:
            self._body_mass: torch.Tensor | None = torch.full((self.num_envs,), cfg.body_mass, **self._tensor_kwargs)
        else:
            self._body_mass = None

    def _init_state_buffers(self, cfg: HydrodynamicsCfg) -> None:
        """Initialize state buffers and ocean current.

        Args:
            cfg: Hydrodynamics configuration.
        """
        # Velocity and acceleration state buffers (used only in finite-difference mode,
        # i.e. when apply_added_mass_force=False; otherwise PhysX acceleration is used)
        self._prev_body_vel = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)
        self._prev_body_acc = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)
        self._alpha = cfg.acceleration_filter_alpha

        # Ocean current state (world frame, 6-DOF)
        self._current_velocity = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)
        self._max_current_vel = torch.tensor(self.current_cfg.max_velocity, **self._tensor_kwargs)
        self._current_noise_scale = torch.tensor(self.current_cfg.noise_scale, **self._tensor_kwargs)

        # PhysX acceleration cache (body frame, updated via update_physx_state)
        self._physx_acc_b = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)

    def _to_env_ids(self, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Convert env_ids to tensor format.

        Args:
            env_ids: Environment indices as tensor, list, or tuple.

        Returns:
            Environment indices as a long tensor on the correct device.
        """
        if not isinstance(env_ids, torch.Tensor):
            return torch.tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids

    def update_physx_state(
        self,
        body_com_acc_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        """Update cached PhysX acceleration after physics step.

        This method should be called after robot.update() to cache the acceleration
        computed by PhysX. Required for hybrid added mass force calculation.

        Args:
            body_com_acc_w: Body center of mass acceleration in world frame.
                Shape: (num_envs, num_bodies, 6) or (num_envs, 6).
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).
        """
        if not self._apply_added_mass:
            return

        # Handle both (num_envs, num_bodies, 6) and (num_envs, 6) shapes
        if body_com_acc_w.dim() == 3:
            acc_w = body_com_acc_w[:, 0, :]  # Root body acceleration
        else:
            acc_w = body_com_acc_w

        # Transform world frame acceleration to body frame
        lin_acc_b = quat_apply_inverse(root_quat_w, acc_w[:, :3])
        ang_acc_b = quat_apply_inverse(root_quat_w, acc_w[:, 3:])
        self._physx_acc_b = torch.cat([lin_acc_b, ang_acc_b], dim=-1)

    def _resolve_volume(self, cfg: HydrodynamicsCfg, articulation_prim_path: str | None) -> float:
        """Resolve volume from config or auto-calculate from collision geometry."""
        if cfg.volume is not None:
            return cfg.volume

        if articulation_prim_path is not None:
            from isaaclab.utils.volume import compute_collision_volume

            body_path = f"{articulation_prim_path}/{cfg.body_name}"
            volume = compute_collision_volume(body_path)
            if volume > 0:
                return volume
            warnings.warn(
                f"Auto-calculated volume is {volume} m^3 for {body_path}. "
                "Using default 0.01 m^3. Consider setting volume explicitly in config."
            )
        else:
            warnings.warn("Volume not specified and no articulation_prim_path provided. Using default 0.01 m^3.")
        return 0.01

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
        # Transform velocities to body frame
        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_vel = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

        # Transform ocean current to body frame
        current_lin_b = quat_apply_inverse(root_quat_w, self._current_velocity[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, self._current_velocity[:, 3:])
        current_b = torch.cat([current_lin_b, current_ang_b], dim=-1)

        # Relative velocity for hydrodynamic calculations
        relative_vel = body_vel - current_b

        # Compute hydrodynamic components
        damping = self._compute_damping(relative_vel)

        # Coriolis: C_RB uses absolute velocity, C_A uses relative velocity (per Fossen)
        if self._use_full_coriolis:
            coriolis = self._compute_coriolis_full(body_vel, relative_vel)
        else:
            coriolis = self._compute_coriolis(relative_vel)

        # Added mass force (M_A * v_dot)
        # Two modes:
        #   1. PhysX acceleration (apply_added_mass_force=True): uses cached PhysX body acceleration
        #      from update_physx_state(). More accurate -- accounts for all forces/constraints.
        #   2. Finite-difference (apply_added_mass_force=False): uses EMA-filtered numerical
        #      differentiation of velocity. Only used as fallback; acceleration_filter_alpha controls
        #      the low-pass filter strength.
        added_mass_force = torch.zeros(self.num_envs, 6, device=self.device)
        if self._apply_added_mass:
            added_mass_force = self._compute_added_mass(self._physx_acc_b) * self._am_stability_factor
        else:
            self._compute_acceleration(relative_vel)

        # Total hydrodynamic wrench: tau = -C(v)*v - D(v)*v - M_A*v_dot + g(eta)
        hydro_wrench = -(coriolis + damping + added_mass_force)
        buoyancy = self._compute_buoyancy_quat(root_quat_w)

        forces_b = hydro_wrench[:, :3] + buoyancy[:, :3]
        torques_b = hydro_wrench[:, 3:] + buoyancy[:, 3:]

        return forces_b, torques_b

    def _compute_acceleration(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute filtered body acceleration via finite-difference.

        Only called when apply_added_mass_force=False (fallback mode).
        When apply_added_mass_force=True, PhysX acceleration is used instead.
        """
        raw_acc = (body_vel - self._prev_body_vel) / self.dt
        filtered_acc = (1.0 - self._alpha) * self._prev_body_acc + self._alpha * raw_acc
        self._prev_body_vel = body_vel.clone()
        self._prev_body_acc = filtered_acc.clone()
        return filtered_acc

    def _compute_damping(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute damping forces (linear + quadratic).

        When cross-coupling is disabled (default), uses diagonal damping:
            D_l * v + D_q * |v| * v

        When cross-coupling is enabled, velocity from coupled DOFs is added
        to the damping computation. For example, coupling (1, 5) means yaw
        velocity also contributes to sway damping, modeling the sway-yaw
        interaction common in slender underwater vehicles.
        """
        vel = body_vel
        if self._damping_cross_coupling is not None:
            vel = body_vel.clone()
            for i, j in self._damping_cross_coupling:
                vel[:, i] = vel[:, i] + body_vel[:, j]

        linear_term = self._linear_damping_diag * vel
        quadratic_term = self._quadratic_damping_diag * torch.abs(vel) * vel
        return linear_term + quadratic_term

    def _compute_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        """Compute added mass forces."""
        added_mass = torch.bmm(self._added_mass_matrix, body_acc.unsqueeze(-1)).squeeze(-1)
        return added_mass

    def _compute_coriolis_added_mass(self, velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Coriolis force and torque from added mass matrix.

        This is the common computation shared between _compute_coriolis and
        _compute_coriolis_full for the C_A (added mass Coriolis) component.

        Args:
            velocity: Body velocity (linear and angular). Shape: (num_envs, 6).

        Returns:
            Tuple of (force, torque) from added mass Coriolis. Each has shape (num_envs, 3).
        """
        lin_vel = velocity[:, :3]
        ang_vel = velocity[:, 3:]

        # M_A * v
        ma_v = torch.bmm(self._added_mass_matrix, velocity.unsqueeze(-1)).squeeze(-1)
        ma_lin = ma_v[:, :3]
        ma_ang = ma_v[:, 3:]

        # C_A force and torque
        force = -torch.cross(ma_lin, ang_vel, dim=-1)
        torque = -(torch.cross(ma_lin, lin_vel, dim=-1) + torch.cross(ma_ang, ang_vel, dim=-1))

        return force, torque

    def _compute_coriolis(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis and centripetal forces (C_A only, legacy method)."""
        force, torque = self._compute_coriolis_added_mass(body_vel)
        return torch.cat([force, torque], dim=-1)

    def _compute_coriolis_full(self, body_vel: torch.Tensor, relative_vel: torch.Tensor) -> torch.Tensor:
        """Compute full Coriolis: C(v) = C_RB(v) + C_A(v_r).

        Per Fossen's formulation:
            - C_RB uses absolute velocity (body_vel)
            - C_A uses relative velocity (body_vel - current)
        """
        # C_RB: Rigid body Coriolis (uses absolute velocity)
        ang_vel_abs = body_vel[:, 3:]
        h_rb = self._rigid_body_inertia * ang_vel_abs
        c_rb_force = torch.zeros_like(body_vel[:, :3])
        c_rb_torque = -torch.cross(ang_vel_abs, h_rb, dim=-1)

        # C_A: Added mass Coriolis (uses relative velocity)
        c_a_force, c_a_torque = self._compute_coriolis_added_mass(relative_vel)

        total_force = c_rb_force + c_a_force
        total_torque = c_rb_torque + c_a_torque

        return torch.cat([total_force, total_torque], dim=-1)

    def _compute_buoyancy_quat(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute buoyancy force and restoring moment.

        Note:
            This method computes ONLY buoyancy force, not weight.
            Weight (gravity) is handled by PhysX with disable_gravity=False.
            This separation allows proper multi-body dynamics where each
            link's mass contributes to the gravitational force naturally.

        The restoring moment arises from the offset between Center of Buoyancy (CoB)
        and Center of Gravity (CoG). When the vehicle tilts, buoyancy acts at CoB
        while gravity acts at CoG, creating a restoring torque.

        Args:
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Buoyancy wrench in body frame [Fx, Fy, Fz, Mx, My, Mz]. Shape: (num_envs, 6).
        """
        # World up direction
        up_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        up_dir_w[:, 2] = 1.0

        # Transform to body frame
        up_dir_b = quat_apply_inverse(root_quat_w, up_dir_w)

        # Buoyancy force: F_b = rho * V * g * up_direction (in body frame)
        buoyancy_force_b = self._buoyancy_force_base.unsqueeze(-1) * up_dir_b

        # Restoring moment from buoyancy acting at CoB
        # M = r_cb x F_buoyancy
        buoyancy_moment_b = torch.cross(self._r_cb, buoyancy_force_b, dim=-1)

        # CoG correction torque for domain randomization.
        # PhysX applies gravity at the nominal (URDF) CoG. When CoG is shifted
        # via randomization, apply: M_corr = delta_cg x F_weight_body
        # where F_weight_body = -m*g*up_dir_b (weight points downward in body frame).
        if self._body_mass is not None:
            delta_cg = self._r_cg - self._r_cg_nominal
            weight_force_b = -(self._body_mass.unsqueeze(-1) * self._gravity) * up_dir_b
            buoyancy_moment_b = buoyancy_moment_b + torch.cross(delta_cg, weight_force_b, dim=-1)

        wrench = torch.cat([buoyancy_force_b, buoyancy_moment_b], dim=-1)
        return wrench

    def set_ocean_current(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Set ocean current velocity for specified environments."""
        env_ids = self._to_env_ids(env_ids)

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
            self._physx_acc_b.zero_()
            self._water_density[:] = self.cfg.water_density
        else:
            env_ids = self._to_env_ids(env_ids)
            self._prev_body_vel[env_ids] = 0.0
            self._prev_body_acc[env_ids] = 0.0
            self._current_velocity[env_ids] = 0.0
            self._physx_acc_b[env_ids] = 0.0
            self._water_density[env_ids] = self.cfg.water_density

    # --- Properties for parameter access (used by environment-specific randomization) ---

    @property
    def added_mass_matrix(self) -> torch.Tensor:
        """Added mass matrix (num_envs, 6, 6)."""
        return self._added_mass_matrix

    @property
    def linear_damping(self) -> torch.Tensor:
        """Linear damping coefficients (num_envs, 6)."""
        return self._linear_damping_diag

    @property
    def quadratic_damping(self) -> torch.Tensor:
        """Quadratic damping coefficients (num_envs, 6)."""
        return self._quadratic_damping_diag

    @property
    def volume(self) -> torch.Tensor:
        """Vehicle volume (num_envs,)."""
        return self._volume

    @property
    def buoyancy_force(self) -> torch.Tensor:
        """Buoyancy force magnitude (num_envs,)."""
        return self._buoyancy_force_base

    @property
    def center_of_buoyancy(self) -> torch.Tensor:
        """Center of buoyancy in body frame (num_envs, 3)."""
        return self._r_cb

    @property
    def center_of_gravity(self) -> torch.Tensor:
        """Center of gravity in body frame (num_envs, 3)."""
        return self._r_cg

    @property
    def water_density(self) -> torch.Tensor:
        """Water density per environment (num_envs,)."""
        return self._water_density

    @property
    def rigid_body_inertia(self) -> torch.Tensor:
        """Rigid body inertia diagonal (num_envs, 3)."""
        return self._rigid_body_inertia

    def update_buoyancy_force(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Update buoyancy force after volume change.

        Call this after modifying volume to ensure buoyancy_force_base is consistent.

        Args:
            env_ids: Environment indices to update. If None, updates all.
        """
        if env_ids is None:
            self._buoyancy_force_base = self._water_density * self._gravity * self._volume
        else:
            env_ids = self._to_env_ids(env_ids)
            self._buoyancy_force_base[env_ids] = self._water_density[env_ids] * self._gravity * self._volume[env_ids]
