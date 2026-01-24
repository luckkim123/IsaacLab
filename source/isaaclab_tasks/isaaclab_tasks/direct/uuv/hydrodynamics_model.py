# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fossen model hydrodynamics for underwater vehicles.

This module implements the 6-DOF hydrodynamic forces and torques based on
the Fossen model for marine craft dynamics. The implementation is adapted
from MarineGym (IROS 2025) for integration with Isaac Lab.

Reference:
    Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from collections.abc import Sequence


@configclass
class HydrodynamicsCfg:
    """Configuration for Fossen model hydrodynamics.

    All coefficients are for a 6-DOF system: [surge, sway, heave, roll, pitch, yaw].
    Diagonal matrices are assumed for simplicity (off-diagonal terms can be added later).

    The model implements the complete Fossen formulation including:
        - Weight-buoyancy difference (W-B) for non-neutral buoyancy vehicles
        - Full Coriolis matrix C(v) = C_RB(v) + C_A(v)
        - Quaternion-based buoyancy calculation (no gimbal lock)

    Reference:
        Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
    """

    # Added mass coefficients (kg for linear, kg*m^2 for angular)
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients (Ns/m for linear, Nms/rad for angular)
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients (Ns^2/m^2 for linear, Nms^2/rad^2 for angular)
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle mass (kg). If None, assumes neutral buoyancy (mass = volume * water_density)
    vehicle_mass: float | None = None

    # Vehicle volume for buoyancy calculation (m^3)
    volume: float = 0.0113459

    # Center of buoyancy position in body frame (m, [x, y, z])
    # For backward compatibility, if only z-offset needed, use center_of_buoyancy_offset
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)

    # Center of gravity position in body frame (m, [x, y, z])
    # Usually (0, 0, 0) if body frame origin is at CoG
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # DEPRECATED: Use center_of_buoyancy instead. Kept for backward compatibility.
    center_of_buoyancy_offset: float | None = None

    # Water density (kg/m^3, default: freshwater, seawater: 1025.0)
    water_density: float = 997.0

    # Acceleration filter alpha for numerical stability (0 < alpha < 1)
    acceleration_filter_alpha: float = 0.3

    # Use full Coriolis matrix C(v) = C_RB(v) + C_A(v) per Fossen model
    # If False, uses simplified added-mass-only Coriolis (legacy behavior)
    use_full_coriolis: bool = True

    # Rigid body inertia for full Coriolis (kg*m^2, [I_xx, I_yy, I_zz])
    # If None, estimates from added mass rotational terms
    rigid_body_inertia: tuple[float, float, float] | None = None


@configclass
class OceanCurrentCfg:
    """Configuration for ocean current disturbances."""

    # Maximum current velocity [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
    max_velocity: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Gaussian noise scale for current velocity
    noise_scale: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
        # Priority: cfg.vehicle_mass > robot_mass > neutral buoyancy assumption
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

        # Center of buoyancy and gravity in body frame (3D vectors)
        # Handle backward compatibility with center_of_buoyancy_offset
        if cfg.center_of_buoyancy_offset is not None:
            # Legacy: scalar offset interpreted as z-component
            self._r_cb = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
            self._r_cb[:, 2] = cfg.center_of_buoyancy_offset
        else:
            self._r_cb = (
                torch.tensor(cfg.center_of_buoyancy, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(num_envs, 1)
            )

        self._r_cg = (
            torch.tensor(cfg.center_of_gravity, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
        )

        # Keep legacy _cob_offset for compatibility
        self._cob_offset = self._r_cb[:, 2]

        # Rigid body inertia for full Coriolis matrix
        if cfg.rigid_body_inertia is not None:
            self._rigid_body_inertia = (
                torch.tensor(cfg.rigid_body_inertia, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(num_envs, 1)
            )
        else:
            # Estimate from added mass rotational terms (rough approximation)
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
            root_quat_w: Root orientation quaternion (w, x, y, z) in world frame. Shape: (num_envs, 4).

        Returns:
            Tuple of (forces_b, torques_b) in body frame. Each has shape (num_envs, 3).
        """
        # Transform world velocities to body frame
        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_vel = torch.cat([lin_vel_b, ang_vel_b], dim=-1)  # (num_envs, 6)

        # Transform ocean current to body frame and compute relative velocity
        current_w = self._current_velocity + torch.randn_like(self._current_velocity) * self._current_noise_scale
        current_lin_b = quat_apply_inverse(root_quat_w, current_w[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, current_w[:, 3:])
        current_b = torch.cat([current_lin_b, current_ang_b], dim=-1)

        # Relative velocity (vehicle velocity - current velocity)
        relative_vel = body_vel - current_b

        # Apply coordinate sign convention for underwater dynamics (Fossen convention)
        # Y and Z axes, and pitch/yaw rates have opposite sign
        relative_vel_fossen = relative_vel.clone()
        relative_vel_fossen[:, [1, 2, 4, 5]] *= -1

        # Compute body acceleration (filtered)
        body_acc = self._compute_acceleration(relative_vel_fossen)

        # Compute individual hydrodynamic force components
        damping = self._compute_damping(relative_vel_fossen)
        added_mass = self._compute_added_mass(body_acc)

        # Choose Coriolis computation method
        if self._use_full_coriolis:
            coriolis = self._compute_coriolis_full(relative_vel_fossen)
        else:
            coriolis = self._compute_coriolis(relative_vel_fossen)

        # Compute buoyancy using quaternion-based method (no gimbal lock)
        buoyancy = self._compute_buoyancy_quat(root_quat_w)

        # Total hydrodynamic wrench (negative because these oppose motion)
        hydro = -(added_mass + coriolis + damping)

        # Restore coordinate sign convention
        hydro[:, [1, 2, 4, 5]] *= -1
        buoyancy[:, [1, 2, 4, 5]] *= -1

        # Split into forces and torques
        forces_b = hydro[:, :3] + buoyancy[:, :3]
        torques_b = hydro[:, 3:] + buoyancy[:, 3:]

        return forces_b, torques_b

    def _compute_acceleration(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute filtered body acceleration.

        Uses exponential moving average filter for numerical stability.

        Args:
            body_vel: Body frame velocity. Shape: (num_envs, 6).

        Returns:
            Filtered body acceleration. Shape: (num_envs, 6).
        """
        # Finite difference acceleration
        raw_acc = (body_vel - self._prev_body_vel) / self.dt

        # Exponential moving average filter
        filtered_acc = (1.0 - self._alpha) * self._prev_body_acc + self._alpha * raw_acc

        # Update state
        self._prev_body_vel = body_vel.clone()
        self._prev_body_acc = filtered_acc.clone()

        return filtered_acc

    def _compute_damping(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute damping forces (linear + quadratic).

        The damping matrix includes coupled terms between sway-yaw and heave-pitch
        as commonly observed in underwater vehicles. This follows MarineGym's
        implementation where velocity values are placed in off-diagonal positions
        to enable coupling through the quadratic damping term.

        Args:
            body_vel: Body frame velocity. Shape: (num_envs, 6).

        Returns:
            Damping wrench. Shape: (num_envs, 6).
        """
        # Build velocity matrix for element-wise multiplication with damping matrices
        # This follows MarineGym's approach: maintained_body_vels = diag(v) + coupling terms
        # Coupling: sway-yaw (1,5), heave-pitch (2,4), and their symmetric counterparts
        maintained_body_vels = torch.diag_embed(body_vel)
        maintained_body_vels[:, 1, 5] = body_vel[:, 5]  # sway-yaw coupling
        maintained_body_vels[:, 2, 4] = body_vel[:, 4]  # heave-pitch coupling
        maintained_body_vels[:, 4, 2] = body_vel[:, 2]  # pitch-heave coupling
        maintained_body_vels[:, 5, 1] = body_vel[:, 1]  # yaw-sway coupling

        # Build damping matrix: D(v) = D_linear + D_quadratic * |v|
        # Element-wise multiplication means coupling only affects quadratic damping
        linear_damping_matrix = torch.diag_embed(self._linear_damping_diag)
        quadratic_damping_matrix = torch.diag_embed(self._quadratic_damping_diag)
        damping_matrix = linear_damping_matrix + quadratic_damping_matrix * torch.abs(maintained_body_vels)

        # Compute damping force: D(v) * v
        damping = torch.bmm(damping_matrix, body_vel.unsqueeze(-1)).squeeze(-1)

        return damping

    def _compute_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        """Compute added mass forces.

        Added mass represents the inertia of the fluid that must be accelerated
        along with the vehicle.

        Args:
            body_acc: Body frame acceleration. Shape: (num_envs, 6).

        Returns:
            Added mass wrench. Shape: (num_envs, 6).
        """
        # M_A * a
        added_mass = torch.bmm(self._added_mass_matrix, body_acc.unsqueeze(-1)).squeeze(-1)
        return added_mass

    def _compute_coriolis(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis and centripetal forces (C_A only, legacy method).

        These forces arise from the coupling between the vehicle's linear and
        angular velocities through the added mass. This is the simplified
        formulation that only includes added mass contribution.

        Args:
            body_vel: Body frame velocity. Shape: (num_envs, 6).

        Returns:
            Coriolis wrench. Shape: (num_envs, 6).
        """
        # M_A * v
        ma_v = torch.bmm(self._added_mass_matrix, body_vel.unsqueeze(-1)).squeeze(-1)

        lin_vel = body_vel[:, :3]
        ang_vel = body_vel[:, 3:]
        ma_lin = ma_v[:, :3]
        ma_ang = ma_v[:, 3:]

        # Coriolis force: -[M_A * v_lin] x omega
        coriolis_force = -torch.cross(ma_lin, ang_vel, dim=-1)

        # Coriolis torque: -[M_A * v_lin] x v_lin - [M_A * omega] x omega
        coriolis_torque = -(torch.cross(ma_lin, lin_vel, dim=-1) + torch.cross(ma_ang, ang_vel, dim=-1))

        return torch.cat([coriolis_force, coriolis_torque], dim=-1)

    def _compute_coriolis_full(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute full Coriolis and centripetal forces: C(v) = C_RB(v) + C_A(v).

        Implements the complete Fossen formulation (Eq. 6.43, 6.53) including
        both rigid body and added mass contributions.

        Reference:
            Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
            Chapter 6, Equations 6.43 and 6.53.

        Args:
            body_vel: Body frame velocity [v_lin, omega]. Shape: (num_envs, 6).

        Returns:
            Full Coriolis wrench. Shape: (num_envs, 6).
        """
        lin_vel = body_vel[:, :3]
        ang_vel = body_vel[:, 3:]

        # ===== C_RB: Rigid body Coriolis (Fossen Eq. 6.43) =====
        # Assuming center of gravity at body frame origin (r_g = 0)

        # Note: For r_g=0 (CoG at body frame origin), C_RB linear-angular coupling is zero
        # p_rb = m * v not needed since C_RB^{12} = -m * S(r_g) = 0

        # Rigid body angular momentum: h_rb = I * omega (diagonal inertia)
        h_rb = self._rigid_body_inertia * ang_vel  # (num_envs, 3)

        # C_RB force component (from linear-angular coupling)
        # For r_g = 0: C_RB^{12} = 0, so c_rb_force = 0
        c_rb_force = torch.zeros_like(lin_vel)

        # C_RB torque component: -omega x (I * omega)
        # Note: -v x (m * v) = 0 (parallel vectors)
        c_rb_torque = -torch.cross(ang_vel, h_rb, dim=-1)

        # ===== C_A: Added mass Coriolis (Fossen Eq. 6.53) =====
        # M_A * v
        ma_v = torch.bmm(self._added_mass_matrix, body_vel.unsqueeze(-1)).squeeze(-1)
        ma_lin = ma_v[:, :3]  # M_A^{11} * v
        ma_ang = ma_v[:, 3:]  # M_A^{22} * omega

        # C_A force: -omega x (M_A * v_lin)
        c_a_force = -torch.cross(ang_vel, ma_lin, dim=-1)

        # C_A torque: -v x (M_A * v) - omega x (M_A * omega)
        c_a_torque = -(torch.cross(lin_vel, ma_lin, dim=-1) + torch.cross(ang_vel, ma_ang, dim=-1))

        # ===== Total Coriolis: C = C_RB + C_A =====
        total_force = c_rb_force + c_a_force
        total_torque = c_rb_torque + c_a_torque

        return torch.cat([total_force, total_torque], dim=-1)

    def _compute_buoyancy(self, rpy: torch.Tensor) -> torch.Tensor:
        """Compute buoyancy force and restoring moment (Euler angle method, legacy).

        The buoyancy force acts upward and the restoring moment tries to
        return the vehicle to a level orientation.

        Note: This method is kept for backward compatibility. The quaternion-based
        method (_compute_buoyancy_quat) is preferred as it avoids gimbal lock.

        Args:
            rpy: Roll, pitch, yaw angles. Shape: (num_envs, 3).

        Returns:
            Buoyancy wrench. Shape: (num_envs, 6).
        """
        roll = rpy[:, 0]
        pitch = rpy[:, 1]

        # Buoyancy force magnitude: rho * g * V
        buoyancy_force_mag = self._water_density * self._gravity * self._volume

        # Buoyancy force components in body frame
        # (depends on orientation relative to gravity)
        buoyancy = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)

        # Force components
        buoyancy[:, 0] = buoyancy_force_mag * torch.sin(pitch)
        buoyancy[:, 1] = -buoyancy_force_mag * torch.sin(roll) * torch.cos(pitch)
        buoyancy[:, 2] = -buoyancy_force_mag * torch.cos(roll) * torch.cos(pitch)

        # Restoring moments (from CoB offset)
        # Moment = r_cb x F_buoyancy
        buoyancy[:, 3] = -self._cob_offset * buoyancy_force_mag * torch.cos(pitch) * torch.sin(roll)
        buoyancy[:, 4] = -self._cob_offset * buoyancy_force_mag * torch.sin(pitch)
        # Yaw moment is zero for vertical CoB offset

        return buoyancy

    def _compute_buoyancy_quat(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute buoyancy force and restoring moment using quaternion rotation.

        IMPORTANT: This method computes ONLY buoyancy force, NOT weight.
        Weight (gravity) is handled by Isaac Sim physics engine when disable_gravity=False.
        Computing weight here would result in double gravity application.

        The buoyancy force always points upward (opposite to gravity) and its magnitude
        depends on the vehicle's displaced volume: F_b = rho * V * g

        The restoring moment comes from the offset between center of buoyancy (CoB)
        and center of gravity (CoG). When CoB is above CoG, tilting the vehicle
        creates a restoring moment that tries to return it to upright orientation.

        Args:
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Buoyancy wrench [force, moment] in body frame. Shape: (num_envs, 6).
        """
        # Up direction in world frame (pointing up, positive z)
        up_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        up_dir_w[:, 2] = 1.0  # Unit vector pointing up

        # Rotate up direction to body frame: up_b = R^T * up_w
        up_dir_b = quat_apply_inverse(root_quat_w, up_dir_w)

        # Buoyancy force in body frame: F_buoyancy = rho * V * g * up_b (points up in body frame)
        # Buoyancy always acts upward in world frame
        buoyancy_force_b = self._buoyancy_force_base.unsqueeze(-1) * up_dir_b

        # Restoring moment from CoB offset
        # M_buoyancy = r_cb x F_buoyancy
        # This creates a restoring moment when the vehicle is tilted
        buoyancy_moment_b = torch.cross(self._r_cb, buoyancy_force_b, dim=-1)

        # Combine into wrench (no weight component - handled by physics engine)
        wrench = torch.cat([buoyancy_force_b, buoyancy_moment_b], dim=-1)

        return wrench

    def _quaternion_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to Euler angles (roll, pitch, yaw).

        Args:
            quat: Quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Euler angles (roll, pitch, yaw). Shape: (num_envs, 3).
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)  # Numerical stability
        pitch = torch.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def set_ocean_current(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Set ocean current velocity for specified environments.

        Args:
            env_ids: Environment indices to update.
            velocity: Current velocity (6-DOF) in world frame. Shape: (len(env_ids), 6).
                     If None, samples random current based on max_velocity config.
        """
        if isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        if velocity is None:
            # Sample random current velocity within [-max, +max]
            rand_vel = torch.rand(len(env_ids), 6, device=self.device)
            velocity = rand_vel * self._max_current_vel * 2 - self._max_current_vel

        self._current_velocity[env_ids] = velocity

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset hydrodynamics state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
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
        """Randomize ocean current for specified environments.

        Args:
            env_ids: Environment indices to randomize.
        """
        self.set_ocean_current(env_ids, velocity=None)

    def randomize_parameters(
        self,
        env_ids: torch.Tensor | Sequence[int],
        added_mass_scale: tuple[float, float] = (1.0, 1.0),
        linear_damping_scale: tuple[float, float] = (1.0, 1.0),
        quadratic_damping_scale: tuple[float, float] = (1.0, 1.0),
        volume_scale: tuple[float, float] = (1.0, 1.0),
        mass_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """Randomize hydrodynamic parameters for specified environments.

        Applies scale factors to the base hydrodynamic parameters from configuration.
        Each environment can have different randomized values.

        Args:
            env_ids: Environment indices to randomize.
            added_mass_scale: Scale range (min, max) for added mass coefficients.
            linear_damping_scale: Scale range (min, max) for linear damping coefficients.
            quadratic_damping_scale: Scale range (min, max) for quadratic damping coefficients.
            volume_scale: Scale range (min, max) for vehicle volume.
            mass_scale: Scale range (min, max) for vehicle mass.
        """
        if isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)

        # Sample random scale factors for added mass (6 DOF)
        am_scales = (
            torch.rand(num_envs, 6, device=self.device) * (added_mass_scale[1] - added_mass_scale[0])
            + added_mass_scale[0]
        )
        base_added_mass = torch.tensor(self.cfg.added_mass, dtype=torch.float32, device=self.device)
        randomized_am = base_added_mass.unsqueeze(0) * am_scales
        self._added_mass_matrix[env_ids] = torch.diag_embed(randomized_am)

        # Sample random scale factors for linear damping
        ld_scales = (
            torch.rand(num_envs, 6, device=self.device) * (linear_damping_scale[1] - linear_damping_scale[0])
            + linear_damping_scale[0]
        )
        base_linear_damping = torch.tensor(self.cfg.linear_damping, dtype=torch.float32, device=self.device)
        self._linear_damping_diag[env_ids] = base_linear_damping.unsqueeze(0) * ld_scales

        # Sample random scale factors for quadratic damping
        qd_scales = (
            torch.rand(num_envs, 6, device=self.device) * (quadratic_damping_scale[1] - quadratic_damping_scale[0])
            + quadratic_damping_scale[0]
        )
        base_quadratic_damping = torch.tensor(self.cfg.quadratic_damping, dtype=torch.float32, device=self.device)
        self._quadratic_damping_diag[env_ids] = base_quadratic_damping.unsqueeze(0) * qd_scales

        # Sample random scale factors for volume
        vol_scales = torch.rand(num_envs, device=self.device) * (volume_scale[1] - volume_scale[0]) + volume_scale[0]
        self._volume[env_ids] = self.cfg.volume * vol_scales

        # Sample random scale factors for vehicle mass
        mass_scales = torch.rand(num_envs, device=self.device) * (mass_scale[1] - mass_scale[0]) + mass_scale[0]
        # Use configured mass or neutral buoyancy assumption as base
        if self.cfg.vehicle_mass is not None:
            base_mass = self.cfg.vehicle_mass
        else:
            base_mass = self.cfg.volume * self.cfg.water_density
        self._vehicle_mass[env_ids] = base_mass * mass_scales
        self._weight[env_ids] = self._vehicle_mass[env_ids] * self._gravity

        # Update buoyancy force base (depends on randomized volume)
        self._buoyancy_force_base[env_ids] = self._water_density * self._gravity * self._volume[env_ids]
