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

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, quat_apply

if TYPE_CHECKING:
    from typing import Sequence


@configclass
class HydrodynamicsCfg:
    """Configuration for Fossen model hydrodynamics.

    All coefficients are for a 6-DOF system: [surge, sway, heave, roll, pitch, yaw].
    Diagonal matrices are assumed for simplicity (off-diagonal terms can be added later).
    """

    # Added mass coefficients (kg for linear, kg*m^2 for angular)
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients (Ns/m for linear, Nms/rad for angular)
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients (Ns^2/m^2 for linear, Nms^2/rad^2 for angular)
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle volume for buoyancy calculation (m^3)
    volume: float = 0.0113459

    # Center of buoyancy offset from center of mass (m, positive = CoB above CoM)
    center_of_buoyancy_offset: float = 0.01

    # Water density (kg/m^3, default: freshwater)
    water_density: float = 997.0

    # Acceleration filter alpha for numerical stability (0 < alpha < 1)
    acceleration_filter_alpha: float = 0.3


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
    ) -> None:
        """Initialize the hydrodynamics model.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Hydrodynamics configuration.
            current_cfg: Ocean current configuration. Defaults to no current.
            dt: Simulation timestep for acceleration calculation.
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.current_cfg = current_cfg or OceanCurrentCfg()
        self.dt = dt

        # Build hydrodynamic matrices (num_envs, 6, 6)
        self._added_mass_matrix = torch.diag(
            torch.tensor(cfg.added_mass, dtype=torch.float32, device=device)
        ).unsqueeze(0).repeat(num_envs, 1, 1)

        self._linear_damping_diag = torch.tensor(
            cfg.linear_damping, dtype=torch.float32, device=device
        ).unsqueeze(0).repeat(num_envs, 1)

        self._quadratic_damping_diag = torch.tensor(
            cfg.quadratic_damping, dtype=torch.float32, device=device
        ).unsqueeze(0).repeat(num_envs, 1)

        # Buoyancy parameters
        self._volume = torch.full((num_envs,), cfg.volume, dtype=torch.float32, device=device)
        self._cob_offset = torch.full((num_envs,), cfg.center_of_buoyancy_offset, dtype=torch.float32, device=device)
        self._water_density = cfg.water_density
        self._gravity = 9.81

        # State buffers for acceleration filtering
        self._prev_body_vel = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._prev_body_acc = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._alpha = cfg.acceleration_filter_alpha

        # Ocean current state (world frame, 6-DOF)
        self._current_velocity = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self._max_current_vel = torch.tensor(
            self.current_cfg.max_velocity, dtype=torch.float32, device=device
        )
        self._current_noise_scale = torch.tensor(
            self.current_cfg.noise_scale, dtype=torch.float32, device=device
        )

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
        lin_vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        body_vel = torch.cat([lin_vel_b, ang_vel_b], dim=-1)  # (num_envs, 6)

        # Transform ocean current to body frame and compute relative velocity
        current_w = self._current_velocity + torch.randn_like(self._current_velocity) * self._current_noise_scale
        current_lin_b = quat_rotate_inverse(root_quat_w, current_w[:, :3])
        current_ang_b = quat_rotate_inverse(root_quat_w, current_w[:, 3:])
        current_b = torch.cat([current_lin_b, current_ang_b], dim=-1)

        # Relative velocity (vehicle velocity - current velocity)
        relative_vel = body_vel - current_b

        # Apply coordinate sign convention for underwater dynamics (Fossen convention)
        # Y and Z axes, and pitch/yaw rates have opposite sign
        relative_vel_fossen = relative_vel.clone()
        relative_vel_fossen[:, [1, 2, 4, 5]] *= -1

        # Compute Euler angles for buoyancy calculation
        rpy = self._quaternion_to_euler(root_quat_w)
        rpy_fossen = rpy.clone()
        rpy_fossen[:, [1, 2]] *= -1  # Pitch and yaw sign convention

        # Compute body acceleration (filtered)
        body_acc = self._compute_acceleration(relative_vel_fossen)

        # Compute individual hydrodynamic force components
        damping = self._compute_damping(relative_vel_fossen)
        added_mass = self._compute_added_mass(body_acc)
        coriolis = self._compute_coriolis(relative_vel_fossen)
        buoyancy = self._compute_buoyancy(rpy_fossen)

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
        as commonly observed in underwater vehicles.

        Args:
            body_vel: Body frame velocity. Shape: (num_envs, 6).

        Returns:
            Damping wrench. Shape: (num_envs, 6).
        """
        # Build velocity-dependent damping matrix with coupling terms
        # D(v) = D_linear + D_quadratic * |v|
        abs_vel = torch.abs(body_vel)

        # Diagonal damping
        damping_diag = self._linear_damping_diag + self._quadratic_damping_diag * abs_vel

        # Build full damping matrix with off-diagonal coupling
        # Coupling: sway-yaw (1,5), heave-pitch (2,4)
        damping_matrix = torch.diag_embed(damping_diag)

        # Add coupling terms (from MarineGym)
        damping_matrix[:, 1, 5] = damping_diag[:, 5]  # sway-yaw coupling
        damping_matrix[:, 2, 4] = damping_diag[:, 4]  # heave-pitch coupling
        damping_matrix[:, 4, 2] = damping_diag[:, 2]  # pitch-heave coupling
        damping_matrix[:, 5, 1] = damping_diag[:, 1]  # yaw-sway coupling

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
        """Compute Coriolis and centripetal forces.

        These forces arise from the coupling between the vehicle's linear and
        angular velocities through the added mass.

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
        coriolis_torque = -(
            torch.cross(ma_lin, lin_vel, dim=-1) +
            torch.cross(ma_ang, ang_vel, dim=-1)
        )

        return torch.cat([coriolis_force, coriolis_torque], dim=-1)

    def _compute_buoyancy(self, rpy: torch.Tensor) -> torch.Tensor:
        """Compute buoyancy force and restoring moment.

        The buoyancy force acts upward and the restoring moment tries to
        return the vehicle to a level orientation.

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
            # Sample random current velocity
            velocity = torch.rand(len(env_ids), 6, device=self.device) * self._max_current_vel * 2 - self._max_current_vel

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
