# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration classes for underwater vehicles.

This module defines the configuration dataclasses for UUV hydrodynamics,
ocean currents, and thruster systems. These are used as base classes
for robot-specific configurations.

Reference:
    Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class HydrodynamicsCfg:
    """Configuration for Fossen model hydrodynamics.

    All coefficients are for a 6-DOF system: [surge, sway, heave, roll, pitch, yaw].
    Diagonal matrices are assumed for simplicity (off-diagonal terms can be added later).

    The model implements the complete Fossen formulation including:
        - Weight-buoyancy difference (W-B) for non-neutral buoyancy vehicles
        - Full Coriolis matrix C(v) = C_RB(v) + C_A(v)
        - Quaternion-based buoyancy calculation (no gimbal lock)
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
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)

    # Center of gravity position in body frame (m, [x, y, z])
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Water density (kg/m^3, default: freshwater, seawater: 1025.0)
    water_density: float = 997.0

    # Acceleration filter alpha for numerical stability (0 < alpha < 1)
    # Higher values (closer to 1) provide stronger low-pass filtering, reducing noise-induced instability
    acceleration_filter_alpha: float = 0.8

    # Use full Coriolis matrix C(v) = C_RB(v) + C_A(v) per Fossen model
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


@configclass
class ThrusterCfg:
    """Configuration for thruster system.

    Defines the thruster allocation matrix and dynamics for converting
    normalized commands to body-frame forces and torques.
    """

    # Number of thrusters
    num_thrusters: int = 6

    # Maximum thrust per thruster (N)
    max_thrust: float = 50.0

    # Thrust coefficient: thrust = coefficient * command
    thrust_coefficient: float = 40.0

    # First-order dynamics time constants (seconds)
    time_constant_up: float = 0.1
    time_constant_down: float = 0.05

    # Thruster allocation matrix: wrench = allocation_matrix @ thrusts
    # Shape: (6, num_thrusters) where 6 = [Fx, Fy, Fz, Mx, My, Mz]
    # Default is BlueROV2 Heavy configuration with 6 thrusters
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        # Fx: forward surge force
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        # Fy: lateral sway force
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        # Fz: vertical heave force
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        # Mx: roll torque
        (0.0, 0.0, 0.0, 0.0, 0.1, -0.1),
        # My: pitch torque
        (0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
        # Mz: yaw torque
        (0.19, -0.19, 0.19, -0.19, 0.0, 0.0),
    )
