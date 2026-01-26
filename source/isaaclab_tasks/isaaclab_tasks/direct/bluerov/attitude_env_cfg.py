# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV attitude-only environment configurations.

This module provides environment configurations for BlueROV attitude control tasks
where the vehicle must maintain a target orientation without position control.
"""

from __future__ import annotations

from isaaclab.utils import configclass

# Import robot configurations from isaaclab_assets
from isaaclab_assets.robots.uuv import (
    BLUEROV_CFG,
    BlueROVHydrodynamicsCfg,
    BlueROVThrusterCfg,
    OceanCurrentCfg,
)

from .bluerov_env_cfg import BlueROVEnvCfg, DomainRandomizationCfg
from .tasks import AttitudeTaskCfg


@configclass
class BlueROVAttitudeEnvCfg(BlueROVEnvCfg):
    """BlueROV attitude control environment configuration.

    The vehicle must achieve and maintain a target orientation.
    Position is not controlled - useful for:
    - Testing attitude control decoupled from position
    - Pre-training attitude stabilization
    - Camera pointing or sensor orientation tasks
    """

    # Robot configuration from isaaclab_assets
    robot = BLUEROV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Hydrodynamics (BlueROV-specific parameters)
    hydrodynamics = BlueROVHydrodynamicsCfg()

    # Thrusters (BlueROV has 6 thrusters)
    thrusters = BlueROVThrusterCfg()

    # Action space matches number of thrusters
    action_space: int = 6

    # Observation space: quat(4) + ang_vel(3) + attitude_error(3) + up(2) = 12
    # Simpler than hover since no position tracking
    observation_space: int = 12

    # Task configuration for attitude control
    task: AttitudeTaskCfg = AttitudeTaskCfg(
        target_attitude_range=(0.5, 0.5, 3.14159),
        base_attitude=(0.0, 0.0, 0.0),
        randomize_target=True,
    )

    # Ocean current (disabled by default)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    # Reward scales for attitude task (no position reward)
    position_reward_scale: float = 0.0  # Disabled for attitude-only
    position_reward_exp_scale: float = 1.0
    orientation_reward_scale: float = 15.0  # Primary objective
    orientation_exp_scale: float = 0.3  # Tighter tolerance
    linear_velocity_penalty_scale: float = -0.005  # Light penalty
    angular_velocity_penalty_scale: float = -0.01  # Moderate penalty
    action_rate_penalty_scale: float = -0.005
    action_magnitude_penalty_scale: float = -0.001
    alive_reward_scale: float = 0.1


@configclass
class BlueROVAttitudeTrainEnvCfg(BlueROVAttitudeEnvCfg):
    """BlueROV attitude training environment with domain randomization.

    Enables domain randomization for robust attitude control policy learning.
    """

    # Enable domain randomization
    randomization = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization (attitude focus)
        position_x_range=(-1.0, 1.0),
        position_y_range=(-1.0, 1.0),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.785, 0.785),  # +/-45 degrees
        pitch_range=(-0.785, 0.785),
        yaw_range=(0.0, 6.283),
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        mass_scale=(0.9, 1.1),
        # Thruster randomization
        thrust_coefficient_scale=(0.9, 1.1),
        time_constant_scale=(0.9, 1.1),
        # Center of Buoyancy and Inertia randomization
        cob_offset_scale=(0.8, 1.2),
        inertia_scale=(0.9, 1.1),
        # Payload randomization
        payload_mass_ratio=(0.0, 0.1),
        payload_cog_offset_z=(-0.02, 0.02),
    )

    # Light ocean currents (attitude is affected by currents too)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
    )
