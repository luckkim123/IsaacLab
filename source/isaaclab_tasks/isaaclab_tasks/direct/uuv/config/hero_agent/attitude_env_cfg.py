# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent attitude-only environment configurations.

This module provides environment configurations for Hero Agent attitude control tasks
where the vehicle must maintain a target orientation without position control.
"""

from __future__ import annotations

from isaaclab.utils import configclass

# Import robot configurations from isaaclab_assets
from isaaclab_assets.robots.uuv import (
    HERO_AGENT_CFG,
    HeroAgentHydrodynamicsCfg,
    HeroAgentThrusterCfg,
    OceanCurrentCfg,
)

# Import base environment configuration
from isaaclab_tasks.direct.uuv.uuv_env_cfg import (
    DomainRandomizationCfg,
    UUVEnvCfg,
)

# Import task configuration
from isaaclab_tasks.direct.uuv.tasks import AttitudeTaskCfg


@configclass
class HeroAgentAttitudeEnvCfg(UUVEnvCfg):
    """Hero Agent attitude control environment configuration.

    The vehicle must achieve and maintain a target orientation.
    Position is not controlled.
    """

    # Robot configuration from isaaclab_assets
    robot = HERO_AGENT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Hydrodynamics (Hero Agent-specific parameters)
    hydrodynamics = HeroAgentHydrodynamicsCfg()

    # Thrusters (Hero Agent has 6 thrusters)
    thrusters = HeroAgentThrusterCfg()

    # Action space matches number of thrusters
    action_space: int = 6

    # Observation space (simpler for attitude-only)
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
    position_reward_scale: float = 0.0
    position_reward_exp_scale: float = 1.0
    orientation_reward_scale: float = 15.0
    orientation_exp_scale: float = 0.3
    linear_velocity_penalty_scale: float = -0.005
    angular_velocity_penalty_scale: float = -0.01
    action_rate_penalty_scale: float = -0.005
    action_magnitude_penalty_scale: float = -0.001
    alive_reward_scale: float = 0.1


@configclass
class HeroAgentAttitudeTrainEnvCfg(HeroAgentAttitudeEnvCfg):
    """Hero Agent attitude training environment with domain randomization.

    Enables domain randomization for robust attitude control policy learning.
    """

    # Enable domain randomization
    randomization = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization (attitude focus)
        position_x_range=(-1.0, 1.0),
        position_y_range=(-1.0, 1.0),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.785, 0.785),  # +-45 degrees
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
    )

    # Light ocean currents (attitude is affected by currents too)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
    )
