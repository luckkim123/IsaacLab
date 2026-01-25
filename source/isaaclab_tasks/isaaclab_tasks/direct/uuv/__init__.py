# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UUV (Underwater Vehicle) environments for Isaac Lab.

This package provides environments for training underwater vehicles with
full 6-DOF hydrodynamics based on the Fossen model.

Available Environments:
    BlueROV2:
        - Isaac-UUV-BlueROV-Hover-v0: Hover task (position + attitude)
        - Isaac-UUV-BlueROV-Hover-Train-v0: Hover with domain randomization
        - Isaac-UUV-BlueROV-Hover-Eval-v0: Hover evaluation
        - Isaac-UUV-BlueROV-Attitude-v0: Attitude-only control
        - Isaac-UUV-BlueROV-Attitude-Train-v0: Attitude with domain randomization

    Hero Agent:
        - Isaac-UUV-HeroAgent-Hover-v0: Hover task (position + attitude)
        - Isaac-UUV-HeroAgent-Hover-Train-v0: Hover with domain randomization
        - Isaac-UUV-HeroAgent-Hover-Eval-v0: Hover evaluation
        - Isaac-UUV-HeroAgent-Attitude-v0: Attitude-only control
        - Isaac-UUV-HeroAgent-Attitude-Train-v0: Attitude with domain randomization

Modular Components:
    - tasks: Task abstraction layer (HoverTask, AttitudeTask)
    - models: Physical models (HydrodynamicsModel, ThrusterModel)
    - rewards: Composition-based reward terms (RewardManager, RewardTermCfg)
    - config: Robot-specific environment configurations
"""

from . import config  # Import config to trigger gym.register() for all environments

# Physical models (from models/)
from .models import (
    HydrodynamicsModel,
    HydrodynamicsCfg,
    OceanCurrentCfg,
    ThrusterModel,
    ThrusterCfg,
)

# Modular reward system
from .rewards import (
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    alive_bonus,
    angular_velocity_penalty,
    linear_velocity_penalty,
    orientation_upright,
    orientation_upright_exp,
    position_tracking_exp,
)

# Task abstraction
from .tasks import AttitudeTask, AttitudeTaskCfg, HoverTask, HoverTaskCfg, TaskBase, TaskBaseCfg

# MDP event functions for domain randomization
from .mdp import (
    randomize_hydrodynamics,
    randomize_ocean_current,
    randomize_robot_pose,
    randomize_thruster_params,
)

# Environment configurations
from .uuv_env_cfg import DomainRandomizationCfg, EventCfg, UUVEnvCfg

# Environment class
from .uuv_env_v2 import UUVEnvV2

# Re-export robot configurations from isaaclab_assets for convenience
from isaaclab_assets.robots.uuv import (
    BLUEROV_CFG,
    BlueROVHydrodynamicsCfg,
    BlueROVThrusterCfg,
    HERO_AGENT_CFG,
    HeroAgentHydrodynamicsCfg,
    HeroAgentThrusterCfg,
)

__all__ = [
    # Environment
    "UUVEnvV2",
    # Configurations
    "UUVEnvCfg",
    "DomainRandomizationCfg",
    "EventCfg",
    # MDP event functions
    "randomize_hydrodynamics",
    "randomize_thruster_params",
    "randomize_ocean_current",
    "randomize_robot_pose",
    # Physical models
    "HydrodynamicsModel",
    "HydrodynamicsCfg",
    "OceanCurrentCfg",
    "ThrusterModel",
    "ThrusterCfg",
    # Tasks
    "TaskBase",
    "TaskBaseCfg",
    "HoverTask",
    "HoverTaskCfg",
    "AttitudeTask",
    "AttitudeTaskCfg",
    # Rewards
    "RewardManager",
    "RewardTermCfg",
    "position_tracking_exp",
    "orientation_upright",
    "orientation_upright_exp",
    "linear_velocity_penalty",
    "angular_velocity_penalty",
    "action_rate_penalty",
    "action_magnitude_penalty",
    "alive_bonus",
    # Robot assets (from isaaclab_assets)
    "BLUEROV_CFG",
    "BlueROVHydrodynamicsCfg",
    "BlueROVThrusterCfg",
    "HERO_AGENT_CFG",
    "HeroAgentHydrodynamicsCfg",
    "HeroAgentThrusterCfg",
]
