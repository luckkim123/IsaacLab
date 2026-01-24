# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UUV (Underwater Vehicle) environments for Isaac Lab.

This package provides environments for training underwater vehicles with
full 6-DOF hydrodynamics based on the Fossen model.

Available Environments:
    - Isaac-UUV-BlueROV-v0: BlueROV2 hover task (no randomization)
    - Isaac-UUV-BlueROV-Current-v0: BlueROV2 hover task with ocean currents
    - Isaac-UUV-BlueROV-Train-v0: BlueROV2 training environment (no randomization)
    - Isaac-UUV-BlueROV-Eval-v0: BlueROV2 evaluation environment (full randomization)
    - Isaac-UUV-BlueROV-v2: BlueROV2 hover task with modular reward/task system

Modular Components (V2):
    - tasks: Task abstraction layer (HoverTask, etc.)
    - rewards: Composition-based reward terms (RewardManager, RewardTermCfg)
    - thrusters: Thruster model interface (ThrusterModel, ThrusterModelCfg)
"""

import gymnasium as gym

from . import agents

# BlueROV configurations
from .bluerov_cfg import (
    BLUEROV_CFG,
    BlueROVCurrentEnvCfg,
    BlueROVEnvCfg,
    BlueROVEvalEnvCfg,
    BlueROVHydrodynamicsCfg,
    BlueROVTrainEnvCfg,
)

# Hydrodynamics
from .hydrodynamics_model import HydrodynamicsCfg, HydrodynamicsModel, OceanCurrentCfg

# Modular components
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
from .tasks import HoverTask, HoverTaskCfg, TaskBase, TaskBaseCfg
from .thrusters import ThrusterModel, ThrusterModelCfg

# Original environment (V1)
from .uuv_env import UUVEnv
from .uuv_env_cfg import DomainRandomizationCfg, ThrusterCfg, UUVEnvCfg

# Modular environment (V2)
from .uuv_env_v2 import UUVEnvV2

__all__ = [
    # Environment (V1 - backward compatible)
    "UUVEnv",
    # Environment (V2 - modular)
    "UUVEnvV2",
    # Configurations
    "UUVEnvCfg",
    "BlueROVEnvCfg",
    "BlueROVCurrentEnvCfg",
    "BlueROVTrainEnvCfg",
    "BlueROVEvalEnvCfg",
    "ThrusterCfg",
    "DomainRandomizationCfg",
    # Hydrodynamics
    "HydrodynamicsModel",
    "HydrodynamicsCfg",
    "OceanCurrentCfg",
    "BlueROVHydrodynamicsCfg",
    # Tasks (V2)
    "TaskBase",
    "TaskBaseCfg",
    "HoverTask",
    "HoverTaskCfg",
    # Rewards (V2)
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
    # Thrusters (V2)
    "ThrusterModel",
    "ThrusterModelCfg",
    # Assets
    "BLUEROV_CFG",
]

##
# Register Gymnasium environments
##

gym.register(
    id="Isaac-UUV-BlueROV-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Current-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVCurrentEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVCurrentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Train-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVTrainPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Eval-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVEvalPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# V2 Environments (modular task/reward system)
##

gym.register(
    id="Isaac-UUV-BlueROV-v2",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
