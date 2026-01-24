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
"""

import gymnasium as gym

from . import agents
from .hydrodynamics_model import HydrodynamicsCfg, HydrodynamicsModel, OceanCurrentCfg
from .uuv_env import UUVEnv
from .uuv_env_cfg import UUVEnvCfg, ThrusterCfg, DomainRandomizationCfg
from .bluerov_cfg import (
    BLUEROV_CFG,
    BlueROVEnvCfg,
    BlueROVCurrentEnvCfg,
    BlueROVTrainEnvCfg,
    BlueROVEvalEnvCfg,
    BlueROVHydrodynamicsCfg,
)

__all__ = [
    # Environment
    "UUVEnv",
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
