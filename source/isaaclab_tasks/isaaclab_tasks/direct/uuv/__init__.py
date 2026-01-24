# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UUV (Underwater Vehicle) environments for Isaac Lab.

This package provides environments for training underwater vehicles with
full 6-DOF hydrodynamics based on the Fossen model.

Available Environments:
    - Isaac-UUV-BlueROV-v0: BlueROV2 hover task without currents
    - Isaac-UUV-BlueROV-Current-v0: BlueROV2 hover task with ocean currents
"""

import gymnasium as gym

from . import agents
from .hydrodynamics_model import HydrodynamicsCfg, HydrodynamicsModel, OceanCurrentCfg
from .uuv_env import UUVEnv
from .uuv_env_cfg import UUVEnvCfg, ThrusterCfg, BlueROVHydrodynamicsCfg
from .bluerov_cfg import BLUEROV_CFG, BlueROVEnvCfg, BlueROVCurrentEnvCfg

__all__ = [
    # Environment
    "UUVEnv",
    # Configurations
    "UUVEnvCfg",
    "BlueROVEnvCfg",
    "BlueROVCurrentEnvCfg",
    "ThrusterCfg",
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
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Current-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:BlueROVCurrentEnvCfg",
    },
)
