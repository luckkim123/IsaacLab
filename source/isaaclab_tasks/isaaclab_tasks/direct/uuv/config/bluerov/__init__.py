# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV2 environment configurations for UUV tasks.

This module registers BlueROV2 environments for hover and attitude control tasks.

Available Environments:
    - Isaac-UUV-BlueROV-Hover-v0: Hover task (position + attitude control)
    - Isaac-UUV-BlueROV-Hover-Train-v0: Hover task with domain randomization
    - Isaac-UUV-BlueROV-Hover-Eval-v0: Hover task for evaluation
    - Isaac-UUV-BlueROV-Attitude-v0: Attitude-only control task
"""

import gymnasium as gym

from . import agents
from .hover_env_cfg import (
    BlueROVHoverEnvCfg,
    BlueROVHoverTrainEnvCfg,
    BlueROVHoverEvalEnvCfg,
)
from .attitude_env_cfg import (
    BlueROVAttitudeEnvCfg,
    BlueROVAttitudeTrainEnvCfg,
)

__all__ = [
    "BlueROVHoverEnvCfg",
    "BlueROVHoverTrainEnvCfg",
    "BlueROVHoverEvalEnvCfg",
    "BlueROVAttitudeEnvCfg",
    "BlueROVAttitudeTrainEnvCfg",
]

##
# Register Gymnasium environments
##

# Hover Task (position + attitude control)
gym.register(
    id="Isaac-UUV-BlueROV-Hover-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:BlueROVHoverEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVHoverPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Hover-Train-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:BlueROVHoverTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVHoverTrainPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Hover-Eval-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:BlueROVHoverEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVHoverEvalPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Attitude Task (attitude-only control)
gym.register(
    id="Isaac-UUV-BlueROV-Attitude-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attitude_env_cfg:BlueROVAttitudeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVAttitudePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-BlueROV-Attitude-Train-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attitude_env_cfg:BlueROVAttitudeTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVAttitudeTrainPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
