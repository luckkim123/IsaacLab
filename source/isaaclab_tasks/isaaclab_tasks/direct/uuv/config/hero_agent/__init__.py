# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent environment configurations for UUV tasks.

This module registers Hero Agent environments for hover and attitude control tasks.

Note:
    Hero Agent hydrodynamic parameters are estimated. For accurate simulation,
    they should be replaced with experimentally identified values.

Available Environments:
    Hover Task (position + attitude control):
        - Isaac-UUV-HeroAgent-Hover-v0: Base hover environment
        - Isaac-UUV-HeroAgent-Hover-Train-v0: Training with domain randomization
        - Isaac-UUV-HeroAgent-Hover-Eval-v0: Evaluation with aggressive randomization

    Attitude Task (attitude-only control):
        - Isaac-UUV-HeroAgent-Attitude-v0: Base attitude environment
        - Isaac-UUV-HeroAgent-Attitude-Train-v0: Training with domain randomization
"""

import gymnasium as gym

from . import agents
from .hover_env_cfg import (
    HeroAgentHoverEnvCfg,
    HeroAgentHoverTrainEnvCfg,
    HeroAgentHoverEvalEnvCfg,
)
from .attitude_env_cfg import (
    HeroAgentAttitudeEnvCfg,
    HeroAgentAttitudeTrainEnvCfg,
)

__all__ = [
    "HeroAgentHoverEnvCfg",
    "HeroAgentHoverTrainEnvCfg",
    "HeroAgentHoverEvalEnvCfg",
    "HeroAgentAttitudeEnvCfg",
    "HeroAgentAttitudeTrainEnvCfg",
]

##
# Register Gymnasium environments
##

# Hover Task (position + attitude control)
gym.register(
    id="Isaac-UUV-HeroAgent-Hover-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:HeroAgentHoverEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroAgentHoverPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-UUV-HeroAgent-Hover-Train-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:HeroAgentHoverTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroAgentHoverTrainPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Hover Evaluation (aggressive randomization)
gym.register(
    id="Isaac-UUV-HeroAgent-Hover-Eval-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:HeroAgentHoverEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroAgentHoverEvalPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Attitude Task (attitude-only control)
gym.register(
    id="Isaac-UUV-HeroAgent-Attitude-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attitude_env_cfg:HeroAgentAttitudeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroAgentAttitudePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Attitude Training (with domain randomization)
gym.register(
    id="Isaac-UUV-HeroAgent-Attitude-Train-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attitude_env_cfg:HeroAgentAttitudeTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroAgentAttitudeTrainPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
