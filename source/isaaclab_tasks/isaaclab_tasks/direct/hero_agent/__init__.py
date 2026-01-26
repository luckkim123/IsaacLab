# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent underwater vehicle environments.

This module provides reinforcement learning environments for the Hero Agent UUV
using ALBC (Active Linear Buoyancy Controller) for attitude control via 2 revolute
joints that position a buoyancy element. No thrusters are used.

Available Tasks:
    - ALBC Attitude: Joint-based attitude control
"""

import gymnasium as gym

from .hero_agent_env import HeroAgentEnv
from .hero_agent_env_cfg import (
    DomainRandomizationCfg,
    HeroAgentEnvCfg,
    HeroAgentTrainEnvCfg,
    HeroAgentEvalEnvCfg,
)

##
# Register Gym environments
##

# Hero Agent ALBC environments
gym.register(
    id="Isaac-HeroAgent-ALBC-Direct-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-HeroAgent-ALBC-Train-Direct-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentTrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-HeroAgent-ALBC-Eval-Direct-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEvalEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEvalPPORunnerCfg",
    },
)

__all__ = [
    # Environment
    "HeroAgentEnv",
    # Configurations
    "HeroAgentEnvCfg",
    "HeroAgentTrainEnvCfg",
    "HeroAgentEvalEnvCfg",
    "DomainRandomizationCfg",
]
