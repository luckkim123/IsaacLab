# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent underwater vehicle environments.

This module provides reinforcement learning environments for the Hero Agent UUV
using ALBC (Active Linear Buoyancy Controller) for attitude control via 2 revolute
joints that position a buoyancy element. No thrusters are used.

Available Tasks:
    - Isaac-HeroAgent-v0: Debug environment (minimal DR, no ocean current)
    - Isaac-HeroAgent-Base-v0: Base training with DR and ocean current
    - Isaac-HeroAgent-TDE-Base-v0: TDE-Base with dynamics mismatch obs
    - Isaac-HeroAgent-Encoder-Base-v0: Encoder training with privileged info
    - Isaac-HeroAgent-TDC-v0: Classical TDC control (no RL)
    - Isaac-HeroAgent-Adapt-Base-v0: Phase 2 adaptation (proprio history -> z_hat, base RL)

MPC environments are in the separate hero_agent_mpc package.
"""

import gymnasium as gym

from .adapt_base_env import HeroAgentAdaptBaseEnv
from .base_env import HeroAgentEnv
from .config import (
    DomainRandomizationCfg,
    HeroAgentAdaptBaseEnvCfg,
    HeroAgentEncoderTrainEnvCfg,
    HeroAgentEnvCfg,
    HeroAgentTDCEnvCfg,
    HeroAgentTDEBaseDebugEnvCfg,
    HeroAgentTDEBaseEnvCfg,
    HeroAgentTrainEnvCfg,
)
from .controllers import TDCControllerCfg
from .tdc_env import HeroAgentTDCEnv

##
# Register Gym environments
##

# Debug environment (no DR, no ocean current)
gym.register(
    id="Isaac-HeroAgent-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

# TDE-Base debug: TDE obs without DR (diagnostic experiment)
gym.register(
    id="Isaac-HeroAgent-TDE-Base-Debug-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTDEBaseDebugEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentTDEBasePPORunnerCfg",
    },
)

# Base training environment
gym.register(
    id="Isaac-HeroAgent-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

# TDE-Base: base RL with dynamics mismatch observation (15D obs)
gym.register(
    id="Isaac-HeroAgent-TDE-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTDEBaseEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentTDEBasePPORunnerCfg",
    },
)

# Encoder training environment
gym.register(
    id="Isaac-HeroAgent-Encoder-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEncoderPPORunnerCfg",
    },
)

# TDC controller environment (no RL, classical control)
gym.register(
    id="Isaac-HeroAgent-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

# Phase 2: Adaptation module (proprio history -> z_hat, base RL pipeline)
gym.register(
    id="Isaac-HeroAgent-Adapt-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentAdaptBaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentAdaptBaseEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentAdaptBaseRunnerCfg",
    },
)

__all__ = [
    # Environments
    "HeroAgentEnv",
    "HeroAgentTDCEnv",
    "HeroAgentAdaptBaseEnv",
    # Configurations
    "HeroAgentEnvCfg",
    "HeroAgentTrainEnvCfg",
    "HeroAgentTDEBaseDebugEnvCfg",
    "HeroAgentTDEBaseEnvCfg",
    "HeroAgentTDCEnvCfg",
    "HeroAgentEncoderTrainEnvCfg",
    "HeroAgentAdaptBaseEnvCfg",
    "DomainRandomizationCfg",
    "TDCControllerCfg",
]
