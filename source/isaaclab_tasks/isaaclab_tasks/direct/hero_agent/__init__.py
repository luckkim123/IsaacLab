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
    - Isaac-HeroAgent-Encoder-Base-v0: Encoder training with privileged info
    - Isaac-HeroAgent-Encoder-TDC-v0: TDC-integrated training with gains output
    - Isaac-HeroAgent-Base-TDC-v0: TDC gain-only training (no encoder, fixed M_hat)
"""

import gymnasium as gym

from .hero_agent_env import HeroAgentEnv
from .hero_agent_env_cfg import (
    DomainRandomizationCfg,
    HeroAgentBaseTDCEnvCfg,
    HeroAgentEncoderTDCEnvCfg,
    HeroAgentEncoderTrainEnvCfg,
    HeroAgentEnvCfg,
    HeroAgentTrainEnvCfg,
    TDCRewardCfg,
)
from .hero_agent_tdc_env import HeroAgentTDCEnv

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

# Base training environment (renamed from Isaac-HeroAgent-Train-v0)
gym.register(
    id="Isaac-HeroAgent-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

# Encoder training environment (renamed from Isaac-HeroAgent-Encoder-v0)
gym.register(
    id="Isaac-HeroAgent-Encoder-Base-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEncoderPPORunnerCfg",
    },
)

# TDC-integrated training environment (with encoder)
gym.register(
    id="Isaac-HeroAgent-Encoder-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEncoderTDCPPORunnerCfg",
    },
)

# TDC gain-only training (no encoder, fixed M_hat)
gym.register(
    id="Isaac-HeroAgent-Base-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentBaseTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentBaseTDCPPORunnerCfg",
    },
)

# Legacy aliases for backward compatibility
gym.register(
    id="Isaac-HeroAgent-Train-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-HeroAgent-Encoder-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEncoderPPORunnerCfg",
    },
)

__all__ = [
    # Environments
    "HeroAgentEnv",
    "HeroAgentTDCEnv",
    # Configurations
    "HeroAgentEnvCfg",
    "HeroAgentTrainEnvCfg",
    "HeroAgentEncoderTrainEnvCfg",
    "HeroAgentEncoderTDCEnvCfg",
    "HeroAgentBaseTDCEnvCfg",
    "DomainRandomizationCfg",
    "TDCRewardCfg",
]
