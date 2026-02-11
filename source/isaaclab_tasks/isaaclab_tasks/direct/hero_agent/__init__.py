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
    - Isaac-HeroAgent-TDC-v0: Classical TDC control (no RL)
    - Isaac-HeroAgent-Encoder-TDC-v0: Encoder-TDC integration (RL adaptive gains + M_hat)
    - Isaac-HeroAgent-Constrained-TDC-v0: Constrained RL with NORBC IPO
    - Isaac-HeroAgent-Adapt-TDC-v0: Phase 2 adaptation (proprio history -> z_hat)
"""

import gymnasium as gym

from .adapt_tdc_env import HeroAgentAdaptTDCEnv
from .base_env import HeroAgentEnv
from .config import (
    DomainRandomizationCfg,
    HeroAgentAdaptTDCEnvCfg,
    HeroAgentConstrainedEncoderTDCEnvCfg,
    HeroAgentEncoderTDCEnvCfg,
    HeroAgentEncoderTrainEnvCfg,
    HeroAgentEnvCfg,
    HeroAgentTDCEnvCfg,
    HeroAgentTrainEnvCfg,
)
from .constrained_encoder_tdc_env import HeroAgentConstrainedEncoderTDCEnv
from .controllers import TDCControllerCfg
from .encoder_tdc_env import HeroAgentEncoderTDCEnv
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

# Encoder-TDC integration (RL adaptive gains + encoder M_hat)
gym.register(
    id="Isaac-HeroAgent-Encoder-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentEncoderTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentEncoderTDCPPORunnerCfg",
    },
)

# Constrained RL: Encoder-TDC with NORBC IPO (log-barrier constraints)
gym.register(
    id="Isaac-HeroAgent-Constrained-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentConstrainedEncoderTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentConstrainedEncoderTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentConstrainedEncoderTDCPPORunnerCfg",
    },
)

# Phase 2: Adaptation module (proprio history -> z_hat, supervised training)
gym.register(
    id="Isaac-HeroAgent-Adapt-TDC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent:HeroAgentAdaptTDCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent:HeroAgentAdaptTDCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent.agents:HeroAgentAdaptTDCRunnerCfg",
    },
)

__all__ = [
    # Environments
    "HeroAgentEnv",
    "HeroAgentTDCEnv",
    "HeroAgentEncoderTDCEnv",
    "HeroAgentConstrainedEncoderTDCEnv",
    "HeroAgentAdaptTDCEnv",
    # Configurations
    "HeroAgentEnvCfg",
    "HeroAgentTrainEnvCfg",
    "HeroAgentTDCEnvCfg",
    "HeroAgentEncoderTrainEnvCfg",
    "HeroAgentEncoderTDCEnvCfg",
    "HeroAgentConstrainedEncoderTDCEnvCfg",
    "HeroAgentAdaptTDCEnvCfg",
    "DomainRandomizationCfg",
    "TDCControllerCfg",
]
