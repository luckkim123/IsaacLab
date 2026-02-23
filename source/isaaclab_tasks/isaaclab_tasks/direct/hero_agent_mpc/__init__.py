# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent SAC-MPC environment: differentiable MPC for UUV attitude control.

This module provides a SAC-based off-policy RL environment using AC-MPC
(Actor-Critic MPC) architecture with learned residual dynamics.

Available Tasks:
    - Isaac-HeroAgent-SAC-MPC-v0: SAC-MPC with ensemble dynamics + error feedback
"""

import gymnasium as gym

from .config import HeroAgentMPCEnvCfg
from .mpc_env import HeroAgentMPCEnv

##
# Register Gym environments
##

gym.register(
    id="Isaac-HeroAgent-SAC-MPC-v0",
    entry_point="isaaclab_tasks.direct.hero_agent_mpc:HeroAgentMPCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.hero_agent_mpc:HeroAgentMPCEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.hero_agent_mpc.agents:HeroAgentSACMPCRunnerCfg",
    },
)

__all__ = [
    "HeroAgentMPCEnv",
    "HeroAgentMPCEnvCfg",
]
