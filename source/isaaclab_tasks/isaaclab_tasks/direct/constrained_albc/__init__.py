# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained ALBC (Active Linear Buoyancy Controller) environment.

Standalone package for constrained RL training with C-TRPO and encoder.
Independent from hero_agent -- can be used without hero_agent installed.

Registered task:
    Isaac-Constrained-ALBC-Encoder-v0: C-TRPO + encoder constrained RL
"""

import gymnasium as gym

from .albc_env import ALBCEnv
from .config import (
    ALBCEnvCfg,
    ALBCEncoderTrainEnvCfg,
    ALBCTrainEnvCfg,
    ConstrainedALBCEncoderEnvCfg,
    DomainRandomizationCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Constrained-ALBC-Encoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ConstrainedALBCEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ConstrainedALBCEncoderRunnerCfg",
    },
)
