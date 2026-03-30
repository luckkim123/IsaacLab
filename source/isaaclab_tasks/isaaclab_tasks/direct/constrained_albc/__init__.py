# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained ALBC (Active Linear Buoyancy Controller) environment.

Standalone package for constrained RL training with TRPO + IPO and encoder.
Independent from hero_agent -- can be used without hero_agent installed.

Registered tasks:
    Isaac-Constrained-ALBC-Encoder-v0:          TRPO + IPO + Encoder (production)
    Isaac-Constrained-ALBC-HardDR-HistOnly-v0:  History-only baseline (hard DR)
    Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0: Frozen encoder fine-tuning (hard DR)
    Isaac-Constrained-ALBC-HardDR-SharedBackbone-v0: Shared backbone encoder (online E2E PPO)
    Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-v0: Asymmetric critic with z (online E2E PPO)
    Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-TRPO-v0: Asymmetric + TRPO + IPO (constrained RL)
"""

import gymnasium as gym

from .albc_env import ALBCEnv
from .config import (
    ALBCEnvCfg,
    ALBCHardDRAsymmetricEncoderConstrainedEnvCfg,
    ALBCHardDRAsymmetricEncoderEnvCfg,
    ALBCHardDRFrozenEncoderEnvCfg,
    ALBCHardDRHistOnlyEnvCfg,
    ALBCHardDRSharedBackboneEnvCfg,
    DomainRandomizationCfg,
    HardDomainRandomizationCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Constrained-ALBC-Encoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ConstrainedALBCEncoderRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-HardDR-HistOnly-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCHardDRHistOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCHardDRHistOnlyRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCHardDRFrozenEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCHardDRFrozenEncoderRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-HardDR-SharedBackbone-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCHardDRSharedBackboneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCHardDRSharedBackboneRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCHardDRAsymmetricEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCHardDRAsymmetricEncoderRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-TRPO-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCHardDRAsymmetricEncoderConstrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCHardDRAsymmetricEncoderTRPORunnerCfg",
    },
)
