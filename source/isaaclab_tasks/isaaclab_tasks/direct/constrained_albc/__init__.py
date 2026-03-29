# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained ALBC (Active Linear Buoyancy Controller) environment.

Standalone package for constrained RL training with TRPO + IPO and encoder.
Independent from hero_agent -- can be used without hero_agent installed.

Registered tasks (ablation ladder):
    Isaac-Constrained-ALBC-Debug-v0:             Step 0  - Pure PPO (no DR/encoder/constraints)
    Isaac-Constrained-ALBC-Debug-DR-v0:          Step 1  - PPO + DR (no encoder/constraints)
    Isaac-Constrained-ALBC-Debug-TRPO-v0:        Step 2  - TRPO + DR (no encoder/barriers)
    Isaac-Constrained-ALBC-Debug-Barrier-v0:     Step 3  - TRPO + IPO + DR (no encoder)
    Isaac-Constrained-ALBC-Debug-Encoder-v0:     Step 4  - TRPO + Encoder + DR (no constraints)
    Isaac-Constrained-ALBC-Debug-PPO-Encoder-v0: Step 4b - PPO + Encoder + DR (no constraints)
    Isaac-Constrained-ALBC-Debug-PPO-Enc-Hist-v0:Step 4c - PPO + Encoder + History + DR (no constraints)
    Isaac-Constrained-ALBC-Debug-PPO-Hist-Only-v0:Step 4d - PPO + History + DR (no encoder)
    Isaac-Constrained-ALBC-Debug-PPO-SB-v0:      Step 5a - PPO + Shared Backbone + DR (Fix A+B)
    Isaac-Constrained-ALBC-Debug-PPO-SB-Hist-v0: Step 5b - PPO + Shared Backbone + History + DR
    Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-v0:    Step 13b - PPO + Static MinMax Norm + DR
    Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-RS-v0: Step 13a - PPO + Static MinMax Norm + reward_scale=0.01
    Isaac-Constrained-ALBC-Encoder-v0:           Full    - TRPO + IPO + Encoder + DR
"""

import gymnasium as gym

from .albc_env import ALBCEnv
from .config import (
    ALBCDebugBarrierEnvCfg,
    ALBCDebugDREnvCfg,
    ALBCDebugEncoderEnvCfg,
    ALBCDebugEncoderHistEnvCfg,
    ALBCDebugEncoderHistStrideEnvCfg,
    ALBCDebugEnvCfg,
    ALBCDebugHistOnlyEnvCfg,
    ALBCEnvCfg,
    DomainRandomizationCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Constrained-ALBC-Debug-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-DR-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugDRRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-Encoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugEncoderRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-Barrier-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugBarrierEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugBarrierRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Encoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOEncoderRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Enc-Hist-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOEncoderHistRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Hist-Only-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugHistOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOHistOnlyRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-TRPO-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugTRPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-SB-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOSharedBackboneRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-SB-Hist-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOSharedBackboneHistRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-SmallEnc-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOSmallEncHistRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Sep-Enc-Hist-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOSeparateEncHistRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOQ1Q3RunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-NoEncNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOQ1Q3NoEncNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Q4-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOQ4RunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-Q4-NoEncNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOQ4NoEncNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-EncScale-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOEncScaleRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-EncScale-NoEncNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOEncScaleNoEncNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-StdUpdate-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOStdUpdateRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-StdUpdate-NoEncNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOStdUpdateNoEncNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-RewardScale-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPORewardScaleRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-RewardScale-NoEncNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPORewardScaleNoEncNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOStaticNormRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-EncFreeze-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOEncFreezeRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-RS-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOStaticNormRSRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-NoClamp-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPONoClampRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-ScalarStd-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOScalarStdRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-SymCritic-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOSymCriticRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-ScalarStdNoClamp-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOScalarStdNoClampRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Debug-PPO-HoraAligned-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCDebugEncoderHistStrideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ALBCDebugPPOHoraAlignedRunnerCfg",
    },
)

gym.register(
    id="Isaac-Constrained-ALBC-Encoder-v0",
    entry_point="isaaclab_tasks.direct.constrained_albc:ALBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config:ALBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ConstrainedALBCEncoderRunnerCfg",
    },
)
