# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides pre-configured hyperparameters for training Hero Agent
joint-based attitude control policies.
"""

from .rsl_rl_ppo_cfg import (
    HeroAgentEncoderEvalPPORunnerCfg,
    HeroAgentEncoderTrainPPORunnerCfg,
    HeroAgentEvalPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    HeroAgentTrainPPORunnerCfg,
    RslRlPpoActorCriticEncoderCfg,
)

__all__ = [
    "HeroAgentPPORunnerCfg",
    "HeroAgentTrainPPORunnerCfg",
    "HeroAgentEvalPPORunnerCfg",
    "HeroAgentEncoderTrainPPORunnerCfg",
    "HeroAgentEncoderEvalPPORunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
]
