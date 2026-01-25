# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent environments.

This module provides pre-configured hyperparameters for training Hero Agent
control policies for hover and attitude tasks.
"""

from .rsl_rl_ppo_cfg import (
    HeroAgentHoverPPORunnerCfg,
    HeroAgentHoverTrainPPORunnerCfg,
    HeroAgentHoverEvalPPORunnerCfg,
    HeroAgentAttitudePPORunnerCfg,
    HeroAgentAttitudeTrainPPORunnerCfg,
)

__all__ = [
    "HeroAgentHoverPPORunnerCfg",
    "HeroAgentHoverTrainPPORunnerCfg",
    "HeroAgentHoverEvalPPORunnerCfg",
    "HeroAgentAttitudePPORunnerCfg",
    "HeroAgentAttitudeTrainPPORunnerCfg",
]
