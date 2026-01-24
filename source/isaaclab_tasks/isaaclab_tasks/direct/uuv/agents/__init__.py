# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for UUV environments.

This module provides pre-configured hyperparameters for training underwater
vehicle control policies using various RL libraries:

- RSL-RL (rsl_rl_ppo_cfg.py): PPO configurations for RSL-RL library
- RL-Games (rl_games_ppo_cfg.yaml): YAML configuration for RL-Games
- SKRL (skrl_ppo_cfg.yaml): YAML configuration for SKRL library

Available configurations:
    - BlueROVPPORunnerCfg: Base configuration for BlueROV hover task
    - BlueROVTrainPPORunnerCfg: Training environment (no randomization)
    - BlueROVEvalPPORunnerCfg: Evaluation environment (full randomization)
    - BlueROVCurrentPPORunnerCfg: Environment with ocean currents
"""

from .rsl_rl_ppo_cfg import (
    BlueROVPPORunnerCfg,
    BlueROVTrainPPORunnerCfg,
    BlueROVEvalPPORunnerCfg,
    BlueROVCurrentPPORunnerCfg,
)

__all__ = [
    "BlueROVPPORunnerCfg",
    "BlueROVTrainPPORunnerCfg",
    "BlueROVEvalPPORunnerCfg",
    "BlueROVCurrentPPORunnerCfg",
]
