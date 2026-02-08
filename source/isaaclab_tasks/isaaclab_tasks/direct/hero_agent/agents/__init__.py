# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides:
    - ActorCriticEncoder: Custom network for HORA Phase 1 teacher training
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
"""

from ..encoder import ActorCriticEncoder
from .rsl_rl_ppo_cfg import (
    HeroAgentEncoderPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    RslRlPpoActorCriticEncoderCfg,
)

__all__ = [
    "ActorCriticEncoder",
    "HeroAgentPPORunnerCfg",
    "HeroAgentEncoderPPORunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
]
