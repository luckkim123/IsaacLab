# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides:
    - ActorCriticEncoder: Custom network for HORA Phase 1 teacher training
    - ActorCriticEncoderTDC: TDC-specific network outputting PD gains
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentEncoderTDCPPORunnerCfg: TDC-integrated training with gains output
    - HeroAgentBaseTDCPPORunnerCfg: TDC gain-only training (no encoder)
"""

from ..encoder import ActorCriticEncoder, ActorCriticEncoderTDC
from .rsl_rl_ppo_cfg import (
    HeroAgentBaseTDCPPORunnerCfg,
    HeroAgentEncoderPPORunnerCfg,
    HeroAgentEncoderTDCPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    RslRlPpoActorCriticEncoderCfg,
    RslRlPpoActorCriticEncoderTDCCfg,
)

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderTDC",
    "HeroAgentPPORunnerCfg",
    "HeroAgentEncoderPPORunnerCfg",
    "HeroAgentEncoderTDCPPORunnerCfg",
    "HeroAgentBaseTDCPPORunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
    "RslRlPpoActorCriticEncoderTDCCfg",
]
