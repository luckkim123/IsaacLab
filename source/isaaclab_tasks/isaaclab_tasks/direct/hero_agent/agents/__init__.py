# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides:
    - ActorCriticEncoder: Custom network for HORA Phase 1 teacher training
    - ActorCriticEncoderTDC: Encoder variant exposing z for TDC M_hat
    - ActorCriticEncoderTDCAdapt: Phase 2 adaptation (proprio history -> z_hat)
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentEncoderTDCPPORunnerCfg: Encoder-TDC integration
    - HeroAgentUnifiedTDCPPORunnerCfg: General encoder + RL-output M_hat/Kp/Kd
    - HeroAgentAdaptTDCRunnerCfg: Phase 2 adaptation (supervised, non-PPO)
    - HeroAgentSinglePhaseTDCRunnerCfg: Single-phase joint training (PPO + aux M_hat loss)
"""

from ..encoder import ActorCriticEncoder, ActorCriticEncoderTDC, ActorCriticEncoderTDCAdapt
from .rsl_rl_ppo_cfg import (
    HeroAgentAdaptTDCRunnerCfg,
    HeroAgentEncoderPPORunnerCfg,
    HeroAgentEncoderTDCPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    HeroAgentSinglePhaseTDCRunnerCfg,
    HeroAgentUnifiedTDCPPORunnerCfg,
    RslRlPpoActorCriticEncoderCfg,
    RslRlPpoActorCriticEncoderTDCAdaptCfg,
    RslRlPpoActorCriticEncoderTDCCfg,
    RslRlPpoActorCriticUnifiedTDCCfg,
)

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderTDC",
    "ActorCriticEncoderTDCAdapt",
    "HeroAgentPPORunnerCfg",
    "HeroAgentEncoderPPORunnerCfg",
    "HeroAgentEncoderTDCPPORunnerCfg",
    "HeroAgentUnifiedTDCPPORunnerCfg",
    "HeroAgentAdaptTDCRunnerCfg",
    "HeroAgentSinglePhaseTDCRunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
    "RslRlPpoActorCriticEncoderTDCCfg",
    "RslRlPpoActorCriticEncoderTDCAdaptCfg",
    "RslRlPpoActorCriticUnifiedTDCCfg",
]
