# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides:
    - ActorCriticEncoder: Custom network for HORA Phase 1 teacher training
    - ActorCriticEncoderAdapt: Phase 2 adaptation (proprio history -> z_hat)
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentTDEBasePPORunnerCfg: TDE-Base with training enhancements
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentAdaptBaseRunnerCfg: Phase 2 adaptation (supervised, base RL)
    - HeroAgentConstrainedEncoderRunnerCfg: IPO + TRPO constrained encoder training
"""

from ..encoder import ActorCriticEncoder, ActorCriticEncoderAdapt, ActorCriticEncoderConstrained
from .rsl_rl_ppo_cfg import (
    HeroAgentAdaptBaseRunnerCfg,
    HeroAgentConstrainedEncoderRunnerCfg,
    HeroAgentEncoderPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    HeroAgentPrivBasePPORunnerCfg,
    HeroAgentTDEBasePPORunnerCfg,
    RslRlConstraintTRPOAlgorithmCfg,
    RslRlPpoActorCriticEncoderAdaptCfg,
    RslRlPpoActorCriticEncoderCfg,
    RslRlPpoActorCriticEncoderConstrainedCfg,
)

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderAdapt",
    "ActorCriticEncoderConstrained",
    "HeroAgentPPORunnerCfg",
    "HeroAgentPrivBasePPORunnerCfg",
    "HeroAgentTDEBasePPORunnerCfg",
    "HeroAgentEncoderPPORunnerCfg",
    "HeroAgentAdaptBaseRunnerCfg",
    "HeroAgentConstrainedEncoderRunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
    "RslRlPpoActorCriticEncoderAdaptCfg",
    "RslRlPpoActorCriticEncoderConstrainedCfg",
    "RslRlConstraintTRPOAlgorithmCfg",
]
