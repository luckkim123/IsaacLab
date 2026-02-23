# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for Hero Agent ALBC environments.

This module provides:
    - ActorCriticEncoder: Custom network for HORA Phase 1 teacher training
    - ActorCriticEncoderAdapt: Phase 2 adaptation (proprio history -> z_hat)
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentAdaptBaseRunnerCfg: Phase 2 adaptation (supervised, base RL)

MPC agent configurations are in the hero_agent_mpc.agents package.
"""

from ..encoder import ActorCriticEncoder, ActorCriticEncoderAdapt
from .rsl_rl_ppo_cfg import (
    HeroAgentAdaptBaseRunnerCfg,
    HeroAgentEncoderPPORunnerCfg,
    HeroAgentPPORunnerCfg,
    RslRlPpoActorCriticEncoderAdaptCfg,
    RslRlPpoActorCriticEncoderCfg,
)

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderAdapt",
    "HeroAgentPPORunnerCfg",
    "HeroAgentEncoderPPORunnerCfg",
    "HeroAgentAdaptBaseRunnerCfg",
    "RslRlPpoActorCriticEncoderCfg",
    "RslRlPpoActorCriticEncoderAdaptCfg",
]
