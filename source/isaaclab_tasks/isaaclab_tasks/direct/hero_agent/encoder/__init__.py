# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encoder networks for HORA/RMA training.

Neural network architectures for the HORA pipeline:
    - ActorCriticEncoder: Base encoder (Phase 1 teacher)
    - ActorCriticEncoderAdapt: Phase 2 adaptation (proprio history -> z_hat)
    - ProprioAdaptTConv: Temporal conv for proprioception history

SAC-MPC encoder networks (ActorCriticMPC, TwinQNetwork) are in hero_agent_mpc.encoder.
"""

from .actor_critic_encoder import ActorCriticEncoder
from .adaptation import ActorCriticEncoderAdapt, ProprioAdaptTConv

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderAdapt",
    "ProprioAdaptTConv",
]
