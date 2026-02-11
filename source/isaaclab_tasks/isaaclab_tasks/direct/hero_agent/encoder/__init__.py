# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encoder networks for HORA/RMA training.

Neural network architectures for the HORA pipeline:
    - ActorCriticEncoder: Base encoder (Phase 1 teacher)
    - ActorCriticEncoderTDC: Encoder with z exposure for TDC M_hat
    - ActorCriticEncoderTDCAdapt: Phase 2 adaptation (proprio history -> z_hat)
    - ProprioAdaptTConv: Temporal conv for proprioception history
    - RunningMeanStd: Welford's online normalization
"""

from .actor_critic_encoder import ActorCriticEncoder, ActorCriticEncoderTDC
from .adaptation import ActorCriticEncoderTDCAdapt, ProprioAdaptTConv
from .normalization import RunningMeanStd

__all__ = [
    "ActorCriticEncoder",
    "ActorCriticEncoderTDC",
    "ActorCriticEncoderTDCAdapt",
    "ProprioAdaptTConv",
    "RunningMeanStd",
]
