# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encoder networks and runner for HORA/RMA Phase 1 training.

This module provides:
    - ActorCriticEncoder: Base encoder network (privileged -> z -> actor/critic)
    - EncoderRunner: OnPolicyRunner with encoder metrics logging
"""

from .actor_critic_encoder import ActorCriticEncoder
from .actor_critic_encoder_tdc import ActorCriticEncoderTDC
from .encoder_runner import EncoderRunner

__all__ = ["ActorCriticEncoder", "ActorCriticEncoderTDC", "EncoderRunner"]
