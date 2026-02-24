# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training runners for Hero Agent environments.

BaseRunner provides shared training enhancements (DORAEMON DR scheduling, adaptive entropy).
EncoderRunner adds HORA Phase 1 encoder metrics on top.
AdaptRunner handles Phase 2 supervised adaptation.
SAC-MPC runner (SACMPCRunner) is in hero_agent_mpc.runners.
"""

from .adapt_runner import AdaptRunner
from .base_runner import BaseRunner
from .encoder_runner import EncoderRunner

__all__ = ["BaseRunner", "EncoderRunner", "AdaptRunner"]
