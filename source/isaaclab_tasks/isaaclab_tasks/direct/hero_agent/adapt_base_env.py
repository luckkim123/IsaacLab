# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Phase 2 Adaptation Environment (base RL pipeline).

History buffer and "proprio_hist" observation are managed by base_env
when proprio_history_len > 0 (set in HeroAgentAdaptBaseEnvCfg via
HeroAgentEncoderTrainEnvCfg inheritance).

Data Flow (Phase 2):
    proprio_hist (N, H, 8) --> adapt_tconv --> z_hat (13D)
    z_hat --> [policy_obs + hist_flat + z_hat] --> Frozen Actor --> 2D velocity actions
    Frozen Encoder([policy_obs, hist_flat, privileged]) --> z_gt (13D)
    Loss = ||z_hat - z_gt||^2
"""

from __future__ import annotations

from .base_env import HeroAgentEnv
from .config import HeroAgentAdaptBaseEnvCfg


class HeroAgentAdaptBaseEnv(HeroAgentEnv):
    """Phase 2 adaptation environment.

    With proprio_history_len=30 set in HeroAgentAdaptBaseEnvCfg,
    the base class automatically manages the history ring buffer and
    includes "proprio_hist" in observations.
    """

    cfg: HeroAgentAdaptBaseEnvCfg
