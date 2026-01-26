# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for Hero Agent ALBC environments.

This module provides a modular reward system following Isaac Lab conventions.
Reward terms are defined as standalone functions or callable classes, composed
through RewardTermCfg configurations.

ALBC (Active Linear Buoyancy Controller) uses joint-based attitude control
without thrusters, so rewards focus on orientation and joint action costs.
"""

from .reward_manager import RewardManager, RewardTermCfg
from .reward_terms import (
    # ALBC-specific reward terms
    albc_action_cost,
    albc_alive_bonus,
    albc_potential_reward,
    albc_progress_reward,
    # Orientation rewards (useful for ALBC attitude tracking)
    orientation_upright,
    orientation_upright_exp,
    # Action penalties (useful for smooth joint control)
    action_rate_penalty,
    action_magnitude_penalty,
    angular_velocity_penalty,
    alive_bonus,
)

__all__ = [
    # Manager
    "RewardManager",
    "RewardTermCfg",
    # ALBC-specific reward terms
    "albc_potential_reward",
    "albc_progress_reward",
    "albc_action_cost",
    "albc_alive_bonus",
    # Orientation rewards
    "orientation_upright",
    "orientation_upright_exp",
    # Action penalties
    "action_rate_penalty",
    "action_magnitude_penalty",
    "angular_velocity_penalty",
    "alive_bonus",
]
