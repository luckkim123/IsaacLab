# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP (Markov Decision Process) functions for Hero Agent ALBC environments.

This module provides:
- Event functions for domain randomization and reset
- Reward system (manager, configuration, reward functions)

Following Isaac Lab conventions, all MDP components are organized under this module.
"""

from .events import (
    randomize_hydrodynamics,
    randomize_joint_positions,
    randomize_ocean_current,
    randomize_payload,
    randomize_robot_pose,
    reset_joint_positions_default,
    reset_robot_pose_default,
)
from .observations import (
    compute_policy_obs,
    compute_privileged_obs,
)
from .rewards import (
    ALBCRewardCfg,
    BaseTDCRewardCfg,
    RewardManager,
    RewardTermCfg,
    TDCRewardCfg,
    albc_action_cost,
    albc_potential_reward,
    albc_progress_reward,
    tdc_attitude_penalty,
    tdc_gain_magnitude_cost,
    tdc_gain_smoothness_reward,
    tdc_stability_reward,
)

__all__ = [
    # Events
    "randomize_hydrodynamics",
    "randomize_joint_positions",
    "randomize_ocean_current",
    "randomize_payload",
    "randomize_robot_pose",
    "reset_joint_positions_default",
    "reset_robot_pose_default",
    # Observations
    "compute_policy_obs",
    "compute_privileged_obs",
    # Rewards (ALBC)
    "ALBCRewardCfg",
    "RewardManager",
    "RewardTermCfg",
    "albc_action_cost",
    "albc_potential_reward",
    "albc_progress_reward",
    # Rewards (TDC)
    "TDCRewardCfg",
    "BaseTDCRewardCfg",
    "tdc_attitude_penalty",
    "tdc_gain_magnitude_cost",
    "tdc_gain_smoothness_reward",
    "tdc_stability_reward",
]
