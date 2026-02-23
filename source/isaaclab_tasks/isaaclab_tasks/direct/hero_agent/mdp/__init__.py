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
    randomize_body_mass,
    randomize_hydrodynamics,
    randomize_joint_friction,
    randomize_joint_gains,
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
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    angular_velocity_penalty,
    compute_stability_gate,
    mhat_accuracy_reward,
    progress_reward,
    progress_reward_pbrs,
    settling_bonus,
    tdc_torque_penalty,
    tracking_reward,
)

__all__ = [
    # Events
    "randomize_body_mass",
    "randomize_hydrodynamics",
    "randomize_joint_friction",
    "randomize_joint_gains",
    "randomize_joint_positions",
    "randomize_ocean_current",
    "randomize_payload",
    "randomize_robot_pose",
    "reset_joint_positions_default",
    "reset_robot_pose_default",
    # Observations
    "compute_policy_obs",
    "compute_privileged_obs",
    # Rewards (configs)
    "ALBCRewardCfg",
    "RewardManager",
    "RewardTermCfg",
    # Rewards (functions)
    "tracking_reward",
    "progress_reward",
    "progress_reward_pbrs",
    "action_rate_penalty",
    "action_magnitude_penalty",
    "angular_velocity_penalty",
    "mhat_accuracy_reward",
    "tdc_torque_penalty",
    "settling_bonus",
    "compute_stability_gate",
]
