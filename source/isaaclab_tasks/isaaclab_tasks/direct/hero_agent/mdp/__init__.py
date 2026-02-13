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
    EncoderTDCRewardCfg,
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    angular_velocity_penalty,
    linear_error_penalty,
    progress_reward,
    tde_residual_penalty,
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
    "EncoderTDCRewardCfg",
    "RewardManager",
    "RewardTermCfg",
    # Rewards (functions)
    "tracking_reward",
    "progress_reward",
    "linear_error_penalty",
    "angular_velocity_penalty",
    "action_rate_penalty",
    "action_magnitude_penalty",
    "tde_residual_penalty",
]
