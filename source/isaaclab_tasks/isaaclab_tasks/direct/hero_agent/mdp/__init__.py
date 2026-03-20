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
    compute_equilibrium_joint_positions,
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
    action_smoothness_penalty,
    command_reward,
    compute_stability_gate,
    energy_penalty,
    mhat_accuracy_reward,
    progress_reward,
    settling_reward,
    tdc_torque_penalty,
)

__all__ = [
    # Events
    "compute_equilibrium_joint_positions",
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
    "action_smoothness_penalty",
    "command_reward",
    "compute_stability_gate",
    "energy_penalty",
    "mhat_accuracy_reward",
    "progress_reward",
    "settling_reward",
    "tdc_torque_penalty",
]
