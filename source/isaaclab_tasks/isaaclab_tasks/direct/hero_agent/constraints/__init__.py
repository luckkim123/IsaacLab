# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained RL components for NORBC-style IPO training.

Implements Interior-point Policy Optimization (IPO) with log-barrier constraints
for the Hero Agent Encoder-TDC environment. Key components:
    - CostManager: Computes per-step cost signals (analogous to RewardManager)
    - LogBarrierManager: Adaptive log-barrier for IPO loss computation
    - ConstrainedPPO: PPO with barrier loss and cost value heads
    - ConstrainedRolloutStorage: Storage with cost buffers and cost GAE
"""

from .barrier import LogBarrierManager
from .config import ConstrainedTrainingCfg, ConstraintCfg
from .constrained_ppo import ConstrainedPPO
from .constrained_storage import ConstrainedRolloutStorage
from .cost_functions import (
    angular_velocity_cost,
    control_smoothness_cost,
    inertia_rate_cost,
    joint_velocity_cost,
    workspace_cost,
)
from .cost_manager import CostManager

__all__ = [
    "ConstraintCfg",
    "ConstrainedTrainingCfg",
    "CostManager",
    "LogBarrierManager",
    "ConstrainedPPO",
    "ConstrainedRolloutStorage",
    "workspace_cost",
    "control_smoothness_cost",
    "inertia_rate_cost",
    "angular_velocity_cost",
    "joint_velocity_cost",
]
