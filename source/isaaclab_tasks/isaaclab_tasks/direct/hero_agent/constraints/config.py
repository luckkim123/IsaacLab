# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for NORBC-style constrained RL training.

Defines per-constraint config and IPO hyperparameters. Each ConstraintCfg
maps to one cost function with a budget limit D_k (NORBC Eq. 7-8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field

import torch

from isaaclab.utils import configclass


@configclass
class ConstraintCfg:
    """Configuration for a single safety/performance constraint.

    Attributes:
        name: Unique identifier for this constraint.
        constraint_type: "probabilistic" (binary indicator) or "average" (continuous).
            Probabilistic constraints use D_k as violation probability limit.
            Average constraints use D_k as mean cost limit.
        cost_func: Function (env, **params) -> (num_envs,) tensor of per-step costs.
        limit_D: Budget limit D_k. For probabilistic: P(violation) <= D_k.
            For average: E[cost] <= D_k.
        params: Additional keyword arguments passed to cost_func.
    """

    name: str = ""
    constraint_type: str = "probabilistic"
    cost_func: Callable[..., torch.Tensor] | None = None
    limit_D: float = 0.01
    params: dict = field(default_factory=dict)


@configclass
class ConstrainedTrainingCfg:
    """IPO (Interior-point Policy Optimization) hyperparameters.

    Based on NORBC (Zhao et al., 2024) Eq. 9-11.

    Attributes:
        barrier_t: Log-barrier steepness parameter. Higher = harder constraint
            enforcement but less smooth optimization landscape.
        adaptive_alpha: Feasible region expansion factor for adaptive thresholding.
            alpha * d_k is added to current cost return as slack (Eq. 11).
        cost_gamma: Discount factor for cost return computation (separate from reward gamma).
        cost_value_coef: Loss coefficient for cost value function heads.
        constraints: List of active constraint configurations.
    """

    barrier_t: float = 10.0
    adaptive_alpha: float = 0.5
    cost_gamma: float = 0.99
    cost_value_coef: float = 0.5
    constraints: list[ConstraintCfg] = field(default_factory=list)
