# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hero Agent SAC-MPC environment.

MPC-specific environment configuration that extends the shared
HeroAgentTrainEnvCfg from hero_agent package.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from isaaclab_tasks.direct.hero_agent.config import (
    DomainRandomizationCfg,
    HeroAgentTrainEnvCfg,
)
from isaaclab_tasks.direct.hero_agent.doraemon import DoraemonCfg
from isaaclab_tasks.direct.hero_agent.mdp import ALBCRewardCfg

from .controllers.mpc import DifferentiableMPCCfg


@configclass
class HeroAgentMPCEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent SAC-MPC environment: learned dynamics + differentiable MPC.

    AC-MPC architecture (Romero et al., 2024):
        - MPC solver lives inside the policy (actor_critic_mpc.py), not env
        - Cost Map Network: obs -> Q_diag
        - DynamicsMLP: x' = x + f(x, u)*dt (end-to-end via SAC actor loss)
        - Differentiable MPC: solves optimal control -> 2D joint velocity

    State (8D): [phi, theta, phi_dot, theta_dot, q1, q2, q1_dot, q2_dot]
    Control (2D): [q1_dot_ref, q2_dot_ref] (MPC output from policy)
    """

    # SAC off-policy replay buffer is structurally incompatible with DORAEMON
    # (stale data from earlier DR stages corrupts training). DORAEMON disabled.
    doraemon: DoraemonCfg = DoraemonCfg(enable=False)

    # MPC solver configuration
    mpc: DifferentiableMPCCfg = DifferentiableMPCCfg()

    # Policy called at 50Hz: decimation=4 -> step_dt = 0.005*4 = 0.02s
    # Each env.step() runs 4 physics sub-steps internally, policy queried once.
    decimation: int = 4
    control_decimation: int = 1

    # 2D joint velocity actions (MPC output from policy)
    action_space: int = 2
    observation_space: int = 13
    state_space: int = 26  # privileged obs for asymmetric critic

    # Joint velocity scaling for MPC output -> position integration (applied at step_dt=0.02s)
    max_joint_velocity: float = math.pi

    reward: ALBCRewardCfg = ALBCRewardCfg(
        tracking_weight=3.0,
        tracking_sigma=0.5,
        joint_oscillation_weight=-0.5,
        joint_velocity_weight=-0.3,
        progress_weight=0.2,
        progress_mode="pbrs",
        progress_gamma=0.99,  # match SAC gamma
        settling_weight=1.0,
        settling_threshold=0.10,
        settling_sharpness=30.0,
    )

    # MPC extracts state directly from physics; obs noise would corrupt MPC state
    observation_noise_model: None = None

    # DR: same as train but no action latency (MPC handles its own timing)
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        action_latency_range=(0, 0),
    )
