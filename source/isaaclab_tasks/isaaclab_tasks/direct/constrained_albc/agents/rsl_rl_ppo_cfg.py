# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configurations for constrained ALBC environments.

Provides runner configurations for TRPO + IPO constrained encoder training:
    - ConstrainedALBCEncoderRunnerCfg: TRPO + IPO with teacher encoder
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

# Register custom classes in RSL-RL runner module namespace.
from ..algorithms import ConstraintTRPO
from ..encoder import (
    ActorCriticEncoder,
    ActorCriticEncoderConstrained,
)
from ..runners import ConstraintEncoderRunner

# Use ALBC-prefixed names to avoid collision with hero_agent registrations.
_runner_module.ALBCActorCriticEncoder = ActorCriticEncoder
_runner_module.ALBCActorCriticEncoderConstrained = ActorCriticEncoderConstrained
_runner_module.ALBCConstraintEncoderRunner = ConstraintEncoderRunner
_runner_module.ALBCConstraintTRPO = ConstraintTRPO


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class _EncoderPolicyCfg(RslRlPpoActorCriticCfg):
    """Shared network architecture for ALBC encoder policies (teacher)."""

    # Actor
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    # Encoder (privileged only -> z)
    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_obs_normalization: bool = True
    # Observation dimensions
    policy_obs_dim: int = 14
    privileged_dim: int = 23


@configclass
class RslRlPpoActorCriticEncoderConstrainedCfg(_EncoderPolicyCfg):
    """Policy config for ActorCriticEncoderConstrained (teacher with cost critic)."""

    class_name: str = "ALBCActorCriticEncoderConstrained"
    num_constraints: int = 0  # Auto-synced from env config by ConstraintEncoderRunner
    cost_critic_hidden_dims: list[int] = [512, 256, 128, 64]


# =============================================================================
# Algorithm Configuration
# =============================================================================


@configclass
class RslRlConstraintTRPOAlgorithmCfg:
    """Algorithm configuration for TRPO + IPO (Interior-Point Optimization).

    Uses log-barrier interior-point method with adaptive thresholding for
    constraint enforcement, combined with TRPO natural gradient for policy update.
    """

    class_name: str = "ALBCConstraintTRPO"

    # TRPO parameters
    max_kl: float = 0.005
    cg_iters: int = 10
    cg_damping: float = 0.1
    line_search_max_backtracks: int = 10
    line_search_shrink_factor: float = 0.5

    # Value function
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    cost_value_loss_coef: float = 1.0
    value_lr: float = 1e-3
    max_grad_norm: float = 1.0

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # Constraint
    num_constraints: int = 0  # Auto-synced from env config
    constraint_budgets: tuple[float, ...] = ()  # Auto-synced from env config
    cost_gamma: float = 0.99
    cost_lam: float = 0.95
    line_search_kl_margin: float = 1.5

    # Log barrier (IPO)
    barrier_t: float = 100.0
    barrier_alpha: float = 0.02

    # Noise floor
    min_std: float = 0.2

    # Encoder update
    num_encoder_epochs: int = 5
    encoder_lr: float = 3e-4


# =============================================================================
# Runner Configuration
# =============================================================================


@configclass
class ConstrainedALBCEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for TRPO + IPO constrained encoder training."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 2500
    save_interval = 50
    experiment_name = "constrained_albc_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = RslRlPpoActorCriticEncoderConstrainedCfg()
