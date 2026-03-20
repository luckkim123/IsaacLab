# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for constrained ALBC environments.

Provides runner configurations for constrained encoder training:
    - ConstrainedALBCEncoderRunnerCfg: C-TRPO barrier-based constrained encoder

For evaluation, use CLI overrides instead of separate config classes:
    --max_iterations 100 --save_interval 25
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

# Register custom classes in RSL-RL runner module namespace.
# The runner resolves policy class_name and runner class dynamically.
# This injection makes custom classes resolvable in that scope.
from ..algorithms import ConstraintTRPO
from ..encoder import (
    ActorCriticEncoder,
    ActorCriticEncoderConstrained,
)
from ..runners import ConstraintEncoderRunner

_runner_module.ActorCriticEncoder = ActorCriticEncoder
_runner_module.ActorCriticEncoderConstrained = ActorCriticEncoderConstrained
_runner_module.ConstraintEncoderRunner = ConstraintEncoderRunner
_runner_module.ConstraintTRPO = ConstraintTRPO


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class _EncoderPolicyCfg(RslRlPpoActorCriticCfg):
    """Shared network architecture for ALBC encoder policies."""

    # Actor/Critic
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
    # Encoder
    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_output_activation: str = "tanh"
    encoder_obs_normalization: bool = True
    policy_obs_dim: int = 13
    privileged_dim: int = 19
    z_bounds_coef: float = 0.3
    z_bounds_soft_bound: float = 0.9
    proprio_history_len: int = 30
    proprio_feature_dim: int = 8


@configclass
class RslRlPpoActorCriticEncoderConstrainedCfg(_EncoderPolicyCfg):
    """Policy config for ActorCriticEncoderConstrained (encoder + cost critic).

    asymmetric_critic=True (default): critics see raw privileged obs (NORBC design).
    """

    class_name: str = "ActorCriticEncoderConstrained"
    num_constraints: int = 0  # Auto-synced from env config by ConstraintEncoderRunner
    cost_critic_hidden_dims: list[int] = [256, 128, 64]
    asymmetric_critic: bool = True


# =============================================================================
# Algorithm Configuration
# =============================================================================


@configclass
class RslRlConstraintTRPOAlgorithmCfg:
    """Algorithm configuration for ConstraintTRPO (C-TRPO barrier-based trust region).

    Implements C-TRPO (Muller et al., ICML 2025, arXiv:2411.02957):
        - No Lagrangian dual variables (lambda removed)
        - Safe mode: barrier-augmented objective + KL-only trust region
        - Recovery mode: cost minimization with standard TRPO trust region
        - Option C: barrier curvature in objective gradient only, FVP stays KL-only

    These fields are forwarded as kwargs to ConstraintTRPO.__init__().
    The class_name tells the runner to instantiate ConstraintTRPO instead of PPO.
    """

    class_name: str = "ConstraintTRPO"

    # TRPO parameters
    max_kl: float = 0.01
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
    num_constraints: int = 0  # Auto-synced from env config by ConstraintEncoderRunner
    constraint_budgets: tuple[float, ...] = ()  # Auto-synced from env config by ConstraintEncoderRunner
    cost_gamma: float = 0.99
    cost_lam: float = 0.95
    line_search_kl_margin: float = 1.5

    # C-TRPO barrier parameters
    beta: float = 0.01
    """Barrier coefficient weighting D_phi relative to D_KL."""

    recovery_threshold_frac: float = 0.8
    """Fraction of budget: if cost_return < budget * frac, exit recovery mode."""

    # Encoder z bounds
    z_bounds_coef: float = 0.3

    # Encoder update
    num_encoder_epochs: int = 1
    """Number of encoder gradient steps per iteration. Must stay at 1: multi-step
    causes uncontrolled KL divergence (encoder changes z -> distribution shift
    not bounded by TRPO trust region). Recovery mode fix is the real encoder fix."""

    encoder_lr: float = 3e-4
    """Encoder Adam learning rate. Matches pre-mod C-TRPO value; higher values
    (1e-3) caused excessive distribution shift even with 1 epoch."""


# =============================================================================
# Runner Configuration
# =============================================================================


@configclass
class ConstrainedALBCEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for constrained encoder training (C-TRPO barrier).

    Uses ConstraintEncoderRunner: DORAEMON + encoder metrics + barrier state.
    """

    class_name: str = "ConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 2500
    save_interval = 50
    experiment_name = "constrained_albc_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged", "proprio_hist"],
    }

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = RslRlPpoActorCriticEncoderConstrainedCfg()
