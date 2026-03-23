# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for constrained ALBC environments.

Provides runner configurations for constrained encoder training:
    - ConstrainedALBCEncoderRunnerCfg: Lagrangian-based constrained TRPO with encoder

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

# Use ALBC-prefixed names to avoid collision with hero_agent registrations.
# Both packages register into the same _runner_module namespace; hero_agent
# imports after constrained_albc (alphabetical) and overwrites unprefixed names.
_runner_module.ALBCActorCriticEncoder = ActorCriticEncoder
_runner_module.ALBCActorCriticEncoderConstrained = ActorCriticEncoderConstrained
_runner_module.ALBCConstraintEncoderRunner = ConstraintEncoderRunner
_runner_module.ALBCConstraintTRPO = ConstraintTRPO


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
    encoder_hidden_dims: list[int] = [128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_obs_normalization: bool = True
    policy_obs_dim: int = 13
    privileged_dim: int = 28
    proprio_history_len: int = 30
    proprio_feature_dim: int = 8


@configclass
class RslRlPpoActorCriticEncoderConstrainedCfg(_EncoderPolicyCfg):
    """Policy config for ActorCriticEncoderConstrained (encoder + cost critic)."""

    class_name: str = "ALBCActorCriticEncoderConstrained"
    num_constraints: int = 0  # Auto-synced from env config by ConstraintEncoderRunner
    cost_critic_hidden_dims: list[int] = [256, 128, 64]


# =============================================================================
# Algorithm Configuration
# =============================================================================


@configclass
class RslRlConstraintTRPOAlgorithmCfg:
    """Algorithm configuration for ConstraintTRPO (Modified IPO with TRPO optimizer).

    Uses log-barrier interior-point method with adaptive thresholding for
    constraint enforcement, combined with TRPO natural gradient for policy update.

    These fields are forwarded as kwargs to ConstraintTRPO.__init__().
    The class_name tells the runner to instantiate ConstraintTRPO instead of PPO.
    """

    class_name: str = "ALBCConstraintTRPO"

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

    # Log barrier constraint parameters (Modified IPO)
    barrier_t: float = 50.0
    """Barrier steepness parameter. Higher t = barrier activates only near boundary.
    Barrier coefficient per constraint: 1/(t * margin). At t=50 with margin=1.29,
    constraint gradient is ~1.5% of reward gradient O(1)."""

    barrier_alpha: float = 0.3
    """Adaptive threshold expansion coefficient. When cost exceeds budget,
    threshold is relaxed: d_k^i = max(d_k, J_C_k + alpha * d_k). Ensures
    log barrier is computable even during initial constraint violations."""

    # Noise floor (exploration maintenance)
    min_std: float = 0.2
    """Minimum action standard deviation. Clamped after TRPO step (outside
    trust region optimization). Prevents exploration collapse without consuming
    KL budget."""

    # Post-encoder KL gating
    max_encoder_kl: float = 0.016
    """Maximum additional KL divergence allowed from encoder update. If an encoder
    step causes KL to exceed pre_encoder_kl + max_encoder_kl, the encoder params
    are reverted. 0.016 = max_kl * line_search_kl_margin (same budget as policy step)."""

    # Encoder update
    num_encoder_epochs: int = 3
    """Number of encoder gradient steps per iteration. KL gating (max_encoder_kl)
    reverts encoder if distribution shift exceeds budget, making multi-step safe."""

    encoder_lr: float = 3e-4
    """Encoder Adam learning rate."""


# =============================================================================
# Runner Configuration
# =============================================================================


@configclass
class ConstrainedALBCEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for constrained encoder training (Lagrangian TRPO)."""

    class_name: str = "ALBCConstraintEncoderRunner"
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
