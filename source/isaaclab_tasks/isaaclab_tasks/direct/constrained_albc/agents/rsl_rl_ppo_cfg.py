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
    privileged_dim: int = 27
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
    max_kl: float = 0.002
    """KL trust region size. Scaled for 2D action space: 0.01 * (2/12) ≈ 0.002.
    Standard 0.01 assumes ~12D actions (locomotion). With 2D, per-dim KL budget
    is 6x larger, causing σ to decay 6x faster. 0.002 normalizes per-dim dynamics
    to match the reference paper's 12D behavior."""
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
    Barrier coefficient per constraint: 1/(t * margin). At t=50 with alpha=0.02,
    barrier gradient for joint_vel_limit (d_k=10): 1/(50*0.2)=0.10 per step."""

    barrier_alpha: float = 0.02
    """Adaptive threshold expansion coefficient (NORBC Section IV-B-1).
    d_k^i = max(d_k, J_C_k + alpha * d_k). Paper recommends alpha=0.02:
    too small -> updates too small/slow, too large -> constraint violations
    persist due to weak barrier gradient. At alpha=0.02, barrier_base = 0.02*d_k
    when violating, giving gradient 1/(t*0.02*d_k) per constraint."""

    # Noise floor (exploration maintenance)
    min_std: float = 0.2
    """Minimum action standard deviation. Clamped after both TRPO and Adam steps."""

    std_lr: float = 1e-4
    """Learning rate for separate log_std Adam optimizer. log_std is decoupled
    from TRPO natural gradient so sigma follows the score-function equilibrium
    (dlogpi/dsigma = ((a-mu)^2 - sigma^2) / sigma^3) without consuming KL budget.
    Conservative value (1/3 of encoder_lr) to prevent sigma oscillation."""

    entropy_coef: float = 0.0
    """Entropy regularization coefficient for TRPO surrogate. Disabled (0.0)
    because min_std=0.2 floor provides sufficient exploration maintenance.
    With fixed entropy_coef > 0, reward plateau causes noise_std runaway
    (constant upward pressure with no counteracting reward gradient)."""

    # Post-encoder KL gating
    max_encoder_kl: float = 0.003
    """Maximum additional KL divergence allowed from encoder update. If an encoder
    step causes KL to exceed pre_encoder_kl + max_encoder_kl, the encoder params
    are reverted. Derived: max_kl * line_search_kl_margin = 0.002 * 1.5 = 0.003."""

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
