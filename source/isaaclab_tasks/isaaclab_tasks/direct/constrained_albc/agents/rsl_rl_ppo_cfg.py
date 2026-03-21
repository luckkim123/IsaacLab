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
    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_obs_normalization: bool = True
    policy_obs_dim: int = 13
    privileged_dim: int = 19
    z_bounds_coef: float = 0.0
    z_bounds_soft_bound: float = 0.85
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
    """Algorithm configuration for ConstraintTRPO (C-TRPO barrier-based trust region).

    Implements C-TRPO (Muller et al., ICML 2025, arXiv:2411.02957):
        - No Lagrangian dual variables (lambda removed)
        - Safe mode: barrier-augmented objective + KL-only trust region
        - Recovery mode: cost minimization with standard TRPO trust region
        - Option C: barrier curvature in objective gradient only, FVP stays KL-only

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

    # C-TRPO barrier parameters
    beta: float = 0.02
    """Barrier coefficient weighting D_phi relative to D_KL.

    Reduced from 0.05 to 0.02: with margin_min=0.1 (phi_pp max=100), max barrier
    gradient = 0.02 * 100 * surr^2 = 2*surr^2. At margin=1: 0.02 * 1 * surr^2 =
    0.02*surr^2 (gentle far from boundary). Combined with recovery exclusion fix
    (barrier stays active through mode transitions), this creates smooth gradient
    proportional to reward surrogate (~0.1)."""

    recovery_threshold_frac: float = 0.4
    """Fraction of budget: if cost_return < budget * frac, exit recovery mode.

    Reduced from 0.6 to 0.4: with d_k=10 (joint_vel_limit budget doubled),
    recovery exits when cost < 4.0. Combined with the smooth barrier fix (barrier
    stays active through mode transitions), recovery should be rare and short.
    Lower threshold means faster exit from recovery -> less attitude damage."""

    ema_cost_alpha: float = 0.3
    """EMA smoothing factor for mean_cost_returns used in margin computation.

    Prevents phantom mode switches caused by cost critic drift when the actor is
    frozen (line search failure). alpha=0.3 gives ~3-iteration lag, which is fast
    enough to detect real constraint violations while filtering single-iteration
    cost value jumps."""

    # Encoder z bounds
    z_bounds_coef: float = 0.0

    # Noise floor (exploration maintenance)
    min_std: float = 0.2
    """Minimum action standard deviation. Clamped after TRPO step (outside
    trust region optimization). Prevents exploration collapse without consuming
    KL budget. Matches hero_agent PPO floor (~0.18)."""

    # Entropy bonus
    entropy_coef: float = 0.0
    """Entropy bonus coefficient. Set to 0.0: entropy in TRPO surrogate is
    structurally unstable (competes with reward for single KL budget step).
    Exploration maintained via min_std floor instead."""

    # EAPO: Entropy Advantage Policy Optimization (arXiv:2407.18143)
    eapo_enabled: bool = True
    """Enable EAPO entropy advantage in surrogate. Replaces entropy_coef
    (keep entropy_coef=0.0 when active)."""

    eapo_tau_init: float = 0.01
    """Initial entropy temperature. Adaptive via SAC v2 dual gradient."""

    eapo_target_entropy: float = 0.5
    """Target entropy (2D action). std~0.34 per dim -> H~0.5.
    Above floor (std=0.2, H=-0.39) but allows convergence."""

    eapo_tau_lr: float = 0.001
    """Dual variable learning rate for tau adaptation."""

    eapo_tau_min: float = 0.001
    """Minimum tau to prevent zero entropy pressure."""

    eapo_tau_max: float = 0.5
    """Maximum tau to prevent entropy dominating reward."""

    # Post-encoder KL gating (Fix 2: prevents encoder-induced KL violation)
    max_encoder_kl: float = 0.016
    """Maximum additional KL divergence allowed from encoder update. If an encoder
    step causes KL to exceed pre_encoder_kl + max_encoder_kl, the encoder params
    are reverted. 0.016 = max_kl * line_search_kl_margin (same budget as policy step)."""

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

    Uses ConstraintEncoderRunner: encoder metrics + barrier state.
    """

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
