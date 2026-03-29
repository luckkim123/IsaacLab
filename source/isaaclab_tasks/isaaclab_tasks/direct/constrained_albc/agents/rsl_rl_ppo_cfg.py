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

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register custom classes in RSL-RL runner module namespace.
from ..algorithms import ConstraintTRPO
from ..encoder import (
    ActorCriticConstrained,
    ActorCriticEncoder,
    ActorCriticEncoderConstrained,
)
from ..runners import ConstraintEncoderRunner

# Use ALBC-prefixed names to avoid collision with hero_agent registrations.
_runner_module.ALBCActorCriticConstrained = ActorCriticConstrained
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
    barrier_alpha: float = 0.05

    # Noise floor
    min_std: float = 0.2


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


# =============================================================================
# Debug Configuration (pure PPO, no encoder, no constraints)
# =============================================================================


@configclass
class _DebugPolicyCfg(RslRlPpoActorCriticCfg):
    """Standard actor-critic for debug (no encoder)."""

    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"


@configclass
class _DebugAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Standard PPO for debug."""

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Debug runner: standard PPO, no encoder, no constraints."""

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug"

    algorithm = _DebugAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class ALBCDebugDRRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 1: PPO + DR (no encoder, no constraints)."""

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_dr"

    algorithm = _DebugAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class ALBCDebugTRPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 2: TRPO + DR (no encoder, no barriers). Tests TRPO vs PPO."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_trpo"

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class _BarrierPolicyCfg(RslRlPpoActorCriticCfg):
    """ActorCritic + cost critic (no encoder) for barrier ablation."""

    class_name: str = "ALBCActorCriticConstrained"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
    num_constraints: int = 0  # Auto-synced by runner


@configclass
class ALBCDebugBarrierRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 3: TRPO + IPO barrier + DR (no encoder). Tests barrier impact."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_barrier"

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = _BarrierPolicyCfg()


@configclass
class ALBCDebugEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4: TRPO + Encoder + DR (no constraints). Tests encoder impact."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = RslRlPpoActorCriticEncoderConstrainedCfg()


@configclass
class _PPOEncoderPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy for PPO (no cost critic)."""

    class_name: str = "ALBCActorCriticEncoder"


@configclass
class ALBCDebugPPOEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4b: PPO + Encoder + DR (no constraints). Tests PPO vs TRPO with encoder."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _DebugAlgorithmCfg()
    policy = _PPOEncoderPolicyCfg()


@configclass
class _PPOEncoderHistPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy with proprio history for PPO.

    Actor input: cat([o_t(14D), hist_flat(240D), z(13D)]) = 267D.
    z/input ratio: 13/267 = 4.9% (was 48% without history).
    """

    class_name: str = "ALBCActorCriticEncoder"
    proprio_hist_dim: int = 240  # 30 steps * 8 features


@configclass
class _PPOEncoderHistAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO for encoder+history ablation."""

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOEncoderHistRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4c: PPO + Encoder + History + DR (no constraints).

    Adds proprio history to actor input to reduce z/input ratio from 48% to 5%.
    Matches NORBC paper architecture: Actor = cat([o_t(+history), z]).
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_encoder_hist"
    normalize_value: bool = False
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _PPOEncoderHistAlgorithmCfg()
    policy = _PPOEncoderHistPolicyCfg()


@configclass
class _PPOHistOnlyAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO for history-only ablation."""

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOHistOnlyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4d: PPO + History + DR (NO encoder). Tests if encoder is the problem."""

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_hist_only"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "proprio_hist"],
        "critic": ["policy", "proprio_hist"],
    }

    algorithm = _PPOHistOnlyAlgorithmCfg()
    policy = _DebugPolicyCfg()


# =============================================================================
# Shared Backbone Experiment (Fix A + B)
# =============================================================================


@configclass
class _SharedBackbonePolicyCfg(_EncoderPolicyCfg):
    """Shared backbone encoder policy (HORA-style).

    Actor and critic share an MLP backbone. Value gradient flows to encoder,
    providing a stable MSE-based learning signal in addition to policy gradient.
    """

    class_name: str = "ALBCActorCriticEncoder"
    shared_backbone: bool = True
    z_bounds_coef: float = 1.0
    z_bounds_soft_bound: float = 0.85


@configclass
class _SharedBackboneAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with HORA-aligned hyperparameters for encoder stability.

    Key differences from _DebugAlgorithmCfg:
    - entropy_coef=0.0: removes sigma upward pressure (HORA default)
    - learning_rate=1e-3: moderate start, 16 LR decreases before min_lr
    - desired_kl=0.02: wider trust region for encoder+actor joint updates
    - Per-epoch LR and mu/sigma refresh auto-enabled by PPO encoder detection
    - Single optimizer group (shared backbone detected): encoder slows with backbone
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 0.25
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOSharedBackboneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 5a: PPO + Encoder + Shared Backbone + DR (no history).

    Fixes applied:
    - Fix A: Shared backbone -> value gradient flows to encoder
    - Fix B: Per-epoch LR + mu/sigma refresh (auto-enabled by PPO)
    - HORA hyperparameters: ent=0, lr=5e-3, kl=0.02, normalize_value=True
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_shared_backbone"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _SharedBackboneAlgorithmCfg()
    policy = _SharedBackbonePolicyCfg()


@configclass
class _SharedBackboneHistPolicyCfg(_SharedBackbonePolicyCfg):
    """Shared backbone with proprio history.

    10 steps * 8D = 80D. Actor input: 14+80+13 = 107D (HORA=104D).
    """

    proprio_hist_dim: int = 80


@configclass
class ALBCDebugPPOSharedBackboneHistRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 5b: PPO + Encoder + Shared Backbone + History + DR.

    Same as 5a but with 30-step proprio history to further reduce z/input ratio.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_shared_backbone_hist"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _SharedBackboneAlgorithmCfg()
    policy = _SharedBackboneHistPolicyCfg()


# =============================================================================
# Step 6: Separate Network + Fix B + Combined Hyperparams
# =============================================================================


@configclass
class _SeparateEncoderPolicyCfg(_EncoderPolicyCfg):
    """Separate encoder policy with z_bounds_loss.

    Standard architecture: separate actor/critic. Critic asymmetric (uses p_t).
    Encoder gradient from surrogate loss only (no value gradient).
    z_bounds_loss prevents softsign saturation.
    """

    class_name: str = "ALBCActorCriticEncoder"
    shared_backbone: bool = False
    z_bounds_coef: float = 1.0
    z_bounds_soft_bound: float = 0.85
    proprio_hist_dim: int = 80  # 10 steps * 8D


@configclass
class _SeparateEncoderAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with combined hyperparams from ablation study.

    Based on Step 4c ablation: normalize_value was the only single variable
    that improved BOTH roll and pitch. Combined with entropy_coef=0 (noise_std
    downtrend) and desired_kl=0.02 (wider trust region).

    Per-minibatch mu/sigma refresh + per-epoch LR auto-enabled by encoder detection.
    Single optimizer group: encoder slows with actor when KL is high.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOSeparateEncHistRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 6: PPO + Encoder + History + Separate Networks + Fix B.

    Separate actor/critic (no shared backbone). Encoder gets surrogate gradient only.
    Key fixes:
    - Fix B: Per-minibatch mu/sigma refresh + per-epoch LR (auto from encoder detection)
    - Single optimizer group: encoder + actor under same adaptive LR (no asymmetric freeze)
    - Combined hyperparams: entropy=0, normalize_value, desired_kl=0.02
    - z_bounds_loss: prevents encoder saturation
    - History 10 steps: actor input 107D, z ratio 12%
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_sep_enc_hist"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _SeparateEncoderAlgorithmCfg()
    policy = _SeparateEncoderPolicyCfg()


# =============================================================================
# Step 7: Small Encoder (reduce encoder param fraction to ~5%)
# =============================================================================


@configclass
class _SmallEncoderPolicyCfg(_EncoderPolicyCfg):
    """HORA-matched encoder: 23D -> [256,128] -> 8D z. ~39K params (~15% of policy).

    Current encoder [256,128,64]->13D is 48K params (41% of policy).
    HORA encoder: 9->[256,128]->8D is 37K params (~15%).
    This encoder: 23->[256,128]->8D is ~39K params (~15%).

    Actor input: 14 + 80 + 8 = 102D (HORA=104D).
    """

    class_name: str = "ALBCActorCriticEncoder"
    shared_backbone: bool = False
    encoder_hidden_dims: list[int] = [256, 128]
    encoder_latent_dim: int = 8
    z_bounds_coef: float = 1.0
    z_bounds_soft_bound: float = 0.85
    proprio_hist_dim: int = 80  # 10 steps * 8D


@configclass
class ALBCDebugPPOSmallEncHistRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 7: PPO + Small Encoder + History + DR.

    Encoder reduced from [256,128,64]->13D (48K, 41%) to [64,32]->8D (4K, 5%).
    One gradient step now changes only 5% of policy params through encoder,
    similar to HORA's encoder fraction (~15%).

    All Fix B improvements retained: per-minibatch mu/sigma refresh, per-epoch LR,
    single optimizer group, entropy=0, normalize_value, desired_kl=0.02.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_small_enc_hist"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _SeparateEncoderAlgorithmCfg()
    policy = _SmallEncoderPolicyCfg()


# =============================================================================
# Phase 1: Q1 (HORA-style normalization) + Q3 (strided proprio history)
# =============================================================================


@configclass
class _Q1Q3EncoderPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy with Q1 (HORA-style norm) + Q3 (strided history).

    Key changes from previous encoder configs:
    - actor_obs_normalizer covers only o_t+hist (134D), excludes z (13D)
    - proprio_hist_dim=120 (15 steps * 8 features, stride=5 at 50Hz -> 10Hz)
    - z/input = 13/147 = 8.8% (vs 48.1% without history)
    - No z_bounds_loss (isolate normalization + history effect)
    """

    class_name: str = "ALBCActorCriticEncoder"
    shared_backbone: bool = False
    proprio_hist_dim: int = 120  # 15 steps * 8 features
    z_bounds_coef: float = 0.0


@configclass
class _Q1Q3AlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO for Q1+Q3 encoder experiment.

    entropy_coef=0.0 (HORA default, prevents sigma upward pressure).
    desired_kl=0.02 (wider trust region for encoder+actor joint updates).
    Per-epoch LR adaptation + per-minibatch mu/sigma refresh auto-enabled
    by encoder detection in ppo.py.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOQ1Q3RunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 8a: PPO + Encoder + Q1 norm fix + Q3 strided history + DR.

    Key changes from Step 4c/6:
    - Q1: actor_obs_normalizer excludes z (HORA-style, normalizes only o_t+hist)
    - Q3: history stride=5, len=15 (1.5s window, 10Hz sampling)
    - entropy_coef=0.0 (HORA default)
    - No z_bounds_loss (isolate effect)
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_q1q3"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _Q1Q3AlgorithmCfg()
    policy = _Q1Q3EncoderPolicyCfg()


# Phase 1b: Same as Q1Q3 but without encoder input normalization (raw p_t)


@configclass
class _Q1Q3NoEncNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Same as Q1Q3 but encoder_obs_normalization=False (raw p_t to encoder, HORA-style)."""

    encoder_obs_normalization: bool = False


@configclass
class ALBCDebugPPOQ1Q3NoEncNormRunnerCfg(ALBCDebugPPOQ1Q3RunnerCfg):
    """Step 8b: Same as 8a but without p_t normalization before encoder."""

    experiment_name = "constrained_albc_debug_ppo_q1q3_no_enc_norm"
    policy = _Q1Q3NoEncNormPolicyCfg()


# =============================================================================
# Phase 2: Q4 (KL Management) - HORA-style LR range
# =============================================================================


@configclass
class _Q4AlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with HORA-style adaptive LR range.

    Key changes from _Q1Q3AlgorithmCfg:
    - learning_rate=5e-3: HORA default, 17x more headroom than 3e-4.
    - min_lr=1e-6: 22 halvings from 5e-3 (vs 9 from 3e-4 with min_lr=1e-5).
    - desired_kl=0.02: unchanged from Q1Q3.

    Rationale:
    iter-0 KL is artificially low (encoder untrained) -> adaptive schedule
    amplifies LR to ~5x init. iter-1 encoder first update -> KL 0.89 spike.
    With init_lr=5e-3, post-spike LR settles at ~3e-4 (still viable).
    With init_lr=3e-4, post-spike LR crashes to ~2.6e-5 (dead).
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 5e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOQ4RunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 9a: Q1+Q3+Q4 with enc norm. HORA-style LR range.

    Based on Step 8a (Q1+Q3, enc norm kept). Adds Q4:
    - init_lr 3e-4 -> 5e-3 (HORA default)
    - min_lr 1e-5 -> 1e-6 (22 halvings from init_lr)
    - Expected: LR survives iter-1 KL spike, policy can learn.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_q4"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _Q4AlgorithmCfg()
    policy = _Q1Q3EncoderPolicyCfg()


@configclass
class _Q4NoEncNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Q4 policy without encoder input normalization (raw p_t, HORA-style)."""

    encoder_obs_normalization: bool = False


@configclass
class ALBCDebugPPOQ4NoEncNormRunnerCfg(ALBCDebugPPOQ4RunnerCfg):
    """Step 9b: Q1+Q3+Q4 without enc norm. Tests HORA-style raw p_t with wider LR range."""

    experiment_name = "constrained_albc_debug_ppo_q4_no_enc_norm"
    policy = _Q4NoEncNormPolicyCfg()


# =============================================================================
# Encoder Gradient Scaling (reduce encoder KL contribution)
# =============================================================================


@configclass
class _EncScaleAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with encoder gradient scaling.

    Encoder gradient scaled by 0.1x after loss.backward(), reducing encoder's
    KL contribution by ~100x (scale^2). This raises the equilibrium LR from
    ~2.6e-5 (encoder-dominated) toward the encoder-free equilibrium (~1e-4+),
    allowing actor to learn normally while encoder learns slowly.

    Key hyperparams:
    - learning_rate=3e-4: proven init_lr (Q4 showed 5e-3 makes iter-0 worse)
    - min_lr=1e-6: wide LR range for safety
    - encoder_grad_scale=0.1: encoder gradient multiplied by 0.1 after backward
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    encoder_grad_scale: float = 0.1
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOEncScaleRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 10a: Q1+Q3 + encoder gradient scaling (0.1x), enc norm kept.

    Encoder gradient scaled 10x down to reduce KL contribution.
    Expected: LR stays above 1e-4, noise_std decreases, policy learns.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_enc_scale"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _EncScaleAlgorithmCfg()
    policy = _Q1Q3EncoderPolicyCfg()


@configclass
class _EncScaleNoEncNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """EncScale policy without encoder input normalization (raw p_t)."""

    encoder_obs_normalization: bool = False


@configclass
class ALBCDebugPPOEncScaleNoEncNormRunnerCfg(ALBCDebugPPOEncScaleRunnerCfg):
    """Step 10b: Same as 10a but without p_t normalization before encoder."""

    experiment_name = "constrained_albc_debug_ppo_enc_scale_no_enc_norm"
    policy = _EncScaleNoEncNormPolicyCfg()


# =============================================================================
# Step 11: Standard update() path for encoder (bypass _update_encoder_ppo)
# =============================================================================


@configclass
class _StdUpdateAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO using standard update() path even with encoder present.

    Bypasses _update_encoder_ppo() by setting use_encoder_update=False.
    Standard path uses per-minibatch LR adaptation (20x/iter) and fixed
    old_mu/sigma from rollout (no per-minibatch refresh).

    Step 4d (PPO+History, no encoder) achieves 3.0 deg with this exact path.
    Steps 8a-10b all use _update_encoder_ppo() and fail (21-32 deg).
    This experiment tests whether the custom encoder update path itself
    is the failure cause, not the encoder architecture.

    Hyperparams identical to Q1Q3 baseline (lr=3e-4, desired_kl=0.02).
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    use_encoder_update: bool = False
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOStdUpdateRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 11a: Encoder uses standard update() path, enc norm kept.

    Tests hypothesis: _update_encoder_ppo() (per-epoch LR + mu/sigma refresh)
    is the cause of encoder training failure, not encoder architecture.
    Uses identical update path as Step 4d (PPO+History success at 3.0 deg).
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_std_update"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _StdUpdateAlgorithmCfg()
    policy = _Q1Q3EncoderPolicyCfg()


@configclass
class _StdUpdateNoEncNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Standard update policy without encoder input normalization."""

    encoder_obs_normalization: bool = False


@configclass
class ALBCDebugPPOStdUpdateNoEncNormRunnerCfg(ALBCDebugPPOStdUpdateRunnerCfg):
    """Step 11a-noenc: Same as 11a but without p_t normalization before encoder."""

    experiment_name = "constrained_albc_debug_ppo_std_update_no_enc_norm"
    policy = _StdUpdateNoEncNormPolicyCfg()


# =============================================================================
# Step 12: HORA-style reward scaling (reward_scale=0.01)
# =============================================================================


@configclass
class _RewardScaleAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with HORA-style reward scaling.

    HORA multiplies all rewards by 0.01 before computing returns/advantages.
    This reduces surrogate gradient by 100x, which:
    - Reduces z change per step by 100x -> KL contribution from encoder ~10000x smaller
    - Allows adaptive LR to stay high (KL well below desired_kl)
    - Actor can learn normally despite encoder shifting z

    Baseline: Step 8a (Q1Q3, enc norm). Only change: reward_scale=0.01.
    All other hyperparams identical to _Q1Q3AlgorithmCfg.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    reward_scale: float = 0.01
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPORewardScaleRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 12a: Q1+Q3 + HORA reward_scale=0.01, enc norm kept.

    Baseline: Step 8a (Q1Q3, best encoder result at 23/20 deg).
    Expected: reward_scale=0.01 reduces gradient 100x -> LR stays high -> policy learns.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_reward_scale"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _RewardScaleAlgorithmCfg()
    policy = _Q1Q3EncoderPolicyCfg()


@configclass
class _RewardScaleNoEncNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Reward scale policy without encoder input normalization."""

    encoder_obs_normalization: bool = False


@configclass
class ALBCDebugPPORewardScaleNoEncNormRunnerCfg(ALBCDebugPPORewardScaleRunnerCfg):
    """Step 12b: Same as 12a but without p_t normalization before encoder."""

    experiment_name = "constrained_albc_debug_ppo_reward_scale_no_enc_norm"
    policy = _RewardScaleNoEncNormPolicyCfg()


# =============================================================================
# Step 13: Static min-max normalization (HORA-style, replaces EmpiricalNorm)
# =============================================================================

# 23D privileged obs bounds derived from DomainRandomizationCfg + HydrodynamicsCfg.
# Each pair is (lower, upper) with ~10% margin beyond actual DR range.
# Layout: hydro(6) + inertia(4) + damping(4) + body(2) + payload(4) + actuator(2) + env(1)
_PRIV_OBS_LOWER: list[float] = [
    # Hydrodynamics (6D): main [volume, CoG_z, CoB_z], buoy [volume, CoG_z, CoB_z]
    0.007,    # [0] main volume: nom=0.009, DR*0.9=0.0081
    -0.08,    # [1] main CoG_z: nom=-0.05, offset=(-0.02,0.02)
    -0.03,    # [2] main CoB_z: nom=0.0, offset=(-0.02,0.02)
    0.002,    # [3] buoy volume: nom=0.00268, DR*0.9=0.00241
    0.03,     # [4] buoy CoG_z: nom=0.059, offset=(-0.02,0.02)
    0.03,     # [5] buoy CoB_z: nom=0.059, offset=(-0.02,0.02)
    # Inertia (4D): main [Ixx, Iyy], buoy [Ixx, Iyy]
    0.06,     # [6] main Ixx: nom=0.0994, DR*0.75=0.0746
    0.06,     # [7] main Iyy: same
    0.0015,   # [8] buoy Ixx: nom=0.00278, DR*0.75=0.00209
    0.0015,   # [9] buoy Iyy: same
    # Damping (4D): linear [roll, pitch], quadratic [roll, pitch]
    0.10,     # [10] lin_damp roll: nom=0.3, DR*0.5=0.15
    0.10,     # [11] lin_damp pitch: same
    0.3,      # [12] quad_damp roll: nom=1.0, DR*0.5=0.5
    0.3,      # [13] quad_damp pitch: same
    # Body properties (2D): mass, added_mass_surge
    7.0,      # [14] body_mass: nom=9.18, DR*0.9=8.26
    5.0,      # [15] added_mass_surge: nom=8.0, DR*0.85=6.8
    # Payload (4D): mass, cog_offset [x, y, z]
    -0.1,     # [16] payload_mass: DR range (0.0, 1.0)
    -0.12,    # [17] payload_cog_x: disk r=0.10
    -0.12,    # [18] payload_cog_y: disk r=0.10
    -0.04,    # [19] payload_cog_z: DR range (-0.03, 0.0)
    # Actuator (2D): stiffness, damping
    30.0,     # [20] joint_stiffness: DR range (40, 120)
    0.3,      # [21] joint_damping: DR range (0.5, 5.0)
    # Environment (1D): water_density
    990.0,    # [22] water_density: DR range (995, 1025)
]

_PRIV_OBS_UPPER: list[float] = [
    # Hydrodynamics (6D)
    0.011,    # [0] main volume: DR*1.1=0.0099
    -0.02,    # [1] main CoG_z: nom=-0.05+0.02=-0.03
    0.03,     # [2] main CoB_z: nom=0.0+0.02=0.02
    0.0035,   # [3] buoy volume: DR*1.1=0.00295
    0.09,     # [4] buoy CoG_z: nom=0.059+0.02=0.079
    0.09,     # [5] buoy CoB_z: nom=0.059+0.02=0.079
    # Inertia (4D)
    0.15,     # [6] main Ixx: DR*1.3=0.1292
    0.15,     # [7] main Iyy: same
    0.004,    # [8] buoy Ixx: DR*1.3=0.00361
    0.004,    # [9] buoy Iyy: same
    # Damping (4D)
    0.50,     # [10] lin_damp roll: DR*1.5=0.45
    0.50,     # [11] lin_damp pitch: same
    1.8,      # [12] quad_damp roll: DR*1.5=1.5
    1.8,      # [13] quad_damp pitch: same
    # Body properties (2D)
    12.0,     # [14] body_mass: DR*1.1=10.10
    11.0,     # [15] added_mass_surge: DR*1.15=9.2
    # Payload (4D)
    1.2,      # [16] payload_mass: DR max=1.0
    0.12,     # [17] payload_cog_x: disk r=0.10
    0.12,     # [18] payload_cog_y: disk r=0.10
    0.01,     # [19] payload_cog_z: DR max=0.0
    # Actuator (2D)
    130.0,    # [20] joint_stiffness: DR max=120
    6.0,      # [21] joint_damping: DR max=5.0
    # Environment (1D)
    1030.0,   # [22] water_density: DR max=1025
]


@configclass
class _StaticNormPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Q1Q3 encoder policy with HORA-style static min-max normalization.

    Replaces EmpiricalNormalization with deterministic min-max scaling.
    EmpiricalNorm causes z drift via running stats updates every env step,
    which is the hidden cause of KL spike (independent of encoder gradient).
    Static normalization eliminates this drift entirely.
    """

    encoder_obs_normalization: bool = False
    encoder_obs_lower: list[float] = _PRIV_OBS_LOWER
    encoder_obs_upper: list[float] = _PRIV_OBS_UPPER


@configclass
class _StaticNormRewardScaleAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with static normalization + HORA reward scaling.

    Combines two key HORA elements:
    - reward_scale=0.01: reduces gradient 100x, prevents KL spike from large updates
    - Static encoder normalization: eliminates normalizer-induced z drift

    Step 12a showed reward_scale=0.01 alone gives first positive signal (noise_std 0.92).
    Step 13a adds static normalization to eliminate the remaining KL source.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    reward_scale: float = 0.01
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOStaticNormRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 13b: Q1+Q3 + static min-max normalization (no reward scaling).

    Isolates the effect of replacing EmpiricalNorm with static min-max.
    Baseline: Step 8a (Q1Q3, EmpiricalNorm, 23/20 deg).
    Expected: static norm eliminates normalizer z drift -> lower KL -> higher LR.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_static_norm"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _Q1Q3AlgorithmCfg()
    policy = _StaticNormPolicyCfg()


@configclass
class ALBCDebugPPOStaticNormRSRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 13a: Q1+Q3 + static min-max + HORA reward_scale=0.01.

    Full HORA alignment: static normalization + reward scaling.
    Baseline: Step 12a (reward_scale=0.01 + EmpiricalNorm, 17.9/43.8 deg).
    Expected: eliminating z drift + small gradient -> encoder and actor both learn.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_static_norm_rs"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _StaticNormRewardScaleAlgorithmCfg()
    policy = _StaticNormPolicyCfg()


# =============================================================================
# Step 14: Encoder freeze (encoder_grad_scale=0.0, validate root cause)
# =============================================================================


@configclass
class _EncFreezeAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with encoder completely frozen (grad_scale=0.0).

    Hypothesis validation: if encoder weight changes cause KL spike,
    then freezing encoder should restore normal PPO learning.
    Actor should learn using o_t + history, treating z as fixed noise.
    Expected: noise_std decreases, LR stays high, roll/pitch approach Step 4d (3 deg).
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    encoder_grad_scale: float = 0.0
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.02
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugPPOEncFreezeRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 14: Encoder completely frozen (grad_scale=0.0).

    Root cause validation experiment.
    If PPO learns normally with frozen encoder -> encoder weight changes ARE the KL cause.
    If still fails -> diagnosis is wrong, other factor at play.
    Baseline: Step 4d (PPO+History, no encoder, 3.0/3.8 deg).
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 300
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_enc_freeze"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _EncFreezeAlgorithmCfg()
    policy = _StaticNormPolicyCfg()


# =============================================================================
# Step 15: Symmetric Critic (Critic Asymmetry KL Test)
# =============================================================================


@configclass
class _SymCriticPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Encoder policy with symmetric critic.

    Critic sees cat([o_t, hist]) instead of cat([o_t, p_t]).
    Same info as Step 4d's critic (134D), matching the actor input minus z.
    Tests whether privileged critic's accurate advantages cause KL spike.
    """

    symmetric_critic: bool = True


@configclass
class ALBCDebugPPOSymCriticRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 15: Encoder + Symmetric Critic.

    Tests hypothesis: privileged critic produces sharp advantages that cause
    large policy gradients and KL spike. With symmetric critic, advantage
    accuracy matches Step 4d (PPO+History, 3.0 deg success).

    If KL normalizes and policy learns -> critic asymmetry interaction confirmed.
    If still fails -> critic asymmetry is not the cause.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_sym_critic"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "proprio_hist"],
    }

    algorithm = _Q1Q3AlgorithmCfg()
    policy = _SymCriticPolicyCfg()


# =============================================================================
# Step 16: Scalar noise_std (log_std -> std parameterization test)
# =============================================================================


@configclass
class _ScalarStdPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Encoder policy with scalar std (matching ActorCritic's parameterization).

    ActorCritic (Step 4d success) uses self.std = Parameter(1.0) (additive gradient).
    ActorCriticEncoder uses self.log_std = Parameter(0.0) (multiplicative gradient via exp).
    Tests whether log-space parameterization prevents noise_std from decreasing.
    """

    noise_std_type: str = "scalar"


@configclass
class ALBCDebugPPOScalarStdRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 16: Encoder + Scalar noise_std.

    Tests hypothesis: log_std parameterization in ActorCriticEncoder structurally
    prevents noise_std from decreasing, keeping policy at random (noise_std CEILING).
    ActorCritic (Step 4d, 3.0 deg) uses scalar std with additive gradient updates.

    If noise_std decreases and policy learns -> log_std is the root cause.
    If still fails -> parameterization is not the cause, try next suspect.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_scalar_std"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _Q1Q3AlgorithmCfg()
    policy = _ScalarStdPolicyCfg()


# =============================================================================
# Step 17: No Action Clamp (action clamp removal test)
# =============================================================================


@configclass
class _NoClampPolicyCfg(_Q1Q3EncoderPolicyCfg):
    """Encoder policy without action clamping.

    ActorCritic (Step 4d success) returns distribution.sample() with no clamp.
    ActorCriticEncoder clamps to [-1, 1], truncating ~32% of samples at noise_std=1.0.
    Clamped actions at boundaries create sharp log_prob gradients that amplify KL.
    """

    clamp_actions: bool = False


@configclass
class ALBCDebugPPONoClampRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 17: Encoder without action clamping.

    Tests hypothesis: sample().clamp(-1,1) concentrates actions at boundaries,
    amplifying log_prob sensitivity to mu changes -> KL spike -> LR death.
    ActorCritic (Step 4d, 3.0 deg) has no action clamp.

    If KL normalizes and policy learns -> action clamp is the root cause.
    If still fails -> clamp is not the cause.
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_no_clamp"
    normalize_value: bool = True
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _Q1Q3AlgorithmCfg()
    policy = _NoClampPolicyCfg()
