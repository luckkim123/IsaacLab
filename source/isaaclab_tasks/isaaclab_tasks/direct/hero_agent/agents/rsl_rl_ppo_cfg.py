# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for Hero Agent ALBC environments.

Provides runner configurations:
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentEncoderTDCPPORunnerCfg: Encoder-TDC integration (adaptive gains + M_hat)
    - HeroAgentUnifiedTDCPPORunnerCfg: General encoder + RL-output M_hat/Kp/Kd
    - HeroAgentAdaptTDCRunnerCfg: Phase 2 adaptation (supervised, non-PPO)

For evaluation, use CLI overrides instead of separate config classes:
    --max_iterations 100 --save_interval 25
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register ActorCriticEncoder in RSL-RL runner module namespace.
# The runner resolves policy class_name dynamically (on_policy_runner.py:416).
# This injection makes custom network classes resolvable in that scope.
from ..encoder import (
    ActorCriticEncoder,
    ActorCriticEncoderTDC,
    ActorCriticEncoderTDCAdapt,
)

_runner_module.ActorCriticEncoder = ActorCriticEncoder
_runner_module.ActorCriticEncoderTDC = ActorCriticEncoderTDC
_runner_module.ActorCriticEncoderTDCAdapt = ActorCriticEncoderTDCAdapt


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class _RslRlPpoEncoderBaseCfg(RslRlPpoActorCriticCfg):
    """Shared encoder architecture fields for all encoder-based policies.

    All encoder variants (base, TDC, TDC-adapt) share the same encoder MLP
    architecture and observation dimensions. This base avoids repeating 6 fields
    across 3 config classes.
    """

    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 6
    encoder_activation: str = "relu"
    encoder_output_activation: str = "softplus"
    z_min: float = 0.01
    policy_obs_dim: int = 13
    privileged_dim: int = 26


@configclass
class RslRlPpoActorCriticEncoderCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration with extrinsics encoder for HORA Phase 1.

    The encoder compresses privileged hydrodynamic parameters into a latent z
    that can later be replaced by a history-based adaptation module (Phase 2).
    """

    class_name: str = "ActorCriticEncoder"
    encoder_latent_dim: int = 13
    encoder_activation: str = "relu"
    encoder_output_activation: str = "softplus"


@configclass
class RslRlPpoActorCriticEncoderTDCCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration with encoder for TDC adaptive gains.

    Uses ActorCriticEncoderTDC which exposes encoder z for M_hat extraction.
    Actor outputs 4D: [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch].
    """

    class_name: str = "ActorCriticEncoderTDC"


@configclass
class RslRlPpoActorCriticEncoderTDCAdaptCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration for Phase 2 adaptation training.

    Uses ActorCriticEncoderTDCAdapt which adds the adapt_tconv module on top
    of ActorCriticEncoderTDC. The adapt module replaces the encoder for z
    estimation using proprioception history.
    """

    class_name: str = "ActorCriticEncoderTDCAdapt"

    # Adaptation module parameters
    proprio_history_len: int = 30
    proprio_feature_dim: int = 12


# =============================================================================
# Runner Configurations
# =============================================================================


@configclass
class HeroAgentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent ALBC (Active Linear Buoyancy Controller).

    Optimized for 2-DOF joint control with potential-based rewards.
    Uses standard ActorCritic without encoder.
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_albc"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentEncoderPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent encoder training (HORA Phase 1).

    Uses ActorCriticEncoder that compresses privileged info into latent z:
        - Encoder: privileged (26D) -> tanh -> z (6D)
        - Actor: cat([policy_obs, z]) = 19D -> actions
        - Critic: cat([policy_obs, z]) = 19D -> value (symmetric, forces encoder learning)
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hero_agent_albc_encoder"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentEncoderTDCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Encoder-TDC integration.

    Uses ActorCriticEncoderTDC that:
        - Encodes privileged (26D) -> z (6D), z[3:6] decomposed -> M_hat via FK
        - Actor: cat([policy_obs, z]) = 19D -> 4D gains (Kp_r, Kp_p, Kd_r, Kd_p)
        - Critic: cat([policy_obs, z]) = 19D -> value (symmetric)
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_encoder_tdc"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderTDCCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentAdaptTDCRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Phase 2 supervised adaptation training.

    This config is consumed by AdaptRunner (not OnPolicyRunner). It extends
    RslRlOnPolicyRunnerCfg for gym.register entry_point compatibility only;
    the inherited PPO fields (num_steps_per_env, max_iterations, algorithm)
    are unused.

    Phase 2 trains adapt_tconv with supervised L2 loss:
        z_hat = adapt_tconv(proprio_hist)
        z_gt  = frozen_encoder(privileged)
        loss  = ||z_hat - z_gt||^2
    """

    seed = 42
    num_steps_per_env = 1  # Unused by AdaptRunner (kept for base class compat)
    max_iterations = 1  # Unused by AdaptRunner (kept for base class compat)
    save_interval = 50  # Unused by AdaptRunner (kept for base class compat)
    experiment_name = "hero_agent_adapt_tdc"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderTDCAdaptCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg()  # Unused by AdaptRunner (kept for base class compat)

    # -- Phase 2 supervised training parameters --

    adapt_lr: float = 3e-4
    """Learning rate for adapt_tconv (Adam optimizer). Matches HORA reference."""

    max_agent_steps: int = 100_000_000
    """Total environment steps before training terminates."""

    save_interval_steps: int = 10_000_000
    """Save checkpoint every N agent steps."""

    log_interval: int = 10
    """Log metrics to writer every N iterations (1 iteration = num_envs steps)."""

    max_grad_norm: float = 10.0
    """Gradient clipping threshold for adapt_tconv. Relaxed vs Phase 1 (1.0) to
    preserve fast initial convergence while catching anomalous gradient spikes."""


# =============================================================================
# Unified TDC: General Encoder + Policy-Output M_hat/Kp/Kd
# =============================================================================


@configclass
class RslRlPpoActorCriticUnifiedTDCCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration for Unified TDC.

    General 13D latent z (no physics decomposition) with tanh output in [-1,1].
    Actor receives [policy_obs(13D) + z(13D)] = 26D and outputs 6D TDC params:
        [M_hat(2), Kp(2), Kd(2)] decoded via sigmoid scaling.
    """

    class_name: str = "ActorCriticEncoderTDC"
    encoder_latent_dim: int = 13
    encoder_output_activation: str = "tanh"
    encoder_activation: str = "elu"


@configclass
class HeroAgentUnifiedTDCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Unified TDC.

    Uses ActorCriticEncoderTDC with general 13D latent (ELU + tanh encoder):
        - Encodes privileged (26D) -> z (13D) in [-1, 1]
        - Actor: cat([policy_obs, z]) = 26D -> 6D [M_hat(2), Kp(2), Kd(2)]
        - Critic: cat([policy_obs, z]) = 26D -> value (symmetric)
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_unified_tdc"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticUnifiedTDCCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
