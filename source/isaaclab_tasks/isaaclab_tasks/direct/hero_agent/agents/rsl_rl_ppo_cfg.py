# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for Hero Agent ALBC environments.

Provides runner configurations:
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentTDEBasePPORunnerCfg: TDE-Base with training enhancements
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentEncoderTDCRunnerCfg: Encoder-TDC (RL M_hat + gains for TDC)
    - HeroAgentAdaptBaseRunnerCfg: Phase 2 adaptation (supervised, base RL)

For evaluation, use CLI overrides instead of separate config classes:
    --max_iterations 100 --save_interval 25
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register custom classes in RSL-RL runner module namespace.
# The runner resolves policy class_name and runner class dynamically.
# This injection makes custom classes resolvable in that scope.
from ..encoder import (
    ActorCriticEncoder,
    ActorCriticEncoderAdapt,
)
from ..runners import BaseRunner, EncoderRunner

_runner_module.ActorCriticEncoder = ActorCriticEncoder
_runner_module.ActorCriticEncoderAdapt = ActorCriticEncoderAdapt
_runner_module.BaseRunner = BaseRunner
_runner_module.EncoderRunner = EncoderRunner


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class _RslRlPpoEncoderBaseCfg(RslRlPpoActorCriticCfg):
    """Shared encoder architecture fields for all encoder-based policies.

    All encoder variants (base, adapt) share the same encoder MLP
    architecture and observation dimensions. This base avoids repeating fields
    across config classes.
    """

    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_output_activation: str = "sigmoid"
    encoder_obs_normalization: bool = True
    z_min: float = 0.01
    z_max: float = 2.0
    policy_obs_dim: int = 13
    privileged_dim: int = 26


@configclass
class RslRlPpoActorCriticEncoderCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration with extrinsics encoder for HORA Phase 1.

    The encoder compresses privileged hydrodynamic parameters into a latent z
    that can later be replaced by a history-based adaptation module (Phase 2).
    """

    class_name: str = "ActorCriticEncoder"


@configclass
class RslRlPpoActorCriticEncoderTDCCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration for Encoder-TDC.

    Same encoder architecture as HORA Phase 1, but the actor outputs 6D
    actions (m_hat + Kp + Kd) instead of 2D joint positions.
    num_actions=6 is set automatically from env.action_space.
    """

    class_name: str = "ActorCriticEncoder"


@configclass
class RslRlPpoActorCriticEncoderAdaptCfg(_RslRlPpoEncoderBaseCfg):
    """PPO actor-critic configuration for Phase 2 adaptation training (base RL).

    Uses ActorCriticEncoderAdapt which adds the adapt_tconv module on top
    of ActorCriticEncoder. The adapt module replaces the encoder for z
    estimation using proprioception history.
    """

    class_name: str = "ActorCriticEncoderAdapt"
    encoder_latent_dim: int = 13  # Must match Phase 1 encoder output dim

    # Adaptation module parameters
    proprio_history_len: int = 30
    proprio_feature_dim: int = 8


# =============================================================================
# Runner Configurations
# =============================================================================


@configclass
class HeroAgentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent ALBC (Active Linear Buoyancy Controller).

    Optimized for 2-DOF joint control with potential-based rewards.
    Uses BaseRunner for DORAEMON DR scheduling.
    No encoder -- serves as baseline for Encoder-Base comparison.
    """

    class_name: str = "BaseRunner"

    seed = 30
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hero_agent_albc"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentTDEBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent TDE-Base training.

    Uses BaseRunner which provides DORAEMON DR scheduling.
    No encoder -- policy learns directly from 15D obs
    (13D policy + 2D TDE dynamics mismatch).
    """

    class_name: str = "BaseRunner"

    seed = 30
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hero_agent_tde_base"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentPrivBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent with raw privileged info (ablation).

    Uses standard ActorCritic with privileged observations concatenated directly
    to the policy input (13D + 26D = 39D). No encoder compression.
    Serves as ablation baseline to isolate encoder's contribution:
        - Base (13D): no privileged info
        - Priv-Base (39D): privileged concatenated, no encoder
        - Encoder-Base (26D): privileged compressed via encoder
    """

    class_name: str = "BaseRunner"

    seed = 30
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hero_agent_priv_base"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
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
        entropy_coef=0.005,
        num_learning_epochs=5,
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

    Uses ActorCriticEncoder with symmetric critic (HORA/RMA standard):
        - Encoder: privileged (26D) -> tanh -> z (13D) in [-1, 1]
        - Actor: cat([policy_obs, z]) = 26D -> actions
        - Critic: cat([policy_obs, z]) = 26D -> value (symmetric, encoder gets critic gradient)

    Uses EncoderRunner (inherits BaseRunner) for DORAEMON + encoder-specific
    metrics logging.
    """

    class_name: str = "EncoderRunner"

    seed = 30
    num_steps_per_env = 128
    max_iterations = 2500
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentEncoderTDCRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Encoder-TDC training.

    Uses EncoderRunner (same as HORA Phase 1) with 6D action space.
    The policy outputs [m_hat(2), Kp(2), Kd(2)] which are linearly
    scaled and fed to the TDC controller.
    """

    class_name: str = "EncoderRunner"

    seed = 30
    num_steps_per_env = 128
    max_iterations = 2500
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentAdaptBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Phase 2 supervised adaptation training (base RL).

    This config is consumed by AdaptRunner (not OnPolicyRunner). It extends
    RslRlOnPolicyRunnerCfg for gym.register entry_point compatibility only;
    the inherited PPO fields (num_steps_per_env, max_iterations, algorithm)
    are unused.

    Phase 2 trains adapt_tconv with supervised L2 loss:
        z_hat = adapt_tconv(proprio_hist)
        z_gt  = frozen_encoder(privileged)
        loss  = ||z_hat - z_gt||^2
    """

    seed = 30
    num_steps_per_env = 1  # Unused by AdaptRunner (kept for base class compat)
    max_iterations = 1  # Unused by AdaptRunner (kept for base class compat)
    save_interval = 50  # Unused by AdaptRunner (kept for base class compat)
    experiment_name = "hero_agent_adapt_base"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderAdaptCfg(
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
