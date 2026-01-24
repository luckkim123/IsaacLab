# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for UUV environments.

Hyperparameters are tuned for underwater vehicle control tasks with:
- 18-dimensional observation space (pose, velocity, goal)
- 6-dimensional action space (thruster commands)
- Slower dynamics compared to aerial vehicles due to hydrodynamic effects
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BlueROVPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for BlueROV hover task.

    This configuration uses larger networks and longer horizons compared to
    quadcopter due to the more complex dynamics of underwater vehicles.

    Note: All BlueROV variants share the same experiment_name to ensure
    checkpoints are compatible across Train/Eval/Current environments.
    """

    num_steps_per_env = 48  # Longer horizon for slower dynamics
    max_iterations = 500
    save_interval = 50
    experiment_name = "bluerov_direct"  # Consistent across all variants (Isaac Lab convention)
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],  # Larger network for complex dynamics
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Small entropy bonus for exploration
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
class BlueROVTrainPPORunnerCfg(BlueROVPPORunnerCfg):
    """PPO configuration for BlueROV training environment (no randomization).

    Inherits experiment_name from base to share checkpoints with other variants.
    """

    max_iterations = 300  # Faster convergence without randomization


@configclass
class BlueROVEvalPPORunnerCfg(BlueROVPPORunnerCfg):
    """PPO configuration for BlueROV evaluation environment (full randomization).

    Uses slightly higher entropy and more iterations to handle domain randomization.
    Inherits experiment_name from base to share checkpoints with other variants.
    """

    max_iterations = 800  # More iterations for robust policy
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for diverse experiences
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
class BlueROVCurrentPPORunnerCfg(BlueROVPPORunnerCfg):
    """PPO configuration for BlueROV with ocean currents.

    Inherits experiment_name from base to share checkpoints with other variants.
    """

    max_iterations = 600
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # Moderate entropy for current disturbances
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
