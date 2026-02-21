# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-phase Encoder-TDC runner: joint adapt_tconv + actor/critic training.

Unlike the two-phase HORA approach (Phase 1: privileged encoder + RL, Phase 2:
supervised adaptation), this runner trains all components jointly in one phase:
    - adapt_tconv: proprio_hist -> z_hat (3D: [m_A, I_roll, I_pitch])
    - actor: policy_obs + z_hat -> Kp/Kd (gradient from PPO surrogate loss)
    - critic: policy_obs + z_hat -> value (gradient from PPO value loss)
    - encoder: UNUSED (exists in module for checkpoint compatibility)

Gradient flow:
    - z_hat is DETACHED in _get_combined_obs(): PPO gradient does NOT reach adapt_tconv
    - Only aux MSE loss trains adapt_tconv (z_hat vs z_true from privileged obs)
    - Separate optimizer: adapt_tconv uses fixed LR (adapt_lr), immune to KL-adaptive schedule
    - Separate gradient clipping (adapt_max_grad_norm=10.0) as safety measure

Extends OnPolicyRunner with:
    1. PPOWithMHatAuxLoss instead of standard PPO (separate grad clip)
    2. RunningMeanStd normalizer for proprio_hist (applied during rollout)
    3. connect_encoder_to_env for M_hat extraction
    4. DR curriculum and TDC metric logging
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import deque

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from ..encoder.adaptation import ActorCriticEncoderTDCAdapt  # noqa: F401 — needed by eval()
from ..encoder.normalization import RunningMeanStd
from ..utils.logging import connect_encoder_to_env, flush_metrics, log_encoder_tdc_metrics, unwrap_env

logger = logging.getLogger(__name__)


class SinglePhaseTDCRunner(OnPolicyRunner):
    """Single-phase joint training: adapt_tconv + actor/critic via PPO + aux M_hat loss.

    Unlike AdaptRunner (which freezes actor/critic/encoder and only trains
    adapt_tconv with supervised L2 loss), this runner trains ALL components
    jointly. The encoder module is present but unused -- z_hat comes from
    adapt_tconv(proprio_hist), not from encoder(privileged).

    Key differences from OnPolicyRunner:
        - PPOWithMHatAuxLoss: aux MSE loss on z_hat for M_hat supervision
        - RunningMeanStd: normalizes proprio_hist before policy forward passes
        - connect_encoder_to_env: wires z_hat -> env for M_hat extraction
        - DR/reward curriculum: ramps difficulty over training iterations
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Extract single-phase config before super().__init__ pops dict keys
        self._aux_mhat_weight = train_cfg.get("aux_mhat_loss_weight", 1.0)
        self._z_true_indices = tuple(train_cfg.get("z_true_privileged_indices", (21, 14, 15)))
        self._adapt_max_grad_norm = train_cfg.get("adapt_max_grad_norm", 10.0)
        self._adapt_lr = train_cfg.get("adapt_lr", 3e-4)
        self._min_lr = train_cfg.get("min_lr", 1e-4)
        self._proprio_shape = (
            train_cfg["policy"].get("proprio_history_len", 30),
            train_cfg["policy"].get("proprio_feature_dim", 12),
        )

        super().__init__(env, train_cfg, log_dir, device)

        # Proprio history normalizer (Welford's online algorithm)
        self.hist_normalizer = RunningMeanStd(self._proprio_shape).to(self.device)

        # Wire encoder policy to env for M_hat extraction by TDC controller
        connect_encoder_to_env(self.env, self.alg.policy, "SinglePhaseTDCRunner")

        # Cache raw env for curriculum updates
        self._raw_env = unwrap_env(self.env)

    def _construct_algorithm(self, obs: TensorDict):
        """Create PPOWithMHatAuxLoss with ActorCriticEncoderTDCAdapt policy."""
        # Resolve RND and symmetry configs (same as parent)
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Handle deprecated empirical_normalization (same as parent)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Create policy (ActorCriticEncoderTDCAdapt)
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Create PPOWithMHatAuxLoss instead of standard PPO
        from .ppo_aux_mhat import PPOWithMHatAuxLoss

        # Pop class_name so it's not forwarded to PPO.__init__
        self.alg_cfg.pop("class_name", None)
        alg = PPOWithMHatAuxLoss(
            actor_critic,
            aux_mhat_weight=self._aux_mhat_weight,
            z_true_priv_indices=self._z_true_indices,
            adapt_max_grad_norm=self._adapt_max_grad_norm,
            adapt_lr=self._adapt_lr,
            min_lr=self._min_lr,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # Initialize rollout storage
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])

        return alg

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Training loop with proprio normalization, DR curriculum, and TDC logging.

        Extends OnPolicyRunner.learn() with:
            1. RunningMeanStd normalization on obs["proprio_hist"] before act/compute_returns
            2. DR and reward curriculum updates per iteration
            3. TDC-specific metric logging
        """
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations and normalize
        obs = self.env.get_observations().to(self.device)
        self._normalize_obs(obs)
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # RND buffers
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Synchronize parameters for distributed training
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        logger.info(
            "Starting single-phase TDC training: %d iterations, %d envs, aux_weight=%.2f",
            num_learning_iterations,
            self.env.num_envs,
            self._aux_mhat_weight,
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # === Rollout phase ===
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Normalize proprio_hist in new observations
                    self._normalize_obs(obs)
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns (obs already normalized)
                self.alg.compute_returns(obs)

            # === Update phase ===
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save final model
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with DR/reward curriculum updates and TDC metrics."""
        super().log(locs, width, pad)

        iteration = locs["it"]

        # Update reward curriculum (sigma annealing)
        if hasattr(self._raw_env, "_reward_manager") and hasattr(self._raw_env.cfg, "reward"):
            self._raw_env._reward_manager.update_curriculum(iteration, self._raw_env.cfg.reward.curriculum_end_iter)

        # Update DR curriculum (perturbation/inertia/mass ramp)
        if hasattr(self._raw_env, "update_dr_curriculum"):
            self._raw_env.update_dr_curriculum(iteration)

        # Log DR curriculum progress
        if (
            hasattr(self._raw_env, "_dr_curriculum_cfg")
            and self._raw_env._dr_curriculum_cfg is not None
            and self.log_dir is not None
            and not self.disable_logs
        ):
            dr_cur = self._raw_env._dr_curriculum_cfg
            progress = min(1.0, iteration / max(1, dr_cur.end_iter))
            rand = self._raw_env.cfg.randomization
            self.writer.add_scalar("DR_Curriculum/progress", progress, iteration)
            self.writer.add_scalar("DR_Curriculum/perturbation_force_max", rand.perturbation_force_range[1], iteration)
            self.writer.add_scalar("DR_Curriculum/inertia_scale_hi", rand.inertia_scale[1], iteration)

        # Log learning rates (adapt_lr fixed, min_lr floor for PPO schedule)
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Loss/adapt_lr", self.alg.adapt_lr, iteration)
            self.writer.add_scalar("Loss/min_lr", self.alg.min_lr, iteration)

        # Log z_hat statistics (from last policy forward pass)
        if self.log_dir is not None and not self.disable_logs:
            z_hat = getattr(self.alg.policy, "_last_z", None)
            if z_hat is not None:
                with torch.no_grad():
                    for i in range(z_hat.shape[-1]):
                        self.writer.add_scalar(f"z_hat/dim{i}_mean", z_hat[:, i].mean().item(), iteration)
                        self.writer.add_scalar(f"z_hat/dim{i}_std", z_hat[:, i].std().item(), iteration)

            # TDC controller metrics (M_hat accuracy, tracking error, etc.)
            metrics: dict[str, float] = {}
            log_encoder_tdc_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=iteration,
                device=self.device,
                logger_type=getattr(self, "logger_type", "tensorboard"),
                metrics=metrics,
            )
            if metrics:
                flush_metrics(self.writer, metrics, iteration, getattr(self, "logger_type", "tensorboard"))

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save checkpoint including hist_normalizer and adapt_optimizer state."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "adapt_optimizer_state_dict": self.alg.adapt_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "hist_normalizer_state_dict": self.hist_normalizer.state_dict(),
        }
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load checkpoint including hist_normalizer and adapt_optimizer state."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "adapt_optimizer_state_dict" in loaded_dict:
                self.alg.adapt_optimizer.load_state_dict(loaded_dict["adapt_optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]

        # Restore normalizer state
        if "hist_normalizer_state_dict" in loaded_dict:
            self.hist_normalizer.load_state_dict(loaded_dict["hist_normalizer_state_dict"])

        # Re-connect encoder to env after loading new weights
        connect_encoder_to_env(self.env, self.alg.policy, "SinglePhaseTDCRunner")

        return loaded_dict["infos"]

    def _normalize_obs(self, obs: TensorDict) -> None:
        """Normalize proprio_hist in observation dict in-place.

        The RunningMeanStd normalizer updates running statistics when in train
        mode, and applies frozen normalization in eval mode.
        """
        key = "proprio_hist"
        if key in obs:
            obs[key] = self.hist_normalizer(obs[key])
