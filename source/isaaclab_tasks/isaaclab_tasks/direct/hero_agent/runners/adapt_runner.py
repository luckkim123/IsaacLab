# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 supervised adaptation runner for Hero Agent (base RL pipeline).

This runner trains the adaptation module (ProprioAdaptTConv) to estimate the
encoder latent z from proprioception history only, using L2 loss against the
frozen Phase 1 encoder output.

Training loop:
    1. Collect on-policy data: obs = env.step(frozen_actor(z_hat))
    2. Compute z_hat = adapt_tconv(proprio_hist)
    3. Compute z_gt = frozen_encoder(privileged)
    4. Minimize L2 loss: ||z_hat - z_gt||^2
    5. Only adapt_tconv parameters are updated

Reference:
    HORA ProprioAdapt (references/hora/hora/algo/padapt/padapt.py)
"""

from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

from ..utils.logging import (
    _WandbTBWriter,
    flush_metrics,
    pearson_r,
    unwrap_env,
)


class AdaptRunner:
    """Phase 2 supervised training runner for adaptation module (base RL).

    Loads a Phase 1 checkpoint, freezes all weights except the adaptation
    module (adapt_tconv), and trains with L2 loss between z_hat (from
    proprioception history) and z_gt (from frozen encoder on privileged info).

    The runner maintains:
        - Frozen actor for on-policy action generation
        - Frozen encoder for ground truth z computation
        - Trainable adapt_tconv for z estimation from history
        - hist_normalizer inside policy (RunningMeanStd, saved in model state_dict)

    Constructor follows the OnPolicyRunner interface: (env, train_cfg, log_dir, device).
    The policy is constructed internally from the config dict.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str | torch.device = "cuda:0",
    ):
        """Initialize the adaptation runner.

        Args:
            env: Wrapped environment (RslRlVecEnvWrapper).
            train_cfg: Runner config dict (from HeroAgentAdaptBaseRunnerCfg.to_dict()).
            log_dir: Directory for logs and checkpoints.
            device: Computation device.
        """
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.logger_type = train_cfg.get("logger_type", "tensorboard")

        # Create policy from config (same pattern as OnPolicyRunner)
        import rsl_rl.runners.on_policy_runner as runner_module

        policy_class_name = train_cfg["policy_class_name"]
        actor_critic_class = getattr(runner_module, policy_class_name)
        obs = env.get_observations()
        obs_groups = train_cfg.get("obs_groups", {})
        policy_cfg = train_cfg["policy"]
        self.policy = actor_critic_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)

        # Freeze base, optimizer only on adapt_tconv
        adapt_lr = train_cfg.get("adapt_lr", 3e-4)
        self.policy.freeze_base()
        self.optimizer = torch.optim.Adam(self.policy.get_adapt_parameters(), lr=adapt_lr)

        # Logging setup
        self.writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._setup_writer()

        # Training stats
        self.agent_steps = 0
        self.total_time = 0.0

        # EMA accumulators for episode metrics from env.extras["log"]
        self._ema_extras: dict[str, float] = {}
        self._ema_alpha = 0.2  # Base EMA alpha (scaled by _num_resets fraction)

    def _setup_writer(self) -> None:
        """Initialize TensorBoard writer, with optional WandB dual-write.

        Always creates a TensorBoard SummaryWriter so that ``add_scalar()``
        works regardless of logger_type.  When WandB is active, wraps it in
        a thin adapter that forwards ``add_scalar`` to both TB and WandB.
        """
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if self.logger_type == "wandb":
            try:
                import wandb

                if wandb.run is not None:
                    self.writer = _WandbTBWriter(tb_writer)
                    return
                else:
                    logger.warning("WandB not initialized. Falling back to TensorBoard.")
                    self.logger_type = "tensorboard"
            except ImportError:
                self.logger_type = "tensorboard"

        self.writer = tb_writer

    def add_git_repo_to_log(self, _file_path: str) -> None:
        """Compatibility stub for OnPolicyRunner interface (no-op)."""
        pass

    def learn(self, num_learning_iterations: int | None = None, init_at_random_ep_len: bool = False) -> None:
        """Run supervised adaptation training loop.

        Args:
            num_learning_iterations: Unused (AdaptRunner uses agent steps from cfg).
            init_at_random_ep_len: Unused (kept for OnPolicyRunner interface compat).
        """
        max_agent_steps = self.cfg.get("max_agent_steps", 100_000_000)
        save_interval = self.cfg.get("save_interval_steps", 10_000_000)
        log_interval = self.cfg.get("log_interval", 10)
        max_grad_norm = self.cfg.get("max_grad_norm", 10.0)
        adapt_lr = self.cfg.get("adapt_lr", 3e-4)

        num_envs = self.env.num_envs
        iteration = 0
        last_save_steps = 0
        start_time = time.time()

        # Accumulators for logging
        loss_acc = 0.0
        reward_acc = 0.0
        log_count = 0

        logger.info(
            "Starting Phase 2 training: %s agent steps, %d envs, save every %s steps, lr=%s",
            f"{max_agent_steps:,}",
            num_envs,
            f"{save_interval:,}",
            adapt_lr,
        )

        trainable_params = sum(p.numel() for p in self.policy.get_adapt_parameters())
        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info("Trainable params: %s / %s total", f"{trainable_params:,}", f"{total_params:,}")

        # DR curriculum support
        raw_env = unwrap_env(self.env)

        # Apply curriculum start values and force re-randomize.
        # Initial envs were spawned with full DR ranges (before curriculum existed).
        # Reset all envs so they sample from curriculum start values.
        if hasattr(raw_env, "update_dr_curriculum"):
            raw_env.update_dr_curriculum(0)

        # Ensure train mode (hist_normalizer updates statistics, adapt_tconv trains)
        self.policy.train()

        # Reset all envs for clean start (matching BaseRunner.learn() pattern)
        self.env.reset()
        obs_dict = self.env.get_observations()

        while self.agent_steps < max_agent_steps:
            obs_dict, loss, rewards, z_hat, z_gt = self._train_step(obs_dict, max_grad_norm)
            self.agent_steps += num_envs
            iteration += 1

            # Update DR curriculum (perturbation/inertia/mass ramp)
            if hasattr(raw_env, "update_dr_curriculum"):
                raw_env.update_dr_curriculum(iteration)

            # Collect episode metrics from env resets (EMA smoothing)
            self._update_ema_extras()

            # Accumulate for logging
            loss_acc += loss.item()
            reward_acc += rewards.mean().item()
            log_count += 1

            # Logging
            if iteration % log_interval == 0 and self.writer is not None:
                self._log_metrics(
                    iteration,
                    loss_acc / log_count,
                    reward_acc / log_count,
                    z_hat,
                    z_gt,
                    time.time() - start_time,
                )
                loss_acc = 0.0
                reward_acc = 0.0
                log_count = 0

            # Save checkpoint
            if self.log_dir is not None and self.agent_steps - last_save_steps >= save_interval:
                self.save(os.path.join(self.log_dir, f"model_{self.agent_steps}.pt"))
                last_save_steps = self.agent_steps

        # Final save
        self.total_time = time.time() - start_time
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "model_final.pt"))
        logger.info("Training complete: %s steps in %.1fs", f"{self.agent_steps:,}", self.total_time)

    def _train_step(
        self,
        obs_dict: dict,
        max_grad_norm: float = 10.0,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single adaptation step: normalize -> z_hat/z_gt -> loss -> env.step.

        Args:
            obs_dict: Current observation dictionary from environment.
            max_grad_norm: Gradient clipping threshold.

        Returns:
            Tuple of (new_obs_dict, loss, rewards, z_hat, z_gt).
        """
        # Compute z_hat (with gradients, includes normalization) and z_gt (frozen)
        z_hat = self.policy.compute_z_hat(obs_dict)
        z_gt = self.policy.compute_z_gt(obs_dict)

        # L2 supervised loss
        loss = ((z_hat - z_gt.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.get_adapt_parameters(), max_grad_norm)
        self.optimizer.step()

        # Generate action from frozen actor using same z_hat (detached)
        actions = self.policy.act_with_z_hat(obs_dict, z_hat)

        # Step environment
        new_obs_dict, rewards, _, _ = self.env.step(actions)
        return new_obs_dict, loss, rewards, z_hat, z_gt

    def _update_ema_extras(self) -> None:
        """Update EMA accumulators from environment episode metrics.

        Reads extras["log"] populated by _collect_episode_metrics() during
        env resets. Uses _num_resets-weighted alpha so high-reset steps
        carry more weight.
        """
        env_extras = getattr(self.env, "extras", {})
        log = env_extras.get("log", {})
        if not log:
            return

        num_resets = log.get("_num_resets", 0.0)
        if num_resets <= 0:
            return

        # Scale alpha by fraction of envs that reset (more resets = more signal)
        alpha = min(1.0, self._ema_alpha * num_resets / max(1, self.env.num_envs))

        for key, value in log.items():
            if key.startswith("_"):
                continue
            # Convert tensor values to float
            if isinstance(value, torch.Tensor):
                value = value.mean().item()
            if not isinstance(value, (int, float)):
                continue
            if key in self._ema_extras:
                self._ema_extras[key] = (1.0 - alpha) * self._ema_extras[key] + alpha * value
            else:
                self._ema_extras[key] = value

    def _log_metrics(
        self,
        iteration: int,
        avg_loss: float,
        avg_reward: float,
        z_hat: torch.Tensor,
        z_gt: torch.Tensor,
        elapsed: float,
    ) -> None:
        """Log all adaptation metrics to writer and console.

        Args:
            iteration: Current training iteration.
            avg_loss: Average L2 loss since last log.
            avg_reward: Average reward since last log.
            z_hat: Latest z_hat tensor for statistics.
            z_gt: Latest z_gt tensor for statistics.
            elapsed: Elapsed time since training start.
        """
        fps = self.agent_steps / elapsed if elapsed > 0 else 0

        metrics: dict[str, float] = {
            "Adapt/l2_loss": avg_loss,
            "Adapt/reward_mean": avg_reward,
            "Adapt/agent_steps": float(self.agent_steps),
            "Adapt/fps": fps,
        }

        # z_hat vs z_gt Pearson correlation (last step snapshot, not interval average)
        if z_hat is not None and z_gt is not None:
            metrics["Adapt/z_hat_z_gt_corr_snapshot"] = pearson_r(z_hat.flatten(), z_gt.detach().flatten()).item()

        # Per-dimension L2 loss (last step snapshot, identifies hardest latent dims)
        if z_hat is not None and z_gt is not None:
            per_dim_loss = (z_hat - z_gt.detach()).pow(2).mean(dim=0)
            for i in range(per_dim_loss.shape[0]):
                metrics[f"Adapt/l2_loss_dim{i}_snapshot"] = per_dim_loss[i].item()

        # Episode metrics from EMA (Episode_Reward/*, Attitude_Error/*, DR/*, etc.)
        metrics.update(self._ema_extras)

        # DR curriculum progress
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_dr_curriculum_cfg") and raw_env._dr_curriculum_cfg is not None:
            dr_cur = raw_env._dr_curriculum_cfg
            progress = min(1.0, iteration / max(1, dr_cur.end_iter))
            metrics["DR_Curriculum/progress"] = progress

        # Single flush: all scalars in one wandb.log() or TensorBoard loop
        flush_metrics(self.writer, metrics, iteration, self.logger_type)

        # Console output
        logger.info(
            "iter=%6d | steps=%12s | loss=%.6f | reward=%.4f | fps=%.0f",
            iteration,
            f"{self.agent_steps:,}",
            avg_loss,
            avg_reward,
            fps,
        )

    def save(self, path: str) -> None:
        """Save checkpoint with adapt_tconv, hist_normalizer, and training state.

        v3: hist_normalizer is part of policy.state_dict() (nn.Module submodule).
        No separate hist_normalizer_state_dict key needed.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "checkpoint_version": 3,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "agent_steps": self.agent_steps,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

        # Save DORAEMON state alongside model checkpoint
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            doraemon_path = os.path.join(os.path.dirname(path), "doraemon_state.pt")
            torch.save(raw_env._doraemon.state_dict(), doraemon_path)

    def load(self, path: str) -> None:
        """Load checkpoint, auto-detecting Phase 1 / v2 / v3 format.

        Checkpoint versions:
            v1 (Phase 1, EncoderRunner): No adapt_tconv/hist_normalizer keys.
                Loaded with strict=False (missing keys get fresh init).
            v2 (Phase 2, old): hist_normalizer_state_dict as separate top-level key.
                Migrated by injecting into model_state_dict with 'hist_normalizer.' prefix.
            v3 (Phase 2, current): hist_normalizer inside policy state_dict.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # Detect checkpoint version
        version = checkpoint.get("checkpoint_version", None)
        if version is not None:
            has_adapt = version >= 2
        else:
            has_adapt = any(k.startswith("adapt_tconv") for k in state_dict)
            logger.warning("Checkpoint lacks version key; using string-based detection (has_adapt=%s).", has_adapt)

        if has_adapt:
            # v2 -> v3 migration: inject standalone hist_normalizer into model_state_dict
            if "hist_normalizer_state_dict" in checkpoint:
                hist_sd = checkpoint["hist_normalizer_state_dict"]
                for k, v in hist_sd.items():
                    state_dict[f"hist_normalizer.{k}"] = v
                logger.info("Migrated v2 hist_normalizer_state_dict into model_state_dict.")

            # Phase 2 checkpoint: load everything strictly
            self.policy.load_state_dict(state_dict, strict=True)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "agent_steps" in checkpoint:
                self.agent_steps = checkpoint["agent_steps"]
            logger.info("Loaded Phase 2 checkpoint: %s (agent_steps=%s)", path, f"{self.agent_steps:,}")
        else:
            # Phase 1 checkpoint: missing adapt_tconv + hist_normalizer keys, strict=False
            self.policy.load_state_dict(state_dict, strict=False)
            logger.info("Loaded Phase 1 checkpoint: %s (adapt_tconv + hist_normalizer initialized fresh)", path)

        self.policy.freeze_base()

        # Restore DORAEMON state if available alongside checkpoint
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            doraemon_path = os.path.join(os.path.dirname(path), "doraemon_state.pt")
            if os.path.exists(doraemon_path):
                state = torch.load(doraemon_path, map_location=self.device, weights_only=False)
                raw_env._doraemon.load_state_dict(state)
                logger.info("[DORAEMON] Loaded distribution state from %s", doraemon_path)

    def get_inference_policy(self, device=None):
        """Return inference-ready policy function.

        Sets eval mode (freezes hist_normalizer statistics) and moves to device.
        Compatible with play.py's runner.get_inference_policy() interface.

        Args:
            device: Target device. If None, keeps current device.

        Returns:
            Callable: policy.act_inference bound method.
        """
        self.policy.eval()
        if device is not None:
            self.policy.to(device)
        return self.policy.act_inference
