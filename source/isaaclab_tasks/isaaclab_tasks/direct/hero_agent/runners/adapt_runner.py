# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 supervised adaptation runner for Hero Agent Encoder-TDC.

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

from ..encoder.normalization import RunningMeanStd
from ..utils.env_utils import connect_encoder_to_env, unwrap_env
from ..utils.logging import flush_metrics, log_encoder_tdc_metrics, pearson_r


class _WandbTBWriter:
    """Thin adapter that forwards ``add_scalar`` to both TensorBoard and WandB.

    ``flush_metrics()`` relies on ``writer.add_scalar()`` for all scalar logging.
    When WandB is the logger backend, we still need TensorBoard records, so this
    adapter wraps a real TB SummaryWriter and additionally calls ``wandb.log()``.
    """

    def __init__(self, tb_writer):
        self._tb = tb_writer

    def add_scalar(self, tag: str, value, global_step: int | None = None, **kw) -> None:
        self._tb.add_scalar(tag, value, global_step, **kw)
        try:
            import wandb

            wandb.log({tag: value}, step=global_step, commit=False)
        except ImportError:
            pass


class AdaptRunner:
    """Phase 2 supervised training runner for adaptation module.

    Loads a Phase 1 checkpoint, freezes all weights except the adaptation
    module (adapt_tconv), and trains with L2 loss between z_hat (from
    proprioception history) and z_gt (from frozen encoder on privileged info).

    The runner maintains:
        - Frozen actor for on-policy action generation
        - Frozen encoder for ground truth z computation
        - Trainable adapt_tconv for z estimation from history
        - RunningMeanStd for proprioception history normalization
    """

    def __init__(
        self,
        env,
        policy,
        runner_cfg,
        device: str | torch.device = "cuda:0",
        log_dir: str | None = None,
        proprio_history_shape: tuple[int, int] = (30, 12),
        logger_type: str = "tensorboard",
    ):
        """Initialize the adaptation runner.

        Args:
            env: Wrapped environment (RslRlVecEnvWrapper).
            policy: ActorCriticEncoderTDCAdapt instance (already loaded with Phase 1 weights).
            runner_cfg: HeroAgentAdaptTDCRunnerCfg with Phase 2 hyperparameters.
            device: Computation device.
            log_dir: Directory for logs and checkpoints.
            proprio_history_shape: Shape of history buffer (H, D) for normalization.
            logger_type: Logger backend ("tensorboard" or "wandb").
        """
        self.env = env
        self.cfg = runner_cfg
        self.device = device
        self.log_dir = log_dir
        self.logger_type = logger_type
        self.policy = policy.to(self.device)

        # Freeze base, optimizer only on adapt_tconv
        self.policy.freeze_base()
        self.optimizer = torch.optim.Adam(self.policy.get_adapt_parameters(), lr=self.cfg.adapt_lr)

        # History normalization (train mode: online updates)
        self.hist_normalizer = RunningMeanStd(proprio_history_shape).to(self.device)

        # Wire encoder policy to env for M_hat extraction
        connect_encoder_to_env(self.env, self.policy, "AdaptRunner")

        # Logging setup
        self.writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._setup_writer()

        # Training stats
        self.agent_steps = 0
        self.total_time = 0.0

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

    def learn(self) -> None:
        """Run supervised adaptation training loop.

        Training parameters are read from self.cfg (HeroAgentAdaptTDCRunnerCfg):
            - max_agent_steps: Total environment steps.
            - save_interval_steps: Checkpoint frequency in agent steps.
            - log_interval: Logging frequency in iterations.
        """
        max_agent_steps = self.cfg.max_agent_steps
        save_interval = self.cfg.save_interval_steps
        log_interval = self.cfg.log_interval

        obs_dict = self.env.get_observations()
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
            self.cfg.adapt_lr,
        )

        trainable_params = sum(p.numel() for p in self.policy.get_adapt_parameters())
        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info("Trainable params: %s / %s total", f"{trainable_params:,}", f"{total_params:,}")

        # Curriculum support: get raw env for reward manager access
        raw_env = unwrap_env(self.env)
        has_curriculum = hasattr(raw_env, "_reward_manager") and hasattr(raw_env.cfg, "reward")

        while self.agent_steps < max_agent_steps:
            obs_dict, loss, rewards, z_hat, z_gt = self._train_step(obs_dict)
            self.agent_steps += num_envs
            iteration += 1

            # Update reward curriculum
            if has_curriculum:
                raw_env._reward_manager.update_curriculum(iteration, raw_env.cfg.reward.curriculum_end_iter)

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
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single adaptation step: normalize -> z_hat/z_gt -> loss -> env.step.

        Args:
            obs_dict: Current observation dictionary from environment.

        Returns:
            Tuple of (new_obs_dict, loss, rewards, z_hat, z_gt).
        """
        # Normalize proprioception history
        obs_dict_normalized = dict(obs_dict)
        if "proprio_hist" in obs_dict:
            obs_dict_normalized["proprio_hist"] = self.hist_normalizer(obs_dict["proprio_hist"])

        # Compute z_hat (with gradients) and z_gt (frozen)
        policy_obs = obs_dict_normalized[self.policy._policy_obs_key]
        z_hat_raw = self.policy.adapt_tconv(obs_dict_normalized[self.policy._proprio_hist_key])
        z_hat = self.policy._softplus_z(z_hat_raw)
        z_gt = self.policy.compute_z_gt(obs_dict_normalized)

        # L2 supervised loss
        loss = ((z_hat - z_gt.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.get_adapt_parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # Generate action from frozen actor using same z_hat (detached)
        with torch.no_grad():
            z_hat_detached = z_hat.detach()
            self.policy._last_z = z_hat_detached  # For env M_hat extraction
            combined_obs = torch.cat([policy_obs, z_hat_detached], dim=-1)
            actor_obs = self.policy.actor_obs_normalizer(combined_obs)
            output = self.policy.actor(actor_obs)
            actions = output[..., 0, :] if self.policy.state_dependent_std else output
            actions = actions.clamp(-1.0, 1.0)

        # Step environment
        new_obs_dict, rewards, _, _ = self.env.step(actions)
        return new_obs_dict, loss, rewards, z_hat, z_gt

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

        # z_hat per-dimension statistics (mean + std)
        if z_hat is not None:
            for i in range(z_hat.shape[-1]):
                metrics[f"Adapt/z_hat_dim{i}_mean"] = z_hat[:, i].mean().item()
                metrics[f"Adapt/z_hat_dim{i}_std"] = z_hat[:, i].std().item()

        # z_gt per-dimension statistics for comparison
        if z_gt is not None:
            for i in range(z_gt.shape[-1]):
                metrics[f"Adapt/z_gt_dim{i}_mean"] = z_gt[:, i].mean().item()
                metrics[f"Adapt/z_gt_dim{i}_std"] = z_gt[:, i].std().item()

        # z_hat vs z_gt Pearson correlation (adaptation tracking quality)
        if z_hat is not None and z_gt is not None:
            metrics["Adapt/z_hat_z_gt_corr"] = pearson_r(z_hat.flatten(), z_gt.detach().flatten()).item()

        # Per-dimension L2 loss
        if z_hat is not None and z_gt is not None:
            per_dim_loss = (z_hat - z_gt.detach()).pow(2).mean(dim=0)
            for i in range(per_dim_loss.shape[0]):
                metrics[f"Adapt/l2_loss_dim{i}"] = per_dim_loss[i].item()

        # M_hat from z_hat[3:5] and estimation error %
        if z_hat is not None and z_hat.shape[-1] > 4:
            metrics["Adapt/m_hat_roll_mean"] = z_hat[:, 3].mean().item()
            metrics["Adapt/m_hat_pitch_mean"] = z_hat[:, 4].mean().item()

            if z_gt is not None and z_gt.shape[-1] > 4:
                for i, axis in enumerate(("roll", "pitch")):
                    idx = 3 + i
                    gt_val = z_gt[:, idx].mean().item()
                    hat_val = z_hat[:, idx].mean().item()
                    pct_err = abs(hat_val - gt_val) / max(abs(gt_val), 1e-8) * 100
                    metrics[f"Adapt/m_hat_{axis}_error_pct"] = pct_err

        # TDC integration metrics (accumulates into same dict)
        log_encoder_tdc_metrics(
            writer=self.writer,
            policy=self.policy,
            env=self.env,
            iteration=iteration,
            device=self.device,
            logger_type=self.logger_type,
            metrics=metrics,
        )

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
        """Save checkpoint with adapt_tconv, normalizer, and training state.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hist_normalizer_state_dict": self.hist_normalizer.state_dict(),
                "agent_steps": self.agent_steps,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load(self, path: str) -> None:
        """Load checkpoint to resume training.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.policy.freeze_base()
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "hist_normalizer_state_dict" in checkpoint:
            self.hist_normalizer.load_state_dict(checkpoint["hist_normalizer_state_dict"])
        if "agent_steps" in checkpoint:
            self.agent_steps = checkpoint["agent_steps"]
        logger.info("Loaded checkpoint: %s (agent_steps=%s)", path, f"{self.agent_steps:,}")
