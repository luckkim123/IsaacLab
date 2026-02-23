# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC-MPC runner: off-policy training loop for AC-MPC architecture.

Off-policy training loop with replay buffer:
    1. Warmup: fill buffer with random actions (no gradient)
    2. Collect: step env with policy actions, store transitions
    3. Update: sample batch, run SAC update (actor/critic/alpha/dynamics)
    4. Log: metrics, save checkpoints, DR curriculum
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[SAC-MPC] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

from isaaclab_tasks.direct.hero_agent.utils.logging import (
    _WandbTBWriter,
    flush_metrics,
    unwrap_env,
)

from ..encoder.actor_critic_mpc import ActorCriticMPC
from ..encoder.sac_critic import TwinQNetwork
from .logging import (
    collect_mpc_metrics,
    collect_reward_components,
)
from .sac import SAC, ReplayBuffer, Transition


class SACMPCRunner:
    """Off-policy SAC runner for AC-MPC training.

    Training loop:
        for each iteration:
            1. Collect 1 env step (num_envs transitions)
            2. Store in replay buffer
            3. Perform `updates_per_step` SAC gradient updates
            4. Log metrics every `log_interval` iterations
            5. Save checkpoint every `save_interval` iterations
            6. Update DR/reward curriculum
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = device
        self.cfg = train_cfg

        # Unwrap to raw env for curriculum and reset access
        self._raw_env = unwrap_env(self.env)

        # ---- Build networks ----
        obs = self.env.get_observations()

        # Policy config
        pcfg = train_cfg["policy"]
        num_actions = self.env.num_actions

        # Actor (AC-MPC policy)
        actor_kwargs = {k: v for k, v in pcfg.items() if k != "class_name"}
        # Derive dynamics_dt from env (physics_dt * decimation) instead of hardcoded config
        actor_kwargs["dynamics_dt"] = self._raw_env.step_dt
        obs_groups = train_cfg.get("obs_groups", {"policy": ["policy", "mpc_state", "mpc_target"]})
        self.actor = ActorCriticMPC(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            **actor_kwargs,
        ).to(device)

        # Log frozen Q_base values once (these never change during training)
        if hasattr(self.actor, "_q_base"):
            q_base = self.actor._q_base
            logger.info("Q_base (frozen): %s", [f"{v:.4f}" for v in q_base.tolist()])

        # Critic (Twin Q-network)
        # critic_obs_keys from obs_groups determines what the critic sees.
        # All selected obs are concatenated into policy_obs_dim (privileged_dim=0),
        # giving the critic flexible, config-driven input composition.
        critic_cfg = train_cfg.get("critic", {})
        critic_obs_keys = obs_groups.get("critic", ["policy"])
        missing = [k for k in critic_obs_keys if k not in obs]
        if missing:
            raise ValueError(f"critic_obs_keys {missing} not found in obs dict (available: {list(obs.keys())})")
        critic_obs_dim = sum(obs[k].shape[-1] for k in critic_obs_keys)
        self._critic_obs_keys = critic_obs_keys
        self.critic = TwinQNetwork(
            policy_obs_dim=critic_obs_dim,
            privileged_dim=0,  # all obs folded into policy_obs_dim
            action_dim=num_actions,
            hidden_dims=critic_cfg.get("hidden_dims", [256, 256]),
            activation=critic_cfg.get("activation", "relu"),
        ).to(device)

        # Target critic (Polyak-averaged copy)
        self.target_critic = copy.deepcopy(self.critic)

        # ---- SAC algorithm ----
        self.sac = SAC(
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            actor_lr=train_cfg.get("actor_lr", 3e-4),
            critic_lr=train_cfg.get("critic_lr", 3e-4),
            alpha_lr=train_cfg.get("alpha_lr", 3e-4),
            dynamics_lr=train_cfg.get("dynamics_lr", 1e-3),
            dynamics_loss_weight=train_cfg.get("dynamics_loss_weight", 1.0),
            multistep_horizon=train_cfg.get("multistep_horizon", 1),
            multistep_decay=train_cfg.get("multistep_decay", 0.5),
            gamma=train_cfg.get("gamma", 0.99),
            tau=train_cfg.get("tau", 0.002),
            init_alpha=train_cfg.get("init_alpha", 0.2),
            target_entropy=train_cfg.get("target_entropy", None),
            alpha_min=train_cfg.get("alpha_min", 0.1),
            actor_delay=train_cfg.get("actor_delay", 1),
            action_dim=num_actions,
            max_q=train_cfg.get("max_q", 0.0),
            critic_grad_clip=train_cfg.get("critic_grad_clip", 1.0),
            policy_obs_key=self.actor.policy_obs_key,
            critic_obs_keys=self._critic_obs_keys,
            dynamics_diag_interval=train_cfg.get("dynamics_diag_interval", 50),
            multistep_eval_steps=train_cfg.get("multistep_eval_steps", None),
            dynamics_dim_weights=train_cfg.get("dynamics_dim_weights", None),
            multistep_weight=train_cfg.get("multistep_weight", 0.5),
            q_aggregation=train_cfg.get("q_aggregation", "min"),
            adaptive_dynamics_weights=train_cfg.get("adaptive_dynamics_weights", False),
            adaptive_weights_warmup=train_cfg.get("adaptive_weights_warmup", 2000),
            adaptive_weights_ema=train_cfg.get("adaptive_weights_ema", 0.99),
        )

        # ---- Replay buffer ----
        obs_shapes = {}
        for key in obs.keys():
            if isinstance(obs[key], torch.Tensor):
                obs_shapes[key] = obs[key].shape[-1]
        buffer_capacity = train_cfg.get("buffer_capacity", 1_000_000)
        pred_error_dim = 8 if getattr(self.actor.dynamics, "use_error_feedback", False) else 0
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            num_envs=self.env.num_envs,
            obs_shapes=obs_shapes,
            action_dim=num_actions,
            device=device,
            pred_error_dim=pred_error_dim,
        )

        # ---- Training parameters ----
        self.max_iterations = train_cfg.get("max_iterations", 10000)
        self.warmup_iters = train_cfg.get("warmup_iters", 10)
        self.updates_per_step = train_cfg.get("updates_per_step", 2)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.save_interval = train_cfg.get("save_interval", 500)
        self.log_interval = train_cfg.get("log_interval", 10)
        self._resume_iteration = 0

        # ---- Logging ----
        self.log_dir = log_dir
        self.writer = None
        self._logger_type = train_cfg.get("logger", "tensorboard")
        if log_dir is not None:
            tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            if self._logger_type == "wandb":
                import wandb

                wandb_project = train_cfg.get("wandb_project", "hero_agent_sac_mpc")
                experiment_name = train_cfg.get("experiment_name", "hero_agent_sac_mpc")
                wandb.init(
                    project=wandb_project,
                    name=os.path.basename(log_dir),
                    dir=log_dir,
                    config=train_cfg,
                    group=experiment_name,
                )
                self.writer = _WandbTBWriter(tb_writer)
            else:
                self.writer = tb_writer

        # Tracking
        self._episode_rewards = torch.zeros(self.env.num_envs, device=device)
        self._episode_lengths = torch.zeros(self.env.num_envs, dtype=torch.long, device=device)
        self._completed_episodes: list[tuple[float, int]] = []
        self._ema_extras: dict[str, float] = {}
        self._step_reward_sum = 0.0
        self._step_reward_count = 0
        self._first_episode_logged = False

        # EMA smoothing for episode metrics (display-only, does not affect learning).
        # Adaptive alpha = 1 - (1 - _EMA_ALPHA)^n suppresses noise from small-batch
        # resets while tracking real policy improvement from large-batch resets.
        self._EMA_ALPHA = 0.002
        self._ema_ep_reward: float | None = None
        self._ema_ep_length: float | None = None

        logger.info(
            "SACMPCRunner: %d envs, buffer=%d, warmup_iters=%d, "
            "updates_per_step=%d, batch=%d, "
            "critic_obs_keys=%s (critic_obs_dim=%d)",
            self.env.num_envs,
            buffer_capacity,
            self.warmup_iters,
            self.updates_per_step,
            self.batch_size,
            self._critic_obs_keys,
            critic_obs_dim,
        )

    def add_git_repo_to_log(self, *_args, **_kwargs) -> None:
        """Compatibility stub for train.py (no-op for SAC runner)."""

    def learn(self, num_learning_iterations: int | None = None, **_kwargs) -> None:
        """Main training loop."""
        if num_learning_iterations is not None:
            self.max_iterations = num_learning_iterations
        obs = self.env.get_observations()
        total_steps = 0
        start_time = time.time()
        metric_history: dict[str, list[float]] = defaultdict(list)

        start_iter = self._resume_iteration
        if start_iter > 0:
            logger.info(
                "Resuming from iteration %d (warmup skipped, buffer refilling with policy).",
                start_iter,
            )
        logger.info(
            "Starting training: %d iterations (from %d), warmup=%d, log_interval=%d",
            self.max_iterations,
            start_iter,
            self.warmup_iters,
            self.log_interval,
        )

        for iteration in range(start_iter, self.max_iterations):
            self._current_iteration = iteration

            # ---- DR / reward curriculum (before data collection) ----
            if hasattr(self._raw_env, "update_dr_curriculum"):
                self._raw_env.update_dr_curriculum(iteration)
            if hasattr(self._raw_env, "_reward_manager") and hasattr(self._raw_env.cfg, "reward"):
                self._raw_env._reward_manager.update_curriculum(iteration, self._raw_env.cfg.reward.curriculum_end_iter)

            # ---- Collect one env step ----
            # Disable EF dropout during rollout: dynamics predictions must be
            # deterministic for consistent PGD convergence and clean error feedback.
            self.actor.dynamics.eval()

            if iteration < self.warmup_iters:
                if iteration == 0:
                    logger.info("Warmup phase: collecting %d steps with random actions...", self.warmup_iters)
                action = torch.rand(self.env.num_envs, self.env.num_actions, device=self.device) * 2 - 1
            else:
                if iteration == self.warmup_iters:
                    logger.info(
                        "Warmup complete. Starting SAC updates (buffer size: %d)...",
                        len(self.replay_buffer),
                    )
                with torch.no_grad():
                    action = self.actor.act(obs)

            # Store dynamics prediction for error feedback (before env.step)
            if self.actor.dynamics.use_error_feedback:
                self.actor.store_prediction(obs["mpc_state"], action)

            next_obs, reward, done, info = self.env.step(action)
            total_steps += self.env.num_envs

            # Update prediction error buffer (after env.step)
            if self.actor.dynamics.use_error_feedback:
                self.actor.update_pred_error(next_obs["mpc_state"])

            self._step_reward_sum += reward.mean().item()
            self._step_reward_count += 1

            self._episode_rewards += reward
            self._episode_lengths += 1
            done_mask = done.bool()
            # For SAC bootstrap: only TRUE terminations zero out Q.
            # Timeouts should bootstrap normally (episode isn't truly over).
            time_outs = info.get("time_outs", torch.zeros_like(done, dtype=torch.bool))
            terminated_mask = done_mask & ~time_outs
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).view(-1)
                for i in done_ids:
                    self._completed_episodes.append(
                        (self._episode_rewards[i].item(), int(self._episode_lengths[i].item()))
                    )

                if not self._first_episode_logged:
                    first_r, first_l = self._completed_episodes[0]
                    logger.info(
                        "First episode: iter %d, length=%d, reward=%.2f",
                        iteration + 1,
                        first_l,
                        first_r,
                    )
                    self._first_episode_logged = True

                # EMA update for episode reward/length (before zeroing accumulators)
                n = len(done_ids)
                batch_r = self._episode_rewards[done_ids].mean().item()
                batch_l = self._episode_lengths[done_ids].float().mean().item()
                alpha = 1.0 - (1.0 - self._EMA_ALPHA) ** n
                if self._ema_ep_reward is None:
                    self._ema_ep_reward = batch_r
                    self._ema_ep_length = batch_l
                else:
                    self._ema_ep_reward += alpha * (batch_r - self._ema_ep_reward)
                    self._ema_ep_length += alpha * (batch_l - self._ema_ep_length)

                self._episode_rewards[done_mask] = 0.0
                self._episode_lengths[done_mask] = 0

                self._accumulate_extras()
                self.actor.reset_mpc(done_ids)

            _pe = self.actor.get_pred_error()
            self.replay_buffer.add(
                Transition(
                    obs={k: v.clone() for k, v in obs.items() if isinstance(v, torch.Tensor)},
                    action=action.clone(),
                    reward=reward.clone(),
                    next_obs={k: v.clone() for k, v in next_obs.items() if isinstance(v, torch.Tensor)},
                    done=terminated_mask,
                    pred_error=_pe.clone() if _pe is not None else None,
                    episode_boundary=done_mask,
                )
            )

            obs = next_obs

            # ---- SAC updates ----
            # Re-enable EF dropout for training (regularizes dynamics learning).
            self.actor.dynamics.train()

            if iteration >= self.warmup_iters and len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.updates_per_step):
                    batch = self.replay_buffer.sample(self.batch_size)
                    seq_batch = None
                    if self.sac.multistep_horizon > 1:
                        seq_batch = self.replay_buffer.sample_sequences(
                            self.batch_size,
                            self.sac.multistep_horizon,
                        )
                    metrics = self.sac.update(batch, seq_batch=seq_batch)
                    for k, v in metrics.items():
                        metric_history[k].append(v)

            # ---- Logging ----
            if (iteration + 1) % self.log_interval == 0:
                self._log(iteration, total_steps, start_time, metric_history)
                metric_history.clear()

            # ---- Save checkpoint ----
            if (iteration + 1) % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))

            # DR/reward curriculum moved to top of loop (before data collection)

        # Final save
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "model_final.pt"))

    def _log(
        self,
        iteration: int,
        total_steps: int,
        start_time: float,
        metrics: dict[str, list[float]],
    ) -> None:
        """Log training metrics."""
        elapsed = time.time() - start_time
        fps = total_steps / max(elapsed, 1e-6)

        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items() if v}

        has_episodes = bool(self._completed_episodes)
        ep_reward = 0.0
        ep_length = 0.0
        if has_episodes:
            rewards, lengths = zip(*self._completed_episodes)
            ep_reward = sum(rewards) / len(rewards)
            ep_length = sum(lengths) / len(lengths)
            self._completed_episodes.clear()

        step_reward_mean = self._step_reward_sum / max(self._step_reward_count, 1)
        self._step_reward_sum = 0.0
        self._step_reward_count = 0

        max_ep_s = getattr(self._raw_env, "max_episode_length_s", 1.0)

        # Use EMA-smoothed values for WandB (fall back to batch values before EMA init)
        ema_r = self._ema_ep_reward if self._ema_ep_reward is not None else ep_reward
        ema_l = self._ema_ep_length if self._ema_ep_length is not None else ep_length
        ema_r_norm = ema_r / max_ep_s if max_ep_s > 0 else ema_r

        all_metrics: dict[str, float] = {
            "Reward/total": ema_r,
            "Reward/total_normalized": ema_r_norm,
            "Train/episode_length": ema_l,
            "Train/fps": fps,
            "Train/total_steps": float(total_steps),
        }
        all_metrics.update(avg_metrics)

        # MPC metrics (Q aggregates, R_diag, solve_cost, state_err)
        collect_mpc_metrics(all_metrics, self.actor, self.env)

        # Env reward components (EMA-smoothed extras with prefix remapping)
        collect_reward_components(all_metrics, self._ema_extras)

        # Console output (2-line format for readability)
        steps_k = total_steps / 1000
        if has_episodes:
            r_str = f"Reward {ep_reward:.2f} (norm {ema_r_norm:.2f})"
        else:
            r_str = f"Reward -- (step_r={step_reward_mean:.3f})"
        line1 = (
            f"Iter {iteration + 1}/{self.max_iterations}  "
            f"Steps {steps_k:.1f}K  FPS {fps:.0f}  "
            f"{r_str}  EpLen {ema_l:.0f}"
        )
        parts = []
        if "Loss/actor" in avg_metrics:
            parts.append(f"A {avg_metrics['Loss/actor']:.2f}")
        if "Loss/critic" in avg_metrics:
            parts.append(f"C {avg_metrics['Loss/critic']:.2f}")
        if "Loss/dynamics" in avg_metrics:
            parts.append(f"D {avg_metrics['Loss/dynamics']:.1e}")
        if "SAC/alpha" in avg_metrics:
            parts.append(f"Alpha {avg_metrics['SAC/alpha']:.2f}")
        if "SAC/target_q_mean" in avg_metrics:
            parts.append(f"Q_mean {avg_metrics['SAC/target_q_mean']:.1f}")
        if "MPC/state_err_total" in all_metrics:
            parts.append(f"StateErr {all_metrics['MPC/state_err_total']:.3f}")
        line2 = "          Loss: " + "  ".join(parts) if parts else ""
        print(f"[SAC-MPC] {line1}")
        if line2:
            print(line2)

        if self.writer is not None:
            flush_metrics(self.writer, all_metrics, iteration, logger_type=self._logger_type)

    def _accumulate_extras(self) -> None:
        """EMA-update extras["log"] from env resets.

        Uses adaptive alpha based on ``_num_resets`` (number of resetting envs).
        Large-batch resets update EMA aggressively; small-batch resets barely move it.
        This eliminates periodic spikes from varying reset batch sizes.
        """
        raw_env = self._raw_env
        if not hasattr(raw_env, "extras") or "log" not in raw_env.extras:
            return

        log = raw_env.extras["log"]
        n = int(log.get("_num_resets", 1))
        alpha = 1.0 - (1.0 - self._EMA_ALPHA) ** n
        for key, value in log.items():
            if key == "_num_resets":
                continue
            if isinstance(value, (int, float)):
                v = float(value)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                v = value.item()
            else:
                continue
            if key in self._ema_extras:
                self._ema_extras[key] += alpha * (v - self._ema_extras[key])
            else:
                self._ema_extras[key] = v

    def save(self, path: str) -> None:
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "sac": self.sac.state_dict(),
                "iteration": getattr(self, "_current_iteration", 0),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    def load(self, path: str) -> None:
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "sac" in state:
            self.sac.load_state_dict(state["sac"])
            self._resume_iteration = state.get("iteration", 0)
        else:
            self.actor.load_state_dict(state)
        logger.info("Loaded checkpoint: %s (resume from iteration %d)", path, self._resume_iteration)
