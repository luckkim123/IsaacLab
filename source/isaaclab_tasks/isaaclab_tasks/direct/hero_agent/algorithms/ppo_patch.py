# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Monkey-patch RSL-RL PPO for Hero Agent encoder training.

The standard RSL-RL PPO class does not support:
1. Separate encoder optimizer with fixed LR and weight decay
2. z_bounds_loss penalty to prevent tanh saturation

This module patches the PPO class at runtime so the modifications
persist in the hero_agent codebase (git-tracked) rather than in the
site-packages directory (lost on container rebuild).

Import this module before constructing PPO instances. It is imported
by rsl_rl_ppo_cfg.py which runs before any training.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def _patched_ppo_init(self, policy, **kwargs):
    """PPO.__init__ replacement with separate encoder param groups.

    Encoder parameters get fixed LR (3e-3) and weight_decay (1e-5).
    Actor/critic parameters get adaptive LR (KL schedule) and NO weight_decay
    (WD increases gradient norm -> KL divergence rises -> adaptive schedule
    pins LR to min_lr, preventing learning).
    """
    # Call the original init first
    self._original_init(policy, **kwargs)

    # Re-create optimizer if encoder params exist
    encoder_params = []
    other_params = []
    for name, param in self.policy.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    if encoder_params:
        self._has_encoder_params = True
        self.encoder_lr = 3e-3
        self.optimizer = optim.Adam([
            {"params": other_params, "weight_decay": 0.0, "lr": self.learning_rate},
            {"params": encoder_params, "weight_decay": 1e-5, "lr": self.encoder_lr},
        ])
    else:
        self._has_encoder_params = False


def _patched_ppo_update(self):
    """PPO.update() replacement with z_bounds_loss integration.

    Adds encoder z bounds loss (soft quadratic penalty) to the PPO objective.
    Prevents tanh activation saturation in the encoder output.
    """
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_z_bounds_loss = 0
    mean_kl = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None

    if self.policy.is_recurrent:
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    for (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hidden_states_batch,
        masks_batch,
    ) in generator:
        num_aug = 1
        original_batch_size = obs_batch.batch_size[0]

        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Symmetric augmentation
        if self.symmetry and self.symmetry["use_data_augmentation"]:
            data_augmentation_func = self.symmetry["data_augmentation_func"]
            obs_batch, actions_batch = data_augmentation_func(
                obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
            )
            num_aug = int(obs_batch.batch_size[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)

        self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
        mu_batch = self.policy.action_mean[:original_batch_size]
        sigma_batch = self.policy.action_std[:original_batch_size]
        entropy_batch = self.policy.entropy[:original_batch_size]

        # KL divergence and adaptive LR
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                mean_kl += kl_mean.item()

                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                # Adaptive LR for actor/critic only (group 0).
                # Encoder (group 1) keeps its fixed LR (managed by runner).
                self.optimizer.param_groups[0]["lr"] = self.learning_rate
                if not self._has_encoder_params:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

        # Encoder z bounds loss
        z_b_loss = torch.tensor(0.0)
        if hasattr(self.policy, "z_bounds_loss"):
            z_b_loss = self.policy.z_bounds_loss()
            loss = loss + z_b_loss

        # Symmetry loss
        if self.symmetry:
            if not self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                num_aug = int(obs_batch.shape[0] / original_batch_size)
            mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
            action_mean_orig = mean_actions_batch[:original_batch_size]
            _, actions_mean_symm_batch = data_augmentation_func(
                obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
            )
            mse_loss = torch.nn.MSELoss()
            symmetry_loss = mse_loss(
                mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
            )
            if self.symmetry["use_mirror_loss"]:
                loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
            else:
                symmetry_loss = symmetry_loss.detach()

        # RND loss
        if self.rnd:
            with torch.no_grad():
                rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
            predicted_embedding = self.rnd.predictor(rnd_state_batch)
            target_embedding = self.rnd.target(rnd_state_batch).detach()
            mseloss = torch.nn.MSELoss()
            rnd_loss = mseloss(predicted_embedding, target_embedding)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        if self.rnd:
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()

        if self.is_multi_gpu:
            self.reduce_parameters()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.rnd_optimizer:
            self.rnd_optimizer.step()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy_batch.mean().item()
        mean_z_bounds_loss += z_b_loss.item() if hasattr(self.policy, "z_bounds_loss") else 0
        if mean_rnd_loss is not None:
            mean_rnd_loss += rnd_loss.item()
        if mean_symmetry_loss is not None:
            mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_z_bounds_loss /= num_updates
    mean_kl /= num_updates
    if mean_rnd_loss is not None:
        mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
        mean_symmetry_loss /= num_updates

    self.storage.clear()

    loss_dict = {
        "value_function": mean_value_loss,
        "surrogate": mean_surrogate_loss,
        "entropy": mean_entropy,
    }
    loss_dict["kl"] = mean_kl
    if hasattr(self.policy, "z_bounds_loss"):
        loss_dict["z_bounds"] = mean_z_bounds_loss
    if self.rnd:
        loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
        loss_dict["symmetry"] = mean_symmetry_loss

    return loss_dict


def apply_ppo_patch():
    """Apply encoder-aware patches to RSL-RL PPO class.

    Safe to call multiple times (idempotent). Patches are only applied
    if not already present.
    """
    try:
        from rsl_rl.algorithms.ppo import PPO
    except ImportError:
        logger.warning("[ppo_patch] rsl_rl not installed. Skipping PPO patch.")
        return

    if getattr(PPO, "_hero_agent_patched", False):
        return

    # Check if already manually patched (e.g., site-packages was edited)
    has_z_bounds = "z_bounds_loss" in PPO.update.__code__.co_names if hasattr(PPO.update, "__code__") else False
    init_code = getattr(PPO.__init__, "__code__", None)
    has_encoder_params = "_has_encoder_params" in init_code.co_names if init_code else False

    if has_z_bounds and has_encoder_params:
        logger.info("[ppo_patch] RSL-RL PPO already patched (site-packages). Marking as patched.")
        PPO._hero_agent_patched = True
        return

    # Apply patches
    PPO._original_init = PPO.__init__
    PPO.__init__ = _patched_ppo_init
    PPO.update = _patched_ppo_update
    PPO._hero_agent_patched = True
    logger.info("[ppo_patch] RSL-RL PPO patched with encoder optimizer + z_bounds_loss.")
