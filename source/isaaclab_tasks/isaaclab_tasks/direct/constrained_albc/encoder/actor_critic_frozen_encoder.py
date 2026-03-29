# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticEncoder with pre-trained frozen encoder for offline encoder pipeline.

The encoder is loaded from an offline-trained checkpoint and frozen (requires_grad=False).
This eliminates the online encoder co-training instability (KL spike, noise_std explosion)
that made 15+ online encoder experiments fail on the 2D action space system.

During fine-tuning:
- Encoder: frozen, provides constant z for given privileged obs
- Actor: trainable, input = cat([o_t_norm, hist_norm, z_frozen])
- Critic: trainable, asymmetric with privileged obs
- z-related actor weights initialized to near-zero for smooth integration
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from tensordict import TensorDict

from .actor_critic_encoder import ActorCriticEncoder

logger = logging.getLogger(__name__)


class ActorCriticFrozenEncoder(ActorCriticEncoder):
    """ActorCriticEncoder with pre-trained frozen encoder.

    Encoder weights loaded from offline training, frozen (requires_grad=False).
    z-related weights in actor's first layer initialized to near-zero so the
    actor initially ignores z and behaves like the history-only baseline.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        pretrained_encoder_path: str = "",
        z_init_scale: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.encoder_obs_normalization:
            for param in self.encoder_obs_normalizer.parameters():
                param.requires_grad = False
        logger.info("Encoder frozen: all encoder parameters have requires_grad=False")

        # Load pre-trained encoder weights
        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path)

        # Initialize z-related actor weights to near-zero
        self._init_z_weights(z_init_scale)

    def _load_pretrained_encoder(self, path: str) -> None:
        """Load encoder weights and normalization bounds from offline training checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder_sd = ckpt["encoder_state_dict"]
        self.encoder.load_state_dict(encoder_sd)
        logger.info("Loaded pre-trained encoder from %s (%d params)", path, len(encoder_sd))

        # Auto-load static normalization bounds from checkpoint.
        # The offline encoder is trained with static min-max normalization;
        # these bounds MUST be applied at inference to match the training input distribution.
        if "enc_obs_lower" in ckpt and "enc_obs_upper" in ckpt:
            lower = ckpt["enc_obs_lower"].float()
            upper = ckpt["enc_obs_upper"].float()
            if self._has_static_enc_norm:
                # Buffers already registered by base class -- just copy values
                self._enc_obs_lower.copy_(lower)
                self._enc_obs_upper.copy_(upper)
            else:
                # Base class did not register buffers -- register them now
                self.register_buffer("_enc_obs_lower", lower)
                self.register_buffer("_enc_obs_upper", upper)
                self._has_static_enc_norm = True
            logger.info("Loaded static normalization bounds from checkpoint")

    def _init_z_weights(self, scale: float) -> None:
        """Initialize actor's first layer weights corresponding to z inputs to near-zero.

        Actor input: cat([o_t_norm, hist_norm, z]) where z occupies the last
        encoder_latent_dim columns. Scaling them to near-zero makes the actor
        initially behave like the history-only baseline.
        """
        if self.shared_backbone_mode:
            first_layer = self.backbone[0]
        else:
            first_layer = self.actor[0]

        z_start = self.policy_obs_dim + self.proprio_hist_dim
        z_end = z_start + self.encoder_latent_dim

        with torch.no_grad():
            first_layer.weight[:, z_start:z_end] *= scale

        logger.info(
            "z-related actor weights (cols %d:%d) scaled by %.4f",
            z_start,
            z_end,
            scale,
        )

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode with frozen encoder (no gradient flow)."""
        with torch.no_grad():
            return super()._encode(obs)

    def load_history_only_weights(self, checkpoint_path: str) -> None:
        """Warm-start actor from history-only checkpoint.

        History-only actor has input dim = policy_obs_dim + proprio_hist_dim (e.g., 254).
        Frozen encoder actor has input dim = policy_obs_dim + proprio_hist_dim + encoder_latent_dim (e.g., 267).
        First layer: copy columns [0:hist_dim] from history-only, keep z columns near-zero.
        Deeper layers: direct copy (hidden dims match).

        Args:
            checkpoint_path: Path to history-only model checkpoint (.pt).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        hist_input_dim = self.policy_obs_dim + self.proprio_hist_dim
        loaded = 0

        for key, param in self.named_parameters():
            if not key.startswith("actor.") and not key.startswith("backbone."):
                continue
            if key.endswith(".weight") and key.split(".")[-2] == "0":
                # First layer: partial copy (different input dims)
                old_key = key
                if old_key in sd:
                    old_w = sd[old_key]  # (hidden, hist_input_dim)
                    if old_w.shape[1] == hist_input_dim:
                        with torch.no_grad():
                            param.data[:, :hist_input_dim] = old_w
                        loaded += 1
            elif key in sd and sd[key].shape == param.shape:
                with torch.no_grad():
                    param.data.copy_(sd[key])
                loaded += 1

        # Transfer actor obs normalizer stats if available
        norm_prefix = "actor_obs_normalizer."
        for key in sd:
            if key.startswith(norm_prefix):
                if hasattr(self, "actor_obs_normalizer"):
                    target = dict(self.actor_obs_normalizer.named_buffers())
                    buf_name = key[len(norm_prefix) :]
                    if buf_name in target and sd[key].shape == target[buf_name].shape:
                        target[buf_name].copy_(sd[key])
                        loaded += 1

        # Transfer log_std / std parameter
        for std_key in ("log_std", "std"):
            if std_key in sd:
                own_param = dict(self.named_parameters()).get(std_key)
                if own_param is not None and sd[std_key].shape == own_param.shape:
                    with torch.no_grad():
                        own_param.copy_(sd[std_key])
                    loaded += 1
                    logger.info("Copied %s from history-only: %s", std_key, sd[std_key].tolist())

        # Transfer critic weights (skip if shape mismatch, e.g. asymmetric critic)
        for key in sd:
            if key.startswith("critic.") or key.startswith("critic_obs_normalizer."):
                if key in dict(self.named_parameters()) or key in dict(self.named_buffers()):
                    target_dict = {**dict(self.named_parameters()), **dict(self.named_buffers())}
                    if key in target_dict and sd[key].shape == target_dict[key].shape:
                        with torch.no_grad():
                            target_dict[key].copy_(sd[key])
                        loaded += 1

        logger.info("Loaded %d params/buffers from history-only checkpoint: %s", loaded, checkpoint_path)
