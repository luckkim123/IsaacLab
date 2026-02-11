# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared policy construction and checkpoint loading for Hero Agent workflows.

Provides factory functions for building ActorCriticEncoderTDCAdapt networks
and loading Phase 1/Phase 2 checkpoints with proper validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from isaaclab.utils.dict import class_to_dict

from ..encoder import ActorCriticEncoderTDCAdapt, RunningMeanStd

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def build_adapt_policy(
    obs_dict: dict[str, torch.Tensor],
    agent_cfg: RslRlBaseRunnerCfg,
    num_actions: int,
) -> ActorCriticEncoderTDCAdapt:
    """Construct ActorCriticEncoderTDCAdapt from runner config and observations.

    Args:
        obs_dict: Observation dictionary from env.get_observations().
        agent_cfg: Runner config with policy sub-config and obs_groups.
        num_actions: Action space dimension.

    Returns:
        Initialized (untrained) ActorCriticEncoderTDCAdapt.
    """
    policy_cfg = class_to_dict(agent_cfg.policy)
    obs_groups = agent_cfg.obs_groups if hasattr(agent_cfg, "obs_groups") else {"policy": ["policy", "privileged"]}

    return ActorCriticEncoderTDCAdapt(
        obs=obs_dict,
        obs_groups=obs_groups,
        num_actions=num_actions,
        **{k: v for k, v in policy_cfg.items() if k != "class_name"},
    )


def load_phase1_checkpoint(
    policy: ActorCriticEncoderTDCAdapt,
    checkpoint_path: str,
    device: str,
) -> None:
    """Load Phase 1 checkpoint with strict=False, validating only adapt_tconv keys are missing.

    Phase 1 models lack adapt_tconv weights (they don't exist yet). All other
    keys must match. Raises RuntimeError if non-adapt keys are missing.

    Args:
        policy: Policy to load weights into.
        checkpoint_path: Path to model_*.pt from Phase 1 training.
        device: Device for checkpoint loading.
    """
    logger.info("Loading Phase 1 checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Invalid Phase 1 checkpoint: expected a dict with 'model_state_dict' key. "
            f"Use model_*.pt (not exported/policy.pt which is TorchScript)."
        )

    # Bypass custom load_state_dict (returns bool for RSL-RL compat) to get missing/unexpected keys
    result = nn.Module.load_state_dict(policy, checkpoint["model_state_dict"], strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys

    # Validate: only adapt_tconv keys should be missing
    invalid_missing = [k for k in missing if "adapt_tconv" not in k]
    if invalid_missing:
        raise RuntimeError(
            f"Phase 1 checkpoint incompatible: missing non-adapt keys {invalid_missing}. "
            f"Verify encoder architecture matches Phase 1 config."
        )
    if unexpected:
        logger.warning("Phase 1 checkpoint has unexpected keys (ignored): %s", unexpected[:5])

    logger.info("Phase 1 load OK: %d adapt_tconv keys (randomly initialized)", len(missing))


def load_phase2_checkpoint(
    policy: ActorCriticEncoderTDCAdapt,
    checkpoint_path: str,
    device: str,
    proprio_history_shape: tuple[int, int],
) -> tuple[RunningMeanStd, dict]:
    """Load Phase 2 checkpoint with strict=True and restore history normalizer.

    Args:
        policy: Policy to load weights into.
        checkpoint_path: Path to model_*.pt from Phase 2 training.
        device: Device for checkpoint loading.
        proprio_history_shape: (history_len, feature_dim) for RunningMeanStd.

    Returns:
        Tuple of (hist_normalizer, checkpoint_dict).
    """
    logger.info("Loading Phase 2 checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Invalid Phase 2 checkpoint: expected dict with 'model_state_dict'. Use model_*.pt from Phase 2 training."
        )

    nn.Module.load_state_dict(policy, checkpoint["model_state_dict"], strict=True)
    policy.to(device)
    policy.eval()

    # Restore history normalizer
    hist_normalizer = RunningMeanStd(proprio_history_shape).to(device)
    if "hist_normalizer_state_dict" in checkpoint:
        hist_normalizer.load_state_dict(checkpoint["hist_normalizer_state_dict"])
    hist_normalizer.eval()

    logger.info("Phase 2 model loaded. Agent steps: %s", checkpoint.get("agent_steps", "N/A"))
    return hist_normalizer, checkpoint


def get_proprio_history_shape(env_cfg: object) -> tuple[int, int]:
    """Extract (history_len, feature_dim) from env config with defaults.

    Args:
        env_cfg: Environment config (may have proprio_history_len/proprio_feature_dim).

    Returns:
        (history_len, feature_dim) tuple.
    """
    history_len = getattr(env_cfg, "proprio_history_len", 30)
    feature_dim = getattr(env_cfg, "proprio_feature_dim", 12)
    return (history_len, feature_dim)
