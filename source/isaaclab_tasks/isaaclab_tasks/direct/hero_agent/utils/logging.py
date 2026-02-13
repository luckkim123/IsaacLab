# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core logging utilities for Hero Agent environment.

Provides foundational helpers used across all logging sub-modules:
    - flush_metrics: Writes scalars + optional WandB extras
    - _collect_tensor_stats: Per-dimension mean/std collection
    - pearson_r: Correlation coefficient
    - _get_wandb: Lazy wandb import
"""

from __future__ import annotations

from typing import Any

import torch

_ROLL_PITCH = ("roll", "pitch")


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation coefficient between two 1D tensors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)


def flush_metrics(
    writer: Any,
    metrics: dict[str, float],
    step: int,
    logger_type: str = "tensorboard",
    wandb_extras: dict[str, Any] | None = None,
) -> None:
    """Flush accumulated metrics dict to TensorBoard and/or WandB in a single call.

    Scalars are always written via ``writer.add_scalar()`` so that TensorBoard
    records are never skipped -- even when the logger backend is WandB (RSL-RL's
    ``WandbSummaryWriter.add_scalar`` writes to both TB and WandB).

    Non-scalar WandB data (histograms, images, etc.) can be passed via
    *wandb_extras* and will be sent in a single ``wandb.log()`` call together
    with scalars.

    Args:
        writer: TensorBoard SummaryWriter (or RSL-RL WandbSummaryWriter).
        metrics: Dict of {tag: scalar_value} to log.
        step: Training step for x-axis.
        logger_type: Backend type ("tensorboard" or "wandb").
        wandb_extras: Optional dict of non-scalar WandB objects (e.g. Histogram).
            Ignored when logger_type is not "wandb".
    """
    if not metrics and not wandb_extras:
        return

    # Always record scalars via writer.add_scalar (works for both TB and WandB writers)
    for tag, value in metrics.items():
        writer.add_scalar(tag, value, step)

    # Send non-scalar extras (histograms, etc.) directly to WandB
    if logger_type == "wandb" and wandb_extras:
        wandb = _get_wandb()
        if wandb is not None:
            wandb.log(wandb_extras, step=step, commit=False)


def _collect_tensor_stats(
    metrics: dict[str, float],
    prefix: str,
    tensor: torch.Tensor,
    dim_names: tuple[str, ...],
) -> None:
    """Collect per-dimension mean/std of a (num_envs, D) tensor into metrics dict.

    Args:
        metrics: Dict to accumulate into.
        prefix: Tag prefix (e.g., "TDC/m_hat").
        tensor: Shape (num_envs, D) tensor.
        dim_names: Names for each dimension (length must equal D).
    """
    for i, name in enumerate(dim_names):
        metrics[f"{prefix}_{name}_mean"] = tensor[:, i].mean().item()
        metrics[f"{prefix}_{name}_std"] = tensor[:, i].std().item()


def _get_wandb() -> Any | None:
    """Import wandb or return None if unavailable."""
    try:
        import wandb

        return wandb
    except ImportError:
        return None
