# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Running normalization utilities for encoder training."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Running mean and variance normalization (Welford's online algorithm).

    Maintains running statistics and normalizes input using:
        output = (input - mean) / sqrt(var + epsilon)

    In train mode, statistics are updated on each forward pass.
    In eval mode, statistics are frozen.

    Based on HORA/IsaacGymEnvs implementation.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.shape[0]
            delta = batch_mean - self.running_mean
            tot_count = self.count + batch_count
            self.running_mean = self.running_mean + delta * batch_count / tot_count
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            self.running_var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count
            self.count = tot_count

        y = (x - self.running_mean.float()) / torch.sqrt(self.running_var.float() + self.epsilon)
        return y.clamp(-5.0, 5.0)
