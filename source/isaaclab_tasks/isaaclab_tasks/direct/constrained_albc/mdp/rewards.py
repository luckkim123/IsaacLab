# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal 3-term reward: r = r_c + r_tau + r_s.

r_c:   -k_c * ||attitude_error||^2    (command tracking)
r_tau: -k_tau * mean(tau^2)            (energy efficiency)
r_s:   -k_s * (mean(da^2) + mean(d2a^2))  (action smoothness)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..albc_env import ALBCEnv


# --- Configuration ---


@configclass
class ALBCRewardCfg:
    """r = r_c + r_tau + r_s (all dt-scaled, negative weights = penalties)."""

    k_c: float = -1.0  # command tracking
    k_tau: float = -0.001  # joint torque
    k_s: float = -0.05  # action smoothness
    termination_penalty: float = -10.0


# --- Reward Functions ---


def command_tracking(env: ALBCEnv) -> torch.Tensor:
    """r_c = ||attitude_error_rp||^2. Quadratic roll/pitch tracking penalty."""
    return env._attitude_error[:, :2].pow(2).sum(dim=-1)


def joint_torque(robot: Articulation, env: ALBCEnv) -> torch.Tensor:
    """r_tau = mean(tau^2). Energy efficiency penalty."""
    return robot.data.computed_torque[:, env._albc_joint_ids].pow(2).mean(dim=-1)


def action_smoothness(env: ALBCEnv) -> torch.Tensor:
    """r_s = mean(da^2) + mean(d2a^2). First + second order action difference."""
    da = env._actions - env._prev_actions
    d2a = env._actions - 2.0 * env._prev_actions + env._prev_prev_actions
    return da.pow(2).mean(dim=-1) + d2a.pow(2).mean(dim=-1)


# --- Reward Manager ---


class RewardManager:
    """Computes r = k_c*r_c + k_tau*r_tau + k_s*r_s with dt-scaling and episode tracking."""

    def __init__(self, cfg: ALBCRewardCfg, num_envs: int, device: str) -> None:
        self._cfg = cfg
        self._buf = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._names = ["command", "torque", "smoothness"]
        self._episode_sums = {n: torch.zeros(num_envs, dtype=torch.float32, device=device) for n in self._names}

    def compute(self, robot: Articulation, dt: float, env: ALBCEnv, **_ctx: Any) -> torch.Tensor:
        """Compute total reward (dt-scaled)."""
        cfg = self._cfg
        self._buf.zero_()

        terms = [
            ("command", cfg.k_c, command_tracking(env)),
            ("torque", cfg.k_tau, joint_torque(robot, env)),
            ("smoothness", cfg.k_s, action_smoothness(env)),
        ]
        for name, weight, value in terms:
            scaled = value * weight * dt
            self._buf += scaled
            self._episode_sums[name] += scaled

        return self._buf

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset episode sums, return means before reset."""
        sums = {n: self._episode_sums[n][env_ids].mean().item() for n in self._names}
        for n in self._names:
            self._episode_sums[n][env_ids] = 0.0
        return sums
