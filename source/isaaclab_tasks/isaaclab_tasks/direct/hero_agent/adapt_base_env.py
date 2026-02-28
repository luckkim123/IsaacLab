# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Phase 2 Adaptation Environment (base RL pipeline).

Extends the base RL env with a proprioception history buffer for training
the adaptation module (student). The history stores per-timestep features
that the temporal convolution network uses to estimate the latent z
without access to privileged information.

Data Flow (Phase 2):
    proprio_hist (N, H, 8) --> AdaptTConv --> z_hat (13D)
    z_hat --> [policy_obs + z_hat] --> Frozen Actor --> 2D velocity actions
    Frozen Encoder(privileged) --> z_gt (13D)
    Loss = ||z_hat - z_gt||^2

History feature vector (8D per timestep, all at 200Hz sim rate):
    [roll(1), pitch(1), p(1), q(1), joint_pos_norm(2), prev_actions(2)]

Design: pure input-output features (HORA-style, no controller internals).
The adapt module learns dynamics from the command-response relationship:
    - prev_actions (command): RL velocity output
    - p, q (response): angular velocity, directly encodes dynamics
    - Same action + different physical params -> different angular velocity response
"""

from __future__ import annotations

import torch

from .base_env import HeroAgentEnv
from .config import HeroAgentAdaptBaseEnvCfg


class HeroAgentAdaptBaseEnv(HeroAgentEnv):
    """Phase 2 adaptation environment with proprioception history buffer (base RL).

    Inherits the full base RL pipeline (velocity-to-position integration,
    hydrodynamics, DR, reward). Adds a ring buffer that accumulates
    proprioception features for the adaptation module to consume.
    """

    cfg: HeroAgentAdaptBaseEnvCfg

    def __init__(self, cfg: HeroAgentAdaptBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        H = cfg.proprio_history_len
        D = cfg.proprio_feature_dim
        self._proprio_hist = torch.zeros(self.num_envs, H, D, device=self.device)

    def _get_observations(self) -> dict:
        """Add proprioception history to the observation dict."""
        obs = super()._get_observations()
        obs["proprio_hist"] = self._proprio_hist.clone()
        return obs

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Update history buffer before running the base RL control pipeline."""
        self._update_proprio_hist()
        super()._pre_physics_step(actions)

    def _update_proprio_hist(self) -> None:
        """Shift ring buffer left and append current proprioception features.

        Features (8D, all updated at 200Hz sim rate):
            - roll, pitch: body euler angles from IMU (rad)
            - p, q: body angular velocities in body frame (rad/s)
            - joint_pos_norm: normalized joint positions [-1, 1]
            - prev_actions: previous RL velocity actions [-1, 1]
        """
        roll, pitch, p, q, joint_pos_norm = self._get_proprio_features()

        new_entry = torch.cat(
            [roll.unsqueeze(-1), pitch.unsqueeze(-1), p, q, joint_pos_norm, self._prev_actions_obs],
            dim=-1,
        )

        # Shift buffer left (oldest entry dropped) and insert new entry at end
        self._proprio_hist[:, :-1] = self._proprio_hist[:, 1:].clone()
        self._proprio_hist[:, -1] = new_entry

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset history buffer for terminated environments."""
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._proprio_hist[env_ids] = 0.0
