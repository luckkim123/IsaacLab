# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Phase 2 Adaptation Environment.

Extends the Encoder-TDC env with a proprioception history buffer for training
the adaptation module (student). The history stores per-timestep proprioception
features that the temporal convolution network uses to estimate the latent z
without access to privileged information.

Data Flow (Single-Phase):
    proprio_hist (N, H, 6) -->  AdaptTConv  -->  z_hat (3D: [m_A, I_roll, I_pitch])
    z_hat + FK --> M_hat --> TDC Controller --> joint_pos_target
    z_hat --> [policy_obs + z_hat] --> Actor --> actions (4D: Kp/Kd gains)

History feature vector (6D per timestep, all at 200Hz sim rate):
    [roll(1), pitch(1), p(1), q(1), joint_pos_target(2)]

Design: pure input-output features (HORA-style, no controller internals).
The adapt module learns dynamics from the command-response relationship:
    - joint_pos_target (command): TDC output, 50Hz staircase in 200Hz history
    - p, q (response): angular velocity, directly encodes M_true
    - Same command + different M_true -> different angular velocity response

Previous 12D design used nu_dot_filtered and u_hat (TDC internals), which
created a circular dependency (u_hat depends on M_hat, M_hat depends on z_hat)
and caused z_hat collapse to the population mean.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from .config import HeroAgentAdaptTDCEnvCfg
from .encoder_tdc_env import HeroAgentEncoderTDCEnv


class HeroAgentAdaptTDCEnv(HeroAgentEncoderTDCEnv):
    """Phase 2 / single-phase adaptation environment with proprioception history buffer.

    Inherits the full Encoder-TDC pipeline (TDC init, kinematics, IK,
    rate-limiting, gain sigmoid scaling, M_hat extraction). Adds a ring
    buffer that accumulates proprioception features for the adaptation
    module to consume.
    """

    def _extract_z_decomposed(self, z: torch.Tensor) -> torch.Tensor:
        """Adapt output IS the decomposed components [m_A, I_roll, I_pitch].

        In single-phase mode, adapt_tconv outputs 3D directly (no unused dims).
        In phase-2 mode with 6D output, fall back to parent's z[:, 3:6] slicing.
        """
        if z.shape[-1] == 3:
            return z
        return super()._extract_z_decomposed(z)

    cfg: HeroAgentAdaptTDCEnvCfg

    def __init__(self, cfg: HeroAgentAdaptTDCEnvCfg, render_mode: str | None = None, **kwargs):
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
        """Update history buffer before running the TDC control pipeline."""
        self._update_proprio_hist()
        super()._pre_physics_step(actions)

    def _update_proprio_hist(self) -> None:
        """Shift ring buffer left and append current proprioception features.

        Features (6D, all updated at 200Hz sim rate):
            - roll, pitch: body euler angles from IMU (rad)
            - p, q: body angular velocities in body frame (rad/s)
            - joint_pos_target: TDC-computed joint position targets (rad)
              Updates at 50Hz (control_decimation=4), appears as staircase
              in 200Hz history. The temporal conv learns to extract dynamics
              from the command-response relationship.
        """
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_b = self._robot.data.root_ang_vel_b
        p = ang_vel_b[:, 0:1]
        q = ang_vel_b[:, 1:2]

        # Joint position targets from TDC controller (50Hz staircase)
        joint_pos_target = self._joint_pos_targets  # (num_envs, 2)

        new_entry = torch.cat([roll.unsqueeze(-1), pitch.unsqueeze(-1), p, q, joint_pos_target], dim=-1)

        # Roll buffer left (oldest entry dropped) and insert new entry at end
        self._proprio_hist = torch.roll(self._proprio_hist, -1, dims=1)
        self._proprio_hist[:, -1] = new_entry

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset history buffer for terminated environments."""
        super()._reset_idx(env_ids)
        if env_ids is not None:
            self._proprio_hist[env_ids] = 0.0
