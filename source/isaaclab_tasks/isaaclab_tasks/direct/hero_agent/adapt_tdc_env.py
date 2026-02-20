# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Phase 2 Adaptation Environment.

Extends the Encoder-TDC env with a proprioception history buffer for training
the adaptation module (student). The history stores per-timestep proprioception
features that the temporal convolution network uses to estimate the latent z
without access to privileged information.

Data Flow (Phase 2):
    proprio_hist (N, H, 12) -->  AdaptTConv  -->  z_hat (6D)
    privileged (25D) --> Frozen Encoder --> z_gt (6D)   [supervision only]
    z_hat --> [policy_obs + z_hat] --> Frozen Actor --> actions (4D)
    z_hat[3:6] + FK --> M_hat --> TDC Controller --> joint targets

History feature vector (12D per timestep):
    [roll(1), pitch(1), p(1), q(1), joint_pos_normalized(2), joint_vel(2), actions(4)]

Body orientation (roll, pitch) and angular rates (p, q) are included because they
directly reveal hydrodynamic parameter effects: restoring torque T_b manifests in
roll/pitch response, and effective inertia M_eff manifests in angular acceleration.
Arm joint state alone provides only indirect information about body dynamics.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from .config import HeroAgentAdaptTDCEnvCfg
from .encoder_tdc_env import HeroAgentEncoderTDCEnv


class HeroAgentAdaptTDCEnv(HeroAgentEncoderTDCEnv):
    """Phase 2 adaptation environment with proprioception history buffer.

    Inherits the full Encoder-TDC pipeline (TDC init, kinematics, IK,
    rate-limiting, gain sigmoid scaling, M_hat extraction). Adds a ring
    buffer that accumulates proprioception features for the adaptation
    module to consume.
    """

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
        self._update_proprio_hist(actions)
        super()._pre_physics_step(actions)

    def _update_proprio_hist(self, actions: torch.Tensor) -> None:
        """Shift ring buffer left and append current proprioception features.

        Features (12D):
            - roll, pitch: body euler angles (rad)
            - p, q: body angular velocities in body frame (rad/s)
            - joint_pos_normalized: joint positions normalized to [-1, 1]
            - joint_vel: raw joint velocities (rad/s)
            - actions: current RL actions (4D: Kp_r, Kp_p, Kd_r, Kd_p)
        """
        # Body orientation: roll, pitch (rad)
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        # Body angular velocity in body frame: p (roll rate), q (pitch rate)
        ang_vel_b = self._robot.data.root_ang_vel_b
        p = ang_vel_b[:, 0:1]
        q = ang_vel_b[:, 1:2]

        # Joint positions normalized to [-1, 1]
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        joint_pos_norm = 2.0 * (joint_pos - self._joint_limits_lower) / self._joint_limits_range - 1.0

        # Joint velocities (raw rad/s)
        joint_vel = self._robot.data.joint_vel[:, self._albc_joint_ids]

        # Actions (clamped to [-1, 1] for consistency)
        act = actions.clamp(-1.0, 1.0)

        # Assemble 12D feature: [roll, pitch, p, q, joint_pos_norm(2), joint_vel(2), actions(4)]
        new_entry = torch.cat([roll.unsqueeze(-1), pitch.unsqueeze(-1), p, q, joint_pos_norm, joint_vel, act], dim=-1)

        # Roll buffer left (oldest entry dropped) and insert new entry at end
        self._proprio_hist = torch.roll(self._proprio_hist, -1, dims=1)
        self._proprio_hist[:, -1] = new_entry

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset history buffer for terminated environments."""
        super()._reset_idx(env_ids)
        if env_ids is not None:
            self._proprio_hist[env_ids] = 0.0
