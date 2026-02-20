# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Encoder-TDC Environment: RL-adaptive gains + encoder M_hat.

This module integrates the HORA encoder with the TDC controller:
- Encoder z[3:6] = [m_A_hat, I_roll_hat, I_pitch_hat] (decomposed constants)
- z[3:6] latched at episode start (physical constants don't change within episode)
- M_hat recomputed each control step from latched z + current FK position
- RL actor outputs 4D gains [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch]
- TDC controller uses adaptive M_hat + gains for attitude stabilization

Data Flow:
    Privileged (18D) -> Encoder -> z (6D)
        z[3:6] latched at episode start -> stored as _latched_z_decomposed
        _latched_z_decomposed + FK(joint_pos) -> M_hat -> TDC.update_controller_params()
    policy_obs (13D) + z (6D) -> Actor -> 4D raw gains
        softplus scaling -> Kp (2D), Kd (2D) -> TDC.update_gains()
    TDC.compute() -> p_EE -> IK -> joint targets -> PhysX

Gain Scaling (softplus + max clamp):
    kp = softplus(raw) * (kp_default / log(2)), clamp(max=kp_max)
    raw=0 -> kp_default (design operating point). No hard minimum;
    low gains allowed but reward incentivizes active control.

M_hat Episode Latch:
    z[3:6] (m_A, I_roll, I_pitch) represent physical constants that are DR'd per-episode
    and don't change within an episode. Latching at episode start prevents drift caused
    by encoder weight updates across the ~94 PPO rollouts that span one episode.
    M_hat is still recomputed each control step because FK position changes as joints move.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import HeroAgentEncoderTDCEnvCfg
from .tdc_env import HeroAgentTDCEnv

# math.log(2) = softplus(0); used to scale softplus so raw=0 -> default gain
_LOG2 = 0.6931471805599453


class HeroAgentEncoderTDCEnv(HeroAgentTDCEnv):
    """Hero Agent environment with encoder-adaptive TDC controller.

    RL actions (4D) are converted to TDC gains via softplus scaling with max clamp.
    Encoder latent z[3:6] provides decomposed constants [m_A_hat, I_roll_hat, I_pitch_hat],
    latched at episode start and combined with current FK position to compute M_hat.
    Inherits TDC initialization, kinematics, and reset logic from HeroAgentTDCEnv.
    """

    cfg: HeroAgentEncoderTDCEnvCfg

    def __init__(self, cfg: HeroAgentEncoderTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Pre-allocate M_hat buffer with config defaults (updated from encoder z each step)
        self._encoder_m_hat = self._create_m_hat_buffer()

        # Episode-latched z[3:6] decomposed constants (m_A, I_roll, I_pitch).
        # Initialized to None; populated on first step after each env reset.
        self._latched_z_decomposed: torch.Tensor | None = None
        # Flag: True = this env needs z[3:6] latched on next control step
        self._needs_z_latch = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert RL actions to TDC gains, extract M_hat from encoder z.

        M_hat uses episode-latched z[3:6] (physical constants) combined with
        current FK position (which changes as joints move). z[3:6] is latched
        on the first control step after each env reset.

        Steps:
            1. Latch z[3:6] for newly-reset envs, compute M_hat from latched z + FK
            2. Convert 4D RL actions to TDC gains via softplus + max clamp
            3. Run TDC control pipeline (inherited mechanics)

        Args:
            actions: RL actions [Kp_r, Kp_p, Kd_r, Kd_p]. Shape: (num_envs, 4).
        """
        self._update_action_buffers(actions, obs_action_slice=slice(0, 2))

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # --- 1. Compute adaptive M_hat from latched encoder z + current FK ---
        if self._encoder_policy is not None:
            z = self._encoder_policy.get_last_z()
            if z is not None:
                # Latch z[3:6] for envs that just reset
                if self._needs_z_latch.any():
                    latch_ids = self._needs_z_latch.nonzero(as_tuple=False).squeeze(-1)
                    if self._latched_z_decomposed is None:
                        self._latched_z_decomposed = z[:, 3:6].clone()
                    else:
                        self._latched_z_decomposed[latch_ids] = z[latch_ids, 3:6]
                    self._needs_z_latch[latch_ids] = False

                # Compute M_hat from latched z[3:6] + current FK position
                if self._latched_z_decomposed is not None:
                    self._encoder_z_decomposed = self._latched_z_decomposed
                    m_hat = self._compute_m_hat_from_encoder_z(self._latched_z_decomposed)
                    self._encoder_m_hat = m_hat
                    self._tdc.update_controller_params(m_hat=m_hat)

        # --- 2. Convert RL actions (4D) to TDC gains via softplus ---
        # softplus(0) = log(2), so raw=0 -> gain = default.
        # Clamped at max only; no hard minimum (reward incentivizes active control).
        kp_raw = actions[:, :2]
        kd_raw = actions[:, 2:]
        kp_scale = self.cfg.kp_default / _LOG2
        kd_scale = self.cfg.kd_default / _LOG2
        kp = F.softplus(kp_raw).mul(kp_scale).clamp(max=self.cfg.kp_max)
        kd = F.softplus(kd_raw).mul(kd_scale).clamp(max=self.cfg.kd_max)
        self._tdc.update_gains(kp=kp, kd=kd)

        # --- 3. TDC control pipeline (shared with parent) ---
        self._run_tdc_pipeline()

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments including cached M_hat."""
        super()._reset_idx(env_ids)
        if env_ids is not None:
            # Mark these envs for z[3:6] latching on next control step
            self._needs_z_latch[env_ids] = True
            # Reset M_hat to defaults until encoder provides new z
            m_hat_default = torch.tensor(self.cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
            self._encoder_m_hat[env_ids] = m_hat_default
            self._reset_m_hat_defaults(env_ids)
