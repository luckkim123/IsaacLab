# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Unified TDC Environment: Encoder + RL-output decomposed M_hat + Kp/Kd.

Identical to Encoder-Base (same encoder, DR, rewards, DR curriculum) except:
1. TDC controller is appended to the control pipeline
2. Actor outputs 7D TDC params instead of 2D joint velocities:
    [m_A(1), I_roll(1), I_pitch(1), Kp(2), Kd(2)] decoded via sigmoid scaling
3. M_hat computed from decomposed components via parallel axis theorem + FK

Data Flow:
    Privileged (26D) -> Encoder [ReLU + softplus] -> z (13D), z > z_min
    policy_obs (13D) + z (13D) -> Actor -> 7D raw params
        sigmoid -> [m_A, I_roll, I_pitch] -> FK -> M_hat(2D)
        sigmoid -> Kp(2D), Kd(2D)
        M_hat -> TDC.update_controller_params()
        Kp, Kd -> TDC.update_gains()
    TDC.compute() -> p_EE -> IK -> joint targets -> PhysX

Key differences vs Encoder-TDC:
- z is general latent (no decomposed m_A/I_roll/I_pitch from encoder)
- Decomposed components come from actor output (not z)
- M_hat computed via FK + parallel axis theorem (same as Encoder-TDC)
- No episode-level z latching (z not physically interpreted)
"""

from __future__ import annotations

import torch

from .config import HeroAgentUnifiedTDCEnvCfg
from .tdc_env import HeroAgentTDCEnv


class HeroAgentUnifiedTDCEnv(HeroAgentTDCEnv):
    """Unified TDC: General encoder latent + policy-output decomposed M_hat + Kp/Kd.

    Actions (7D) are decoded into TDC parameters via sigmoid scaling:
        - actions[:, 0:1] -> m_A (added mass estimate)
        - actions[:, 1:2] -> I_roll (roll inertia estimate)
        - actions[:, 2:3] -> I_pitch (pitch inertia estimate)
        - actions[:, 3:5] -> Kp (proportional gains)
        - actions[:, 5:7] -> Kd (derivative gains)

    M_hat is computed from [m_A, I_roll, I_pitch] via parallel axis theorem
    using current FK joint positions (same as Encoder-TDC).

    The encoder policy is registered via set_encoder_policy() and produces
    a general 13D latent z that the actor uses as context.

    Inherits TDC initialization, kinematics, IK, and reset logic from HeroAgentTDCEnv.
    """

    cfg: HeroAgentUnifiedTDCEnvCfg

    def __init__(self, cfg: HeroAgentUnifiedTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Pre-allocate M_hat buffer with config defaults
        self._unified_m_hat = self._create_m_hat_buffer()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Decode 7D actions into TDC params and run TDC pipeline.

        Action latency is applied before TDC param decoding (same mechanism as
        Encoder-Base). The delayed raw logits are then sigmoid-decoded into
        decomposed M_hat components and Kp/Kd values.

        Args:
            actions: RL actions [m_A, I_r, I_p, Kp_r, Kp_p, Kd_r, Kd_p]. Shape: (num_envs, 7).
        """
        self._update_action_buffers(actions, obs_action_slice=slice(0, 2))

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # Apply action latency (same as Encoder-Base)
        effective_actions = self._get_delayed_actions(self._actions)

        # --- 1. Decode decomposed M_hat components via sigmoid scaling ---
        comp_raw = effective_actions[:, :3]
        m_A_min, m_A_max = self.cfg.m_A_range
        I_min, I_max = self.cfg.I_range
        m_A = m_A_min + torch.sigmoid(comp_raw[:, 0:1]) * (m_A_max - m_A_min)
        I_roll = I_min + torch.sigmoid(comp_raw[:, 1:2]) * (I_max - I_min)
        I_pitch = I_min + torch.sigmoid(comp_raw[:, 2:3]) * (I_max - I_min)
        z_decomposed = torch.cat([m_A, I_roll, I_pitch], dim=-1)

        # M_hat via parallel axis theorem (existing method in tdc_env.py)
        m_hat = self._compute_m_hat_from_encoder_z(z_decomposed)
        self._unified_m_hat = m_hat
        self._tdc.update_controller_params(m_hat=m_hat)

        # --- 2. Decode Kp, Kd from actions via sigmoid scaling ---
        kp_raw = effective_actions[:, 3:5]
        kd_raw = effective_actions[:, 5:7]
        kp_min, kp_max = self.cfg.kp_range
        kd_min, kd_max = self.cfg.kd_range
        kp = kp_min + torch.sigmoid(kp_raw) * (kp_max - kp_min)
        kd = kd_min + torch.sigmoid(kd_raw) * (kd_max - kd_min)
        self._tdc.update_gains(kp=kp, kd=kd)

        # --- 3. TDC control pipeline (shared with parent) ---
        self._run_tdc_pipeline()

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments including cached M_hat."""
        super()._reset_idx(env_ids)
        if env_ids is not None:
            m_hat_default = torch.tensor(self.cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
            self._unified_m_hat[env_ids] = m_hat_default
            self._reset_m_hat_defaults(env_ids)
