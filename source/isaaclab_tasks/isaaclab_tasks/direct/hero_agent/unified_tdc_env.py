# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Unified TDC Environment: General encoder + RL-output M_hat/Kp/Kd.

This module combines encoder-based context compression with direct TDC parameter
output from the RL policy:
- Encoder produces general 13D latent z (no physics interpretation)
- Policy receives [policy_obs(13D) + z(13D)] = 26D and outputs 6D:
    [M_hat(2), Kp(2), Kd(2)] decoded via sigmoid scaling
- TDC controller uses these adaptive parameters for attitude stabilization

Data Flow:
    Privileged (18D) -> Encoder [ELU + tanh] -> z (13D), z in [-1, 1]
    policy_obs (13D) + z (13D) -> Actor -> 6D raw params
        sigmoid -> M_hat(2D), Kp(2D), Kd(2D)
        M_hat -> TDC.update_controller_params()
        Kp, Kd -> TDC.update_gains()
    TDC.compute() -> p_EE -> IK -> joint targets -> PhysX

Key simplifications vs Encoder-TDC:
- z is general latent (no decomposed m_A/I_roll/I_pitch)
- M_hat from actor output directly (no FK + parallel axis theorem)
- Encoder uses ELU+tanh (not ReLU+softplus)
"""

from __future__ import annotations

import torch

from .config import HeroAgentUnifiedTDCEnvCfg
from .tdc_env import HeroAgentTDCEnv


class HeroAgentUnifiedTDCEnv(HeroAgentTDCEnv):
    """Unified TDC: General encoder latent + policy-output M_hat/Kp/Kd.

    Actions (6D) are decoded into TDC parameters via sigmoid scaling:
        - actions[:, 0:2] -> M_hat (design inertia)
        - actions[:, 2:4] -> Kp (proportional gains)
        - actions[:, 4:6] -> Kd (derivative gains)

    The encoder policy is registered via set_encoder_policy() and produces
    a general 13D latent z that the actor uses as context. Unlike Encoder-TDC,
    M_hat is NOT extracted from z -- it comes directly from the actor output.

    Inherits TDC initialization, kinematics, IK, and reset logic from HeroAgentTDCEnv.
    """

    cfg: HeroAgentUnifiedTDCEnvCfg

    def __init__(self, cfg: HeroAgentUnifiedTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Pre-allocate M_hat buffer with config defaults
        self._unified_m_hat = self._create_m_hat_buffer()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Decode 6D actions into TDC params and run TDC pipeline.

        Steps:
            1. Decode M_hat from actions[:, 0:2] via sigmoid
            2. Decode Kp from actions[:, 2:4] via sigmoid
            3. Decode Kd from actions[:, 4:6] via sigmoid
            4. Run TDC control pipeline

        Args:
            actions: RL actions [M_hat_r, M_hat_p, Kp_r, Kp_p, Kd_r, Kd_p]. Shape: (num_envs, 6).
        """
        self._update_action_buffers(actions, obs_action_slice=slice(0, 2))

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # --- 1. Decode M_hat from actions via sigmoid scaling ---
        m_hat_raw = actions[:, :2]
        m_min, m_max = self.cfg.m_hat_range
        m_hat = m_min + torch.sigmoid(m_hat_raw) * (m_max - m_min)
        self._unified_m_hat = m_hat
        self._tdc.update_controller_params(m_hat=m_hat)

        # --- 2. Decode Kp, Kd from actions via sigmoid scaling ---
        kp_raw = actions[:, 2:4]
        kd_raw = actions[:, 4:6]
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
