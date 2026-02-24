# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Encoder-TDC Environment.

The RL policy outputs 6D actions that parameterize the TDC controller:
    [m_hat_roll, m_hat_pitch, Kp_roll, Kp_pitch, Kd_roll, Kd_pitch]

Actions are linearly scaled from [-1, 1] to physical ranges defined in config.
The encoder compresses privileged observations (26D) into a latent z (13D),
which augments the policy observation space.

Control Flow (50 Hz):
    1. Parse 6D actions -> m_hat(2), Kp(2), Kd(2)
    2. Linear scale [-1, 1] -> physical ranges
    3. Update TDC controller params (m_hat, Kp, Kd)
    4. Run TDC pipeline (orientation -> TDC -> IK -> rate limit -> anti-windup)
"""

from __future__ import annotations

import torch

from .config import HeroAgentEncoderTDCEnvCfg
from .tdc_env import HeroAgentTDCEnv


class HeroAgentEncoderTDCEnv(HeroAgentTDCEnv):
    """Encoder-TDC environment: RL policy outputs M_hat + gains for TDC controller.

    Action Space (6D):
        [0:2] m_hat_raw -> linear scale -> [m_hat_min, m_hat_max]
        [2:4] kp_raw    -> linear scale -> [kp_min, kp_max]
        [4:6] kd_raw    -> linear scale -> [kd_min, kd_max]

    The first 2D (m_hat portion) is used as prev_actions_obs for observation,
    matching base RL's 2D action convention for observation continuity.
    """

    cfg: HeroAgentEncoderTDCEnvCfg

    def __init__(self, cfg: HeroAgentEncoderTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Pre-compute scaling constants: value = mid + action * half_range
        self._m_hat_mid = (cfg.m_hat_min + cfg.m_hat_max) / 2.0
        self._m_hat_half = (cfg.m_hat_max - cfg.m_hat_min) / 2.0
        self._kp_mid = (cfg.kp_min + cfg.kp_max) / 2.0
        self._kp_half = (cfg.kp_max - cfg.kp_min) / 2.0
        self._kd_mid = (cfg.kd_min + cfg.kd_max) / 2.0
        self._kd_half = (cfg.kd_max - cfg.kd_min) / 2.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Parse 6D actions, scale to physical ranges, update TDC, then run pipeline.

        Args:
            actions: RL actions. Shape: (num_envs, 6).
                [0:2] m_hat, [2:4] Kp, [4:6] Kd (all in [-1, 1]).
        """
        # Update action buffers; use m_hat (first 2D) as prev_actions_obs
        self._update_action_buffers(actions, obs_action_slice=slice(0, 2))

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # Apply action latency (delayed actions for control, raw for obs)
        effective_actions = self._get_delayed_actions(self._actions)

        # Linear scale: value = mid + action * half_range
        m_hat = self._m_hat_mid + effective_actions[:, 0:2] * self._m_hat_half
        kp = self._kp_mid + effective_actions[:, 2:4] * self._kp_half
        kd = self._kd_mid + effective_actions[:, 4:6] * self._kd_half

        # Update TDC controller parameters
        self._tdc.update_controller_params(m_hat=m_hat)
        self._tdc.update_gains(kp=kp, kd=kd)

        # Run TDC pipeline (shared with parent)
        self._run_tdc_pipeline()

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset environments and restore default M_hat.

        Args:
            env_ids: Environment indices to reset. None = all.
        """
        super()._reset_idx(env_ids)
        env_ids_ = self._coerce_env_ids(env_ids)
        self._reset_m_hat_defaults(env_ids_)
