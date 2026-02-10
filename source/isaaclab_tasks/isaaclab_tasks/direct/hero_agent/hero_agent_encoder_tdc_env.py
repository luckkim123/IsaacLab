# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent Encoder-TDC Environment: RL-adaptive gains + encoder M_hat.

This module integrates the HORA encoder with the TDC controller:
- Encoder z[3:5] -> adaptive design inertia M_hat for TDC
- RL actor outputs 4D gains [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch]
- TDC controller uses adaptive M_hat + gains for attitude stabilization

Data Flow:
    Privileged (22D) -> Encoder -> z (6D)
        z[3:5] -> M_hat -> TDC.update_controller_params()
    policy_obs (13D) + z (6D) -> Actor -> 4D raw gains
        sigmoid scaling -> Kp (2D), Kd (2D) -> TDC.update_gains()
    TDC.compute() -> p_EE -> IK -> joint targets -> PhysX

M_hat Transfer Mechanism:
    The env stores a reference to the policy network (set by the runner via
    set_encoder_policy()). In _pre_physics_step(), it calls policy.get_last_z()
    to extract z[3:5] as M_hat. The z was computed during the preceding act()
    call in the training loop (obs -> act -> step -> _pre_physics_step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from .hero_agent_env_cfg import HeroAgentEncoderTDCEnvCfg
from .hero_agent_tdc_env import HeroAgentTDCEnv

if TYPE_CHECKING:
    from .encoder.actor_critic_encoder_tdc import ActorCriticEncoderTDC


class HeroAgentEncoderTDCEnv(HeroAgentTDCEnv):
    """Hero Agent environment with encoder-adaptive TDC controller.

    RL actions (4D) are converted to TDC gains via sigmoid scaling.
    Encoder latent z[3:5] provides adaptive M_hat for TDC.
    Inherits TDC initialization, kinematics, and reset logic from HeroAgentTDCEnv.
    """

    cfg: HeroAgentEncoderTDCEnvCfg

    def __init__(self, cfg: HeroAgentEncoderTDCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._encoder_policy: ActorCriticEncoderTDC | None = None
        # Pre-allocate M_hat buffer with config defaults (updated from encoder z each step)
        m_hat_default = torch.tensor(cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
        self._encoder_m_hat = m_hat_default.unsqueeze(0).expand(self.num_envs, -1).clone()

    def set_encoder_policy(self, policy: ActorCriticEncoderTDC) -> None:
        """Register the encoder policy for z -> M_hat extraction.

        Called by the runner after policy creation to enable M_hat transfer.

        Args:
            policy: ActorCriticEncoderTDC instance.
        """
        self._encoder_policy = policy

    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert RL actions to TDC gains, extract M_hat from encoder z.

        Steps:
            1. Extract M_hat from encoder z (if policy registered)
            2. Convert 4D RL actions to TDC gains via sigmoid scaling
            3. Run TDC control pipeline (inherited mechanics)

        Args:
            actions: RL actions [Kp_r, Kp_p, Kd_r, Kd_p]. Shape: (num_envs, 4).
        """
        # Store actions for observation/logging compatibility
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # Store first 2 dims as prev_actions_obs to maintain 13D obs compatibility
        self._prev_actions_obs = self._actions[:, :2].clone()
        self._control_step_counter += 1

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # --- 1. Extract adaptive M_hat from encoder z ---
        if self._encoder_policy is not None:
            z = self._encoder_policy.get_last_z()
            if z is not None:
                m_hat = z[:, 3:5].clamp(self.cfg.m_hat_min, self.cfg.m_hat_max)
                self._encoder_m_hat = m_hat
                self._tdc.update_controller_params(m_hat=m_hat)

        # --- 2. Convert RL actions (4D) to TDC gains via sigmoid scaling ---
        kp_raw = actions[:, :2]
        kd_raw = actions[:, 2:]
        kp_min, kp_max = self.cfg.kp_range
        kd_min, kd_max = self.cfg.kd_range
        kp = kp_min + torch.sigmoid(kp_raw) * (kp_max - kp_min)
        kd = kd_min + torch.sigmoid(kd_raw) * (kd_max - kd_min)
        self._tdc.update_gains(kp=kp, kd=kd)

        # --- 3. TDC control pipeline (same as parent) ---
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_body = self._robot.data.root_ang_vel_b

        p_EE_desired = self._tdc.compute(
            roll=roll,
            pitch=pitch,
            ang_vel_body=ang_vel_body,
            target_euler=self._target_euler,
        )

        # --- 4. DLS inverse kinematics -> joint angles ---
        current_joints = self._joint_pos_targets.clone()
        joint_targets = self._kinematics.inverse(
            p_EE_desired,
            current_joint_angles=current_joints,
            lambda_dls=self.cfg.tdc.ik_dls_lambda,
        )

        # --- 5. Clamp to joint limits ---
        joint_targets = torch.clamp(
            joint_targets,
            self._joint_limits_lower,
            self._joint_limits_upper,
        )

        # --- 6. Rate limiting ---
        max_delta = self.cfg.tdc.max_joint_velocity * self._tdc_dt
        delta = joint_targets - self._joint_pos_targets
        delta = torch.clamp(delta, -max_delta, max_delta)
        self._joint_pos_targets = self._joint_pos_targets + delta

        # --- 7. Anti-windup: feed actual (rate-limited) EE back to controller ---
        p_EE_actual = self._kinematics.forward(self._joint_pos_targets)
        self._tdc.update_ee_position(p_EE_actual)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments including cached M_hat."""
        super()._reset_idx(env_ids)
        # Reset M_hat to defaults for reset envs (will be re-populated from encoder z)
        if env_ids is not None:
            m_hat_default = torch.tensor(self.cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
            self._encoder_m_hat[env_ids] = m_hat_default
