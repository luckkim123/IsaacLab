# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent TDC (Time Delay Control) Environment.

This module implements a non-RL control environment for Hero Agent that uses
TDC for attitude stabilization. The TDC controller replaces RL actions with
computed end-effector positions, which are then converted to joint angles
via inverse kinematics.

Control Flow (50 Hz, every control_decimation steps):
    1. Read roll, pitch from quaternion
    2. Read body angular velocity [p, q]
    3. TDCController.compute() -> desired p_EE
    4. ALBCKinematics.inverse(p_EE, current_joints) -> joint angles (DLS IK)
    5. Rate limit + clamp to joint limits -> set as joint_pos_targets
    6. FK(rate-limited) -> update_ee_position() (anti-windup)
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from .base_env import HeroAgentEnv
from .config import HeroAgentTDCEnvCfg
from .controllers import ALBCKinematics, TDCController
from .controllers.tdc import compute_M_hat_from_z
from .mdp import RewardTermCfg, compute_stability_gate, mhat_accuracy_reward, tdc_torque_penalty
from .utils.logging import log_tdc_control_state, log_tdc_init, log_tdc_reset_info


class HeroAgentTDCEnv(HeroAgentEnv):
    """Hero Agent environment with TDC attitude controller.

    This environment overrides the RL action pipeline with a classical TDC
    controller. RL actions are ignored; the controller directly computes
    joint position targets from the current state.

    The TDC controller runs at 50 Hz (every control_decimation steps = 0.02s)
    matching the C++ reference implementation.
    """

    cfg: HeroAgentTDCEnvCfg

    def __init__(self, cfg: HeroAgentTDCEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize TDC environment.

        Args:
            cfg: TDC environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(cfg, render_mode, **kwargs)

        tdc_cfg = cfg.tdc

        # ALBC kinematics for IK and FK
        self._kinematics = ALBCKinematics(
            num_envs=self.num_envs,
            device=self.device,
            link1_length=tdc_cfg.link1_length,
            link2_length=tdc_cfg.link2_length,
        )

        # Get buoyancy force from buoy hydrodynamics model
        F_bu = self._buoy_hydro.buoyancy_force

        # TDC dt = step_dt * control_decimation (50Hz with dec=1, ctrl_dec=4)
        self._tdc_dt = self.step_dt * self.cfg.control_decimation

        # TDC controller
        self._tdc = TDCController(
            num_envs=self.num_envs,
            device=self.device,
            cfg=tdc_cfg,
            F_bu=F_bu.mean().item() if F_bu.dim() > 0 else float(F_bu),
            dt=self._tdc_dt,
        )

        # Console logging
        self._log_interval = tdc_cfg.log_interval
        self._log_env_id = 0

        if self._log_interval > 0:
            log_tdc_init(tdc_cfg, self._tdc.F_bu[0].item(), self._tdc_dt)

        # Encoder policy reference (set by runner via set_encoder_policy)
        self._encoder_policy = None

        # Stability gate episode tracking (gate=1 means reward passed through)
        if cfg.reward.stability_gate_enable:
            self._gate_episode_sum = torch.zeros(self.num_envs, device=self.device)
            self._gate_episode_steps = torch.zeros(self.num_envs, device=self.device)

    def _build_reward_terms(self) -> dict[str, RewardTermCfg]:
        """Build reward terms: base terms + M_hat accuracy if configured."""
        terms = super()._build_reward_terms()
        rcfg = self.cfg.reward

        if hasattr(rcfg, "mhat_accuracy_weight") and rcfg.mhat_accuracy_weight != 0.0:
            kernel = getattr(rcfg, "mhat_accuracy_kernel", "cauchy")
            terms["mhat_accuracy"] = RewardTermCfg(
                func=mhat_accuracy_reward,
                weight=rcfg.mhat_accuracy_weight,
                params={"sigma": rcfg.mhat_accuracy_sigma, "kernel": kernel},
            )

        if hasattr(rcfg, "tdc_torque_weight") and rcfg.tdc_torque_weight != 0.0:
            terms["tdc_torque"] = RewardTermCfg(
                func=tdc_torque_penalty,
                weight=rcfg.tdc_torque_weight,
            )

        return terms

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with optional TDC stability gate.

        When stability_gate_enable is True, total reward is multiplied by a
        binary gate: 1.0 if |1 - M_hat/M_true| < 1 on all axes, 0.0 otherwise.
        This forces the policy to learn M_hat values within the stable region.

        Based on Baek et al., "RL-based Adaptive TDC" (ACC 2022).
        """
        reward = super()._get_rewards()

        if self.cfg.reward.stability_gate_enable:
            gate = compute_stability_gate(self)
            reward = reward * gate

            # Track gate fraction for instantaneous logging
            self._stability_gate_frac = 1.0 - gate.mean().item()

            # Accumulate per-episode gate statistics
            self._gate_episode_sum += gate
            self._gate_episode_steps += 1

        return reward

    def _collect_episode_metrics(
        self, env_ids: torch.Tensor, reward_sums: dict[str, torch.Tensor]
    ) -> dict[str, float | torch.Tensor]:
        """Add stability gate open fraction to episode metrics."""
        log = super()._collect_episode_metrics(env_ids, reward_sums)

        if self.cfg.reward.stability_gate_enable and len(env_ids) > 0:
            steps = self._gate_episode_steps[env_ids]
            sums = self._gate_episode_sum[env_ids]
            valid = steps > 0
            open_frac = torch.where(valid, sums / steps, torch.ones_like(steps))
            log["Episode_Reward/stability_gate_open_frac"] = open_frac.mean().item()

        return log

    def set_encoder_policy(self, policy: object) -> None:
        """Register the encoder policy for z -> M_hat extraction.

        Called by the runner after policy creation to enable M_hat transfer.

        Args:
            policy: Policy instance with get_last_z() method.
        """
        self._encoder_policy = policy

    def _create_m_hat_buffer(self) -> torch.Tensor:
        """Create an M_hat buffer initialized with config defaults.

        Returns:
            M_hat buffer. Shape: (num_envs, 2).
        """
        m_hat_default = torch.tensor(self.cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
        return m_hat_default.unsqueeze(0).expand(self.num_envs, -1).clone()

    def _reset_m_hat_defaults(self, env_ids: torch.Tensor) -> None:
        """Reset M_hat to config defaults for specified envs and push to TDC controller.

        Args:
            env_ids: Environment indices to reset.
        """
        m_hat_default = torch.tensor(self.cfg.tdc.m_hat, device=self.device, dtype=torch.float32)
        self._tdc.update_controller_params(
            m_hat=m_hat_default.unsqueeze(0).expand(len(env_ids), -1),
            env_ids=env_ids,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Override RL actions with TDC control output.

        TDC runs every control_decimation steps (50Hz). Between TDC steps,
        existing joint targets are held.

        Args:
            actions: RL actions (ignored). Shape: (num_envs, 2).
        """
        self._update_action_buffers(actions)

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        self._run_tdc_pipeline()

    def _run_tdc_pipeline(self, **compute_kwargs) -> None:
        """Run the TDC control pipeline: orientation -> TDC -> IK -> rate limit -> anti-windup.

        Shared between HeroAgentTDCEnv, HeroAgentEncoderTDCEnv, and HeroAgentNeuralTDCEnv.
        Subclasses should update TDC params (M_hat, gains) before calling this.

        Args:
            **compute_kwargs: Extra keyword arguments forwarded to TDCController.compute().
                Used by Neural TDC to pass external_u_pd.
        """
        # 1. Get current orientation
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)

        # 2. Get body angular velocity [p, q, r]
        ang_vel_body = self._robot.data.root_ang_vel_b

        # 3. TDC compute -> desired EE position
        p_EE_desired = self._tdc.compute(
            roll=roll,
            pitch=pitch,
            ang_vel_body=ang_vel_body,
            target_euler=self._target_euler,
            **compute_kwargs,
        )

        # 4. DLS inverse kinematics -> joint angles
        current_joints = self._joint_pos_targets.clone()
        joint_targets = self._kinematics.inverse(
            p_EE_desired,
            current_joint_angles=current_joints,
            lambda_dls=self.cfg.tdc.ik_dls_lambda,
            num_iterations=self.cfg.tdc.ik_num_iterations,
            learning_rate=self.cfg.tdc.ik_learning_rate,
        )

        # 5. Clamp to joint limits
        joint_targets = torch.clamp(
            joint_targets,
            self._joint_limits_lower,
            self._joint_limits_upper,
        )

        # 6. Rate limiting
        max_delta = self.cfg.tdc.max_joint_velocity * self._tdc_dt
        delta = joint_targets - self._joint_pos_targets
        delta = torch.clamp(delta, -max_delta, max_delta)
        self._joint_pos_targets = self._joint_pos_targets + delta

        # 7. Anti-windup: feed actual (rate-limited) EE back to controller
        p_EE_actual = self._kinematics.forward(self._joint_pos_targets)
        self._tdc.update_ee_position(p_EE_actual)

        # 8. Console logging
        if self._log_interval > 0 and self._control_step_counter % self._log_interval == 0:
            log_tdc_control_state(
                step_counter=self._control_step_counter,
                step_dt=self.step_dt,
                log_env_id=self._log_env_id,
                roll=roll,
                pitch=pitch,
                ang_vel_body=ang_vel_body,
                p_EE_desired=p_EE_desired,
                joint_pos_targets=self._joint_pos_targets,
                target_euler=self._target_euler,
            )

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments including TDC controller state.

        Args:
            env_ids: Environment indices to reset. None = all.
        """
        if self._log_interval > 0 and env_ids is not None and len(env_ids) < self.num_envs:
            log_tdc_reset_info(
                env_ids=env_ids,
                terminated=self.reset_terminated,
                time_outs=self.reset_time_outs,
                root_pos_w=self._robot.data.root_pos_w,
                env_origins=self.scene.env_origins,
                episode_length_buf=self.episode_length_buf,
                step_dt=self.step_dt,
            )

        super()._reset_idx(env_ids)

        # Reset TDC controller history for reset environments
        env_ids_ = self._coerce_env_ids(env_ids)
        self._tdc.reset(env_ids_)

        # Update buoyancy force for reset envs (may have changed from DR)
        self._tdc.update_controller_params(F_bu=self._buoy_hydro.buoyancy_force[env_ids_], env_ids=env_ids_)

        # Reset stability gate episode accumulators (logging already happened in super)
        if self.cfg.reward.stability_gate_enable:
            self._gate_episode_sum[env_ids_] = 0
            self._gate_episode_steps[env_ids_] = 0

    def _compute_m_hat_from_encoder_z(self, z_decomposed: torch.Tensor) -> torch.Tensor:
        """Compute M_hat from decomposed encoder z components + current FK joint positions.

        Shared helper for encoder_tdc_env, unified_tdc_env, and adapt_tdc_env.
        Uses parallel axis theorem: M_hat = I_hat + m_A_hat * (p_EE^2 + h^2).

        Args:
            z_decomposed: Decomposed z = [m_A_hat, I_roll_hat, I_pitch_hat].
                Shape: (num_envs, 3).

        Returns:
            Design inertia M_hat [roll, pitch]. Shape: (num_envs, 2).
        """
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        p_EE = self._kinematics.forward(joint_pos)
        return compute_M_hat_from_z(z_decomposed, p_EE, self.cfg.tdc.h)
