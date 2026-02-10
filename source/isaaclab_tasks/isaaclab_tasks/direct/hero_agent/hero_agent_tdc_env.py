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

import logging

import torch

from isaaclab.utils.math import euler_xyz_from_quat

logger = logging.getLogger(__name__)

from .controllers import ALBCKinematics, TDCController
from .hero_agent_env import HeroAgentEnv
from .hero_agent_env_cfg import HeroAgentTDCEnvCfg


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
            self._setup_logger(tdc_cfg, self._tdc_dt)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Override RL actions with TDC control output.

        TDC runs every control_decimation steps (50Hz). Between TDC steps,
        existing joint targets are held.

        Args:
            actions: RL actions (ignored). Shape: (num_envs, 2).
        """
        # Store actions for observation/logging compatibility
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._prev_actions_obs = self._actions.clone()
        self._control_step_counter += 1

        # Only run TDC at control_decimation interval
        if self._control_step_counter % self.cfg.control_decimation != 0:
            return

        # --- TDC control pipeline ---
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
        )

        # 4. DLS inverse kinematics -> joint angles
        current_joints = self._joint_pos_targets.clone()
        joint_targets = self._kinematics.inverse(
            p_EE_desired,
            current_joint_angles=current_joints,
            lambda_dls=self.cfg.tdc.ik_dls_lambda,
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
            self._log_control_state(roll, pitch, ang_vel_body, p_EE_desired)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments including TDC controller state.

        Args:
            env_ids: Environment indices to reset. None = all.
        """
        if self._log_interval > 0 and env_ids is not None and len(env_ids) < self.num_envs:
            self._log_reset_info(env_ids)

        super()._reset_idx(env_ids)

        # Reset TDC controller history for reset environments
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._tdc.reset(env_ids)

        # Update buoyancy force for reset envs (may have changed from DR)
        self._tdc.update_buoyancy_force(self._buoy_hydro.buoyancy_force, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _setup_logger(self, tdc_cfg, tdc_dt: float) -> None:
        """Configure console logger for TDC diagnostics."""
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[TDC] %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
        logger.info(
            "TDC Controller initialized | M_hat=(%.3f, %.3f) Kp=%.1f Kd=%.1f "
            "F_bu=%.2f h=%.3f dt=%.4f max_jvel=%.1f base=(%.3f,%.3f)",
            *tdc_cfg.m_hat,
            tdc_cfg.kp,
            tdc_cfg.kd,
            self._tdc.F_bu[0].item(),
            tdc_cfg.h,
            tdc_dt,
            tdc_cfg.max_joint_velocity,
            *tdc_cfg.base_position,
        )

    def _log_control_state(
        self,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        ang_vel_body: torch.Tensor,
        p_EE_desired: torch.Tensor,
    ) -> None:
        """Log control diagnostics for a single environment."""
        i = self._log_env_id
        t = self._control_step_counter * self.step_dt
        r_deg = torch.rad2deg(roll[i]).item()
        p_deg = torch.rad2deg(pitch[i]).item()
        err_r = torch.rad2deg(self._target_euler[i, 0] - roll[i]).item()
        err_p = torch.rad2deg(self._target_euler[i, 1] - pitch[i]).item()
        pq = ang_vel_body[i, :2]
        ee = p_EE_desired[i]
        jt = self._joint_pos_targets[i]
        logger.info(
            "t=%5.2fs | roll=%+6.2f pitch=%+6.2f | err_r=%+6.2f err_p=%+6.2f | "
            "p=%.3f q=%.3f | pEE=(%.4f,%.4f) | j=(%.3f,%.3f)",
            t,
            r_deg,
            p_deg,
            err_r,
            err_p,
            pq[0].item(),
            pq[1].item(),
            ee[0].item(),
            ee[1].item(),
            jt[0].item(),
            jt[1].item(),
        )

    def _log_reset_info(self, env_ids: torch.Tensor) -> None:
        """Log termination reason before reset clears state."""
        terminated, time_out = self._get_dones()
        for eid in env_ids:
            i = eid.item()
            reason = "timeout" if time_out[i] else ("terminated" if terminated[i] else "unknown")
            height = self._robot.data.root_pos_w[i, 2].item()
            xy = self._robot.data.root_pos_w[i, :2] - self.scene.env_origins[i, :2]
            dist = torch.linalg.norm(xy).item()
            ep_len = self.episode_length_buf[i].item()
            logger.info(
                "RESET env=%d | reason=%s | ep_steps=%d (%.1fs) | h=%.2f dist=%.2f",
                i,
                reason,
                ep_len,
                ep_len * self.step_dt,
                height,
                dist,
            )
