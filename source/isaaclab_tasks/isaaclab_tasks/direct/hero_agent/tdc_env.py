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
from .utils.logging_tdc import log_tdc_control_state, log_tdc_init, log_tdc_reset_info


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

    def _run_tdc_pipeline(self) -> None:
        """Run the TDC control pipeline: orientation -> TDC -> IK -> rate limit -> anti-windup.

        Shared between HeroAgentTDCEnv and HeroAgentEncoderTDCEnv.
        Subclasses should update TDC params (M_hat, gains) before calling this.
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
