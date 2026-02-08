# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent TDC (Time Delay Control) Environment.

This module implements a non-RL control environment for Hero Agent that uses
TDC for attitude stabilization. The TDC controller replaces RL actions with
computed end-effector positions, which are then converted to joint angles
via inverse kinematics.

Control Flow (100 Hz):
    1. Read roll, pitch from quaternion
    2. Read body angular velocity [p, q]
    3. Compute current EE position via FK
    4. TDCController.compute() -> desired p_EE
    5. ALBCKinematics.inverse(p_EE) -> joint angles
    6. Clamp to joint limits -> set as joint_pos_targets
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

    The TDC controller runs at 100 Hz (every RL step, ignoring control_decimation)
    to maintain accurate Time Delay Estimation.
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

        # ALBC kinematics for IK (p_EE -> joint angles) and FK (joint angles -> p_EE)
        self._kinematics = ALBCKinematics(
            num_envs=self.num_envs,
            device=self.device,
        )

        # Get buoyancy force from buoy hydrodynamics model
        F_bu = self._buoy_hydro.buoyancy_force

        # TDC controller
        self._tdc = TDCController(
            num_envs=self.num_envs,
            device=self.device,
            m_hat=self.cfg.tdc_m_hat,
            kp=self.cfg.tdc_kp,
            kd=self.cfg.tdc_kd,
            F_bu=F_bu.mean().item() if F_bu.dim() > 0 else float(F_bu),
            h=self.cfg.tdc_h,
            dls_damping=self.cfg.tdc_dls_damping,
            dt=self.step_dt,  # TDC dt = RL step dt (100 Hz with dec=2)
            workspace_radius=self.cfg.tdc_workspace_radius,
            nu_dot_ema_alpha=self.cfg.tdc_nu_dot_ema_alpha,
            tde_gain=self.cfg.tdc_tde_gain,
            h_hat_filter_alpha=self.cfg.tdc_h_hat_filter_alpha,
        )

        # Console logging config
        self._log_interval = self.cfg.tdc_log_interval
        self._log_env_id = 0  # Which env to log (env 0)

        if self._log_interval > 0:
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[TDC] %(message)s"))
                logger.addHandler(handler)
                logger.propagate = False
            logger.info(
                "TDC Controller initialized | M_hat=(%.3f, %.3f) Kp=%.1f Kd=%.1f F_bu=%.2f h=%.3f dt=%.4f tde_gain=%.2f",
                *self.cfg.tdc_m_hat, self.cfg.tdc_kp, self.cfg.tdc_kd,
                self._tdc._F_bu[0].item(), self.cfg.tdc_h, self.step_dt,
                self.cfg.tdc_tde_gain,
            )

    def _pre_physics_step(self, actions: torch.Tensor):
        """Override RL actions with TDC control output.

        TDC runs every step (ignoring control_decimation) to maintain
        accurate TDE. RL actions are stored but not used for control.

        Args:
            actions: RL actions (ignored). Shape: (num_envs, 2).
        """
        # Store actions for observation/logging compatibility
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._prev_actions_obs = self._actions.clone()
        self._control_step_counter += 1

        # --- TDC control pipeline ---
        # 1. Get current orientation
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)

        # 2. Get body angular velocity [p, q, r]
        ang_vel_body = self._robot.data.root_ang_vel_b

        # 3. Get current joint positions
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]

        # 4. TDC compute -> desired EE position
        p_EE_desired = self._tdc.compute(
            roll=roll,
            pitch=pitch,
            ang_vel_body=ang_vel_body,
            target_euler=self._target_euler,
            joint_pos=joint_pos,
            kinematics=self._kinematics,
        )

        # 5. Inverse kinematics -> joint angles
        joint_targets = self._kinematics.inverse(p_EE_desired)

        # 6. Clamp to joint limits and set targets
        self._joint_pos_targets = torch.clamp(
            joint_targets,
            self._joint_limits_lower,
            self._joint_limits_upper,
        )

        # 7. Console logging
        if self._log_interval > 0 and self._control_step_counter % self._log_interval == 0:
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
                t, r_deg, p_deg, err_r, err_p,
                pq[0].item(), pq[1].item(),
                ee[0].item(), ee[1].item(),
                jt[0].item(), jt[1].item(),
            )
            # Debug: TDC internal term magnitudes
            dbg = self._tdc._debug
            def _n(x):
                return torch.norm(x[i]).item()
            logger.info(
                "  terms | Lam*pEE=%+.4f | M*nu_dot=%+.4f | M*u_pd=%+.4f | "
                "dT_b=%+.4f | H_hat=%+.4f | tau=%+.4f | pEE_raw_r=%.4f",
                _n(dbg["tde_lambda_p"]), _n(dbg["tde_m_nu_dot"]),
                _n(dbg["m_hat_u_pd"]), _n(dbg["tde_delta_T_b"]),
                _n(dbg["tde_H_hat"]), _n(dbg["tau_desired"]),
                dbg["p_EE_raw_norm"][i].item(),
            )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments including TDC controller state.

        Args:
            env_ids: Environment indices to reset. None = all.
        """
        super()._reset_idx(env_ids)

        # Reset TDC controller history for reset environments
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._tdc.reset(env_ids)

        # Update buoyancy force for reset envs only (may have changed from DR)
        self._tdc.update_buoyancy_force(self._buoy_hydro.buoyancy_force, env_ids=env_ids)
