# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent MPC Environment: 2D joint velocity actions, MPC state in obs.

SAC-MPC Architecture:
    MPC solver lives inside the policy (actor_critic_mpc.py), NOT in this env.
    The env provides:
        - 2D action space: joint velocity commands (MPC output applied directly)
        - mpc_state (10D): [phi, theta, p, q, q1, q2, q1_dot, q2_dot, q1_target, q2_target]
        - mpc_target (10D): [target_phi, target_theta, 0, 0, 0, 0, 0, 0, 0, 0]

    The policy uses mpc_state + mpc_target to run MPC.solve() and produce 2D actions.

Control Flow (50Hz):
    1. Policy receives obs dict with "policy" (13D), "mpc_state" (10D), "mpc_target" (10D)
    2. Policy internally: cost_map(policy_obs)->Q, MPC.solve->u
    3. Env receives 2D joint velocity commands
    4. Integrate velocity -> position targets, clamp to limits

The environment inherits from HeroAgentEnv (base_env), NOT tdc_env.
"""

from __future__ import annotations

import logging

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_tasks.direct.hero_agent.base_env import HeroAgentEnv

from .config import HeroAgentMPCEnvCfg

logger = logging.getLogger(__name__)


class HeroAgentMPCEnv(HeroAgentEnv):
    """Hero Agent environment for SAC-MPC: 2D actions with MPC state in observations.

    The RL policy (SAC actor) contains the MPC solver internally.
    This env simply receives 2D joint velocity commands and provides
    the necessary state information for the policy's MPC.

    Observation Space (13D): same as base_env (policy obs)
    Action Space (2D): joint velocity commands from policy's MPC
    Additional obs keys: "mpc_state" (10D), "mpc_target" (10D)
    """

    cfg: HeroAgentMPCEnvCfg

    def __init__(self, cfg: HeroAgentMPCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # MPC control frequency: with decimation=4, step_dt = 0.02s = control_dt
        self._mpc_dt = self.step_dt

        logger.info(
            "MPC env (SAC): action_space=%d, step_dt=%.4fs (policy=control freq)",
            cfg.action_space, self._mpc_dt,
        )

    def _extract_mpc_state(self) -> torch.Tensor:
        """Extract the 10D MPC state from the simulation.

        State: [phi, theta, phi_dot, theta_dot, q1, q2, q1_dot, q2_dot, q1_target, q2_target]

        Physical state (8D) from sensors available on real hardware:
        - phi, theta: IMU orientation (Euler angles)
        - phi_dot, theta_dot: IMU angular velocity (body frame p, q)
        - q1, q2: joint encoders (position)
        - q1_dot, q2_dot: joint encoders (velocity)

        Actuator state (2D) from PD controller:
        - q1_target, q2_target: current joint position targets (known to controller)

        Returns:
            MPC state. Shape: (num_envs, 10).
        """
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel = self._robot.data.root_ang_vel_b  # (num_envs, 3)

        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]  # (num_envs, 2)
        joint_vel = self._robot.data.joint_vel[:, self._albc_joint_ids]  # (num_envs, 2)

        return torch.cat([
            roll.unsqueeze(-1),       # phi
            pitch.unsqueeze(-1),      # theta
            ang_vel[:, 0:1],          # phi_dot (p)
            ang_vel[:, 1:2],          # theta_dot (q)
            joint_pos,                # q1, q2
            joint_vel,                # q1_dot, q2_dot
            self._joint_pos_targets,  # q1_target, q2_target
        ], dim=-1)

    def _build_target_state(self) -> torch.Tensor:
        """Build the 10D MPC target state.

        Target: [target_phi, target_theta, 0, 0, 0, 0, 0, 0, 0, 0]

        Joint targets are neutral [0, 0] so MPC cost penalizes deviation from
        center, preventing singularity drift and enabling Q[4:5] learning.
        q_target targets are also 0 (neutral position).

        Returns:
            Target state. Shape: (num_envs, 10).
        """
        return torch.cat([
            self._target_euler[:, 0:1],   # target roll
            self._target_euler[:, 1:2],   # target pitch
            torch.zeros(self.num_envs, 8, device=self.device),  # zero: ang vel, joint pos/vel, q_target
        ], dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply 2D joint velocity commands from policy's MPC.

        With decimation=4, each env.step() = one control step at 50Hz.
        The policy has already run MPC internally and output 2D joint velocities.
        We integrate velocity -> position targets every step (no modulo gating).

        Args:
            actions: Joint velocity commands from policy. Shape: (num_envs, 2).
        """
        self._update_action_buffers(actions)

        # Apply action latency if configured
        effective_actions = self._get_delayed_actions(self._actions)

        # Integrate velocity -> position targets (step_dt = 0.02s = control_dt)
        position_delta = self._mpc_dt * self.cfg.max_joint_velocity * effective_actions
        self._joint_pos_targets += position_delta
        self._joint_pos_targets = torch.clamp(
            self._joint_pos_targets,
            self._joint_limits_lower,
            self._joint_limits_upper,
        )

    def _get_observations(self) -> dict:
        """Compute observations including MPC state and target.

        Returns dict with keys: "policy", "mpc_state", "mpc_target".
        """
        observations = super()._get_observations()
        observations["mpc_state"] = self._extract_mpc_state()
        observations["mpc_target"] = self._build_target_state()
        return observations
