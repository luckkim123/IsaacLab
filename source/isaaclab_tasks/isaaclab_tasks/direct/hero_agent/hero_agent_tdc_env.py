# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent TDC (Time Delay Control) Environment.

This module implements the TDC-integrated attitude control for Hero Agent,
based on the IROS 2026 RL-ALBC paper methodology.

Control Flow:
    1. RL Actor outputs PD gains: [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
    2. Encoder outputs z (6D) -> M_hat = diag(z[:2]) for inertia estimation
    3. TDC controller: gains + M_hat + attitude_error -> p_EE_desired
    4. IK: p_EE_desired -> target_joint_angles
    5. Joint position control: apply target to robot joints

The TDC controller uses Time Delay Estimation (TDE) to compensate for
model uncertainty, combined with learned adaptive PD gains.

Key Differences from HeroAgentEnv:
    - Action space: 4D gains instead of 2D joint velocities
    - Control path: TDC -> FK/IK pipeline instead of direct velocity integration
    - Reward terms: Additional TDC stability and gain smoothness rewards
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_HEIGHT_OFFSET,
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
)

from .controllers import ALBCKinematics, TDCController, TDCControllerCfg
from .hero_agent_env import HeroAgentEnv
from .hero_agent_env_cfg import HeroAgentEncoderTDCEnvCfg
from .mdp import (
    RewardTermCfg,
    albc_potential_reward,
    albc_progress_reward,
    tdc_gain_magnitude_cost,
    tdc_gain_smoothness_reward,
    tdc_stability_reward,
)


class HeroAgentTDCEnv(HeroAgentEnv):
    """Hero Agent environment with TDC-based attitude control.

    This environment extends HeroAgentEnv to use TDC (Time Delay Control)
    for attitude stabilization. The RL policy outputs PD gains instead of
    direct joint velocities, and the TDC controller computes desired
    end-effector positions.

    Observation Space (15 dims):
        [0:3]   roll, pitch, yaw (Euler angles)
        [3:6]   angular velocity in body frame
        [6:9]   attitude errors (target - current)
        [9:11]  joint positions (normalized)
        [11:15] previous actions (all 4 gains)

    Action Space (4 dims):
        [0] K_p_roll   - Proportional gain for roll
        [1] K_d_roll   - Derivative gain for roll
        [2] K_p_pitch  - Proportional gain for pitch
        [3] K_d_pitch  - Derivative gain for pitch

    All actions are in [-1, 1] and scaled to gain ranges by TDC controller.
    """

    cfg: HeroAgentEncoderTDCEnvCfg

    def __init__(self, cfg: HeroAgentEncoderTDCEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the Hero Agent TDC environment.

        Args:
            cfg: TDC environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        # Initialize parent class first
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize TDC-specific components
        self._init_tdc_components()

    def _init_tdc_components(self) -> None:
        """Initialize TDC controller and kinematics.

        Link lengths are sourced from robot config constants (hero_agent.py),
        which are derived from URDF joint origins. This ensures consistency
        with the robot model and eliminates hardcoded values in env config.
        """
        # Use link lengths from robot config (derived from URDF)
        link1_length = HERO_AGENT_ALBC_LINK1_LENGTH
        link2_length = HERO_AGENT_ALBC_LINK2_LENGTH
        height_offset = HERO_AGENT_ALBC_HEIGHT_OFFSET

        # Compute workspace radius from link lengths (circular reachable region)
        workspace_radius_min = abs(link1_length - link2_length) + 1e-4
        workspace_radius_max = link1_length + link2_length - 1e-4

        # Build TDC controller configuration
        tdc_cfg = TDCControllerCfg(
            k_p_min=self.cfg.tdc_k_p_min,
            k_p_max=self.cfg.tdc_k_p_max,
            k_d_min=self.cfg.tdc_k_d_min,
            k_d_max=self.cfg.tdc_k_d_max,
            tde_delay_steps=self.cfg.tdc_tde_delay_steps,
            workspace_radius_min=workspace_radius_min,
            workspace_radius_max=workspace_radius_max,
            default_m_hat=self.cfg.tdc_default_m_hat,
            height_offset=height_offset,
        )

        # Initialize TDC controller
        self._tdc_controller = TDCController(
            cfg=tdc_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Initialize ALBC kinematics
        self._kinematics = ALBCKinematics(
            num_envs=self.num_envs,
            device=self.device,
            link1_length=link1_length,
            link2_length=link2_length,
            height_offset=height_offset,
        )

        # Buffer for storing true inertia (for stability reward)
        self._true_inertia = torch.ones(self.num_envs, 2, device=self.device)

        # Buffer for encoder z (set by external policy during training)
        self._encoder_z: torch.Tensor | None = None

    def _init_task_and_rewards(self) -> None:
        """Initialize attitude task buffers and TDC-specific reward manager."""
        from .mdp import RewardManager

        self._init_attitude_buffers()

        # Build TDC-specific reward configuration
        self._reward_manager = RewardManager(
            cfg={
                "potential": RewardTermCfg(
                    func=albc_potential_reward,
                    weight=self.cfg.reward.potential_weight,
                    params={"scale": self.cfg.reward.potential_scale},
                ),
                "progress": RewardTermCfg(
                    func=albc_progress_reward,
                    weight=self.cfg.reward.progress_weight,
                    scale_by_dt=False,
                ),
                "tdc_stability": RewardTermCfg(
                    func=tdc_stability_reward,
                    weight=self.cfg.reward.stability_weight,
                    params={"scale": self.cfg.reward.stability_scale},
                ),
                "gain_smoothness": RewardTermCfg(
                    func=tdc_gain_smoothness_reward,
                    weight=self.cfg.reward.gain_smoothness_weight,
                ),
                "gain_magnitude": RewardTermCfg(
                    func=tdc_gain_magnitude_cost,
                    weight=self.cfg.reward.gain_magnitude_weight,
                ),
            },
            num_envs=self.num_envs,
            device=self.device,
        )

    def _init_state_buffers(self) -> None:
        """Initialize action buffers for TDC (4D gains instead of 2D velocities)."""
        # Initialize force/torque buffers from parent
        super()._init_state_buffers()

        # Override action buffers for 4D gain space
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device)
        # Override prev_actions_obs to 4D (all gains) for 15D observation
        self._prev_actions_obs = torch.zeros(self.num_envs, 4, device=self.device)

    def set_encoder_z(self, z: torch.Tensor) -> None:
        """Set encoder latent z for TDC inertia estimation.

        This method should be called by the training loop after the policy
        forward pass to provide M_hat to the TDC controller.

        Args:
            z: Encoder latent output (positive values via softplus).
                Shape: (num_envs, latent_dim) where latent_dim >= 2.
        """
        self._encoder_z = z
        self._tdc_controller.set_inertia_estimate(z)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process gain actions and compute TDC output.

        Args:
            actions: PD gain commands [-1, 1]. Shape: (num_envs, 4).
                [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
        """
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._control_step_counter += 1

        if self._control_step_counter % self.cfg.control_decimation == 0:
            # Store all 4 gains for 15D TDC observation
            self._prev_actions_obs = self._actions.clone()

            # Set gains in TDC controller
            self._tdc_controller.set_gains(self._actions)

            # Compute attitude error for TDC
            attitude_error = self._get_tdc_attitude_error()

            # Get angular velocity (body frame, roll/pitch only)
            angular_vel = self._robot.data.root_ang_vel_b[:, :2]

            # Extract roll and pitch from quaternion for Lambda and T_b
            roll, pitch, _yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)

            # Get buoyancy force magnitude from buoy hydrodynamics
            f_bu = self._buoy_hydro.buoyancy_force

            # Compute desired end-effector position via TDC
            control_dt = self.physics_dt * self.cfg.control_decimation
            p_ee_desired = self._tdc_controller.compute(
                attitude_error=attitude_error,
                angular_velocity=angular_vel,
                dt=control_dt,
                roll=roll,
                pitch=pitch,
                f_bu=f_bu,
            )

            # Compute target joint angles via IK
            target_joint_angles = self._kinematics.inverse(target_position=p_ee_desired)

            # Update joint position targets
            self._joint_pos_targets = torch.clamp(
                target_joint_angles,
                self._joint_limits_lower,
                self._joint_limits_upper,
            )

    def _get_tdc_attitude_error(self) -> torch.Tensor:
        """Get attitude error for TDC controller (roll, pitch only).

        Returns:
            Attitude error [roll_error, pitch_error] in radians.
                Shape: (num_envs, 2).
        """
        # Compute full attitude error
        full_error = self._compute_attitude_error(self._robot.data.root_quat_w)
        # Return only roll and pitch (TDC cannot control yaw with buoyancy)
        return full_error[:, :2]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments including TDC state.

        Args:
            env_ids: Environment indices to reset.
        """
        # Call parent reset first (handles None and full batch conversion)
        super()._reset_idx(env_ids)

        # Convert to tensor for TDC-specific reset
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Reset TDC controller state
        self._tdc_controller.reset(env_ids)

        # Reset encoder z buffer
        if self._encoder_z is not None:
            self._encoder_z[env_ids] = 1.0  # Reset to neutral value

        # Update true inertia from hydro properties for TDC stability reward
        # Use effective inertia (rigid body + added mass) for accurate underwater dynamics
        # Hydrodynamics model:
        #   - rigid_body_inertia: (num_envs, 3) -> [Ixx, Iyy, Izz]
        #   - added_mass_matrix: (num_envs, 6, 6) diagonal -> [u, v, w, p, q, r]
        #   - Indices 3, 4 correspond to roll (p) and pitch (q) added mass
        rigid_inertia = self._hydro.rigid_body_inertia[env_ids, :2]  # Ixx, Iyy
        added_mass_roll = self._hydro.added_mass_matrix[env_ids, 3, 3]  # M_44 (roll)
        added_mass_pitch = self._hydro.added_mass_matrix[env_ids, 4, 4]  # M_55 (pitch)
        added_mass_rot = torch.stack([added_mass_roll, added_mass_pitch], dim=-1)
        self._true_inertia[env_ids] = rigid_inertia + added_mass_rot

    @property
    def tdc_controller(self) -> TDCController:
        """Get the TDC controller instance.

        Returns:
            TDC controller for external inspection/modification.
        """
        return self._tdc_controller

    @property
    def kinematics(self) -> ALBCKinematics:
        """Get the kinematics instance.

        Returns:
            ALBC kinematics for external inspection.
        """
        return self._kinematics
