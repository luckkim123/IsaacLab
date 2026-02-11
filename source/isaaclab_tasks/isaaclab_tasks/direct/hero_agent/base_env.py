# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent ALBC (Active Linear Buoyancy Controller) Environment.

This module implements joint-based attitude control for Hero Agent without thrusters.
The ALBC uses 2 revolute joints (joint1, joint2) to position a buoyancy element
for attitude stabilization.

Control Flow:
    actions [-1, 1] -> accumulate with dt*scale -> clamp to limits -> position target

Hero Agent has a unique buoy body (link3) that requires separate hydrodynamic
force calculations.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

# Import models from common isaaclab_tasks.models
from isaaclab_tasks.models import HydrodynamicsModel

from .config import HeroAgentEnvCfg
from .mdp import (
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    angular_velocity_penalty,
    compute_policy_obs,
    compute_privileged_obs,
    progress_reward,
    tracking_reward,
)
from .mdp.events import (
    randomize_body_mass,
    randomize_hydrodynamics,
    randomize_joint_friction,
    randomize_joint_gains,
    randomize_joint_positions,
    randomize_ocean_current,
    randomize_payload,
    randomize_robot_pose,
    reset_joint_positions_default,
    reset_robot_pose_default,
)
from .utils import DebugVisualization, log_dr_metrics, log_tdc_diagnostics


class HeroAgentEnvWindow(BaseEnvWindow):
    """Window manager for the Hero Agent environment."""

    def __init__(self, env: HeroAgentEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


class HeroAgentEnv(DirectRLEnv):
    """Hero Agent ALBC environment for attitude control using joint-based buoyancy control.

    This environment implements:
    - Joint position control (no thrusters)
    - Multi-body hydrodynamics (main body + buoy)
    - Potential-based reward system
    - Decimated control (default: every physics step, configurable via control_decimation)

    Observation Space (13 dims):
        [0:3]   roll, pitch, yaw (Euler angles from quaternion)
        [3:6]   angular velocity in body frame
        [6:9]   attitude errors (target - current, wrapped)
        [9:11]  joint positions (normalized to [-1, 1])
        [11:13] previous actions

    Action Space (2 dims):
        [0] joint1 velocity command [-1, 1]
        [1] joint2 velocity command [-1, 1]

    Physical Parameters:
        - sim_dt: 1/200 s (200 Hz physics), decimation: 1, control_decimation: 4 (50 Hz control)
        - max_joint_velocity: 2*pi rad/s (360 deg/s)
        - joint stiffness: 100.0, damping: 3.0 (ImplicitActuator default)
        - joint_limits: from URDF (±2*pi rad, i.e. ±360 deg)
    """

    cfg: HeroAgentEnvCfg

    def __init__(self, cfg: HeroAgentEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the Hero Agent ALBC environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        # Convert noise config tuples to tensors before DirectRLEnv creates the noise model.
        # Tuples are used in config for OmegaConf/Hydra serialization compatibility.
        self._convert_noise_cfg_tuples(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        # Pre-expand the bias buffer to match observation dimensions.
        # NoiseModelWithAdditiveBias initializes bias as (num_envs, 1) and only expands
        # on first __call__. But the wrapper calls env.reset() before any step, which
        # triggers noise_model.reset() while bias is still (N, 1). With per-dimension
        # n_min/n_max tensors, the reset produces (N, obs_dim) which can't fit in (N, 1).
        if self.cfg.observation_noise_model is not None:
            nm = self._observation_noise_model
            if nm._sample_bias_per_component and nm._num_components is None:
                nm._num_components = self.cfg.observation_space
                nm._bias = nm._bias.repeat(1, nm._num_components)

        # Validate state_space vs enable_payload consistency
        if self.cfg.state_space >= 24 and not self.cfg.enable_payload:
            raise ValueError(
                f"state_space={self.cfg.state_space} requires enable_payload=True "
                f"(payload provides 4D of the {self.cfg.state_space}D privileged obs)"
            )

        self._init_body_ids()
        self._init_hydrodynamics()
        self._init_payload()
        self._init_joints()
        self._init_task_and_rewards()
        self._init_state_buffers()

        # Debug visualization manager
        self._debug_vis = DebugVisualization(self.num_envs, self.device)
        self.set_debug_vis(self.cfg.debug_vis)

    @staticmethod
    def _convert_noise_cfg_tuples(cfg: HeroAgentEnvCfg) -> None:
        """Convert noise config tuple/list values to torch.Tensor in-place.

        Config uses tuples for OmegaConf/Hydra serialization compatibility.
        The noise model functions require float or torch.Tensor for arithmetic.
        Must be called before DirectRLEnv.__init__() which instantiates noise models.
        """
        noise_model = getattr(cfg, "observation_noise_model", None)
        if noise_model is None:
            return
        for sub_cfg_attr in ("noise_cfg", "bias_noise_cfg"):
            sub_cfg = getattr(noise_model, sub_cfg_attr, None)
            if sub_cfg is None:
                continue
            for param in ("std", "mean", "n_min", "n_max"):
                val = getattr(sub_cfg, param, None)
                if isinstance(val, (list, tuple)):
                    setattr(sub_cfg, param, torch.tensor(val))

    def _init_body_ids(self) -> None:
        """Initialize body IDs and physics parameters."""
        self._body_id = self._robot.find_bodies(self.cfg.hydrodynamics.body_name)[0]
        self._buoy_body_id = self._robot.find_bodies(self.cfg.buoy_hydrodynamics.body_name)[0]
        self._gripper_body_id = self._robot.find_bodies("gripper")[0]
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

    def _init_hydrodynamics(self) -> None:
        """Initialize hydrodynamics models for main body and buoy."""
        prim_path = self.cfg.robot.prim_path.replace("env_.*", "env_0")
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            current_cfg=self.cfg.ocean_current,
            dt=self.physics_dt,
            articulation_prim_path=prim_path,
        )
        self._buoy_hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.buoy_hydrodynamics,
            current_cfg=None,  # Buoy shares current with main body
            dt=self.physics_dt,
            articulation_prim_path=prim_path,
        )

    def _init_payload(self) -> None:
        """Initialize payload parameters if enabled.

        Payload is applied to the gripper body (fixed to base via base_to_gripper joint).
        """
        self._payload_enabled = self.cfg.enable_payload
        if not self._payload_enabled:
            self._payload_mass = None
            self._payload_attachment_offset = None
            self._payload_cog_offset = None
            self._gravity_vec = None
            return

        self._payload_mass = torch.full((self.num_envs,), self.cfg.payload_mass, device=self.device)
        offset_tensor = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
        self._payload_attachment_offset = offset_tensor.expand(self.num_envs, -1).clone()
        self._payload_cog_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self._gravity_vec = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)

    def _init_joints(self) -> None:
        """Initialize ALBC joint IDs and limits."""
        self._albc_joint_ids = self._robot.find_joints(self.cfg.albc_joint_names)[0]
        if len(self._albc_joint_ids) != 2:
            raise ValueError(
                f"Expected 2 ALBC joints, found {len(self._albc_joint_ids)}. Joint names: {self.cfg.albc_joint_names}"
            )
        joint_limits = self._robot.data.soft_joint_pos_limits[:, self._albc_joint_ids]
        self._joint_limits_lower = joint_limits[0, :, 0]
        self._joint_limits_upper = joint_limits[0, :, 1]
        self._joint_limits_range = self._joint_limits_upper - self._joint_limits_lower

    def _init_task_and_rewards(self) -> None:
        """Initialize attitude task buffers and reward manager.

        Reward terms are built by ``_build_reward_terms()`` (overridable hook).
        """
        self._init_attitude_buffers()
        self._reward_manager = RewardManager(
            cfg=self._build_reward_terms(),
            num_envs=self.num_envs,
            device=self.device,
        )

    def _build_reward_terms(self) -> dict[str, RewardTermCfg]:
        """Build the reward terms dict. Override in subclasses to add/modify terms.

        Base terms (5):
            1. tracking: Gaussian kernel exp(-e^2/sigma^2), dt-scaled
            2. progress: telescoping (prev_e - e), NOT dt-scaled
            3. angular_velocity: squared omega penalty, dt-scaled, curriculum
            4. action_magnitude: squared action penalty, dt-scaled
            5. action_rate: squared delta-action penalty, NOT dt-scaled, curriculum
        """
        rcfg = self.cfg.reward
        return {
            "tracking": RewardTermCfg(
                func=tracking_reward,
                weight=rcfg.tracking_weight,
                params={"sigma": rcfg.tracking_sigma},
            ),
            "progress": RewardTermCfg(
                func=progress_reward,
                weight=rcfg.progress_weight,
                scale_by_dt=False,
            ),
            "angular_velocity": RewardTermCfg(
                func=angular_velocity_penalty,
                weight=rcfg.angular_velocity_weight,
                curriculum_start_weight=rcfg.angular_velocity_weight / 10.0,
            ),
            "action_magnitude": RewardTermCfg(
                func=action_magnitude_penalty,
                weight=rcfg.action_magnitude_weight,
            ),
            "action_rate": RewardTermCfg(
                func=action_rate_penalty,
                weight=rcfg.action_rate_weight,
                scale_by_dt=False,
                curriculum_start_weight=rcfg.action_rate_weight / 10.0,
            ),
        }

    def _init_state_buffers(self) -> None:
        """Initialize action and force/torque buffers."""
        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions_obs = torch.zeros(self.num_envs, 2, device=self.device)
        self._joint_pos_targets = torch.zeros(self.num_envs, 2, device=self.device)
        # Global step counter (not per-env). With control_decimation=1 (default),
        # this modulo always passes. If control_decimation > 1, all envs share
        # the same control phase.
        self._control_step_counter = 0

        # Energy accumulator for control effort tracking
        self._cumulative_effort = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Force/torque buffers
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

    def _init_attitude_buffers(self) -> None:
        """Initialize attitude task buffers for potential-based reward."""
        # Target Euler angles (roll, pitch, yaw)
        target_tensor = torch.tensor(self.cfg.target_attitude, device=self.device)
        self._target_euler = target_tensor.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Attitude error and potential tracking buffers
        self._attitude_error = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_attitude_error(
        self,
        quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attitude error from quaternion orientation.

        Args:
            quat: Quaternion orientation (w, x, y, z). Shape: (N, 4).
            env_ids: Environment indices. If None, computes for all envs.

        Returns:
            Attitude error (target - current), wrapped to [-pi, pi]. Shape: (N, 3).
        """
        current_euler = torch.stack(euler_xyz_from_quat(quat), dim=-1)
        target = self._target_euler if env_ids is None else self._target_euler[env_ids]
        error = target - current_euler

        # Wrap angles to [-pi, pi]
        return torch.atan2(torch.sin(error), torch.cos(error))

    def _get_attitude_error(self) -> torch.Tensor:
        """Compute attitude error for observations.

        Returns the difference between target and current orientation
        as roll, pitch, yaw errors.

        Returns:
            Attitude error (target - current). Shape: (num_envs, 3).
        """
        self._attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w)
        return self._attitude_error

    def _update_potentials(self) -> None:
        """Update potential values for reward computation.

        Call once per step before reward computation. Saves current potential
        as prev_potential and computes new potential from roll/pitch errors.
        Yaw is excluded because buoyancy control cannot generate Z-axis torque.

        Also caches _attitude_error so that logging in _reset_idx (which runs
        after _get_rewards but before _get_observations) uses the current step's
        error, not the previous step's stale value.
        """
        self._prev_potentials = self._potentials.clone()
        self._attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w)
        self._potentials = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)

    def _reset_attitude_task(self, env_ids: torch.Tensor) -> None:
        """Reset target attitudes and potentials for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        num_reset = len(env_ids)
        base_attitude = torch.tensor(self.cfg.target_attitude, device=self.device)

        if self.cfg.randomize_target_attitude:
            attitude_range = torch.tensor(self.cfg.target_attitude_range, device=self.device)
            random_offset = (torch.rand(num_reset, 3, device=self.device) * 2 - 1) * attitude_range
            self._target_euler[env_ids] = base_attitude + random_offset
        else:
            self._target_euler[env_ids] = base_attitude.unsqueeze(0).expand(num_reset, -1)

        # Reset potentials (will be properly initialized by _initialize_potentials after pose reset)
        self._potentials[env_ids] = 0.0
        self._prev_potentials[env_ids] = 0.0

    def _initialize_potentials(self, env_ids: torch.Tensor) -> None:
        """Initialize potential values after robot pose reset.

        Sets both prev_potentials and potentials to the same value to prevent
        spurious progress reward on the first step.

        Args:
            env_ids: Environment indices that were reset.
        """
        attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w[env_ids], env_ids)
        initial_potential = torch.linalg.norm(attitude_error[:, :2], dim=-1)
        self._potentials[env_ids] = initial_potential
        self._prev_potentials[env_ids] = initial_potential

    def _setup_scene(self):
        """Setup simulation scene with robot and underwater lighting."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Dark underwater-style background with dim ambient lighting
        # visible_in_primary_ray=False makes the background black (no sky texture)
        light_cfg = sim_utils.DomeLightCfg(
            intensity=800.0,
            color=(0.3, 0.5, 0.7),
            visible_in_primary_ray=False,
        )
        light_cfg.func("/World/Light", light_cfg)

    def _update_action_buffers(self, actions: torch.Tensor, obs_action_slice: slice | None = None) -> None:
        """Update action history buffers. Called at the start of _pre_physics_step().

        Args:
            actions: Raw actions from RL. Shape: (num_envs, action_space).
            obs_action_slice: Slice for _prev_actions_obs. None = full clone.
        """
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if obs_action_slice is not None:
            self._prev_actions_obs = self._actions[:, obs_action_slice].clone()
        else:
            self._prev_actions_obs = self._actions.clone()
        self._control_step_counter += 1

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step with control decimation.

        Velocity commands are integrated to position targets at control frequency,
        reflecting real hardware actuator constraints.

        Args:
            actions: Joint velocity commands [-1, 1]. Shape: (num_envs, 2).
        """
        self._update_action_buffers(actions)

        if self._control_step_counter % self.cfg.control_decimation == 0:
            # Integrate velocity to position: delta_pos = dt * max_vel * action
            # step_dt = physics_dt * decimation (time per RL step)
            control_dt = self.step_dt * self.cfg.control_decimation
            position_delta = control_dt * self.cfg.max_joint_velocity * self._actions
            self._joint_pos_targets += position_delta

            self._joint_pos_targets = torch.clamp(
                self._joint_pos_targets,
                self._joint_limits_lower,
                self._joint_limits_upper,
            )

    def _apply_action(self):
        """Apply joint position targets and hydrodynamic forces."""
        # Joint position control
        self._robot.set_joint_position_target(self._joint_pos_targets, joint_ids=self._albc_joint_ids)

        # Update PhysX acceleration cache for added mass force (M_A * v_dot).
        # Uses previous step's acceleration to avoid circular dependency.
        # Stability factor must satisfy: factor * max(M_A_i / M_rigid_i) < 1
        if self._hydro._apply_added_mass:
            self._hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w,
                root_quat_w=self._robot.data.root_quat_w,
            )
        if self._buoy_hydro._apply_added_mass:
            buoy_body_idx = self._buoy_body_id[0]
            self._buoy_hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w[:, buoy_body_idx, :],
                root_quat_w=self._robot.data.body_quat_w[:, buoy_body_idx, :],
            )

        # Main body hydrodynamics (no payload -- payload is on gripper body)
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._hydro_forces.unsqueeze(1),
            torques=self._hydro_torques.unsqueeze(1),
        )

        # Buoy hydrodynamics
        buoy_idx = self._buoy_body_id[0]
        self._buoy_hydro_forces, self._buoy_hydro_torques = self._buoy_hydro.compute_forces(
            root_lin_vel_w=self._robot.data.body_lin_vel_w[:, buoy_idx, :],
            root_ang_vel_w=self._robot.data.body_ang_vel_w[:, buoy_idx, :],
            root_quat_w=self._robot.data.body_quat_w[:, buoy_idx, :],
        )
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._buoy_body_id,
            forces=self._buoy_hydro_forces.unsqueeze(1),
            torques=self._buoy_hydro_torques.unsqueeze(1),
        )

        # Gripper payload (weight force applied at attachment point + CoG offset)
        payload_forces, payload_torques = self._compute_payload_wrench()
        if payload_forces is not None and payload_torques is not None:
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                body_ids=self._gripper_body_id,
                forces=payload_forces.unsqueeze(1),
                torques=payload_torques.unsqueeze(1),
            )

    def _compute_payload_wrench(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute payload weight force and torque in the gripper body frame.

        The payload hangs from the gripper (fixed to base via base_to_gripper joint).
        Force = m*g transformed to gripper body frame.
        Torque = (attachment_offset + cog_offset) x F_body.

        Returns:
            Tuple of (forces, torques) in gripper body frame, or (None, None) if disabled.
        """
        if not self._payload_enabled:
            return None, None

        # Gripper body orientation (same as base since fixed joint, but correct API)
        gripper_idx = self._gripper_body_id[0]
        gripper_quat = self._robot.data.body_quat_w[:, gripper_idx, :]

        # Payload weight in world frame, then transform to gripper body frame
        payload_weight_w = self._payload_mass.unsqueeze(-1) * self._gravity_vec
        payload_weight_b = quat_apply_inverse(gripper_quat, payload_weight_w)

        # Effective offset = attachment point + CoG offset (both in gripper body frame)
        effective_offset = self._payload_attachment_offset + self._payload_cog_offset

        # Torque in body frame: tau = r_body x F_body
        payload_torque_b = torch.cross(effective_offset, payload_weight_b, dim=-1)

        return payload_weight_b, payload_torque_b

    def _get_observations(self) -> dict:
        """Compute ALBC-specific observations.

        Returns 13-dim policy observation and optional privileged observations.
        See mdp.observations for implementation details.

        Returns:
            Observation dictionary with "policy" key and optional "privileged" key.
        """
        observations = {"policy": compute_policy_obs(self, self._robot)}
        if self.cfg.state_space > 0:
            observations["privileged"] = compute_privileged_obs(self)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute ALBC rewards using potential-based system.

        Updates potentials before computing rewards to ensure progress reward
        is correctly calculated as the difference between previous and current
        potential values.

        Returns:
            Reward tensor. Shape: (num_envs,).
        """
        # Update potentials before reward computation
        # This must be called exactly once per step to correctly compute progress reward
        self._update_potentials()

        # Accumulate control effort: sum(||a||^2 * dt) over episode
        self._cumulative_effort += torch.sum(self._actions**2, dim=-1) * self.step_dt

        return self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            actions=self._actions,
            prev_actions=self._prev_actions,
            env=self,  # Pass env for accessing potentials
        )

    def _collect_episode_metrics(
        self,
        env_ids: torch.Tensor,
        reward_sums: dict[str, torch.Tensor],
    ) -> dict[str, float | torch.Tensor]:
        """Collect episode metrics for TensorBoard/WandB logging.

        Called from ``_reset_idx()`` before resetting state.  Replaces the former
        standalone ``log_episode_metrics()`` function so that all env-internal
        attribute access goes through ``self``.

        Args:
            env_ids: Environment indices being reset.
            reward_sums: Accumulated rewards per term from RewardManager.

        Returns:
            Dict of metric tag -> value, written to ``self.extras["log"]``.
        """
        log: dict[str, float | torch.Tensor] = {}

        # Reward sums
        for name, value in reward_sums.items():
            log[f"Episode_Reward/{name}"] = value

        # Termination counts
        log["Episode_Termination/terminated"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if len(env_ids) == 0:
            return log

        # Attitude errors (use cached values to avoid side effects)
        attitude_errors_deg = torch.rad2deg(self._attitude_error[env_ids])
        log["Attitude_Error/roll_deg"] = attitude_errors_deg[:, 0].abs().mean().item()
        log["Attitude_Error/pitch_deg"] = attitude_errors_deg[:, 1].abs().mean().item()
        log["Attitude_Error/total_deg"] = attitude_errors_deg[:, :2].abs().sum(dim=-1).mean().item()

        # Cumulative control effort
        log["Performance/cumulative_effort"] = self._cumulative_effort[env_ids].mean().item()

        # --- Per-step reward diagnostics (last step snapshot) ---
        # Raw magnitude: unweighted function output (for cross-term comparison)
        for term_name, raw_mean in self._reward_manager.step_raw_means.items():
            log[f"Reward_Magnitude/{term_name}"] = raw_mean
        # Active weight: curriculum-adjusted weight (for tracking ramp progress)
        for term_name, weight in self._reward_manager.active_weights.items():
            log[f"Reward_Weight/{term_name}"] = weight

        # --- Action diagnostics ---
        log["Action/magnitude_mean"] = torch.linalg.norm(self._actions[env_ids], dim=-1).mean().item()
        log["Action/rate_mean"] = (
            torch.linalg.norm(self._actions[env_ids] - self._prev_actions[env_ids], dim=-1).mean().item()
        )

        # --- Dynamics diagnostics ---
        ang_vel_rp = self._robot.data.root_ang_vel_b[env_ids, :2]
        log["Dynamics/angular_velocity_rms"] = ang_vel_rp.pow(2).mean().sqrt().item()

        # TDC diagnostics (for any env with TDC controller)
        if hasattr(self, "_tdc"):
            log_tdc_diagnostics(log, self._tdc)

        # DR parameters (when randomization is enabled)
        if hasattr(self.cfg, "randomization") and self.cfg.randomization.enable:
            # log_dr_metrics expects extras["log"] dict -- pass a wrapper
            extras_wrapper: dict = {"log": log}
            log_dr_metrics(extras_wrapper, self, self._robot, self._albc_joint_ids)

        return log

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Height bounds check
        height = self._robot.data.root_pos_w[:, 2]
        out_of_height_bounds = (height < self.cfg.min_height) | (height > self.cfg.max_height)

        # Horizontal distance from origin check
        xy_displacement = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        too_far = torch.linalg.norm(xy_displacement, dim=1) > self.cfg.max_distance_from_origin

        return out_of_height_bounds | too_far, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments.

        Execution order:
            1. Logging (episode metrics before reset)
            2. Component reset (robot, parent class, action buffers)
            3. Hydrodynamics reset + domain randomization
            4. Task reset
            5. Robot state reset (joints, then pose)
            6. Potential initialization
        """
        # Use all indices if None or full batch (separate branches for type narrowing)
        if env_ids is None:  # noqa: SIM114
            env_ids = self._robot._ALL_INDICES
        elif len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        env_ids_: torch.Tensor = env_ids  # type: ignore[assignment]

        # --- 1. Logging ---
        reward_sums = self._reward_manager.reset(env_ids_)
        self.extras["log"] = self._collect_episode_metrics(env_ids_, reward_sums)

        # --- 2. Component reset ---
        self._robot.reset(env_ids_)
        super()._reset_idx(env_ids_)

        # Randomize episode lengths to decorrelate environment terminations
        if len(env_ids_) == self.num_envs:
            # Full batch (initial reset): spread across entire episode range
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        else:
            # Individual resets: small jitter prevents re-synchronization
            max_jitter = max(1, int(self.max_episode_length * 0.1))
            self.episode_length_buf[env_ids_] = torch.randint_like(self.episode_length_buf[env_ids_], high=max_jitter)

        # Reset action buffers
        for buf in (self._actions, self._prev_actions, self._prev_actions_obs):
            buf[env_ids_] = 0.0
        self._cumulative_effort[env_ids_] = 0.0

        # --- 3. Hydrodynamics reset ---
        self._hydro.reset(env_ids_)
        self._buoy_hydro.reset(env_ids_)

        if self._payload_enabled:
            self._payload_mass[env_ids_] = self.cfg.payload_mass
            offset_tensor = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
            self._payload_attachment_offset[env_ids_] = offset_tensor
            self._payload_cog_offset[env_ids_] = 0.0

        rand_cfg = self.cfg.randomization
        if rand_cfg.enable:
            randomize_hydrodynamics(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)
            randomize_body_mass(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)
            if self._payload_enabled:
                randomize_payload(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)

        has_ocean_current = any(v > 0 for v in self.cfg.ocean_current.max_velocity)
        if has_ocean_current:
            randomize_ocean_current(env=self, env_ids=env_ids_)

        # --- 4. Attitude task reset ---
        self._reset_attitude_task(env_ids_)

        # --- 5. Robot state reset (joint state must precede root pose) ---
        if rand_cfg.enable:
            randomize_joint_positions(env=self, env_ids=env_ids_, joint_pos_range=self.cfg.initial_joint_pos_range)
            randomize_robot_pose(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)
        else:
            reset_joint_positions_default(env=self, env_ids=env_ids_)
            reset_robot_pose_default(env=self, env_ids=env_ids_, initial_height=self.cfg.initial_height)

        # --- 5.5. Joint actuator DR ---
        # Always applied: when DR disabled, ranges collapse to defaults (no randomization).
        # TDC envs override stiffness/damping in their own _reset_idx().
        randomize_joint_gains(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)
        randomize_joint_friction(env=self, env_ids=env_ids_, rand_cfg=rand_cfg)

        # --- 6. Potential initialization (must be after pose reset) ---
        self._initialize_potentials(env_ids_)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup or toggle visibility of debug visualization markers."""
        if debug_vis:
            self._debug_vis.setup(enable_payload=self._payload_enabled)
        self._debug_vis.set_visibility(debug_vis)

    def _debug_vis_callback(self, _event):
        """Update debug marker positions each frame."""
        self._debug_vis.update(
            robot=self._robot,
            body_id=self._body_id,
            buoy_body_id=self._buoy_body_id,
            hydro=self._hydro,
            buoy_hydro=self._buoy_hydro,
            payload_mass=self._payload_mass,
            payload_offset=self._payload_attachment_offset,
            default_payload_mass=self.cfg.payload_mass,
        )
