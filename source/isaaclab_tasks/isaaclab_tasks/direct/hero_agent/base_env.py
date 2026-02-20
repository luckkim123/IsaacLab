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
        - max_joint_velocity: pi rad/s (180 deg/s)
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
        if self.cfg.state_space >= 18 and not self.cfg.enable_payload:
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
        """Initialize payload physics buffers if enabled.

        Payload is applied to the gripper body (fixed to base via base_to_gripper joint).
        When disabled, all payload attributes are set to None.
        """
        if self.cfg.enable_payload:
            self._payload_mass = torch.full((self.num_envs,), self.cfg.payload_mass, device=self.device)
            offset = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
            self._payload_attachment_offset = offset.expand(self.num_envs, -1).clone()
            self._payload_cog_offset = torch.zeros(self.num_envs, 3, device=self.device)
            self._payload_gravity_vec = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)
        else:
            self._payload_mass = None
            self._payload_attachment_offset = None
            self._payload_cog_offset = None
            self._payload_gravity_vec = None

    @property
    def _payload_enabled(self) -> bool:
        """Whether payload physics is enabled."""
        return self._payload_mass is not None

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
        # Attitude task state (inlined from AttitudeTask)
        self._randomize_targets = self.cfg.randomize_target_attitude
        self._base_attitude = torch.tensor(self.cfg.target_attitude, device=self.device)
        self._target_range = torch.tensor(self.cfg.target_attitude_range, device=self.device)
        self._target_euler = self._base_attitude.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._attitude_error = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._reward_manager = RewardManager(
            cfg=self._build_reward_terms(),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._init_dr_curriculum()

    def _build_reward_terms(self) -> dict[str, RewardTermCfg]:
        """Build the reward terms dict. Override in subclasses to add/modify terms.

        Base terms (4):
            1. tracking: Gaussian kernel exp(-e^2/sigma^2), dt-scaled
            2. progress: potential-based shaping (prev-curr), NOT dt-scaled
            3. action_magnitude: squared action penalty, dt-scaled
            4. action_rate: squared delta-action penalty, NOT dt-scaled, curriculum
        """
        rcfg = self.cfg.reward
        tracking_params = {"sigma": rcfg.tracking_sigma}
        if rcfg.tracking_sigma_start is not None:
            tracking_params["_sigma_start"] = rcfg.tracking_sigma_start
        terms = {
            "tracking": RewardTermCfg(
                func=tracking_reward,
                weight=rcfg.tracking_weight,
                params=tracking_params,
            ),
            "action_magnitude": RewardTermCfg(
                func=action_magnitude_penalty,
                weight=rcfg.action_magnitude_weight,
            ),
            "action_rate": RewardTermCfg(
                func=action_rate_penalty,
                weight=rcfg.action_rate_weight,
                scale_by_dt=False,  # per-step delta naturally scales with frequency
                curriculum_start_weight=rcfg.action_rate_weight / 10.0,
            ),
        }
        if rcfg.progress_weight != 0.0:
            terms["progress"] = RewardTermCfg(
                func=progress_reward,
                weight=rcfg.progress_weight,
                params={"scale": rcfg.progress_scale},
                scale_by_dt=False,
            )
        if rcfg.angular_velocity_weight != 0.0:
            terms["angular_velocity"] = RewardTermCfg(
                func=angular_velocity_penalty,
                weight=rcfg.angular_velocity_weight,
            )
        return terms

    def _init_dr_curriculum(self) -> None:
        """Store original DR ranges for curriculum interpolation."""
        dr_cur = getattr(self.cfg, "dr_curriculum", None)
        if dr_cur is None or not dr_cur.enable:
            self._dr_curriculum_cfg = None
            return

        self._dr_curriculum_cfg = dr_cur
        # Snapshot the full (target) DR ranges from current config
        rand = self.cfg.randomization
        self._dr_full_ranges = {
            "perturbation_force_range": rand.perturbation_force_range,
            "perturbation_torque_range": rand.perturbation_torque_range,
            "inertia_scale": rand.inertia_scale,
            "body_mass_scale": rand.body_mass_scale,
            "volume_scale": rand.volume_scale,
            "added_mass_scale": rand.added_mass_scale,
            "payload_mass_range": rand.payload_mass_range,
            "cog_offset_z": rand.cog_offset_z,
            "cob_offset_z": rand.cob_offset_z,
            "action_latency_range": rand.action_latency_range,
        }
        # Apply start ranges immediately (iteration 0 uses easy DR)
        self.update_dr_curriculum(0)

    def update_dr_curriculum(self, iteration: int) -> None:
        """Linearly ramp DR ranges from start to full over curriculum period."""
        if self._dr_curriculum_cfg is None:
            return

        dr_cur = self._dr_curriculum_cfg
        progress = min(1.0, iteration / max(1, dr_cur.end_iter))
        rand = self.cfg.randomization

        _FIELD_MAP = {
            "perturbation_force_range": "perturbation_force_range_start",
            "perturbation_torque_range": "perturbation_torque_range_start",
            "inertia_scale": "inertia_scale_start",
            "body_mass_scale": "body_mass_scale_start",
            "volume_scale": "volume_scale_start",
            "added_mass_scale": "added_mass_scale_start",
            "payload_mass_range": "payload_mass_range_start",
            "cog_offset_z": "cog_offset_z_start",
            "cob_offset_z": "cob_offset_z_start",
            "action_latency_range": "action_latency_range_start",
        }

        for rand_field, cur_field in _FIELD_MAP.items():
            start = getattr(dr_cur, cur_field)
            full = self._dr_full_ranges[rand_field]
            lo = start[0] + (full[0] - start[0]) * progress
            hi = start[1] + (full[1] - start[1]) * progress
            # Integer fields (action_latency_range) need rounding
            if isinstance(full[0], int):
                lo, hi = int(round(lo)), int(round(hi))
            setattr(rand, rand_field, (lo, hi))

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

        # Random perturbation buffers (Tan et al. 2018)
        self._perturb_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._perturb_torques = torch.zeros(self.num_envs, 3, device=self.device)
        rand_cfg = self.cfg.randomization
        perturb_cycle = max(1, rand_cfg.perturbation_interval + rand_cfg.perturbation_duration)
        self._perturb_timer = torch.randint(0, perturb_cycle, (self.num_envs,), device=self.device)

        # Action latency buffer (ring buffer for delayed action application)
        max_latency = rand_cfg.action_latency_range[1]
        self._max_action_latency = max_latency
        if max_latency > 0:
            self._action_history = torch.zeros(
                self.num_envs, max_latency + 1, self.cfg.action_space, device=self.device
            )
            self._action_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self._action_history = None
            self._action_latency = None

    # ------------------------------------------------------------------
    # Attitude task methods
    # ------------------------------------------------------------------

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
        return torch.atan2(torch.sin(error), torch.cos(error))

    def _get_attitude_error(self) -> torch.Tensor:
        """Compute and cache attitude error for observations."""
        self._attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w)
        return self._attitude_error

    def _update_potentials(self, quat: torch.Tensor) -> None:
        """Update potential values for reward computation.

        Saves current potential as prev_potential and computes new potential
        from roll/pitch errors. Yaw is excluded because buoyancy control
        cannot generate Z-axis torque.

        Args:
            quat: Current root quaternion. Shape: (num_envs, 4).
        """
        self._prev_potentials = self._potentials.clone()
        self._attitude_error = self.compute_attitude_error(quat)
        self._potentials = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)

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

    def _get_delayed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply action latency by returning delayed actions from the history buffer.

        The history buffer stores recent actions in order [newest, ..., oldest].
        Each env has a per-env latency (sampled at reset). Latency=0 returns
        the current action (no delay).

        Args:
            actions: Current raw actions. Shape: (num_envs, action_space).

        Returns:
            Delayed actions. Same shape as input.
        """
        if self._action_history is None or not self.cfg.randomization.enable:
            return actions

        # Shift history: move existing entries one slot older
        if self._action_history.shape[1] > 1:
            self._action_history[:, 1:] = self._action_history[:, :-1].clone()
        # Insert newest action at index 0
        self._action_history[:, 0] = actions

        # Read delayed actions using per-env latency as index
        env_idx = torch.arange(self.num_envs, device=self.device)
        return self._action_history[env_idx, self._action_latency]

    def _update_perturbation(self) -> None:
        """Update per-step random perturbation forces on the base body.

        Uses a per-env timer that cycles through [0, interval+duration).
        Phase [0, duration): perturbation active. Phase [duration, cycle): cooldown.
        New random wrench is generated at the start of each active phase.

        Forces are stored in ``_perturb_forces`` / ``_perturb_torques`` and
        added to hydro forces in ``_apply_action()``.
        """
        rand_cfg = self.cfg.randomization
        if not rand_cfg.enable or not rand_cfg.enable_perturbation:
            return

        interval = rand_cfg.perturbation_interval
        duration = rand_cfg.perturbation_duration
        cycle = interval + duration

        # Advance per-env timer (wraps around)
        self._perturb_timer = (self._perturb_timer + 1) % cycle

        # Generate new perturbation at the start of active phase (timer == 0)
        trigger = self._perturb_timer == 0
        if trigger.any():
            n = trigger.sum().item()
            # Random force: uniform magnitude, random direction (unit sphere)
            f_dir = torch.randn(n, 3, device=self.device)
            f_dir = f_dir / f_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            f_lo, f_hi = rand_cfg.perturbation_force_range
            f_mag = torch.rand(n, device=self.device) * (f_hi - f_lo) + f_lo
            self._perturb_forces[trigger] = f_dir * f_mag.unsqueeze(1)

            # Random torque: uniform magnitude, random direction
            t_dir = torch.randn(n, 3, device=self.device)
            t_dir = t_dir / t_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            t_lo, t_hi = rand_cfg.perturbation_torque_range
            t_mag = torch.rand(n, device=self.device) * (t_hi - t_lo) + t_lo
            self._perturb_torques[trigger] = t_dir * t_mag.unsqueeze(1)

        # Clear forces when active phase ends (timer == duration)
        deactivate = self._perturb_timer == duration
        if deactivate.any():
            self._perturb_forces[deactivate] = 0.0
            self._perturb_torques[deactivate] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step with control decimation.

        Velocity commands are integrated to position targets at control frequency,
        reflecting real hardware actuator constraints. Action latency is applied
        before control integration.

        Args:
            actions: Joint velocity commands [-1, 1]. Shape: (num_envs, 2).
        """
        self._update_action_buffers(actions)

        if self._control_step_counter % self.cfg.control_decimation == 0:
            # Apply action latency (delayed actions for control, raw actions kept for obs)
            effective_actions = self._get_delayed_actions(self._actions)

            # Integrate velocity to position: delta_pos = dt * max_vel * action
            # step_dt = physics_dt * decimation (time per RL step)
            control_dt = self.step_dt * self.cfg.control_decimation
            position_delta = control_dt * self.cfg.max_joint_velocity * effective_actions
            self._joint_pos_targets += position_delta

            self._joint_pos_targets = torch.clamp(
                self._joint_pos_targets,
                self._joint_limits_lower,
                self._joint_limits_upper,
            )

    def _apply_action(self):
        """Apply joint position targets, hydrodynamic forces, and random perturbation."""
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

        # Update random perturbation state (per-step event, independent of control freq)
        self._update_perturbation()

        # Main body hydrodynamics + random perturbation
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )
        total_forces = self._hydro_forces + self._perturb_forces
        total_torques = self._hydro_torques + self._perturb_torques
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces.unsqueeze(1),
            torques=total_torques.unsqueeze(1),
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

        Returns:
            Tuple of (forces, torques) in gripper body frame, or (None, None) if disabled.
        """
        if self._payload_mass is None:
            return None, None

        gripper_idx = self._gripper_body_id[0]
        gripper_quat = self._robot.data.body_quat_w[:, gripper_idx, :]
        payload_weight_w = self._payload_mass.unsqueeze(-1) * self._payload_gravity_vec
        payload_weight_b = quat_apply_inverse(gripper_quat, payload_weight_w)
        effective_offset = self._payload_attachment_offset + self._payload_cog_offset
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
        """Compute ALBC rewards: Gaussian tracking + small action regularizers.

        Returns:
            Reward tensor. Shape: (num_envs,).
        """
        # Update error potentials before reward computation
        self._update_potentials(self._robot.data.root_quat_w)

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

        # Reward sums (normalized by max episode duration for episode-length-independent metrics)
        for name, value in reward_sums.items():
            log[f"Episode_Reward/{name}"] = value / self.max_episode_length_s

        # Termination counts
        log["Episode_Termination/terminated"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if len(env_ids) == 0:
            return log

        # Attitude errors (use cached values to avoid side effects)
        attitude_errors_deg = torch.rad2deg(self._attitude_error[env_ids])
        log["Attitude_Error/roll_deg"] = attitude_errors_deg[:, 0].abs().mean().item()
        log["Attitude_Error/pitch_deg"] = attitude_errors_deg[:, 1].abs().mean().item()

        # --- Action diagnostics ---
        log["Action/magnitude_mean"] = torch.linalg.norm(self._actions[env_ids], dim=-1).mean().item()

        # --- Dynamics diagnostics ---
        ang_vel_rp = self._robot.data.root_ang_vel_b[env_ids, :2]
        log["Dynamics/angular_velocity_rms"] = ang_vel_rp.pow(2).mean().sqrt().item()

        # TDC diagnostics (for any env with TDC controller)
        if hasattr(self, "_tdc"):
            log_tdc_diagnostics(log, self._tdc, env=self)

        # DR parameters (when randomization is enabled)
        if hasattr(self.cfg, "randomization") and self.cfg.randomization.enable:
            # log_dr_metrics expects extras["log"] dict -- pass a wrapper
            extras_wrapper: dict = {"log": log}
            log_dr_metrics(extras_wrapper, self, self._robot, self._albc_joint_ids)

        return log

    def get_eval_snapshot(self) -> dict[str, float]:
        """Return current evaluation metrics for play-mode diagnostics.

        Provides instantaneous per-env averages of key quantities, useful for
        printing periodic summaries during play without needing episode resets.

        Returns:
            Dict with keys: attitude_error_deg, action_magnitude, action_rate,
            angular_velocity_rms, joint_pos (mean absolute).
        """
        err = self._attitude_error[:, :2]
        return {
            "attitude_error_deg": torch.rad2deg(torch.linalg.norm(err, dim=-1)).mean().item(),
            "action_magnitude": torch.linalg.norm(self._actions, dim=-1).mean().item(),
            "action_rate": torch.linalg.norm(self._actions - self._prev_actions, dim=-1).mean().item(),
            "angular_velocity_rms": self._robot.data.root_ang_vel_b[:, :2].pow(2).mean().sqrt().item(),
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions.

        Termination triggers:
            1. Height out of bounds (z < min_height or z > max_height)
            2. Horizontal distance from origin exceeds max_distance_from_origin
            3. Angular velocity exceeds max_angular_velocity (simulation instability)
            4. NaN detected in root state (PhysX failure)
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Height bounds check
        height = self._robot.data.root_pos_w[:, 2]
        out_of_height_bounds = (height < self.cfg.min_height) | (height > self.cfg.max_height)

        # Horizontal distance from origin check
        xy_displacement = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        too_far = torch.linalg.norm(xy_displacement, dim=1) > self.cfg.max_distance_from_origin

        # Angular velocity check (roll/pitch rate)
        ang_vel_rp = self._robot.data.root_ang_vel_b[:, :2]
        too_fast = ang_vel_rp.abs().max(dim=1).values > self.cfg.max_angular_velocity

        # NaN/Inf check on root state (position + quaternion)
        bad_state = (
            torch.isnan(self._robot.data.root_pos_w).any(dim=1)
            | torch.isnan(self._robot.data.root_quat_w).any(dim=1)
            | torch.isinf(self._robot.data.root_lin_vel_w).any(dim=1)
        )

        # Attitude angle check: terminate if roll or pitch exceeds limit.
        # Beyond ~90 deg, cos(angle) flips sign -> Lambda matrix reversal -> unrecoverable.
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        excessive_tilt = (roll.abs() > self.cfg.max_attitude_angle) | (pitch.abs() > self.cfg.max_attitude_angle)

        return out_of_height_bounds | too_far | too_fast | bad_state | excessive_tilt, time_out

    def _coerce_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Normalize env_ids to a concrete tensor.

        Returns ALL_INDICES for None or full-batch inputs.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            return self._robot._ALL_INDICES
        return env_ids

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments.

        Phases:
            1. Logging and reward reset
            2. Framework reset (robot, parent class, episode jitter, action buffers)
            3. Physics reset (hydrodynamics, payload, domain randomization, ocean current)
            4. Task and state reset (attitude targets, robot pose, joint DR, potentials)
        """
        env_ids_ = self._coerce_env_ids(env_ids)
        self._log_and_reset_rewards(env_ids_)
        self._reset_framework(env_ids_)
        self._reset_physics(env_ids_)
        self._reset_task_and_state(env_ids_)

    def _log_and_reset_rewards(self, env_ids: torch.Tensor) -> None:
        """Collect episode metrics and reset reward accumulators."""
        reward_sums = self._reward_manager.reset(env_ids)
        self.extras["log"] = self._collect_episode_metrics(env_ids, reward_sums)

    def _reset_framework(self, env_ids: torch.Tensor) -> None:
        """Reset robot, parent class, jitter episode lengths, and zero action buffers."""
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Randomize episode lengths to decorrelate environment terminations
        if len(env_ids) == self.num_envs:
            # Full batch (initial reset): spread across 0~50% of episode range.
            # This decorrelates terminations while ensuring every env collects
            # at least half an episode of meaningful experience.
            half_ep = max(1, int(self.max_episode_length * 0.5))
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=half_ep)
        else:
            # Individual resets: small jitter prevents re-synchronization
            max_jitter = max(1, int(self.max_episode_length * 0.1))
            self.episode_length_buf[env_ids] = torch.randint_like(self.episode_length_buf[env_ids], high=max_jitter)

        for buf in (self._actions, self._prev_actions, self._prev_actions_obs):
            buf[env_ids] = 0.0
        self._cumulative_effort[env_ids] = 0.0

        # Reset perturbation state: randomize timer phase to decorrelate envs
        rand_cfg = self.cfg.randomization
        perturb_cycle = max(1, rand_cfg.perturbation_interval + rand_cfg.perturbation_duration)
        self._perturb_forces[env_ids] = 0.0
        self._perturb_torques[env_ids] = 0.0
        self._perturb_timer[env_ids] = torch.randint(0, perturb_cycle, (len(env_ids),), device=self.device)

        # Reset action latency: sample new per-env latency and clear history
        if self._action_history is not None and self._action_latency is not None:
            lo, hi = rand_cfg.action_latency_range
            self._action_history[env_ids] = 0.0
            self._action_latency[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    def _reset_physics(self, env_ids: torch.Tensor) -> None:
        """Reset hydrodynamics, payload, and apply domain randomization."""
        self._hydro.reset(env_ids)
        self._buoy_hydro.reset(env_ids)

        if self._payload_mass is not None:
            self._payload_mass[env_ids] = self.cfg.payload_mass
            offset = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
            self._payload_attachment_offset[env_ids] = offset
            self._payload_cog_offset[env_ids] = 0.0

        rand_cfg = self.cfg.randomization
        if rand_cfg.enable:
            randomize_hydrodynamics(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
            randomize_body_mass(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
            if self._payload_enabled:
                randomize_payload(env=self, env_ids=env_ids, rand_cfg=rand_cfg)

        has_ocean_current = any(v > 0 for v in self.cfg.ocean_current.max_velocity)
        if has_ocean_current:
            randomize_ocean_current(env=self, env_ids=env_ids)

    def _reset_task_and_state(self, env_ids: torch.Tensor) -> None:
        """Reset attitude targets, robot pose, joint DR, and initialize potentials."""
        # Reset attitude targets
        num_reset = len(env_ids)
        if self._randomize_targets:
            random_offset = (torch.rand(num_reset, 3, device=self.device) * 2 - 1) * self._target_range
            self._target_euler[env_ids] = self._base_attitude + random_offset
        else:
            self._target_euler[env_ids] = self._base_attitude.unsqueeze(0).expand(num_reset, -1)
        self._potentials[env_ids] = 0.0
        self._prev_potentials[env_ids] = 0.0

        rand_cfg = self.cfg.randomization
        if rand_cfg.enable:
            randomize_joint_positions(env=self, env_ids=env_ids, joint_pos_range=self.cfg.initial_joint_pos_range)
            randomize_robot_pose(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
        else:
            reset_joint_positions_default(env=self, env_ids=env_ids)
            reset_robot_pose_default(env=self, env_ids=env_ids, initial_height=self.cfg.initial_height)

        # Joint actuator DR: always applied (when DR disabled, ranges collapse to defaults).
        # TDC envs override stiffness/damping in their own _reset_idx().
        randomize_joint_gains(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
        randomize_joint_friction(env=self, env_ids=env_ids, rand_cfg=rand_cfg)

        # Potential initialization (must be after pose reset).
        # write_root_pose_to_sim() immediately updates internal data cache,
        # so root_quat_w reflects the new pose without needing an explicit update() call.
        attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w[env_ids], env_ids)
        initial_potential = torch.linalg.norm(attitude_error[:, :2], dim=-1)
        self._potentials[env_ids] = initial_potential
        self._prev_potentials[env_ids] = initial_potential

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
            gripper_body_id=self._gripper_body_id,
            payload_mass=self._payload_mass,
            payload_offset=self._payload_attachment_offset,
            payload_cog_offset=self._payload_cog_offset,
            default_payload_mass=self.cfg.payload_mass,
        )
