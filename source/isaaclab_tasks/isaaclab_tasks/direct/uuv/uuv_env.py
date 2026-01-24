# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UUV (Underwater Vehicle) environment for hover and position tracking tasks.

This environment implements hydrodynamic forces based on the Fossen model,
including added mass, damping, Coriolis, and buoyancy effects.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow

# Pre-defined marker configs
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from .hydrodynamics_model import HydrodynamicsModel
from .uuv_env_cfg import UUVEnvCfg


class UUVEnvWindow(BaseEnvWindow):
    """Window manager for the UUV environment."""

    def __init__(self, env: UUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window.
        """
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


class UUVEnv(DirectRLEnv):
    """UUV environment for underwater vehicle control.

    This environment simulates an underwater vehicle with full 6-DOF hydrodynamics
    based on the Fossen model. The vehicle must learn to hover at a target position
    while counteracting hydrodynamic forces and optional ocean currents.

    Observation Space:
        - Root position in world frame (3)
        - Root orientation quaternion (4)
        - Root linear velocity in body frame (3)
        - Root angular velocity in body frame (3)
        - Goal position in body frame (3)
        - Up vector projection (2)
        Total: 18 dimensions

    Action Space:
        - Thruster commands normalized to [-1, 1] (num_thrusters)
        Default: 6 thrusters for BlueROV

    Reward:
        - Position tracking reward (exponential)
        - Orientation reward (upright bonus)
        - Velocity penalties
        - Action smoothness penalties
        - Alive bonus
    """

    cfg: UUVEnvCfg

    def __init__(self, cfg: UUVEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the UUV environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Get body ID for force application (configurable body link name)
        self._body_id = self._robot.find_bodies(self.cfg.body_link_name)[0]

        # Get robot mass for normalization and hydrodynamics
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize hydrodynamics model with robot mass from physics engine
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            current_cfg=self.cfg.ocean_current,
            dt=self.physics_dt,
            robot_mass=self._robot_mass.item(),
        )

        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Thruster state (for first-order dynamics)
        self._thruster_state = torch.zeros(self.num_envs, self.cfg.thrusters.num_thrusters, device=self.device)

        # Force/torque buffers
        self._thrust_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

        # Goal position
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_reward",
                "orientation_reward",
                "lin_vel_penalty",
                "ang_vel_penalty",
                "action_rate_penalty",
                "action_magnitude_penalty",
                "alive_reward",
            ]
        }

        # Build thruster allocation matrix from config
        self._allocation_matrix = torch.tensor(
            self.cfg.thrusters.allocation_matrix, dtype=torch.float32, device=self.device
        )  # (6, num_thrusters)

        # Domain randomization buffers (per-environment)
        if self.cfg.randomization.enable:
            self._randomized_thrust_coeff = torch.full(
                (self.num_envs,), self.cfg.thrusters.thrust_coefficient, device=self.device
            )
            self._randomized_time_constant_up = torch.full(
                (self.num_envs,), self.cfg.thrusters.time_constant_up, device=self.device
            )
            self._randomized_time_constant_down = torch.full(
                (self.num_envs,), self.cfg.thrusters.time_constant_down, device=self.device
            )
        else:
            self._randomized_thrust_coeff = None
            self._randomized_time_constant_up = None
            self._randomized_time_constant_down = None

        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """Setup the simulation scene with robot and terrain."""
        # Create robot articulation
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add underwater lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.4, 0.6, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.

        Applies first-order thruster dynamics and clamps actions.

        Args:
            actions: Raw actions from policy. Shape: (num_envs, num_thrusters).
        """
        # Store previous actions for smoothness penalty
        self._prev_actions = self._actions.clone()

        # Clamp and store current actions
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Apply first-order thruster dynamics
        target_state = self._actions

        # Use randomized or default time constants
        if self._randomized_time_constant_up is not None:
            tau_up = self._randomized_time_constant_up.unsqueeze(-1)
            tau_down = self._randomized_time_constant_down.unsqueeze(-1)
        else:
            tau_up = self.cfg.thrusters.time_constant_up
            tau_down = self.cfg.thrusters.time_constant_down

        # Select time constant based on direction
        tau = torch.where(
            target_state > self._thruster_state,
            tau_up,
            tau_down,
        )

        # Update thruster state with first-order dynamics
        alpha = 1.0 - torch.exp(-self.step_dt / tau)
        self._thruster_state = self._thruster_state + alpha * (target_state - self._thruster_state)

    def _apply_action(self):
        """Apply thruster forces and hydrodynamic forces to the robot.

        Uses the allocation matrix from config to convert thruster commands
        to body wrench: wrench = allocation_matrix @ thrusts
        """
        # Compute individual thruster forces from thruster state
        if self._randomized_thrust_coeff is not None:
            thrust_magnitude = self._thruster_state * self._randomized_thrust_coeff.unsqueeze(-1)
        else:
            thrust_magnitude = self._thruster_state * self.cfg.thrusters.thrust_coefficient

        # Clamp to max thrust
        thrust_magnitude = thrust_magnitude.clamp(-self.cfg.thrusters.max_thrust, self.cfg.thrusters.max_thrust)

        # Apply allocation matrix to convert thruster forces to body wrench
        # wrench = allocation_matrix @ thrusts
        # allocation_matrix: (6, num_thrusters), thrust_magnitude: (num_envs, num_thrusters)
        # Result: (num_envs, 6) = [Fx, Fy, Fz, Mx, My, Mz]
        body_wrench = torch.matmul(thrust_magnitude, self._allocation_matrix.T)

        # Extract forces and torques from wrench
        self._thrust_forces[:, 0, :] = body_wrench[:, :3]
        self._thrust_torques[:, 0, :] = body_wrench[:, 3:]

        # Compute hydrodynamic forces
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )

        # Combine thrust and hydrodynamic forces
        total_forces = self._thrust_forces.clone()
        total_forces[:, 0, :] += self._hydro_forces

        total_torques = self._thrust_torques.clone()
        total_torques[:, 0, :] += self._hydro_torques

        # Apply forces via wrench composer
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=total_torques,
        )

    def _get_observations(self) -> dict:
        """Compute observations for the policy.

        Returns:
            Dictionary containing observation tensor.
        """
        # Goal position in body frame
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal_pos_w,
        )

        # Up vector in body frame (for orientation reward)
        up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        up_b = quat_apply_inverse(self._robot.data.root_quat_w, up_w)

        # Compile observation
        obs = torch.cat(
            [
                self._robot.data.root_pos_w,  # 3
                self._robot.data.root_quat_w,  # 4
                self._robot.data.root_lin_vel_b,  # 3
                self._robot.data.root_ang_vel_b,  # 3
                goal_pos_b,  # 3
                up_b[:, :2],  # 2 (xy components of up vector)
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the current step.

        Returns:
            Total reward tensor. Shape: (num_envs,).
        """
        # Position tracking reward (exponential)
        distance_to_goal = torch.linalg.norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=1)
        position_reward = (
            torch.exp(-distance_to_goal / self.cfg.position_reward_exp_scale)
            * self.cfg.position_reward_scale
            * self.step_dt
        )

        # Orientation reward (upright bonus)
        # Compute how aligned the robot's Z-axis is with world Z
        up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        up_b = quat_apply_inverse(self._robot.data.root_quat_w, up_w)
        orientation_reward = (
            (up_b[:, 2] + 1.0)
            / 2.0  # Map [-1, 1] to [0, 1]
            * self.cfg.orientation_reward_scale
            * self.step_dt
        )

        # Velocity penalties
        lin_vel_penalty = (
            torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
            * self.cfg.linear_velocity_penalty_scale
            * self.step_dt
        )
        ang_vel_penalty = (
            torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
            * self.cfg.angular_velocity_penalty_scale
            * self.step_dt
        )

        # Action smoothness penalty (rate of change)
        action_rate = self._actions - self._prev_actions
        action_rate_penalty = (
            torch.sum(torch.square(action_rate), dim=1) * self.cfg.action_rate_penalty_scale * self.step_dt
        )

        # Action magnitude penalty (penalize large actions)
        action_magnitude_penalty = (
            torch.sum(torch.square(self._actions), dim=1) * self.cfg.action_magnitude_penalty_scale * self.step_dt
        )

        # Alive bonus
        alive_reward = self.cfg.alive_reward_scale * self.step_dt

        # Total reward
        reward = (
            position_reward
            + orientation_reward
            + lin_vel_penalty
            + ang_vel_penalty
            + action_rate_penalty
            + action_magnitude_penalty
            + alive_reward
        )

        # Log episode sums
        self._episode_sums["position_reward"] += position_reward
        self._episode_sums["orientation_reward"] += orientation_reward
        self._episode_sums["lin_vel_penalty"] += lin_vel_penalty
        self._episode_sums["ang_vel_penalty"] += ang_vel_penalty
        self._episode_sums["action_rate_penalty"] += action_rate_penalty
        self._episode_sums["action_magnitude_penalty"] += action_magnitude_penalty
        self._episode_sums["alive_reward"] += alive_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions.

        Returns:
            Tuple of (terminated, time_out) tensors.
        """
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Termination conditions
        height = self._robot.data.root_pos_w[:, 2]
        too_low = height < self.cfg.min_height
        too_high = height > self.cfg.max_height

        distance_from_origin = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        too_far = distance_from_origin > self.cfg.max_distance_from_origin

        terminated = too_low | too_high | too_far

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log episode statistics
        final_distance = torch.linalg.norm(
            self._goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._thruster_state[env_ids] = 0.0

        # Reset hydrodynamics
        self._hydro.reset(env_ids)

        # Apply domain randomization if enabled
        if self.cfg.randomization.enable:
            # Randomize hydrodynamic parameters including vehicle mass
            self._hydro.randomize_parameters(
                env_ids=env_ids,
                added_mass_scale=self.cfg.randomization.added_mass_scale,
                linear_damping_scale=self.cfg.randomization.linear_damping_scale,
                quadratic_damping_scale=self.cfg.randomization.quadratic_damping_scale,
                volume_scale=self.cfg.randomization.volume_scale,
                mass_scale=self.cfg.randomization.mass_scale,
            )

            # Randomize thrust coefficient
            tc_scale = self.cfg.randomization.thrust_coefficient_scale
            thrust_scales = torch.rand(len(env_ids), device=self.device) * (tc_scale[1] - tc_scale[0]) + tc_scale[0]
            self._randomized_thrust_coeff[env_ids] = self.cfg.thrusters.thrust_coefficient * thrust_scales

            # Randomize thruster time constants
            time_const_scales = (
                torch.rand(len(env_ids), device=self.device)
                * (self.cfg.randomization.time_constant_scale[1] - self.cfg.randomization.time_constant_scale[0])
                + self.cfg.randomization.time_constant_scale[0]
            )
            self._randomized_time_constant_up[env_ids] = self.cfg.thrusters.time_constant_up * time_const_scales
            self._randomized_time_constant_down[env_ids] = self.cfg.thrusters.time_constant_down * time_const_scales

            # Randomize robot mass (applied via scaling gravity compensation)
            # Note: Actual mass change requires physics engine support
            # This affects buoyancy calculation through volume scaling already applied above

        # Randomize ocean current if enabled
        if any(v > 0 for v in self.cfg.ocean_current.max_velocity):
            self._hydro.randomize_current(env_ids)

        # Sample new goal position
        goal_range = torch.tensor(self.cfg.goal_pos_range, device=self.device)
        self._goal_pos_w[env_ids, :2] = (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * goal_range[:2]
        self._goal_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._goal_pos_w[env_ids, 2] = (
            torch.rand(len(env_ids), device=self.device) * goal_range[2] + self.cfg.initial_height
        )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Apply domain randomization to initial pose if enabled
        if self.cfg.randomization.enable:
            num_reset = len(env_ids)

            # Randomize initial position
            pos_x = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.position_x_range[1] - self.cfg.randomization.position_x_range[0])
                + self.cfg.randomization.position_x_range[0]
            )
            pos_y = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.position_y_range[1] - self.cfg.randomization.position_y_range[0])
                + self.cfg.randomization.position_y_range[0]
            )
            pos_z = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.position_z_range[1] - self.cfg.randomization.position_z_range[0])
                + self.cfg.randomization.position_z_range[0]
            )
            default_root_state[:, 0] += pos_x
            default_root_state[:, 1] += pos_y
            default_root_state[:, 2] = pos_z

            # Randomize initial orientation
            roll = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.roll_range[1] - self.cfg.randomization.roll_range[0])
                + self.cfg.randomization.roll_range[0]
            )
            pitch = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.pitch_range[1] - self.cfg.randomization.pitch_range[0])
                + self.cfg.randomization.pitch_range[0]
            )
            yaw = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.randomization.yaw_range[1] - self.cfg.randomization.yaw_range[0])
                + self.cfg.randomization.yaw_range[0]
            )

            # Convert euler angles to quaternion and apply
            random_quat = quat_from_euler_xyz(roll, pitch, yaw)
            default_root_state[:, 3:7] = quat_mul(default_root_state[:, 3:7], random_quat)
        else:
            default_root_state[:, 2] = self.cfg.initial_height

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            # Goal position marker (red cube)
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            # Ocean current arrow marker (bright yellow for visibility)
            if not hasattr(self, "current_visualizer"):
                arrow_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/ocean_current",
                    markers={
                        "arrow": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(1.0, 0.5, 0.5),  # Thicker arrow
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 1.0, 0.0),  # Bright yellow
                                emissive_color=(0.3, 0.3, 0.0),  # Slight glow
                            ),
                        )
                    },
                )
                self.current_visualizer = VisualizationMarkers(arrow_cfg)
            self.current_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "current_visualizer"):
                self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        # Update goal position markers
        self.goal_pos_visualizer.visualize(self._goal_pos_w)

        # Update ocean current arrows
        self._visualize_ocean_current()

    def _visualize_ocean_current(self):
        """Visualize ocean current as arrows above robot positions.

        The arrow direction indicates current flow direction, and length
        indicates magnitude (scaled: 1 m/s = 1 meter arrow length).
        """
        # Get current velocity from hydrodynamics model (world frame)
        current_vel = self._hydro._current_velocity[:, :3]  # Linear velocity only (x, y, z)
        current_magnitude = torch.linalg.norm(current_vel, dim=1, keepdim=True)

        # Skip if no current - hide arrows with zero scale
        if current_magnitude.max() < 1e-6:
            scales = torch.full((self.num_envs, 3), 0.001, device=self.device)
            self.current_visualizer.visualize(
                translations=self._robot.data.root_pos_w,
                scales=scales,
            )
            return

        # Normalize current direction
        current_dir = current_vel / (current_magnitude + 1e-6)

        # Compute quaternion to rotate X-axis to current direction
        orientations = self._direction_to_quat(current_dir)

        # Scale arrow: length proportional to magnitude, thick for visibility
        # Arrow length = magnitude * 5 (so 0.3 m/s current = 1.5m arrow)
        # Minimum length 1.0m for visibility, max 4.0m
        arrow_length = torch.clamp(current_magnitude * 5.0, 1.0, 4.0)
        arrow_thickness = torch.full_like(arrow_length, 0.4)  # Very thick for visibility
        scales = torch.cat([arrow_length, arrow_thickness, arrow_thickness], dim=1)

        # Position arrows above robot (1.0m offset in Z for clear visibility)
        arrow_positions = self._robot.data.root_pos_w.clone()
        arrow_positions[:, 2] += 1.0

        self.current_visualizer.visualize(
            translations=arrow_positions,
            orientations=orientations,
            scales=scales,
        )

    def _direction_to_quat(self, direction: torch.Tensor) -> torch.Tensor:
        """Convert direction vectors to quaternions that rotate X-axis to the direction.

        Args:
            direction: Normalized direction vectors. Shape: (num_envs, 3).

        Returns:
            Quaternions (w, x, y, z). Shape: (num_envs, 4).
        """
        # Reference direction (X-axis)
        x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Compute rotation axis (cross product)
        axis = torch.cross(x_axis, direction, dim=-1)
        axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)

        # Compute rotation angle (dot product)
        cos_angle = torch.sum(x_axis * direction, dim=1, keepdim=True)

        # Handle parallel and anti-parallel cases
        parallel_mask = axis_norm.squeeze(-1) < 1e-6

        # For non-parallel cases: use axis-angle to quaternion
        axis_normalized = axis / (axis_norm + 1e-6)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        half_angle = angle / 2.0

        w = torch.cos(half_angle)
        xyz = axis_normalized * torch.sin(half_angle)

        quat = torch.cat([w, xyz], dim=-1)

        # For parallel case (same direction): identity quaternion
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # For anti-parallel case (opposite direction): 180 degree rotation around Y or Z
        anti_parallel_mask = parallel_mask & (cos_angle.squeeze(-1) < 0)
        rotation_180_y = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Apply masks
        result = torch.where(parallel_mask.unsqueeze(-1), identity_quat, quat)
        result = torch.where(anti_parallel_mask.unsqueeze(-1), rotation_180_y, result)

        return result

    def get_ocean_current(self) -> torch.Tensor:
        """Get current ocean current velocity for all environments.

        Returns:
            Ocean current velocity in world frame. Shape: (num_envs, 6).
            Format: [vx, vy, vz, wx, wy, wz] where v is linear and w is angular.
        """
        return self._hydro._current_velocity.clone()

    def get_ocean_current_info(self) -> dict:
        """Get detailed ocean current information for logging/debugging.

        Returns:
            Dictionary with current statistics:
            - linear_velocity: Mean linear current velocity (x, y, z)
            - magnitude: Mean current magnitude
            - max_magnitude: Maximum current magnitude across environments
        """
        current = self._hydro._current_velocity[:, :3]
        magnitude = torch.linalg.norm(current, dim=1)
        return {
            "linear_velocity": current.mean(dim=0).cpu().tolist(),
            "magnitude": magnitude.mean().item(),
            "max_magnitude": magnitude.max().item(),
        }
