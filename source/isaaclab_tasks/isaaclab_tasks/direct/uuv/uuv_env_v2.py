# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UUV Environment V2 with modular task and reward systems.

This environment builds on the original UUVEnv with improved modularity:
- Pluggable task system (HoverTask, TrackTask, etc.)
- Composition-based reward terms via RewardManager
- Separated ThrusterModel for cleaner abstraction

While maintaining backward compatibility with existing configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .hydrodynamics_model import HydrodynamicsModel
from .rewards import (
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    alive_bonus,
    angular_velocity_penalty,
    linear_velocity_penalty,
    orientation_upright_exp,
    position_tracking_exp,
)
from .tasks import HoverTask, HoverTaskCfg, TaskBase
from .thrusters import ThrusterModel, ThrusterModelCfg
from .uuv_env_cfg import UUVEnvCfg

if TYPE_CHECKING:
    pass


class UUVEnvV2Window(BaseEnvWindow):
    """Window manager for the UUV V2 environment."""

    def __init__(self, env: UUVEnvV2, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


class UUVEnvV2(DirectRLEnv):
    """UUV environment V2 with modular task and reward systems.

    This environment provides improved modularity over UUVEnv:
    - Task abstraction for easy addition of new tasks (hover, track, dock)
    - Composition-based reward terms via RewardManager
    - Separated ThrusterModel for cleaner code structure

    Observation Space (configurable via task):
        Base observations:
        - Root position in world frame (3)
        - Root orientation quaternion (4)
        - Root linear velocity in body frame (3)
        - Root angular velocity in body frame (3)
        - Up vector projection (2)
        Task observations:
        - Goal position in body frame (3) [for HoverTask]
        Total: 18 dimensions (default)

    Action Space:
        - Thruster commands normalized to [-1, 1] (num_thrusters)
        Default: 6 thrusters for BlueROV
    """

    cfg: UUVEnvCfg

    def __init__(self, cfg: UUVEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the UUV V2 environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Get body ID for force application
        self._body_id = self._robot.find_bodies(self.cfg.body_link_name)[0]

        # Get robot mass for hydrodynamics
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # Initialize hydrodynamics model
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            current_cfg=self.cfg.ocean_current,
            dt=self.physics_dt,
            robot_mass=self._robot_mass.item(),
        )

        # Initialize thruster model
        self._thruster = ThrusterModel(
            cfg=ThrusterModelCfg(
                num_thrusters=self.cfg.thrusters.num_thrusters,
                max_thrust=self.cfg.thrusters.max_thrust,
                time_constant_up=self.cfg.thrusters.time_constant_up,
                time_constant_down=self.cfg.thrusters.time_constant_down,
                thrust_coefficient=self.cfg.thrusters.thrust_coefficient,
                allocation_matrix=self.cfg.thrusters.allocation_matrix,
            ),
            num_envs=self.num_envs,
            device=self.device,
            enable_randomization=self.cfg.randomization.enable,
        )

        # Initialize task (default: HoverTask)
        self._task: TaskBase = HoverTask(
            cfg=HoverTaskCfg(
                goal_pos_range=self.cfg.goal_pos_range,
                initial_height=self.cfg.initial_height,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

        # Initialize reward manager with configurable terms
        self._reward_manager = RewardManager(
            cfg=self._build_reward_cfg(),
            num_envs=self.num_envs,
            device=self.device,
        )

        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Force/torque buffers for visualization
        self._thrust_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _build_reward_cfg(self) -> dict[str, RewardTermCfg]:
        """Build reward configuration from environment config.

        Returns:
            Dictionary mapping term names to RewardTermCfg.
        """
        return {
            "position_tracking": RewardTermCfg(
                func=position_tracking_exp,
                weight=self.cfg.position_reward_scale,
                params={"exp_scale": self.cfg.position_reward_exp_scale},
            ),
            "orientation_upright": RewardTermCfg(
                func=orientation_upright_exp,
                weight=self.cfg.orientation_reward_scale,
                params={"exp_scale": getattr(self.cfg, "orientation_exp_scale", 0.5)},
            ),
            "linear_velocity": RewardTermCfg(
                func=linear_velocity_penalty,
                weight=self.cfg.linear_velocity_penalty_scale,  # Already negative in config
            ),
            "angular_velocity": RewardTermCfg(
                func=angular_velocity_penalty,
                weight=self.cfg.angular_velocity_penalty_scale,  # Already negative in config
            ),
            "action_rate": RewardTermCfg(
                func=action_rate_penalty,
                weight=self.cfg.action_rate_penalty_scale,  # Already negative in config
            ),
            "action_magnitude": RewardTermCfg(
                func=action_magnitude_penalty,
                weight=self.cfg.action_magnitude_penalty_scale,  # Already negative in config
            ),
            "alive_bonus": RewardTermCfg(
                func=alive_bonus,
                weight=self.cfg.alive_reward_scale,
            ),
        }

    def _setup_scene(self):
        """Setup the simulation scene with robot and terrain."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.4, 0.6, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Apply thruster dynamics
        self._thruster.apply_dynamics(self._actions, self.physics_dt)

    def _apply_action(self):
        """Apply thruster forces and hydrodynamic forces to the robot."""
        # Compute thruster wrench
        thrust_forces, thrust_torques = self._thruster.compute_wrench()
        self._thrust_forces[:, 0, :] = thrust_forces
        self._thrust_torques[:, 0, :] = thrust_torques

        # Compute hydrodynamic forces (using world-frame velocities as per HydrodynamicsModel API)
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )

        # Combine forces in body frame
        total_forces = self._thrust_forces.clone()
        total_forces[:, 0, :] += self._hydro_forces

        total_torques = self._thrust_torques.clone()
        total_torques[:, 0, :] += self._hydro_torques

        # Apply to robot via wrench composer (Isaac Lab API)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=total_torques,
        )

    def _get_observations(self) -> dict:
        """Compute observations."""
        # Base state observations
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        # Up vector in body frame for orientation
        up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        up_b = quat_apply_inverse(root_quat_w, up_w)

        # Task-specific goal observations
        goal_obs = self._task.get_goal_observations(self._robot)

        # Concatenate observations
        obs = torch.cat(
            [
                root_pos_w,  # 3
                root_quat_w,  # 4
                root_lin_vel_b,  # 3
                root_ang_vel_b,  # 3
                goal_obs,  # 3 (task-dependent)
                up_b[:, :2],  # 2
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using the reward manager."""
        # Get goal position from task
        goal_state = self._task.get_goal_state()

        # Compute rewards with context
        return self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            goal_pos_w=goal_state.get("goal_pos_w", torch.zeros(self.num_envs, 3, device=self.device)),
            actions=self._actions,
            prev_actions=self._prev_actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        height = self._robot.data.root_pos_w[:, 2]
        too_low = height < self.cfg.min_height
        too_high = height > self.cfg.max_height

        distance_from_origin = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        too_far = distance_from_origin > self.cfg.max_distance_from_origin

        # Task-specific terminations
        task_terminated = self._task.get_terminations(self._robot)

        terminated = too_low | too_high | too_far | task_terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log episode statistics
        reward_sums = self._reward_manager.reset(env_ids)

        self.extras["log"] = {}
        for name, value in reward_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = value

        final_distance = torch.linalg.norm(
            self._task.goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # Reset components
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Reset thruster and hydrodynamics
        self._thruster.reset(env_ids)
        self._hydro.reset(env_ids)

        # Apply domain randomization
        if self.cfg.randomization.enable:
            self._hydro.randomize_parameters(
                env_ids=env_ids,
                added_mass_scale=self.cfg.randomization.added_mass_scale,
                linear_damping_scale=self.cfg.randomization.linear_damping_scale,
                quadratic_damping_scale=self.cfg.randomization.quadratic_damping_scale,
                volume_scale=self.cfg.randomization.volume_scale,
                mass_scale=self.cfg.randomization.mass_scale,
            )
            self._thruster.randomize_parameters(
                env_ids=env_ids,
                thrust_coeff_scale=self.cfg.randomization.thrust_coefficient_scale,
                time_constant_scale=self.cfg.randomization.time_constant_scale,
            )

        # Randomize ocean current
        if any(v > 0 for v in self.cfg.ocean_current.max_velocity):
            self._hydro.randomize_current(env_ids)

        # Reset task (samples new goals)
        self._task.reset(env_ids, self._terrain.env_origins)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Apply initial pose randomization
        if self.cfg.randomization.enable:
            num_reset = len(env_ids)

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
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "current_visualizer"):
                arrow_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/ocean_current",
                    markers={
                        "arrow": sim_utils.UsdFileCfg(
                            usd_path=f"{sim_utils.ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(0.3, 0.3, 0.3),
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
        goal_state = self._task.get_goal_state()
        goal_pos = goal_state.get("goal_pos_w", torch.zeros(self.num_envs, 3, device=self.device))
        self.goal_pos_visualizer.visualize(goal_pos)

        # Visualize ocean current (access internal attribute, linear components only)
        current_vel = self._hydro._current_velocity[:, :3]
        current_magnitude = torch.linalg.norm(current_vel, dim=1, keepdim=True)
        current_scale = current_magnitude.clamp(min=0.01) * 2.0

        robot_pos = self._robot.data.root_pos_w
        arrow_pos = robot_pos + torch.tensor([[0.0, 0.0, 0.5]], device=self.device)

        from isaaclab.utils.math import quat_from_angle_axis

        default_dir = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        current_dir = current_vel / current_magnitude.clamp(min=1e-6)

        cross = torch.cross(default_dir, current_dir, dim=1)
        dot = torch.sum(default_dir * current_dir, dim=1)
        angle = torch.atan2(torch.linalg.norm(cross, dim=1), dot)
        axis = cross / torch.linalg.norm(cross, dim=1, keepdim=True).clamp(min=1e-6)

        arrow_quat = quat_from_angle_axis(angle, axis)

        self.current_visualizer.visualize(
            translations=arrow_pos,
            orientations=arrow_quat,
            scales=current_scale.expand(-1, 3),
        )
