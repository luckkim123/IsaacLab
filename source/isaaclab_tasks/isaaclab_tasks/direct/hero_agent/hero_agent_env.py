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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_mul

# Import models from common isaaclab_tasks.models
from isaaclab_tasks.models import HydrodynamicsModel

from .rewards import (
    RewardManager,
    RewardTermCfg,
    albc_action_cost,
    albc_potential_reward,
    albc_progress_reward,
)
from .tasks import ALBCAttitudeTask, ALBCAttitudeTaskCfg
from .hero_agent_env_cfg import HeroAgentEnvCfg


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
    - Decimated control (25 Hz default)

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
        - control_decimation: 4 (25 Hz control at 100 Hz sim)
        - action_speed_scale: pi rad/s
        - joint_limits: from URDF (-6.0 to 6.0 rad)
    """

    cfg: HeroAgentEnvCfg

    def __init__(self, cfg: HeroAgentEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the Hero Agent ALBC environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Get body IDs for force application
        self._body_id = self._robot.find_bodies(self.cfg.body_link_name)[0]
        self._buoy_body_id = self._robot.find_bodies(self.cfg.buoy_link_name)[0]

        # Get robot mass for hydrodynamics
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # Initialize main body hydrodynamics model
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            current_cfg=self.cfg.ocean_current,
            dt=self.physics_dt,
            robot_mass=self._robot_mass.item(),
        )

        # Initialize buoy hydrodynamics (Hero Agent specific)
        self._buoy_hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.buoy_hydrodynamics,
            current_cfg=None,  # Buoy shares current with main body
            dt=self.physics_dt,
            robot_mass=self.cfg.buoy_hydrodynamics.vehicle_mass,
        )

        # Find ALBC joint indices
        self._albc_joint_ids = self._robot.find_joints(self.cfg.albc_joint_names)[0]
        if len(self._albc_joint_ids) != 2:
            raise ValueError(
                f"Expected 2 ALBC joints, found {len(self._albc_joint_ids)}. "
                f"Joint names: {self.cfg.albc_joint_names}"
            )

        # Get joint limits from robot
        joint_limits_low = self._robot.data.soft_joint_pos_limits[:, self._albc_joint_ids, 0]
        joint_limits_high = self._robot.data.soft_joint_pos_limits[:, self._albc_joint_ids, 1]
        self._joint_limits_lower = joint_limits_low[0]  # Same for all envs
        self._joint_limits_upper = joint_limits_high[0]

        # Initialize task
        self._task = self._create_task_from_config()

        # Initialize reward manager
        self._reward_manager = RewardManager(
            cfg=self._build_reward_cfg(),
            num_envs=self.num_envs,
            device=self.device,
        )

        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Joint position target buffer (accumulated from actions)
        self._joint_pos_targets = torch.zeros(self.num_envs, 2, device=self.device)

        # Control step counter for decimation
        self._control_step_counter = 0

        # Previous actions for observation
        self._prev_actions_obs = torch.zeros(self.num_envs, 2, device=self.device)

        # Force/torque buffers
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

        # Initialize child body buoyancy (simple vertical force, no torque)
        # This avoids joint constraint conflicts while maintaining buoyancy balance
        self._child_body_ids: list[torch.Tensor] = []
        self._child_body_net_forces: list[float] = []
        gravity = 9.81

        for body_name, params in self.cfg.child_body_buoyancy.items():
            body_id = self._robot.find_bodies(body_name)[0]
            self._child_body_ids.append(body_id)

            # Net buoyancy force: buoyancy - weight (positive = upward)
            buoyancy = self.cfg.water_density * params["volume"] * gravity
            weight = params["mass"] * gravity
            net_force = buoyancy - weight
            self._child_body_net_forces.append(net_force)

        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _create_task_from_config(self) -> ALBCAttitudeTask:
        """Create ALBC attitude task instance.

        Returns:
            ALBCAttitudeTask instance.
        """
        task_cfg = self.cfg.task
        if isinstance(task_cfg, ALBCAttitudeTaskCfg):
            return ALBCAttitudeTask(cfg=task_cfg, num_envs=self.num_envs, device=self.device)
        else:
            return ALBCAttitudeTask(
                cfg=ALBCAttitudeTaskCfg(
                    target_attitude=(0.0, 0.0, 0.0),
                    randomize_target=False,
                ),
                num_envs=self.num_envs,
                device=self.device,
            )

    def _build_reward_cfg(self) -> dict[str, RewardTermCfg]:
        """Build ALBC-specific reward configuration.

        Returns:
            Dictionary mapping term names to RewardTermCfg.
        """
        return {
            "potential": RewardTermCfg(
                func=albc_potential_reward,
                weight=1.0,
                params={"task": self._task, "scale": 8.0},
            ),
            "progress": RewardTermCfg(
                func=albc_progress_reward,
                weight=1.0,
                params={"task": self._task},
            ),
            "action_cost": RewardTermCfg(
                func=albc_action_cost,
                weight=self.cfg.albc_action_cost_weight,
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
        """Process actions before physics step with decimation.

        Actions are accumulated over control_decimation steps before
        being applied to joint position targets.

        Args:
            actions: Joint velocity commands [-1, 1]. Shape: (num_envs, 2).
        """
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Increment control step counter
        self._control_step_counter += 1

        # Apply actions at control frequency (decimation)
        if self._control_step_counter % self.cfg.control_decimation == 0:
            # Store for observation
            self._prev_actions_obs = self._actions.clone()

            # Accumulate joint position targets
            dt_control = self.physics_dt * self.cfg.control_decimation
            delta = dt_control * self.cfg.action_speed_scale * self._actions

            self._joint_pos_targets += delta

            # Clamp to joint limits
            self._joint_pos_targets = torch.clamp(
                self._joint_pos_targets,
                self._joint_limits_lower,
                self._joint_limits_upper,
            )

    def _apply_action(self):
        """Apply joint position targets and hydrodynamic forces."""
        # Apply joint position targets
        self._robot.set_joint_position_target(
            self._joint_pos_targets, joint_ids=self._albc_joint_ids
        )

        # Compute and apply main body hydrodynamic forces
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )

        # Apply to main body
        total_forces = self._hydro_forces.unsqueeze(1)
        total_torques = self._hydro_torques.unsqueeze(1)

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=total_torques,
        )

        # Compute and apply buoy hydrodynamics
        buoy_idx = self._buoy_body_id[0]
        buoy_lin_vel_w = self._robot.data.body_lin_vel_w[:, buoy_idx, :]
        buoy_ang_vel_w = self._robot.data.body_ang_vel_w[:, buoy_idx, :]
        buoy_quat_w = self._robot.data.body_quat_w[:, buoy_idx, :]

        self._buoy_hydro_forces, self._buoy_hydro_torques = self._buoy_hydro.compute_forces(
            root_lin_vel_w=buoy_lin_vel_w,
            root_ang_vel_w=buoy_ang_vel_w,
            root_quat_w=buoy_quat_w,
        )

        buoy_forces = self._buoy_hydro_forces.unsqueeze(1)
        buoy_torques = self._buoy_hydro_torques.unsqueeze(1)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._buoy_body_id,
            forces=buoy_forces,
            torques=buoy_torques,
        )

        # Apply simple buoyancy to child bodies (force only, no torque)
        # This maintains buoyancy balance without fighting joint constraints
        for i, (body_id, net_force) in enumerate(
            zip(self._child_body_ids, self._child_body_net_forces)
        ):
            body_idx = body_id[0]
            body_quat = self._robot.data.body_quat_w[:, body_idx, :]

            # World up direction with net buoyancy force magnitude
            world_up = torch.zeros(self.num_envs, 3, device=self.device)
            world_up[:, 2] = net_force

            # Transform world force to body frame
            force_body = quat_apply_inverse(body_quat, world_up)

            # Apply force only (no torque to avoid joint conflicts)
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                body_ids=body_id,
                forces=force_body.unsqueeze(1),
                torques=torch.zeros(self.num_envs, 1, 3, device=self.device),
            )

    def _get_observations(self) -> dict:
        """Compute ALBC-specific observations.

        Returns 13-dim observation:
            [0:3]   roll, pitch, yaw
            [3:6]   angular velocity body frame
            [6:9]   attitude errors
            [9:11]  joint positions (normalized)
            [11:13] previous actions

        Returns:
            Observation dictionary with "policy" key.
        """
        # Get current orientation as Euler angles
        root_quat_w = self._robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_w)
        euler = torch.stack([roll, pitch, yaw], dim=-1)

        # Angular velocity in body frame
        ang_vel_b = self._robot.data.root_ang_vel_b

        # Attitude errors from task
        attitude_errors = self._task.get_goal_observations(self._robot)

        # Joint positions (normalized to [-1, 1])
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        joint_pos_normalized = 2.0 * (joint_pos - self._joint_limits_lower) / (
            self._joint_limits_upper - self._joint_limits_lower
        ) - 1.0

        # Concatenate observations
        obs = torch.cat(
            [
                euler,                 # 3: roll, pitch, yaw
                ang_vel_b,             # 3: angular velocity
                attitude_errors,       # 3: attitude errors
                joint_pos_normalized,  # 2: joint positions
                self._prev_actions_obs,  # 2: previous actions
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute ALBC rewards using potential-based system.

        Returns:
            Reward tensor. Shape: (num_envs,).
        """
        return self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            goal_pos_w=torch.zeros(self.num_envs, 3, device=self.device),
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

        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()

        # Log attitude errors (in degrees) for terminated environments
        if len(env_ids) > 0:
            attitude_errors = self._task.get_goal_observations(self._robot)
            attitude_errors_deg = torch.rad2deg(attitude_errors[env_ids])
            self.extras["log"]["Attitude_Error/roll_deg"] = attitude_errors_deg[:, 0].mean().item()
            self.extras["log"]["Attitude_Error/pitch_deg"] = attitude_errors_deg[:, 1].mean().item()
            self.extras["log"]["Attitude_Error/yaw_deg"] = attitude_errors_deg[:, 2].mean().item()
            self.extras["log"]["Attitude_Error/total_deg"] = attitude_errors_deg.abs().sum(dim=-1).mean().item()

            # Log joint positions (in radians)
            joint_pos = self._robot.data.joint_pos[env_ids][:, self._albc_joint_ids]
            self.extras["log"]["Joint_Position/joint1_rad"] = joint_pos[:, 0].mean().item()
            self.extras["log"]["Joint_Position/joint2_rad"] = joint_pos[:, 1].mean().item()
            self.extras["log"]["Joint_Position/target1_rad"] = self._joint_pos_targets[env_ids, 0].mean().item()
            self.extras["log"]["Joint_Position/target2_rad"] = self._joint_pos_targets[env_ids, 1].mean().item()

        # Reset components
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_actions_obs[env_ids] = 0.0

        # Reset hydrodynamics
        self._hydro.reset(env_ids)
        self._buoy_hydro.reset(env_ids)

        # Apply domain randomization
        if self.cfg.randomization.enable:
            self._hydro.randomize_parameters(
                env_ids=env_ids,
                added_mass_scale=self.cfg.randomization.added_mass_scale,
                linear_damping_scale=self.cfg.randomization.linear_damping_scale,
                quadratic_damping_scale=self.cfg.randomization.quadratic_damping_scale,
                volume_scale=self.cfg.randomization.volume_scale,
                mass_scale=self.cfg.randomization.mass_scale,
                cob_offset_scale=self.cfg.randomization.cob_offset_scale,
                inertia_scale=self.cfg.randomization.inertia_scale,
                payload_mass_ratio=self.cfg.randomization.payload_mass_ratio,
                payload_cog_offset_z=self.cfg.randomization.payload_cog_offset_z,
            )

        # Randomize ocean current
        if any(v > 0 for v in self.cfg.ocean_current.max_velocity):
            self._hydro.randomize_current(env_ids)

        # Randomize initial joint positions
        if self.cfg.randomization.enable:
            initial_range = self.cfg.initial_joint_pos_range
            random_pos = torch.rand(len(env_ids), 2, device=self.device)
            initial_joint_pos = (
                random_pos * (initial_range[1] - initial_range[0]) + initial_range[0]
            )
        else:
            initial_joint_pos = torch.zeros(len(env_ids), 2, device=self.device)

        # Clamp to limits
        initial_joint_pos = torch.clamp(
            initial_joint_pos,
            self._joint_limits_lower,
            self._joint_limits_upper,
        )

        # Reset joint position targets
        self._joint_pos_targets[env_ids] = initial_joint_pos

        # Reset control step counter
        self._control_step_counter = 0

        # Reset task (samples new targets)
        self._task.reset(env_ids, self._terrain.env_origins)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._albc_joint_ids] = initial_joint_pos
        joint_vel = torch.zeros_like(joint_pos)

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
            default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + pos_z

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
            default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + self.cfg.initial_height

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        # ALBC doesn't have goal position visualization (attitude only)
        pass

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        # ALBC doesn't have goal position visualization (attitude only)
        pass
