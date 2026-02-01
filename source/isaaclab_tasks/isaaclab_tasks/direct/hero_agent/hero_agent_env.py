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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

# Import models from common isaaclab_tasks.models
from isaaclab_tasks.models import HydrodynamicsModel

from .hero_agent_env_cfg import HeroAgentEnvCfg
from .mdp.events import (
    randomize_buoy_hydrodynamics,
    randomize_hydrodynamics,
    randomize_joint_positions,
    randomize_ocean_current,
    randomize_robot_pose,
    reset_joint_positions_default,
    reset_robot_pose_default,
)
from .rewards import (
    RewardManager,
    RewardTermCfg,
    albc_action_cost,
    albc_potential_reward,
    albc_progress_reward,
)
from .tasks import ALBCAttitudeTask


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
    - Decimated control (12.5 Hz default)

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
        - control_decimation: 4 (12.5 Hz control at 100 Hz sim with decimation=2)
        - action_speed_scale: pi rad/s (effective pi/2 rad/s, see cfg note)
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
        self._task = ALBCAttitudeTask(
            cfg=self.cfg.task, num_envs=self.num_envs, device=self.device
        )

        # Initialize reward manager
        self._reward_manager = RewardManager(
            cfg={
                "potential": RewardTermCfg(
                    func=albc_potential_reward,
                    weight=self.cfg.albc_potential_reward_weight,
                    params={"task": self._task, "scale": self.cfg.albc_potential_reward_scale},
                ),
                "progress": RewardTermCfg(
                    func=albc_progress_reward,
                    weight=self.cfg.albc_progress_reward_weight,
                    params={"task": self._task},
                ),
                "action_cost": RewardTermCfg(
                    func=albc_action_cost,
                    weight=self.cfg.albc_action_cost_weight,
                ),
            },
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
        gravity = self._gravity_magnitude.item()

        for body_name, params in self.cfg.child_body_buoyancy.items():
            body_id = self._robot.find_bodies(body_name)[0]
            self._child_body_ids.append(body_id)

            # Net buoyancy force: buoyancy - weight (positive = upward)
            buoyancy = self.cfg.water_density * params["volume"] * gravity
            weight = params["mass"] * gravity
            net_force = buoyancy - weight
            self._child_body_net_forces.append(net_force)

        # Pre-allocate reusable buffers for child body force application
        self._child_world_up = torch.zeros(self.num_envs, 3, device=self.device)
        self._child_zero_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

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
        for body_id, net_force in zip(self._child_body_ids, self._child_body_net_forces):
            body_idx = body_id[0]
            body_quat = self._robot.data.body_quat_w[:, body_idx, :]

            # World up direction with net buoyancy force magnitude
            self._child_world_up[:, 2] = net_force
            force_body = quat_apply_inverse(body_quat, self._child_world_up)

            # Apply force only (no torque to avoid joint conflicts)
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                body_ids=body_id,
                forces=force_body.unsqueeze(1),
                torques=self._child_zero_torque,
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

        observations = {"policy": obs}

        if self.cfg.state_space > 0:
            main_priv = self._hydro.get_privileged_info(include_current=True)
            buoy_priv = self._buoy_hydro.get_privileged_info(include_current=False)
            observations["privileged"] = torch.cat([main_priv, buoy_priv], dim=-1)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute ALBC rewards using potential-based system.

        Returns:
            Reward tensor. Shape: (num_envs,).
        """
        return self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
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
        """Reset specified environments.

        Execution order:
            1. Logging
            2. Component reset (robot, super, action buffers)
            3. Hydrodynamics reset + optional randomization
            4. Ocean current randomization
            5. Task reset
            6. Joint state reset (randomized or default)
            7. Robot pose reset (randomized or default)
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # --- 1. Logging ---
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

        if len(env_ids) > 0:
            # Use cached attitude_error to avoid get_goal_observations() side effect
            # (which would update potentials before task.reset)
            attitude_errors_deg = torch.rad2deg(self._task.attitude_error[env_ids])
            self.extras["log"]["Attitude_Error/roll_deg"] = attitude_errors_deg[:, 0].mean().item()
            self.extras["log"]["Attitude_Error/pitch_deg"] = attitude_errors_deg[:, 1].mean().item()
            # total_deg only includes roll and pitch (yaw excluded from control)
            self.extras["log"]["Attitude_Error/total_deg"] = attitude_errors_deg[:, :2].abs().sum(dim=-1).mean().item()

            joint_pos = self._robot.data.joint_pos[env_ids][:, self._albc_joint_ids]
            self.extras["log"]["Joint_Position/joint1_rad"] = joint_pos[:, 0].mean().item()
            self.extras["log"]["Joint_Position/joint2_rad"] = joint_pos[:, 1].mean().item()
            self.extras["log"]["Joint_Position/target1_rad"] = self._joint_pos_targets[env_ids, 0].mean().item()
            self.extras["log"]["Joint_Position/target2_rad"] = self._joint_pos_targets[env_ids, 1].mean().item()

        # --- 2. Component reset ---
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_actions_obs[env_ids] = 0.0

        # --- 3. Hydrodynamics reset ---
        self._hydro.reset(env_ids)
        self._buoy_hydro.reset(env_ids)

        rand_cfg = self.cfg.randomization
        if rand_cfg.enable:
            randomize_hydrodynamics(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
            randomize_buoy_hydrodynamics(env=self, env_ids=env_ids, rand_cfg=rand_cfg)

        # --- 4. Ocean current ---
        if any(v > 0 for v in self.cfg.ocean_current.max_velocity):
            randomize_ocean_current(env=self, env_ids=env_ids)

        # --- 5. Task reset ---
        self._task.reset(env_ids, self._terrain.env_origins)

        # --- 6. Joint state reset (must precede root pose write) ---
        if rand_cfg.enable:
            randomize_joint_positions(
                env=self,
                env_ids=env_ids,
                joint_pos_range=self.cfg.initial_joint_pos_range,
            )
        else:
            reset_joint_positions_default(env=self, env_ids=env_ids)

        # --- 7. Robot pose reset ---
        if rand_cfg.enable:
            randomize_robot_pose(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
        else:
            reset_robot_pose_default(
                env=self, env_ids=env_ids, initial_height=self.cfg.initial_height
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup CoM and CoB visualization markers.

        Creates red sphere for Center of Mass (CoM) and blue sphere for
        Center of Buoyancy (CoB) to visualize the system's hydrostatic properties.
        """
        if debug_vis:
            if not hasattr(self, "_com_marker"):
                # CoM marker (red sphere, 3cm radius)
                com_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/CoM",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    },
                )
                self._com_marker = VisualizationMarkers(com_cfg)

            if not hasattr(self, "_cob_marker"):
                # CoB marker (blue sphere, 3cm radius)
                cob_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/CoB",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                        ),
                    },
                )
                self._cob_marker = VisualizationMarkers(cob_cfg)

            self._com_marker.set_visibility(True)
            self._cob_marker.set_visibility(True)
        else:
            if hasattr(self, "_com_marker"):
                self._com_marker.set_visibility(False)
            if hasattr(self, "_cob_marker"):
                self._cob_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update CoM and CoB marker positions each frame.

        Computes the system-level Center of Mass and Center of Buoyancy
        by combining main body and buoy contributions, then transforms
        from body frame to world frame for visualization.
        """
        # Get robot pose in world frame
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w

        # Main body parameters (from hydrodynamics model)
        m_main = self._hydro._vehicle_mass  # (num_envs,)
        V_main = self._hydro._volume  # (num_envs,)
        r_cg_main = self._hydro._r_cg  # (num_envs, 3) body frame
        r_cb_main = self._hydro._r_cb  # (num_envs, 3) body frame

        # Buoy body parameters
        m_buoy = self._buoy_hydro._vehicle_mass  # (num_envs,)
        V_buoy = self._buoy_hydro._volume  # (num_envs,)
        r_cg_buoy = self._buoy_hydro._r_cg  # (num_envs, 3) body frame relative to buoy
        r_cb_buoy = self._buoy_hydro._r_cb  # (num_envs, 3) body frame relative to buoy

        # Get buoy link position in body frame
        buoy_idx = self._buoy_body_id[0]
        buoy_pos_w = self._robot.data.body_pos_w[:, buoy_idx]
        buoy_offset_w = buoy_pos_w - root_pos
        buoy_offset_b = quat_apply_inverse(root_quat, buoy_offset_w)

        # Calculate system CoM in body frame
        m_total = m_main + m_buoy
        r_cg_system_b = (
            m_main.unsqueeze(-1) * r_cg_main
            + m_buoy.unsqueeze(-1) * (buoy_offset_b + r_cg_buoy)
        ) / m_total.unsqueeze(-1)

        # Calculate system CoB in body frame
        V_total = V_main + V_buoy
        r_cb_system_b = (
            V_main.unsqueeze(-1) * r_cb_main
            + V_buoy.unsqueeze(-1) * (buoy_offset_b + r_cb_buoy)
        ) / V_total.unsqueeze(-1)

        # Transform to world frame
        com_world = root_pos + quat_apply(root_quat, r_cg_system_b)
        cob_world = root_pos + quat_apply(root_quat, r_cb_system_b)

        # Update markers
        self._com_marker.visualize(translations=com_world)
        self._cob_marker.visualize(translations=cob_world)
