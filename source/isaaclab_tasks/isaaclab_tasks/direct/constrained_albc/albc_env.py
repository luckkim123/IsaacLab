# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ALBC (Active Linear Buoyancy Controller) Environment.

Joint-based attitude control using 2 revolute joints to position a buoyancy
element. No thrusters. Joint PD target action mode (q_des = q_nominal + sigma_a * a_t).
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from isaaclab_tasks.models import HydrodynamicsModel

from .config import ALBCEnvCfg
from .mdp.constraints import compute_all_costs
from .mdp.events import (
    DRSampler,
    randomize_body_mass,
    randomize_hydrodynamics,
    randomize_joint_effort_limit,
    randomize_joint_friction,
    randomize_joint_gains,
    randomize_joint_positions,
    randomize_ocean_current,
    randomize_payload,
    randomize_robot_pose,
    reset_joint_positions_default,
    reset_robot_pose_default,
)
from .mdp.observations import compute_policy_obs, compute_privileged_obs
from .mdp.rewards import RewardManager
from .utils import log_dr_metrics


class ALBCEnv(DirectRLEnv):
    """ALBC environment: 2-joint buoyancy attitude control with constrained RL.

    Obs (14D): euler(3), ang_vel(3), att_err(2), joint_pos(2), joint_vel(2), prev_actions(2).
    Action (2D): Delta joint targets via q_des += delta_scale * a_t.
    """

    cfg: ALBCEnvCfg

    def __init__(self, cfg: ALBCEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the ALBC environment.

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

        # Validate state_space value
        if self.cfg.state_space < 0:
            raise ValueError(f"state_space={self.cfg.state_space} must be non-negative")

        # Validate control frequency: control_decimation must be a positive divisor
        # of episode steps. step_dt * control_decimation gives the control period.
        if cfg.control_decimation < 1:
            raise ValueError(f"control_decimation={cfg.control_decimation} must be >= 1")
        control_dt = self.step_dt * cfg.control_decimation
        control_freq = 1.0 / control_dt
        if control_freq < 10.0 or control_freq > 1000.0:
            raise ValueError(
                f"Control frequency {control_freq:.1f}Hz (step_dt={self.step_dt}, "
                f"control_decimation={cfg.control_decimation}) outside valid range [10, 1000]Hz"
            )

        self._init_body_ids()
        self._init_hydrodynamics()
        self._init_payload()
        self._init_joints()
        self._init_task_and_rewards()
        self._init_state_buffers()
        self._init_doraemon()

        # Cache constraint config (avoids getattr on every _get_rewards call)
        self._constraints_cfg = getattr(self.cfg, "constraints", None)

        # Per-condition termination flags (for diagnostics logging)
        self._term_too_fast = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_bad_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_excessive_tilt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @staticmethod
    def _iter_noise_params(cfg: ALBCEnvCfg):
        """Yield (sub_cfg, param_name, value) for all tuple/list noise params."""
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
                    yield sub_cfg, param, val

    @staticmethod
    def _convert_noise_cfg_tuples(cfg: ALBCEnvCfg) -> None:
        """Convert noise config tuple/list values to torch.Tensor in-place.

        Config uses tuples for OmegaConf/Hydra serialization compatibility.
        The noise model functions require float or torch.Tensor for arithmetic.
        Must be called before DirectRLEnv.__init__() which instantiates noise models.
        """
        for sub_cfg, param, val in ALBCEnv._iter_noise_params(cfg):
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
        """Initialize payload physics buffers.

        Payload is applied to the gripper body (fixed to base via base_to_gripper joint).
        """
        self._payload_mass = torch.full((self.num_envs,), self.cfg.payload_mass, device=self.device)
        offset = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
        self._payload_attachment_offset = offset.expand(self.num_envs, -1).clone()
        self._payload_cog_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self._payload_gravity_vec = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)

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
        """Initialize attitude task buffers and reward manager."""
        self._randomize_targets = self.cfg.randomize_target_attitude
        self._base_attitude = torch.tensor(self.cfg.target_attitude, device=self.device)
        self._target_range = torch.tensor(self.cfg.target_attitude_range, device=self.device)
        self._target_euler = self._base_attitude.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._attitude_error = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self._reward_manager = RewardManager(
            cfg=self.cfg.reward,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _init_state_buffers(self) -> None:
        """Initialize action and force/torque buffers."""
        # Action buffers (3-deep history for smoothness penalty)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions_obs = torch.zeros(self.num_envs, 2, device=self.device)

        # Joint PD target: delta accumulation q_des += delta_scale * a_t
        self._nominal_joint_pos = torch.tensor(self.cfg.nominal_joint_pos, device=self.device)
        self._delta_scale = self.cfg.delta_scale
        self._joint_pos_targets = self._nominal_joint_pos.expand(self.num_envs, -1).clone()
        self._control_step_counter = 0

        # Proprioceptive history buffer for actor input (reduces z/input ratio)
        self._proprio_history_len = self.cfg.proprio_history_len
        self._proprio_stride = getattr(self.cfg, "proprio_history_stride", 1)
        if self._proprio_history_len > 0:
            self._proprio_hist = torch.zeros(
                self.num_envs,
                self._proprio_history_len,
                self.cfg.proprio_feature_dim,
                device=self.device,
            )
            self._proprio_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self._proprio_hist = None
            self._proprio_step_counter = None

        # Force/torque buffers
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

    def _init_doraemon(self) -> None:
        """Initialize DORAEMON adaptive DR scheduler if enabled."""
        doraemon_cfg = getattr(self.cfg, "doraemon", None)
        if doraemon_cfg is not None and doraemon_cfg.enable:
            from .doraemon import NDIMS, DoraemonScheduler

            self._doraemon = DoraemonScheduler(doraemon_cfg, self.device)
            self._doraemon_ndims = NDIMS
        else:
            self._doraemon = None
            self._doraemon_ndims = 0

        if self._doraemon is not None:
            ndims = self._doraemon_ndims
            self._episode_dr_xi = torch.zeros(self.num_envs, ndims, device=self.device)
            self._episode_dr_log_probs = torch.zeros(self.num_envs, device=self.device)
            self._episode_return_accum = torch.zeros(self.num_envs, device=self.device)
            self._settling_window = 50  # 1 second at 50Hz control
            self._settling_errors = torch.zeros(self.num_envs, self._settling_window, device=self.device)
            self._settling_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        """Return cached attitude error (computed in _update_attitude_error during reward step)."""
        return self._attitude_error

    def _update_attitude_error(self, quat: torch.Tensor) -> None:
        """Update cached attitude error from current orientation.

        Args:
            quat: Current root quaternion. Shape: (num_envs, 4).
        """
        self._attitude_error = self.compute_attitude_error(quat)

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
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if obs_action_slice is not None:
            self._prev_actions_obs = self._prev_actions[:, obs_action_slice].clone()
        else:
            self._prev_actions_obs = self._prev_actions.clone()
        self._control_step_counter += 1

    def _get_proprio_features(self) -> torch.Tensor:
        """Extract proprioception features for history buffer (8D).

        Features: roll, pitch, p (ang_vel_x), q (ang_vel_y),
                  joint_pos_norm(2), prev_actions(2).
        """
        roll, pitch, _yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel = self._robot.data.root_ang_vel_b
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        joint_pos_norm = 2.0 * (joint_pos - self._joint_limits_lower) / self._joint_limits_range - 1.0
        return torch.cat(
            [roll.unsqueeze(-1), pitch.unsqueeze(-1), ang_vel[:, :2], joint_pos_norm, self._prev_actions_obs],
            dim=-1,
        )

    def _update_proprio_hist(self) -> None:
        """Record proprioception features into ring buffer with stride.

        With stride=1, records every control step (original behavior).
        With stride=N, records every N-th step for wider temporal coverage.
        Effective span = history_len * stride * step_dt (e.g., 15 * 5 * 0.02 = 1.5s).
        """
        if self._proprio_hist is None:
            return
        self._proprio_step_counter += 1
        record_mask = (self._proprio_step_counter % self._proprio_stride) == 0
        if not record_mask.any():
            return
        new_entry = self._get_proprio_features()
        ids = record_mask.nonzero(as_tuple=True)[0]
        self._proprio_hist[ids, :-1] = self._proprio_hist[ids, 1:].clone()
        self._proprio_hist[ids, -1] = new_entry[ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions: compute joint PD targets from policy output.

        Called once per env step (50Hz). With decimation=40, the subsequent
        _apply_action() runs 40 times (2000Hz PD) tracking these targets.

        Args:
            actions: Action commands [-1, 1]. Shape: (num_envs, 2).
        """
        self._update_action_buffers(actions)
        self._update_proprio_hist()

        if self._control_step_counter % self.cfg.control_decimation == 0:
            self._apply_joint_pd_action(self._actions)

    def _apply_joint_pd_action(self, actions: torch.Tensor) -> None:
        """Accumulate delta joint targets: q_des += delta_scale * a_t.

        Delta parameterization limits per-step position change, preventing
        PD actuator saturation while allowing any absolute position via
        accumulation over multiple steps.

        Args:
            actions: Normalized actions [-1, 1]. Shape: (num_envs, 2).
        """
        self._joint_pos_targets += self._delta_scale * actions
        self._joint_pos_targets.clamp_(self._joint_limits_lower, self._joint_limits_upper)

    def _apply_action(self):
        """Apply joint position targets and hydrodynamic forces."""
        self._robot.set_joint_position_target(self._joint_pos_targets, joint_ids=self._albc_joint_ids)

        # Update PhysX acceleration cache for added mass force
        if self._hydro.apply_added_mass:
            self._hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w,
                root_quat_w=self._robot.data.root_quat_w,
            )
        if self._buoy_hydro.apply_added_mass:
            buoy_body_idx = self._buoy_body_id[0]
            self._buoy_hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w[:, buoy_body_idx, :],
                root_quat_w=self._robot.data.body_quat_w[:, buoy_body_idx, :],
            )

        # Main body hydrodynamics
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
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._gripper_body_id,
            forces=payload_forces.unsqueeze(1),
            torques=payload_torques.unsqueeze(1),
        )

    def _compute_payload_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute payload weight force and torque in the gripper body frame.

        Returns:
            Tuple of (forces, torques) in gripper body frame.
        """
        gripper_idx = self._gripper_body_id[0]
        gripper_quat = self._robot.data.body_quat_w[:, gripper_idx, :]
        payload_weight_w = self._payload_mass.unsqueeze(-1) * self._payload_gravity_vec
        payload_weight_b = quat_apply_inverse(gripper_quat, payload_weight_w)
        effective_offset = self._payload_attachment_offset + self._payload_cog_offset
        payload_torque_b = torch.cross(effective_offset, payload_weight_b, dim=-1)
        return payload_weight_b, payload_torque_b

    def _get_observations(self) -> dict:
        """Compute observations: o_t (14D policy) and p_t (23D privileged).

        Returns:
            Observation dictionary with "policy" and "privileged" keys.
        """
        observations = {"policy": compute_policy_obs(self, self._robot)}
        if self.cfg.state_space > 0:
            observations["privileged"] = compute_privileged_obs(self)
        if self._proprio_hist is not None:
            observations["proprio_hist"] = self._proprio_hist.clone().flatten(start_dim=-2)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute ALBC rewards: Gaussian tracking + small action regularizers.

        Returns:
            Reward tensor. Shape: (num_envs,).
        """
        # Update attitude error before reward computation
        self._update_attitude_error(self._robot.data.root_quat_w)

        reward = self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            env=self,
        )

        # Termination penalty: large one-time penalty on early termination
        if self.cfg.reward.termination_penalty != 0.0:
            reward += self.reset_terminated * self.cfg.reward.termination_penalty

        # Compute constraint costs for TRPO + IPO (if constraints configured)
        if self._constraints_cfg is not None and self._constraints_cfg.num_constraints > 0:
            self.extras["costs"] = compute_all_costs(self._robot, self, self._constraints_cfg)

        # DORAEMON: accumulate episode return and settling error
        if self._doraemon is not None:
            self._episode_return_accum += reward
            err = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)
            idx = self._settling_idx % self._settling_window
            self._settling_errors.scatter_(1, idx.unsqueeze(1), err.unsqueeze(1))
            self._settling_idx += 1

        return reward

    def _collect_episode_metrics(
        self,
        env_ids: torch.Tensor,
        reward_sums: dict[str, float],
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
        n = len(env_ids)

        # Weight for downstream weighted averaging (SAC runner uses this)
        log["_num_resets"] = float(n)

        # Reward sums (normalized by max episode duration for episode-length-independent metrics)
        total = 0.0
        for name, value in reward_sums.items():
            normalized = value / self.max_episode_length_s
            log[f"Episode_Reward/{name}"] = normalized
            total += normalized
        log["Episode_Reward/total"] = total

        # Termination rates
        self._collect_termination_metrics(log, env_ids, n)

        if n == 0:
            return log

        # Attitude errors (all resetting envs; weighted averaging handles noise)
        errors_deg = torch.rad2deg(self._attitude_error[env_ids, :2]).abs()
        log["Attitude_Error/roll_deg"] = errors_deg[:, 0].mean().item()
        log["Attitude_Error/pitch_deg"] = errors_deg[:, 1].mean().item()

        # --- Action diagnostics ---
        log["Action/size_mean"] = torch.linalg.norm(self._actions[env_ids], dim=-1).mean().item()
        da = self._actions[env_ids] - self._prev_actions[env_ids]
        log["Action/rate_mean"] = torch.linalg.norm(da, dim=-1).mean().item()

        # --- Dynamics & actuator diagnostics ---
        ang_vel = self._robot.data.root_ang_vel_b[env_ids]
        log["Dynamics/angular_velocity_rp_rms"] = ang_vel[:, :2].pow(2).mean().sqrt().item()
        log["Dynamics/angular_velocity_yaw_rms"] = ang_vel[:, 2].pow(2).mean().sqrt().item()

        jids = self._albc_joint_ids
        joint_vel = self._robot.data.joint_vel[env_ids][:, jids]
        joint_pos = self._robot.data.joint_pos[env_ids][:, jids]
        log["Dynamics/joint_pos_mean_abs"] = joint_pos.abs().mean().item()
        log["Dynamics/joint_vel_abs_max"] = joint_vel.abs().max().item()

        effort_lim = self._robot.data.joint_effort_limits[env_ids][:, jids]
        computed = self._robot.data.computed_torque[env_ids][:, jids]
        applied = self._robot.data.applied_torque[env_ids][:, jids]
        log["Dynamics/effort_limit_mean"] = effort_lim.mean().item()
        log["Dynamics/computed_torque_abs_max"] = computed.abs().max().item()
        log["Dynamics/applied_torque_abs_max"] = applied.abs().max().item()
        log["Dynamics/effort_saturation_frac"] = (computed.abs() >= effort_lim * 0.99).float().mean().item()

        vel_lim = self._robot.data.joint_vel_limits[env_ids][:, jids]
        log["Dynamics/vel_saturation_frac"] = (joint_vel.abs() >= vel_lim.clamp(min=1e-6) * 0.95).float().mean().item()

        # DR parameters (when randomization is enabled)
        if hasattr(self.cfg, "randomization") and self.cfg.randomization.enable:
            # log_dr_metrics expects extras["log"] dict -- pass a wrapper
            extras_wrapper: dict = {"log": log}
            log_dr_metrics(extras_wrapper, self)

        return log

    def _collect_termination_metrics(self, log: dict[str, float | torch.Tensor], env_ids: torch.Tensor, n: int) -> None:
        """Collect termination rate metrics (0.0~1.0, scale-invariant)."""

        def _term_rate(flag: torch.Tensor) -> float:
            return torch.count_nonzero(flag[env_ids]).item() / n if n > 0 else 0.0

        log["Episode_Termination/terminated"] = _term_rate(self.reset_terminated)
        log["Episode_Termination/time_out"] = _term_rate(self.reset_time_outs)
        log["Episode_Termination/too_fast"] = _term_rate(self._term_too_fast)
        log["Episode_Termination/bad_state"] = _term_rate(self._term_bad_state)
        log["Episode_Termination/excessive_tilt"] = _term_rate(self._term_excessive_tilt)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions.

        Termination triggers:
            1. Angular velocity exceeds max_angular_velocity (simulation instability)
            2. NaN detected in root state (PhysX failure)
            3. Attitude angle exceeds max_attitude_angle (prevents Lambda sign reversal)

        Per-condition flags are stored for diagnostics logging.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

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
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        excessive_tilt = (roll.abs() > self.cfg.max_attitude_angle) | (pitch.abs() > self.cfg.max_attitude_angle)

        # Store per-condition flags for diagnostics
        self._term_too_fast = too_fast
        self._term_bad_state = bad_state
        self._term_excessive_tilt = excessive_tilt

        return too_fast | bad_state | excessive_tilt, time_out

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
            4. Task and state reset (attitude targets, robot pose, joint DR, error buffers)
        """
        env_ids_ = self._coerce_env_ids(env_ids)
        self._log_and_reset_rewards(env_ids_)
        self._reset_framework(env_ids_)
        self._reset_physics(env_ids_)
        self._reset_task_and_state(env_ids_)

    def _log_and_reset_rewards(self, env_ids: torch.Tensor) -> None:
        """Collect episode metrics, record DORAEMON episodes, and reset accumulators."""
        # Record completed episodes to DORAEMON buffer before resetting
        if self._doraemon is not None and len(env_ids) > 0:
            current_threshold = self._doraemon._current_threshold_deg
            threshold_rad = torch.deg2rad(torch.tensor(current_threshold, device=self.device))
            filled = self._settling_idx[env_ids].clamp(max=self._settling_window).float()
            mean_settling_err = self._settling_errors[env_ids].sum(dim=-1) / filled.clamp(min=1.0)
            # Soft traversability: sigmoid gives smooth [0,1] instead of binary {0,1}.
            # Improves IS estimator gradient for scipy trust-constr optimizer.
            tau_rad = math.radians(self._doraemon.cfg.traversability_tau_deg)
            success = torch.sigmoid(-(mean_settling_err - threshold_rad) / tau_rad)

            self._doraemon.record_episodes(
                xi=self._episode_dr_xi[env_ids],
                returns=self._episode_return_accum[env_ids],
                success=success,
                log_probs=self._episode_dr_log_probs[env_ids],
            )

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

        self._reset_action_buffers(env_ids)

    def _reset_action_buffers(self, env_ids: torch.Tensor) -> None:
        """Reset action buffers and proprio history."""
        for buf in (self._actions, self._prev_actions, self._prev_prev_actions, self._prev_actions_obs):
            buf[env_ids] = 0.0
        self._joint_pos_targets[env_ids] = self._robot.data.joint_pos[env_ids][:, self._albc_joint_ids]
        if self._proprio_hist is not None:
            self._proprio_hist[env_ids] = 0.0
            self._proprio_step_counter[env_ids] = 0

    def _reset_physics(self, env_ids: torch.Tensor) -> None:
        """Reset hydrodynamics, payload, and apply domain randomization."""
        self._hydro.reset(env_ids)
        self._buoy_hydro.reset(env_ids)

        self._payload_mass[env_ids] = self.cfg.payload_mass
        offset = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
        self._payload_attachment_offset[env_ids] = offset
        self._payload_cog_offset[env_ids] = 0.0

        rand_cfg = self.cfg.randomization
        if not rand_cfg.enable:
            return

        # Create DRSampler (bundles rand_cfg + num_envs + device)
        dr = DRSampler(rand_cfg, num_envs=len(env_ids), device=self.device)
        # Store for _reset_task_and_state (joint gains/friction)
        self._current_dr_sampler = dr

        # DORAEMON: sample from Beta distribution for curriculum-managed parameters
        sampled: dict[str, torch.Tensor] | None = None
        if self._doraemon is not None:
            from .doraemon import PARAM_SPECS

            n = len(env_ids)
            xi_physical, log_probs = self._doraemon.sample(n)
            sampled = {spec.name: xi_physical[:, i] for i, spec in enumerate(PARAM_SPECS)}
            self._episode_dr_xi[env_ids] = xi_physical
            self._episode_dr_log_probs[env_ids] = log_probs
            self._episode_return_accum[env_ids] = 0.0
            self._settling_errors[env_ids] = 0.0
            self._settling_idx[env_ids] = 0

        randomize_hydrodynamics(env=self, env_ids=env_ids, dr=dr, sampled=sampled)
        randomize_body_mass(env=self, env_ids=env_ids, dr=dr)
        randomize_payload(env=self, env_ids=env_ids, dr=dr, sampled=sampled)

        has_ocean_current = any(v > 0 for v in self.cfg.ocean_current.max_velocity)
        if has_ocean_current:
            randomize_ocean_current(env=self, env_ids=env_ids)

    def _reset_task_and_state(self, env_ids: torch.Tensor) -> None:
        """Reset attitude targets, robot pose, joint DR, and initialize error buffers."""
        # Reset attitude targets
        num_reset = len(env_ids)
        if self._randomize_targets:
            random_offset = (torch.rand(num_reset, 3, device=self.device) * 2 - 1) * self._target_range
            self._target_euler[env_ids] = self._base_attitude + random_offset
        else:
            self._target_euler[env_ids] = self._base_attitude.unsqueeze(0).expand(num_reset, -1)

        rand_cfg = self.cfg.randomization
        # Pose must be set BEFORE joints
        if rand_cfg.enable:
            randomize_robot_pose(env=self, env_ids=env_ids, rand_cfg=rand_cfg)
        else:
            reset_robot_pose_default(env=self, env_ids=env_ids, initial_height=self.cfg.initial_height)

        # Joint initialization: random or default
        if rand_cfg.enable:
            randomize_joint_positions(env=self, env_ids=env_ids, joint_pos_range=self.cfg.initial_joint_pos_range)
        else:
            reset_joint_positions_default(env=self, env_ids=env_ids)

        # Joint actuator DR: only when DR enabled.
        # TDC envs override stiffness/damping in their own _reset_idx().
        if rand_cfg.enable:
            dr = getattr(self, "_current_dr_sampler", None)
            if dr is None:
                dr = DRSampler(rand_cfg, num_envs=len(env_ids), device=self.device)
            randomize_joint_gains(env=self, env_ids=env_ids, dr=dr)
            randomize_joint_effort_limit(env=self, env_ids=env_ids, dr=dr)
            randomize_joint_friction(env=self, env_ids=env_ids, dr=dr)
