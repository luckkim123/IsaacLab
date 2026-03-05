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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from isaaclab_tasks.models import HydrodynamicsModel

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
)

from .config import HeroAgentEnvCfg
from .mdp import (
    RewardManager,
    RewardTermCfg,
    compute_policy_obs,
    compute_privileged_obs,
    joint_oscillation_penalty,
    joint_velocity_penalty,
    linear_error_penalty,
    progress_reward,
    progress_reward_pbrs,
    settling_bonus,
    tracking_reward,
)
from .mdp.events import (
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
from .utils import DebugVisualization, log_dr_metrics, log_tdc_diagnostics


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
        - max_joint_velocity: 4*pi/3 rad/s (40 RPM at 12V, ~240 deg/s)
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
        # Adjust observation_space for optional TDE obs (must be before super().__init__)
        if cfg.enable_tde_obs:
            cfg.observation_space += 2
            self._pad_noise_cfg_for_tde(cfg)

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

        # Per-condition termination flags (for diagnostics logging)
        self._term_too_fast = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_bad_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_excessive_tilt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Debug visualization manager
        self._debug_vis = DebugVisualization(self.num_envs, self.device)
        self.set_debug_vis(self.cfg.debug_vis)

    @staticmethod
    def _iter_noise_params(cfg: HeroAgentEnvCfg):
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
    def _pad_noise_cfg_for_tde(cfg: HeroAgentEnvCfg) -> None:
        """Pad observation noise config by 2 dims (zeros) for TDE obs channels.

        TDE obs (H_hat) has no sensor noise model -- it's a computed signal
        whose noise comes from nu_dot estimation and is handled by the EMA filter.
        Must be called before _convert_noise_cfg_tuples() to preserve tuple format.
        """
        for sub_cfg, param, val in HeroAgentEnv._iter_noise_params(cfg):
            setattr(sub_cfg, param, type(val)(list(val) + [0.0, 0.0]))

    @staticmethod
    def _convert_noise_cfg_tuples(cfg: HeroAgentEnvCfg) -> None:
        """Convert noise config tuple/list values to torch.Tensor in-place.

        Config uses tuples for OmegaConf/Hydra serialization compatibility.
        The noise model functions require float or torch.Tensor for arithmetic.
        Must be called before DirectRLEnv.__init__() which instantiates noise models.
        """
        for sub_cfg, param, val in HeroAgentEnv._iter_noise_params(cfg):
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
            penalty_curriculum_ratio=self.cfg.reward.penalty_curriculum_ratio,
        )
        self._init_doraemon()

    def _build_reward_terms(self) -> dict[str, RewardTermCfg]:
        """Build the reward terms dict. Override in subclasses to add/modify terms.

        Base terms (dt-scaled via RewardTermCfg.scale_by_dt, default True):
            1. tracking: Laplacian kernel exp(-||e||/sigma)
            2. linear_error: -min(||err||, max)/max (constant gradient tail)
            3. joint_oscillation: EMA high-pass joint vel^2
            4. joint_velocity: quadratic joint velocity penalty
            5. progress: PBRS (scale_by_dt=False, per-transition not per-time)
        """
        rcfg = self.cfg.reward
        terms = {
            "tracking": RewardTermCfg(
                func=tracking_reward,
                weight=rcfg.tracking_weight,
                params={"sigma": rcfg.tracking_sigma},
            ),
        }
        if rcfg.linear_error_weight != 0.0:
            terms["linear_error"] = RewardTermCfg(
                func=linear_error_penalty,
                weight=rcfg.linear_error_weight,
                params={"max_err": rcfg.linear_error_max},
            )
        if rcfg.joint_oscillation_weight != 0.0:
            terms["joint_oscillation"] = RewardTermCfg(
                func=joint_oscillation_penalty,
                weight=rcfg.joint_oscillation_weight,
            )
        if rcfg.joint_velocity_weight != 0.0:
            terms["joint_velocity"] = RewardTermCfg(
                func=joint_velocity_penalty,
                weight=rcfg.joint_velocity_weight,
            )
        if rcfg.progress_weight != 0.0:
            if rcfg.progress_mode == "pbrs":
                func = progress_reward_pbrs
                params = {"gamma": rcfg.progress_gamma}
            else:
                func = progress_reward
                params = {"scale": rcfg.progress_scale}
            terms["progress"] = RewardTermCfg(
                func=func,
                weight=rcfg.progress_weight,
                params=params,
                scale_by_dt=False,  # PBRS is per-transition, not a rate
            )
        if rcfg.settling_weight != 0.0:
            terms["settling"] = RewardTermCfg(
                func=settling_bonus,
                weight=rcfg.settling_weight,
                params={"threshold": rcfg.settling_threshold},
            )
        return terms

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

        # Per-env DORAEMON tracking buffers
        if self._doraemon is not None:
            ndims = self._doraemon_ndims
            self._episode_dr_xi = torch.zeros(self.num_envs, ndims, device=self.device)
            self._episode_dr_log_probs = torch.zeros(self.num_envs, device=self.device)
            self._episode_return_accum = torch.zeros(self.num_envs, device=self.device)
            # Settling window: last 1 second (50 steps at 50Hz control)
            self._settling_window = 50
            self._settling_errors = torch.zeros(self.num_envs, self._settling_window, device=self.device)
            self._settling_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _init_state_buffers(self) -> None:
        """Initialize action and force/torque buffers."""
        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions_obs = torch.zeros(self.num_envs, 2, device=self.device)

        # EMA joint velocity (for high-pass oscillation penalty)
        self._ema_joint_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._ema_joint_vel_alpha = self.cfg.reward.joint_oscillation_alpha
        self._joint_pos_targets = torch.zeros(self.num_envs, 2, device=self.device)
        # Global step counter (not per-env). With control_decimation=1 (default),
        # this modulo always passes. If control_decimation > 1, all envs share
        # the same control phase.
        self._control_step_counter = 0

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

        # Buoy perturbation buffers (independent phase from main body)
        self._buoy_perturb_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_perturb_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._buoy_perturb_timer = torch.randint(0, perturb_cycle, (self.num_envs,), device=self.device)

        # TDE observation buffers (optional dynamics mismatch signal)
        if self.cfg.enable_tde_obs:
            self._tde_m_hat = torch.tensor(self.cfg.tde_m_hat, device=self.device, dtype=torch.float32)
            self._tde_nu_prev = torch.zeros(self.num_envs, 2, device=self.device)
            self._tde_nu_dot_filtered = torch.zeros(self.num_envs, 2, device=self.device)
            self._tde_h_hat = torch.zeros(self.num_envs, 2, device=self.device)
            self._tde_is_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tde_h = self.cfg.tde_h
            self._tde_ema_alpha = self.cfg.tde_nu_dot_ema_alpha
            self._tde_l1 = HERO_AGENT_ALBC_LINK1_LENGTH
            self._tde_l2 = HERO_AGENT_ALBC_LINK2_LENGTH

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

    def _get_proprio_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract proprioception features shared by policy obs and adaptation history.

        Returns:
            roll, pitch, p (ang_vel_x), q (ang_vel_y), joint_pos_normalized
        """
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        ang_vel_b = self._robot.data.root_ang_vel_b
        p = ang_vel_b[:, 0:1]
        q = ang_vel_b[:, 1:2]
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        joint_pos_norm = 2.0 * (joint_pos - self._joint_limits_lower) / self._joint_limits_range - 1.0
        return roll, pitch, p, q, joint_pos_norm

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

    def _apply_perturbation_cycle(
        self,
        timer: torch.Tensor,
        forces: torch.Tensor,
        torques: torch.Tensor,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        cycle: int,
        duration: int,
    ) -> torch.Tensor:
        """Advance perturbation timer and generate/clear random wrench.

        Returns updated timer.
        """
        timer = (timer + 1) % cycle

        trigger = timer == 0
        if trigger.any():
            n = trigger.sum().item()
            f_dir = torch.randn(n, 3, device=self.device)
            f_dir = f_dir / f_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            f_lo, f_hi = force_range
            f_mag = torch.rand(n, device=self.device) * (f_hi - f_lo) + f_lo
            forces[trigger] = f_dir * f_mag.unsqueeze(1)

            t_dir = torch.randn(n, 3, device=self.device)
            t_dir = t_dir / t_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
            t_lo, t_hi = torque_range
            t_mag = torch.rand(n, device=self.device) * (t_hi - t_lo) + t_lo
            torques[trigger] = t_dir * t_mag.unsqueeze(1)

        deactivate = timer == duration
        if deactivate.any():
            forces[deactivate] = 0.0
            torques[deactivate] = 0.0

        return timer

    def _update_perturbation(self) -> None:
        """Update per-step random perturbation forces on the base body and buoy.

        Uses per-env timers that cycle through [0, interval+duration).
        Phase [0, duration): perturbation active. Phase [duration, cycle): cooldown.
        New random wrench is generated at the start of each active phase.

        Main body and buoy have independent timers (decorrelated phases) but
        share the same interval/duration timing parameters.

        Forces are stored in ``_perturb_forces`` / ``_perturb_torques`` (main body)
        and ``_buoy_perturb_forces`` / ``_buoy_perturb_torques`` (buoy), then
        added to hydro forces in ``_apply_action()``.
        """
        rand_cfg = self.cfg.randomization
        if not rand_cfg.enable:
            return

        main_perturb = rand_cfg.enable_perturbation
        buoy_perturb = rand_cfg.enable_buoy_perturbation
        if not main_perturb and not buoy_perturb:
            return

        interval = rand_cfg.perturbation_interval
        duration = rand_cfg.perturbation_duration
        cycle = interval + duration

        if main_perturb:
            self._perturb_timer = self._apply_perturbation_cycle(
                self._perturb_timer,
                self._perturb_forces,
                self._perturb_torques,
                rand_cfg.perturbation_force_range,
                rand_cfg.perturbation_torque_range,
                cycle,
                duration,
            )

        if buoy_perturb:
            self._buoy_perturb_timer = self._apply_perturbation_cycle(
                self._buoy_perturb_timer,
                self._buoy_perturb_forces,
                self._buoy_perturb_torques,
                rand_cfg.buoy_perturbation_force_range,
                rand_cfg.buoy_perturbation_torque_range,
                cycle,
                duration,
            )

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
            # control_dt = physics_dt * control_decimation (50Hz = 0.005 * 4 = 0.02s)
            control_dt = self.physics_dt * self.cfg.control_decimation
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
        buoy_total_forces = self._buoy_hydro_forces + self._buoy_perturb_forces
        buoy_total_torques = self._buoy_hydro_torques + self._buoy_perturb_torques
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._buoy_body_id,
            forces=buoy_total_forces.unsqueeze(1),
            torques=buoy_total_torques.unsqueeze(1),
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

    def _compute_tde_obs(self) -> torch.Tensor:
        """Compute TDE-based dynamics mismatch observation H_hat (2D).

        H_hat encodes all unmodeled dynamics using the TDE identity:
            H = Lambda * p_EE + T_b - M_bar * nu_dot

        Where H = (M_true - M_bar)*nu_dot + B_t captures inertia error,
        coupling, Coriolis, damping, gravity, and external disturbances.

        Uses EMA-filtered angular acceleration to reduce sensor noise.
        On the first step after reset, returns zeros (no valid finite diff).

        Returns:
            H_hat tensor of shape (num_envs, 2).
        """
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        nu = self._robot.data.root_ang_vel_b[:, :2]  # [p, q]

        # Angular acceleration via EMA-filtered finite difference
        nu_dot_raw = (nu - self._tde_nu_prev) / self.step_dt
        self._tde_nu_dot_filtered = (
            self._tde_ema_alpha * nu_dot_raw + (1.0 - self._tde_ema_alpha) * self._tde_nu_dot_filtered
        )

        # Buoyancy force from buoy hydro model (per-env, DR'd)
        F_bu = self._buoy_hydro.buoyancy_force

        # Lambda * p_EE (2x2 anti-diagonal @ 2D EE position)
        lf = torch.cos(pitch) * torch.cos(roll) * F_bu
        joint_pos = self._robot.data.joint_pos[:, self._albc_joint_ids]
        g1 = joint_pos[:, 0]
        g12 = joint_pos[:, 0] + joint_pos[:, 1]
        p_EE_x = self._tde_l1 * torch.cos(g1) + self._tde_l2 * torch.cos(g12)
        p_EE_y = self._tde_l1 * torch.sin(g1) + self._tde_l2 * torch.sin(g12)
        # Lambda = [[0, lf], [-lf, 0]] -> Lambda @ [x, y] = [lf*y, -lf*x]
        Lambda_p_EE_roll = lf * p_EE_y
        Lambda_p_EE_pitch = -lf * p_EE_x

        # Restoring torque T_b
        h = self._tde_h
        T_b_roll = -torch.cos(pitch) * torch.sin(roll) * F_bu * h
        T_b_pitch = -torch.sin(pitch) * F_bu * h

        # H_hat = Lambda*p_EE + T_b - M_bar*nu_dot
        H_hat_roll = Lambda_p_EE_roll + T_b_roll - self._tde_m_hat[0] * self._tde_nu_dot_filtered[:, 0]
        H_hat_pitch = Lambda_p_EE_pitch + T_b_pitch - self._tde_m_hat[1] * self._tde_nu_dot_filtered[:, 1]

        h_hat = torch.stack([H_hat_roll, H_hat_pitch], dim=-1)

        # Zero out for envs without valid history (first step after reset)
        h_hat = torch.where(self._tde_is_initialized.unsqueeze(-1), h_hat, torch.zeros_like(h_hat))

        # Update history
        self._tde_nu_prev.copy_(nu)
        self._tde_is_initialized[:] = True
        self._tde_h_hat = h_hat

        return h_hat

    def _get_observations(self) -> dict:
        """Compute ALBC-specific observations.

        Returns 13-dim (or 15-dim with TDE) policy observation
        and optional privileged observations.
        See mdp.observations for implementation details.

        Returns:
            Observation dictionary with "policy" key and optional "privileged" key.
        """
        policy_obs = compute_policy_obs(self, self._robot)
        if self.cfg.enable_tde_obs:
            tde_obs = self._compute_tde_obs()
            policy_obs = torch.cat([policy_obs, tde_obs], dim=-1)

        observations = {"policy": policy_obs}
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

        # Update EMA joint velocity (low-pass) before reward computation
        vel = self._robot.data.joint_vel[:, self._albc_joint_ids]
        alpha = self._ema_joint_vel_alpha
        self._ema_joint_vel = alpha * vel + (1.0 - alpha) * self._ema_joint_vel

        reward = self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            actions=self._actions,
            prev_actions=self._prev_actions,
            env=self,  # Pass env for accessing potentials, EMA state
        )

        # Termination penalty: large one-time penalty on early termination
        if self.cfg.reward.termination_penalty != 0.0:
            reward += self.reset_terminated * self.cfg.reward.termination_penalty

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

        # Penalty curriculum scale (0.0 ~ 1.0)
        log["Reward/penalty_scale"] = self._reward_manager.penalty_scale

        # Reward sums (normalized by max episode duration for episode-length-independent metrics)
        total = 0.0
        for name, value in reward_sums.items():
            normalized = value / self.max_episode_length_s
            log[f"Episode_Reward/{name}"] = normalized
            total += normalized
        log["Episode_Reward/total"] = total

        # Termination rates (0.0~1.0, scale-invariant for weighted averaging)
        def _term_rate(flag: torch.Tensor) -> float:
            return torch.count_nonzero(flag[env_ids]).item() / n if n > 0 else 0.0

        log["Episode_Termination/terminated"] = _term_rate(self.reset_terminated)
        log["Episode_Termination/time_out"] = _term_rate(self.reset_time_outs)
        log["Episode_Termination/too_fast"] = _term_rate(self._term_too_fast)
        log["Episode_Termination/bad_state"] = _term_rate(self._term_bad_state)
        log["Episode_Termination/excessive_tilt"] = _term_rate(self._term_excessive_tilt)

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

        # Joint oscillation high-freq component
        hf = joint_vel - self._ema_joint_vel[env_ids]
        log["Dynamics/joint_oscillation_hf_rms"] = hf.pow(2).mean().sqrt().item()
        log["Dynamics/joint_pos_mean_abs"] = joint_pos.abs().mean().item()
        log["Dynamics/joint_vel_abs_max"] = joint_vel.abs().max().item()

        # Effort limit saturation (verify DR'd effort limits are active)
        effort_lim = self._robot.data.joint_effort_limits[env_ids][:, jids]
        computed = self._robot.data.computed_torque[env_ids][:, jids]
        log["Dynamics/effort_limit_mean"] = effort_lim.mean().item()
        log["Dynamics/computed_torque_abs_max"] = computed.abs().max().item()
        log["Dynamics/effort_saturation_frac"] = (computed.abs() >= effort_lim * 0.99).float().mean().item()

        # Velocity limit saturation (verify PhysX velocity limits are active)
        vel_lim = self._robot.data.joint_vel_limits[env_ids][:, jids]
        log["Dynamics/vel_saturation_frac"] = (joint_vel.abs() >= vel_lim.clamp(min=1e-6) * 0.95).float().mean().item()

        # TDC diagnostics (for any env with TDC controller)
        if hasattr(self, "_tdc"):
            log_tdc_diagnostics(log, self._tdc, env=self)

        # DR parameters (when randomization is enabled)
        if hasattr(self.cfg, "randomization") and self.cfg.randomization.enable:
            # log_dr_metrics expects extras["log"] dict -- pass a wrapper
            extras_wrapper: dict = {"log": log}
            log_dr_metrics(extras_wrapper, self)

        return log

    def get_eval_snapshot(self) -> dict[str, float]:
        """Return current evaluation metrics for play-mode diagnostics.

        Provides instantaneous per-env averages of key quantities, useful for
        printing periodic summaries during play without needing episode resets.

        Returns:
            Dict with keys: attitude_error_deg, action_rate,
            angular_velocity_rp_rms, angular_velocity_yaw_rms,
            joint_oscillation_hf_rms, joint_pos_mean_abs.
        """
        err = self._attitude_error[:, :2]
        da = self._actions - self._prev_actions
        joint_vel = self._robot.data.joint_vel[:, self._albc_joint_ids]
        hf = joint_vel - self._ema_joint_vel
        return {
            "attitude_error_deg": torch.rad2deg(torch.linalg.norm(err, dim=-1)).mean().item(),
            "action_rate": torch.linalg.norm(da, dim=-1).mean().item(),
            "angular_velocity_rp_rms": self._robot.data.root_ang_vel_b[:, :2].pow(2).mean().sqrt().item(),
            "angular_velocity_yaw_rms": self._robot.data.root_ang_vel_b[:, 2].pow(2).mean().sqrt().item(),
            "joint_oscillation_hf_rms": hf.pow(2).mean().sqrt().item(),
            "joint_pos_mean_abs": self._robot.data.joint_pos[:, self._albc_joint_ids].abs().mean().item(),
        }

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
            4. Task and state reset (attitude targets, robot pose, joint DR, potentials)
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
            # Success = mean settling error < threshold (no timed_out requirement).
            # Early-terminated episodes that achieved low error before termination
            # still count as successful for DR distribution optimization.
            current_threshold = self._doraemon._current_threshold_deg
            threshold_rad = torch.deg2rad(torch.tensor(current_threshold, device=self.device))
            # Use the filled portion of settling ring buffer
            filled = self._settling_idx[env_ids].clamp(max=self._settling_window).float()
            mean_settling_err = self._settling_errors[env_ids].sum(dim=-1) / filled.clamp(min=1.0)
            success = mean_settling_err < threshold_rad

            self._doraemon.record_episodes(
                xi=self._episode_dr_xi[env_ids],
                returns=self._episode_return_accum[env_ids],
                success=success.float(),
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

        for buf in (self._actions, self._prev_actions, self._prev_actions_obs):
            buf[env_ids] = 0.0
        # Reset EMA to zero (velocity is reset to 0 in _reset_task_and_state after this)
        self._ema_joint_vel[env_ids] = 0.0

        # Reset perturbation state: randomize timer phase to decorrelate envs
        rand_cfg = self.cfg.randomization
        perturb_cycle = max(1, rand_cfg.perturbation_interval + rand_cfg.perturbation_duration)
        self._perturb_forces[env_ids] = 0.0
        self._perturb_torques[env_ids] = 0.0
        self._perturb_timer[env_ids] = torch.randint(0, perturb_cycle, (len(env_ids),), device=self.device)
        self._buoy_perturb_forces[env_ids] = 0.0
        self._buoy_perturb_torques[env_ids] = 0.0
        self._buoy_perturb_timer[env_ids] = torch.randint(0, perturb_cycle, (len(env_ids),), device=self.device)

        # Reset TDE observation buffers
        if self.cfg.enable_tde_obs:
            self._tde_nu_prev[env_ids] = 0.0
            self._tde_nu_dot_filtered[env_ids] = 0.0
            self._tde_h_hat[env_ids] = 0.0
            self._tde_is_initialized[env_ids] = False

        # Reset action latency: sample new per-env latency and clear history
        if self._action_history is not None and self._action_latency is not None:
            lo, hi = rand_cfg.action_latency_range
            self._action_history[env_ids] = 0.0
            self._action_latency[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    def _reset_physics(self, env_ids: torch.Tensor) -> None:
        """Reset hydrodynamics, payload, and apply domain randomization.

        When DORAEMON is active, samples DR parameters from the Beta distribution
        and passes them to randomize functions via the ``sampled`` dict.
        """
        self._hydro.reset(env_ids)
        self._buoy_hydro.reset(env_ids)

        if self._payload_mass is not None:
            self._payload_mass[env_ids] = self.cfg.payload_mass
            offset = torch.tensor(self.cfg.payload_attachment_offset, device=self.device, dtype=torch.float32)
            self._payload_attachment_offset[env_ids] = offset
            self._payload_cog_offset[env_ids] = 0.0

        rand_cfg = self.cfg.randomization
        if not rand_cfg.enable:
            return

        # Build sampled dict from DORAEMON Beta distribution
        sampled: dict[str, torch.Tensor] | None = None
        if self._doraemon is not None:
            from .doraemon import PARAM_SPECS

            n = len(env_ids)
            xi_physical, log_probs = self._doraemon.sample(n)
            sampled = {spec.name: xi_physical[:, i] for i, spec in enumerate(PARAM_SPECS)}

            # Store for episode tracking
            self._episode_dr_xi[env_ids] = xi_physical
            self._episode_dr_log_probs[env_ids] = log_probs
            self._episode_return_accum[env_ids] = 0.0
            self._settling_errors[env_ids] = 0.0
            self._settling_idx[env_ids] = 0

        # Store sampled dict for _reset_task_and_state (joint gains/friction)
        self._current_sampled = sampled

        randomize_hydrodynamics(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)
        randomize_body_mass(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)
        if self._payload_enabled:
            randomize_payload(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)

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
        sampled = getattr(self, "_current_sampled", None)
        randomize_joint_gains(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)
        randomize_joint_effort_limit(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)
        randomize_joint_friction(env=self, env_ids=env_ids, rand_cfg=rand_cfg, sampled=sampled)

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
