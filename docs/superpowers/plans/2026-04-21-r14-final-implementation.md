# r14 Final Training Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize r14 configuration from r13_B baseline with entropy reduction, aggressive DR widening, and action-latency DR, then launch 4096-env × 20000-iter training.

**Architecture:** Modify `constrained_full_albc` config + environment. No new files in the task package. Port 1 code block (~40 lines) from `hero_agent/base_env.py` for action-latency buffer. Single training run invoked by existing `train.py`.

**Tech Stack:** Isaac Lab 5.1, RSL-RL, PyTorch, ConstraintTRPO + Asymmetric Encoder, DORAEMON curriculum, WandB.

**Spec reference:** `docs/superpowers/specs/2026-04-21-r14-final-design.md`

---

## CRITICAL DISCOVERY (pre-plan)

Spec says "`entropy_coef` 0.003 → 0.001" but the current code (`rsl_rl_ppo_cfg.py:206`) has `entropy_coef_per_dim = (0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)`. Per `constraint_trpo.py:107`, when this tuple is non-empty it **overrides** the scalar `entropy_coef`. Thruster dims are **already at 0.001**. Dropping the scalar changes nothing.

**Correct intervention for the roll-oscillation root cause**: reduce the **thruster entries** of `entropy_coef_per_dim`, not the scalar. Plan uses `(0.01, 0.01, 0.0005×6)` — half thruster pressure, arm unchanged. This directly attacks the thruster std ~0.25 → target ~0.15.

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py` | Modify | encoder_latent_dim 9→16, entropy_coef_per_dim thrusters 0.001→0.0005, save_interval 50→100 |
| `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py` | Modify | HardDR widened, DORAEMON step_interval 250→500, ocean_current noise_scale expanded, action_latency_range added |
| `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py` | Modify | Port action latency buffer (init + reset + pre_physics hook) |
| `scripts/launch_r14.sh` | Create | Launch script pinning CLI args and enabling auto-resume |

---

## Task 1: Update policy config (encoder latent, entropy per-dim, save_interval)

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py:122`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py:206`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py:232`

- [ ] **Step 1.1: Change encoder_latent_dim 9 → 16**

File: `rsl_rl_ppo_cfg.py`
Find (line 122):
```python
    encoder_latent_dim: int = 9
```
Replace with:
```python
    encoder_latent_dim: int = 16
```

- [ ] **Step 1.2: Reduce thruster entropy pressure**

File: `rsl_rl_ppo_cfg.py`
Find (lines 204-206):
```python
    # Validated in Round 2 experiments (2026-04-14): PerDimEnt outperformed Baseline
    # and ArmOnly on reward, noise stability, and DORAEMON success.
    entropy_coef_per_dim: tuple[float, ...] = (0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
```
Replace with:
```python
    # Validated in Round 2 experiments (2026-04-14): PerDimEnt outperformed Baseline
    # and ArmOnly on reward, noise stability, and DORAEMON success.
    # r14: thrusters 0.001 -> 0.0005 to attack roll-osc root cause (thruster std ~0.25
    # drove 0.68-0.87 Hz limit cycle via weak TAM roll arm 0.007m). Target std ~0.15.
    entropy_coef_per_dim: tuple[float, ...] = (0.01, 0.01, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005)
```

- [ ] **Step 1.3: Increase save_interval 50 → 100**

File: `rsl_rl_ppo_cfg.py`
Find (line 232):
```python
    save_interval = 50
```
Replace with:
```python
    save_interval = 100
```

- [ ] **Step 1.4: Syntax check**

Run from `/workspace/isaaclab`:
```bash
python3 -c "import ast; ast.parse(open('source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py').read())"
```
Expected output: no error (empty).

---

## Task 2: Update DORAEMON step_interval

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py:401`

- [ ] **Step 2.1: DORAEMON step_interval 250 → 500**

File: `config.py`
Find (line 401):
```python
    doraemon: DoraemonCfg = DoraemonCfg(enable=True, kl_ub=0.06, performance_lb=90.0, step_interval=250)
```
Replace with:
```python
    doraemon: DoraemonCfg = DoraemonCfg(enable=True, kl_ub=0.06, performance_lb=90.0, step_interval=500)
```

---

## Task 3: Aggressive HardDR widening (13 tuple/scalar params)

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py:150-186` (HardDomainRandomizationCfg class body)
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py:146` (parent DomainRandomizationCfg `ocean_current_strength_range` upper bound)

Rationale: spec `Aggressive HardDR (17 params, 1.5-3x wider)`. r13_B uses `HardDomainRandomizationCfg` which inherits from `DomainRandomizationCfg` and overrides a subset. Parent's defaults apply where not overridden (e.g., `yaw_damping_scale`, `joint_effort_limit_range`, `water_density_range`, `joint_stiffness_range`, `joint_damping_range`, `joint_static_friction_range`, `joint_viscous_friction_range`, `joint_static_friction_range`, `joint_viscous_friction_range`, `joint_damping_range`, `thrust_coefficient_scale`, `time_constant_scale`, `ocean_current_strength_range`). Only some of those have overrides; to cleanly widen all spec-listed fields we set them directly on `HardDomainRandomizationCfg` (even if the value equals an inherited field). This is explicit and future-proof.

- [ ] **Step 3.1: Replace HardDomainRandomizationCfg body with widened ranges**

File: `config.py`
Find (lines 149-186, the entire class body — **after** the docstring, keep docstring):
```python
@configclass
class HardDomainRandomizationCfg(DomainRandomizationCfg):
    """Aggressive DR for encoder training. Widens all ranges significantly.

    Expanded 2026-04-10: DORAEMON saturated all 15 parameters at Beta(1,1)=UNIFORM
    in run 2026-04-09_16-41-45. All bounds widened by ~30-50% beyond prior limits.
    Physics stability constraints: added_mass/inertia ratio < 1.0 (init validation),
    post-DR per-axis clamp (0.95*I) ensures stability.
    """

    enable: bool = True
    added_mass_scale: tuple[float, float] = (0.5, 1.5)
    linear_damping_scale: tuple[float, float] = (0.4, 1.7)
    quadratic_damping_scale: tuple[float, float] = (0.4, 1.7)
    volume_scale: tuple[float, float] = (0.75, 1.25)
    cob_offset_x: tuple[float, float] = (-0.02, 0.02)
    cob_offset_y: tuple[float, float] = (-0.02, 0.02)
    cob_offset_z: tuple[float, float] = (-0.04, 0.04)
    cog_offset_x: tuple[float, float] = (-0.02, 0.02)
    cog_offset_y: tuple[float, float] = (-0.02, 0.02)
    cog_offset_z: tuple[float, float] = (-0.04, 0.04)
    inertia_scale: tuple[float, float] = (0.4, 2.0)
    body_mass_scale: tuple[float, float] = (0.75, 1.25)
    payload_mass_range: tuple[float, float] = (0.0, 3.0)
    # Reduced 2026-04-19 from 0.15 -> 0.08: outlier-env analysis of r9_tightrates
    # (eval_dr hard) showed 3 kg payload at 0.15 m offset generates ~4.5 Nm
    # gravitational torque, far exceeding roll TAM authority (4 x 50 N x 0.007 m
    # = 1.4 Nm). Combinations in that regime are physically uncontrollable and
    # dominated per-env SS_std (CV 2.18). 0.08 caps torque at ~2.4 Nm so roll
    # can still stabilize within authority while keeping pitch/yaw challenge.
    payload_cog_offset_xy_radius: float = 0.08
    payload_cog_offset_z: tuple[float, float] = (-0.05, 0.0)
    # -- Joint Actuator --
    joint_stiffness_range: tuple[float, float] = (30.0, 150.0)
    joint_damping_range: tuple[float, float] = (0.3, 7.0)
    # -- Thruster --
    thrust_coefficient_scale: tuple[float, float] = (0.7, 1.3)
    time_constant_scale: tuple[float, float] = (0.7, 1.3)
```

Replace with:
```python
@configclass
class HardDomainRandomizationCfg(DomainRandomizationCfg):
    """Aggressive DR for encoder training. Widens all ranges significantly.

    Expanded 2026-04-21 (r14): r13_B achieved survival=100% + clean SS on all DR
    levels, indicating policy capacity was under-utilized by previous ranges.
    All spec-listed params widened 1.5-3x to stress beyond nominal robot limits.
    Eval filters extreme tail (see spec for DR_SCALE rescaling strategy).

    Physics stability constraints: added_mass/inertia ratio < 1.0 (init validation),
    post-DR per-axis clamp (0.95*I) ensures stability.
    """

    enable: bool = True
    # -- Hydrodynamics --
    added_mass_scale: tuple[float, float] = (0.3, 1.8)
    linear_damping_scale: tuple[float, float] = (0.2, 2.2)
    quadratic_damping_scale: tuple[float, float] = (0.2, 2.2)
    volume_scale: tuple[float, float] = (0.6, 1.4)
    # -- COB/COG --
    cob_offset_x: tuple[float, float] = (-0.02, 0.02)
    cob_offset_y: tuple[float, float] = (-0.02, 0.02)
    cob_offset_z: tuple[float, float] = (-0.04, 0.04)
    cog_offset_x: tuple[float, float] = (-0.02, 0.02)
    cog_offset_y: tuple[float, float] = (-0.02, 0.02)
    cog_offset_z: tuple[float, float] = (-0.04, 0.04)
    # -- Inertia / Mass --
    inertia_scale: tuple[float, float] = (0.3, 3.0)
    body_mass_scale: tuple[float, float] = (0.5, 1.5)
    water_density_range: tuple[float, float] = (970.0, 1050.0)
    # -- Payload (radius kept at 0.08 per user decision) --
    payload_mass_range: tuple[float, float] = (0.0, 5.0)
    payload_cog_offset_xy_radius: float = 0.08
    payload_cog_offset_z: tuple[float, float] = (-0.05, 0.0)
    # -- Joint Actuator --
    joint_stiffness_range: tuple[float, float] = (20.0, 200.0)
    joint_damping_range: tuple[float, float] = (0.1, 10.0)
    yaw_damping_scale: tuple[float, float] = (0.2, 2.0)
    joint_effort_limit_range: tuple[float, float] = (0.3, 1.0)
    joint_static_friction_range: tuple[float, float] = (0.0, 0.1)
    joint_viscous_friction_range: tuple[float, float] = (0.0, 0.5)
    # -- Thruster --
    thrust_coefficient_scale: tuple[float, float] = (0.3, 1.5)
    time_constant_scale: tuple[float, float] = (0.3, 2.0)
    # -- Ocean current (parent bounds (0,1); widened here to (0, 2)) --
    ocean_current_strength_range: tuple[float, float] = (0.0, 2.0)
    # -- Action latency (new, see albc_env.py port from hero_agent) --
    action_latency_range: tuple[int, int] = (0, 6)
```

**Note:** `action_latency_range` appears here but the field must also be declared on the parent `DomainRandomizationCfg`. See Task 4 Step 4.1.

- [ ] **Step 3.2: Syntax check**

Run from `/workspace/isaaclab`:
```bash
python3 -c "import ast; ast.parse(open('source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py').read())"
```
Expected: no output (success).

---

## Task 4: Add action_latency_range to DomainRandomizationCfg

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py:146` (end of parent DomainRandomizationCfg)

- [ ] **Step 4.1: Add action_latency_range to parent DR config**

File: `config.py`
Find (lines 142-146, the ocean current block at end of DomainRandomizationCfg):
```python
    # -- Ocean Current (DORAEMON-managed) --
    # Scalar strength [0, 1] multiplier on ocean_current.max_velocity.
    # DORAEMON nominal=0 (no current at curriculum start) -> expands as policy
    # masters easier variants. Bounds mirrored in HardDomainRandomizationCfg.
    ocean_current_strength_range: tuple[float, float] = (0.0, 1.0)
```

Replace with (append action_latency_range):
```python
    # -- Ocean Current (DORAEMON-managed) --
    # Scalar strength [0, 1] multiplier on ocean_current.max_velocity.
    # DORAEMON nominal=0 (no current at curriculum start) -> expands as policy
    # masters easier variants. Bounds mirrored in HardDomainRandomizationCfg.
    ocean_current_strength_range: tuple[float, float] = (0.0, 1.0)

    # -- Action latency (physics steps delay on policy output before env application) --
    # 0 = disabled (no delay). Range (lo, hi) sampled per-env at reset.
    # Physics dt = 5ms; 6 steps = 30ms delay, real-hardware communication scale.
    # Implemented via ring buffer in albc_env.py (ported from hero_agent).
    action_latency_range: tuple[int, int] = (0, 0)
```

Default `(0, 0)` on the parent means feature is OFF unless HardDR overrides it (Task 3 Step 3.1 sets `(0, 6)` on HardDR).

---

## Task 5: Expand ocean current noise_scale (6-channel)

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py:312-315` (ocean_current init in ALBCEnvCfg)

- [ ] **Step 5.1: Widen ocean_current.noise_scale and OU delta**

File: `config.py`
Find (lines 312-315):
```python
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.25, 0.0, 0.0, 0.0),
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )
```

Replace with:
```python
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.25, 0.0, 0.0, 0.0),
        # r14: angular channels activated, linear widened 2x. Adds rotational current
        # disturbance (torque DR). Previously zero-angular = purely translational.
        noise_scale=(0.2, 0.2, 0.1, 0.05, 0.05, 0.05),
    )
```

- [ ] **Step 5.2: Increase OU delta_scale (ocean current time-variance)**

File: `config.py`
Find (line 325):
```python
    delta_scale: float = 0.10
```

**STOP** — verify context. The `delta_scale: float = 0.10` at line 325 is the **arm joint delta** (`delta_scale` under ALBC Joint Control section, line 323-325), NOT the OU ocean delta. The OU params are at lines 383-388 (`ou_theta`, `ou_sigma`, `ou_enable`).

Re-read spec: "OU ocean current noise (delta_scale 0.1) 0.1 → 0.2". The spec miscalled the param — there's no `ou_delta_scale`. The OU process uses `ou_sigma=0.05` and `ou_theta=0.15`. Doubling the time-variance means raising `ou_sigma` (noise injection rate).

Actual change: `ou_sigma: 0.05 → 0.10` (doubles steady-state std from ~0.091 m/s to ~0.18 m/s, keeping ou_theta reversion rate constant). Additionally, `ou_enable: False → True` to actually activate mid-episode drift.

File: `config.py`
Find (lines 383-388):
```python
    # -- Ocean Current OU Drift --
    ou_theta: float = 0.15
    """OU mean reversion rate (1/s). 0.15 gives ~6.7s time constant."""
    ou_sigma: float = 0.05
    """OU noise scale (m/s per sqrt(s)). 0.05 gives steady-state std ~0.091 m/s."""
    ou_enable: bool = False
    """Enable OU process drift on ocean current (False = fixed per episode)."""
```

Replace with:
```python
    # -- Ocean Current OU Drift (r14: enabled, doubled noise) --
    ou_theta: float = 0.15
    """OU mean reversion rate (1/s). 0.15 gives ~6.7s time constant."""
    ou_sigma: float = 0.10
    """OU noise scale (m/s per sqrt(s)). 0.10 gives steady-state std ~0.18 m/s (r14: 2x)."""
    ou_enable: bool = True
    """Enable OU process drift on ocean current (r14: True for mid-episode time-variance)."""
```

---

## Task 6: Port action-latency buffer to albc_env.py

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py:212-220` (init)
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py:337-346` (update buffers)
- Modify: `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py:_reset_framework section` (reset hook)

Reference implementation (read-only): `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py:420-430, 540-561, 1143-1147`.

- [ ] **Step 6.1: Allocate latency buffer in _init_action_buffers**

File: `albc_env.py`
Find (lines 212-220):
```python
    def _init_action_buffers(self) -> None:
        """Action history (3-deep for smoothness penalty) and joint PD targets."""
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._nominal_joint_pos = torch.tensor(self.cfg.nominal_joint_pos, device=self.device)
        self._delta_scale = self.cfg.delta_scale
        self._joint_pos_targets = self._nominal_joint_pos.expand(self.num_envs, -1).clone()
        self._control_step_counter = 0
```

Replace with:
```python
    def _init_action_buffers(self) -> None:
        """Action history (3-deep for smoothness penalty) and joint PD targets.

        Also allocates the action-latency ring buffer when enabled via
        ``randomization.action_latency_range`` (max > 0). The buffer stores
        recent raw actions so that each env can retrieve a delayed action
        indexed by its per-env latency (sampled at reset).
        """
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._nominal_joint_pos = torch.tensor(self.cfg.nominal_joint_pos, device=self.device)
        self._delta_scale = self.cfg.delta_scale
        self._joint_pos_targets = self._nominal_joint_pos.expand(self.num_envs, -1).clone()
        self._control_step_counter = 0

        # Action latency ring buffer (ported from hero_agent/base_env.py).
        # max_latency > 0 allocates buffer; otherwise feature is disabled.
        max_latency = self.cfg.randomization.action_latency_range[1]
        self._max_action_latency = max_latency
        if max_latency > 0:
            self._action_history = torch.zeros(
                self.num_envs, max_latency + 1, self.cfg.action_space, device=self.device
            )
            self._action_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self._action_history = None
            self._action_latency = None
```

- [ ] **Step 6.2: Apply delay in _update_action_buffers**

File: `albc_env.py`
Find (lines 337-346):
```python
    def _update_action_buffers(self, actions: torch.Tensor) -> None:
        """Update action history buffers. Called at the start of _pre_physics_step().

        Args:
            actions: Raw actions from RL. Shape: (num_envs, action_space).
        """
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._control_step_counter += 1
```

Replace with:
```python
    def _update_action_buffers(self, actions: torch.Tensor) -> None:
        """Update action history buffers. Called at the start of _pre_physics_step().

        When action-latency DR is enabled (``randomization.action_latency_range[1] > 0``),
        applies a per-env delay: stores the current raw action into the ring buffer
        and reads back the action N physics-steps old (N sampled per-env at reset).

        Args:
            actions: Raw actions from RL. Shape: (num_envs, action_space).
        """
        clamped = actions.clamp(-1.0, 1.0)

        # Apply action latency delay (ported from hero_agent/base_env.py:540-561).
        if self._action_history is not None and self.cfg.randomization.enable:
            if self._action_history.shape[1] > 1:
                self._action_history[:, 1:] = self._action_history[:, :-1].clone()
            self._action_history[:, 0] = clamped
            env_idx = torch.arange(self.num_envs, device=self.device)
            delayed = self._action_history[env_idx, self._action_latency]
        else:
            delayed = clamped

        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = delayed.clone()
        self._control_step_counter += 1
```

- [ ] **Step 6.3: Sample per-env latency at reset**

File: `albc_env.py`
Find `_reset_action_buffers` method (lines 1120-1140):
```python
    def _reset_action_buffers(self, env_ids: torch.Tensor) -> None:
        """Reset action buffers, temporal history, and cumulative yaw."""
        for buf in (self._actions, self._prev_actions, self._prev_prev_actions):
            buf[env_ids] = 0.0
        self._joint_pos_targets[env_ids] = self._robot.data.joint_pos[env_ids][:, self._albc_joint_ids]
        if self._hist_buf is not None:
            self._hist_buf[env_ids] = 0.0
            self._hist_step_counter[env_ids] = 0

        # Reset cumulative yaw tracking
        self._cumulative_yaw[env_ids] = 0.0
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        self._prev_yaw[env_ids] = yaw[env_ids]

        # Reset previous-step velocity buffers (settling cost constraints)
        self._prev_root_lin_vel_b[env_ids] = self._robot.data.root_lin_vel_b[env_ids]
        self._prev_root_ang_vel_z[env_ids] = self._robot.data.root_ang_vel_b[env_ids, 2]

        # Reset mid-episode payload toggle state
        self._payload_toggle_counter[env_ids] = 0
        self._payload_toggled[env_ids] = False
```

Replace with (append latency reset before the method ends):
```python
    def _reset_action_buffers(self, env_ids: torch.Tensor) -> None:
        """Reset action buffers, temporal history, and cumulative yaw."""
        for buf in (self._actions, self._prev_actions, self._prev_prev_actions):
            buf[env_ids] = 0.0
        self._joint_pos_targets[env_ids] = self._robot.data.joint_pos[env_ids][:, self._albc_joint_ids]
        if self._hist_buf is not None:
            self._hist_buf[env_ids] = 0.0
            self._hist_step_counter[env_ids] = 0

        # Reset cumulative yaw tracking
        self._cumulative_yaw[env_ids] = 0.0
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        self._prev_yaw[env_ids] = yaw[env_ids]

        # Reset previous-step velocity buffers (settling cost constraints)
        self._prev_root_lin_vel_b[env_ids] = self._robot.data.root_lin_vel_b[env_ids]
        self._prev_root_ang_vel_z[env_ids] = self._robot.data.root_ang_vel_b[env_ids, 2]

        # Reset mid-episode payload toggle state
        self._payload_toggle_counter[env_ids] = 0
        self._payload_toggled[env_ids] = False

        # Reset action latency: clear history and resample per-env latency (r14).
        # Ported from hero_agent/base_env.py:1143-1147.
        if self._action_history is not None and self._action_latency is not None:
            lo, hi = self.cfg.randomization.action_latency_range
            self._action_history[env_ids] = 0.0
            self._action_latency[env_ids] = torch.randint(
                lo, hi + 1, (len(env_ids),), device=self.device
            )
```

- [ ] **Step 6.4: Syntax check**

Run from `/workspace/isaaclab`:
```bash
python3 -c "import ast; ast.parse(open('source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py').read())"
```
Expected: no output.

- [ ] **Step 6.5: Env instantiation smoke test (no simulation step)**

Run from `/workspace/isaaclab`:
```bash
./isaaclab.sh -p scripts/demos/test_full_dof_env.py --headless --num_envs 4 2>&1 | tail -30
```
Expected: env constructs and prints obs/privileged shapes; no tracebacks involving `_action_history` or `action_latency_range`.

---

## Task 7: Pre-launch sanity checks

- [ ] **Step 7.1: Check encoder latent dim propagates**

Run from `/workspace/isaaclab`:
```bash
python3 -c "
from isaaclab_tasks.direct.constrained_full_albc.agents.rsl_rl_ppo_cfg import _FullDOFPolicyCfg
p = _FullDOFPolicyCfg()
assert p.encoder_latent_dim == 16, p.encoder_latent_dim
print('OK latent=', p.encoder_latent_dim)
"
```
Expected: `OK latent= 16`

- [ ] **Step 7.2: Check entropy_coef_per_dim**

Run from `/workspace/isaaclab`:
```bash
python3 -c "
from isaaclab_tasks.direct.constrained_full_albc.agents.rsl_rl_ppo_cfg import RslRlConstraintTRPOAlgorithmCfg
a = RslRlConstraintTRPOAlgorithmCfg()
assert a.entropy_coef_per_dim[2:] == (0.0005,) * 6, a.entropy_coef_per_dim
print('OK entropy thruster=', a.entropy_coef_per_dim[2])
"
```
Expected: `OK entropy thruster= 0.0005`

- [ ] **Step 7.3: Check HardDR widened values**

Run from `/workspace/isaaclab`:
```bash
python3 -c "
from isaaclab_tasks.direct.constrained_full_albc.config import HardDomainRandomizationCfg
h = HardDomainRandomizationCfg()
checks = [
    ('thrust_coefficient_scale', h.thrust_coefficient_scale, (0.3, 1.5)),
    ('time_constant_scale', h.time_constant_scale, (0.3, 2.0)),
    ('yaw_damping_scale', h.yaw_damping_scale, (0.2, 2.0)),
    ('body_mass_scale', h.body_mass_scale, (0.5, 1.5)),
    ('joint_effort_limit_range', h.joint_effort_limit_range, (0.3, 1.0)),
    ('ocean_current_strength_range', h.ocean_current_strength_range, (0.0, 2.0)),
    ('payload_mass_range', h.payload_mass_range, (0.0, 5.0)),
    ('action_latency_range', h.action_latency_range, (0, 6)),
    ('payload_cog_offset_xy_radius', h.payload_cog_offset_xy_radius, 0.08),
]
for name, got, expected in checks:
    assert got == expected, f'{name}: got {got} expected {expected}'
    print(f'OK {name}={got}')
"
```
Expected: 9 `OK` lines.

- [ ] **Step 7.4: Check DORAEMON step_interval**

Run from `/workspace/isaaclab`:
```bash
python3 -c "
from isaaclab_tasks.direct.constrained_full_albc.config import ALBCEnvCfg
c = ALBCEnvCfg()
assert c.doraemon.step_interval == 500, c.doraemon.step_interval
assert c.ou_enable is True
assert c.ou_sigma == 0.10
print('OK doraemon.step_interval=', c.doraemon.step_interval, 'ou_sigma=', c.ou_sigma)
"
```
Expected: `OK doraemon.step_interval= 500 ou_sigma= 0.10`

---

## Task 8: Create launch script

**Files:**
- Create: `scripts/launch_r14.sh`

- [ ] **Step 8.1: Write launch script**

File: `scripts/launch_r14.sh`
```bash
#!/usr/bin/env bash
# r14 final training run: r13_B baseline + entropy reduction + aggressive DR + action latency.
# See docs/superpowers/specs/2026-04-21-r14-final-design.md.
set -e

cd /workspace/isaaclab

EXPERIMENT_NAME="r14"
LOG_DIR_ROOT="/workspace/isaaclab/logs/rsl_rl/full_dof_trpo"
STAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="/workspace/isaaclab/logs/archive/launch_scripts/${EXPERIMENT_NAME}_${STAMP}.log"
mkdir -p "$(dirname "$STDOUT_LOG")"

echo "[r14 $(date)] START experiment_name=${EXPERIMENT_NAME}" | tee -a "$STDOUT_LOG"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-FullDOF-TRPO-v0 \
    --num_envs 4096 \
    --max_iterations 20000 \
    --headless \
    --logger wandb \
    --log_project_name full_dof_trpo \
    --experiment_name "${EXPERIMENT_NAME}" \
    2>&1 | tee -a "$STDOUT_LOG"
RC=${PIPESTATUS[0]}
echo "[r14 $(date)] END rc=${RC}" | tee -a "$STDOUT_LOG"
exit "$RC"
```

- [ ] **Step 8.2: chmod executable**

```bash
chmod +x /workspace/isaaclab/scripts/launch_r14.sh
```

---

## Task 9: Commit all changes

- [ ] **Step 9.1: Stage and commit**

Run from `/workspace/isaaclab`:
```bash
git add source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/albc_env.py \
        scripts/launch_r14.sh \
        docs/superpowers/plans/2026-04-21-r14-final-implementation.md

git commit -m "$(cat <<'EOF'
feat(r14): final run config - entropy reduction + aggressive DR + action latency

- encoder_latent_dim 9 -> 16 (r13_B baseline)
- entropy_coef_per_dim thrusters 0.001 -> 0.0005 (attack roll osc root cause)
- save_interval 50 -> 100, DORAEMON step_interval 250 -> 500
- HardDR widened 1.5-3x on 13 params (thrust/time/yaw_damp/mass/etc.)
- Ocean current noise_scale 6-channel expanded, OU enabled with 2x sigma
- Action latency DR (0-30ms) ported from hero_agent/base_env.py

Spec:  docs/superpowers/specs/2026-04-21-r14-final-design.md
Plan:  docs/superpowers/plans/2026-04-21-r14-final-implementation.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9.2: Verify commit**

```bash
git log --oneline -3
```
Expected: newest commit is the r14 feat commit.

---

## Launch (user action, not part of plan execution)

After plan completion the user runs:
```bash
nohup /workspace/isaaclab/scripts/launch_r14.sh > /workspace/isaaclab/logs/archive/launch_scripts/r14_nohup.out 2>&1 &
```

Expected wall-clock: ~30-40 hours on one GPU (4x more iters than r13_B at 2x envs).

## Verification during training

At iter ~2000:
```bash
python3 /workspace/.claude/skills/train-analyze/analyze_training.py --deep \
    $(ls -td /workspace/isaaclab/logs/rsl_rl/full_dof_trpo/*_r14 | head -1)
```
Watch for:
- `Policy/mean_noise_std` dropping below 0.20 (sign entropy reduction is working)
- `DORAEMON/entropy_before` not collapsing below -30 (exploration floor)
- Constraint violations stable (not runaway due to aggressive DR)
- `line_search_success >= 0.5`

At iter ~5000 and ~10000: re-run same command, track trends.
